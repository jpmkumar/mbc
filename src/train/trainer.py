"""Two-stage training: classical head first, then VQC head."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.preprocessing import preprocess_cache_tag
from src.utils.metrics import compute_metrics

from .accelerator import amp_device_type, configure_runtime, maybe_compile_model
from .feature_cache import build_feature_loader, extract_compressed_features
from .losses import compute_class_weights


def filter_compatible_state_dict(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop checkpoint keys whose tensor shapes differ from the current model."""
    model_sd = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        if key not in model_sd:
            continue
        if model_sd[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    return filtered, skipped


class HybridTrainer:
    STAGES = ("stage_a", "stage_b", "stage_c")

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        test_loader=None,
        device: str | None = None,
        experiment_name: str = "default",
        train_eval_loader=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_eval_loader = train_eval_loader
        self.config = config
        train_cfg = config.get("training", {})

        self._setup_devices(device, train_cfg)
        self.experiment_name = experiment_name
        self.results_dir = Path(config.get("paths", {}).get("results", "results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(
            config.get("paths", {}).get("checkpoints", "results/checkpoints")
        )
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.feature_cache_dir = self.results_dir / "feature_cache"
        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

        self.stage_a_epochs = train_cfg.get("stage_a_epochs", 20)
        self.stage_b_epochs = train_cfg.get("stage_b_epochs", 30)
        self.stage_c_epochs = train_cfg.get("stage_c_epochs", 10)
        self.freeze_encoder_epochs = train_cfg.get("freeze_encoder_epochs", 5)
        self.freeze_backbone_in_stage_b = train_cfg.get(
            "freeze_backbone_in_stage_b", True
        )
        self.backbone_eval_in_stage_b = train_cfg.get(
            "backbone_eval_in_stage_b", True
        )
        self.cache_frozen_backbone_features = train_cfg.get(
            "cache_frozen_backbone_features", True
        )
        self.quantum_weight_decay = train_cfg.get(
            "quantum_weight_decay", train_cfg.get("weight_decay", 1e-4)
        )
        self.selection_metric = train_cfg.get(
            "selection_metric", "balanced_accuracy"
        )
        self.val_interval = max(1, int(train_cfg.get("val_interval", 1)))
        self.checkpoint_interval = max(
            1, int(train_cfg.get("checkpoint_interval", 1))
        )
        self.batch_size = train_cfg.get("batch_size", 16)

        runtime = configure_runtime(config)
        self.use_amp = runtime["use_amp"]
        self.amp_device_type = amp_device_type(self.classical_device)
        self.scaler = torch.amp.GradScaler(
            self.amp_device_type, enabled=self.use_amp
        )
        self.cache_batch_size = runtime["cache_batch_size"]
        if self.use_amp:
            print(f"Mixed precision (AMP) enabled on {self.amp_device_type}")

        use_weights = train_cfg.get("use_class_weights", True)
        malignant_mult = float(train_cfg.get("malignant_weight_multiplier", 1.0))
        if use_weights:
            weights = compute_class_weights(
                train_loader,
                malignant_multiplier=malignant_mult,
            ).to(self.classical_device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_metrics": []}
        self.best_score = -1.0
        self.best_state = None
        self.best_stage = "stage_a"
        self.stage_epochs_done = {s: 0 for s in self.STAGES}
        self.total_epochs = 0
        self._feature_loaders: dict[str, DataLoader] = {}

    @property
    def device(self):
        """Primary device for labels, loss, and classical modules."""
        return self.classical_device

    def _setup_devices(self, device: str | None, train_cfg: dict):
        requested = device or train_cfg.get("classical_device", "auto")
        if requested == "auto":
            classical = self._default_device()
        else:
            classical = requested

        if getattr(self.model, "use_quantum", False):
            self.classical_device = torch.device(classical)
            self.quantum_device = torch.device(
                train_cfg.get("quantum_device", "cpu")
            )
            if hasattr(self.model, "set_devices"):
                self.model.set_devices(self.classical_device, self.quantum_device)
            else:
                self.model.to(self.classical_device)
            print(
                f"Hybrid devices: classical={self.classical_device}, "
                f"quantum={self.quantum_device}"
            )
        else:
            self.classical_device = torch.device(classical)
            self.quantum_device = self.classical_device
            self.model.to(self.classical_device)

    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_stage_plan(self, stages_filter: list[str] | None) -> list[tuple[str, int]]:
        stages = [
            ("stage_a", self.stage_a_epochs),
            ("stage_b", self.stage_b_epochs),
            ("stage_c", self.stage_c_epochs),
        ]
        if not getattr(self.model, "use_quantum", False):
            stages = [("stage_a", self.stage_a_epochs + self.stage_b_epochs)]

        if stages_filter:
            allowed = set(stages_filter)
            stages = [(name, n) for name, n in stages if name in allowed]
        return stages

    def _configure_stage(self, stage: str):
        if hasattr(self.model, "set_training_stage"):
            self.model.set_training_stage(stage)

        if not getattr(self.model, "use_quantum", False):
            return

        if stage == "stage_a":
            self.model.set_backbone_trainable(True)
            self.model.set_classical_head_trainable(True)
            self.model.set_vqc_head_trainable(False)
            if hasattr(self.model, "set_backbone_eval_mode"):
                self.model.set_backbone_eval_mode(False)
        elif stage == "stage_b":
            freeze_backbone = self.freeze_backbone_in_stage_b
            self.model.set_backbone_trainable(not freeze_backbone)
            self.model.set_classical_head_trainable(False)
            self.model.set_vqc_head_trainable(True)
            if hasattr(self.model, "set_backbone_eval_mode"):
                self.model.set_backbone_eval_mode(
                    freeze_backbone and self.backbone_eval_in_stage_b
                )
        else:
            if hasattr(self.model, "set_backbone_eval_mode"):
                self.model.set_backbone_eval_mode(False)
            self.model.set_backbone_trainable(True)
            self.model.set_classical_head_trainable(False)
            self.model.set_vqc_head_trainable(True)
        self._feature_loaders.clear()

    def _should_cache_features(self, stage: str) -> bool:
        return (
            stage == "stage_b"
            and self.cache_frozen_backbone_features
            and self.freeze_backbone_in_stage_b
            and getattr(self.model, "use_quantum", False)
        )

    def _feature_cache_path(self) -> Path:
        tag = preprocess_cache_tag(self.config.get("data", {}).get("preprocessing"))
        return self.feature_cache_dir / f"{self.experiment_name}_{tag}_features.pt"

    def _prepare_feature_loaders(self, stage: str):
        if not self._should_cache_features(stage):
            return

        cache_path = self._feature_cache_path()
        if cache_path.exists():
            print(f"Loading cached Stage B features from {cache_path.parent}")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            self._feature_loaders["train"] = build_feature_loader(
                cached["train"], self.batch_size, shuffle=True
            )
            self._feature_loaders["val"] = build_feature_loader(
                cached["val"], self.batch_size, shuffle=False
            )
            if cached.get("test") is not None:
                self._feature_loaders["test"] = build_feature_loader(
                    cached["test"], self.batch_size, shuffle=False
                )
            print(
                f"  train={len(cached['train']['features'])} "
                f"val={len(cached['val']['features'])} samples"
            )
            return

        print("Pre-extracting frozen backbone features (one-time, ~2-5 min on GPU)...")
        train_source = self.train_eval_loader or self.train_loader
        cache_loader = self._loader_with_batch_size(train_source, self.cache_batch_size)
        val_cache_loader = self._loader_with_batch_size(
            self.val_loader, self.cache_batch_size
        )
        cached_train = extract_compressed_features(
            self.model,
            cache_loader,
            self.classical_device,
            desc="Cache train",
            use_amp=self.use_amp,
            amp_device_type=self.amp_device_type,
        )
        cached_val = extract_compressed_features(
            self.model,
            val_cache_loader,
            self.classical_device,
            desc="Cache val",
            use_amp=self.use_amp,
            amp_device_type=self.amp_device_type,
        )
        cached_test = None
        if self.test_loader is not None:
            test_cache_loader = self._loader_with_batch_size(
                self.test_loader, self.cache_batch_size
            )
            cached_test = extract_compressed_features(
                self.model,
                test_cache_loader,
                self.classical_device,
                desc="Cache test",
                use_amp=self.use_amp,
                amp_device_type=self.amp_device_type,
            )

        payload = {"train": cached_train, "val": cached_val, "test": cached_test}
        torch.save(payload, cache_path)
        print(f"Saved feature cache to {cache_path}")

        self._feature_loaders["train"] = build_feature_loader(
            cached_train, self.batch_size, shuffle=True
        )
        self._feature_loaders["val"] = build_feature_loader(
            cached_val, self.batch_size, shuffle=False
        )
        if cached_test is not None:
            self._feature_loaders["test"] = build_feature_loader(
                cached_test, self.batch_size, shuffle=False
            )

    def _loader_with_batch_size(
        self, loader: DataLoader, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        if getattr(loader, "batch_size", None) == batch_size:
            return loader
        kwargs: dict = {
            "dataset": loader.dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": loader.num_workers,
            "pin_memory": loader.pin_memory,
        }
        if loader.num_workers > 0:
            kwargs["prefetch_factor"] = getattr(loader, "prefetch_factor", 2)
            kwargs["persistent_workers"] = True
        return DataLoader(**kwargs)

    def _compute_logits(self, batch, use_cache: bool, stage: str):
        if use_cache:
            features, labels, _mods = batch
            labels = labels.to(self.classical_device, non_blocking=True)
            return self.model.forward_from_features(features), labels

        images = batch["image"].to(self.classical_device, non_blocking=True)
        labels = batch["label"].to(self.classical_device, non_blocking=True)
        modality_ids = batch["modality_id"].to(
            self.classical_device, non_blocking=True
        )

        quantum_stage = (
            getattr(self.model, "use_quantum", False)
            and stage in ("stage_b", "stage_c")
            and hasattr(self.model, "forward_from_features")
            and not getattr(self.model, "_use_classical_head", True)
        )

        if quantum_stage and self.use_amp:
            with torch.autocast(self.amp_device_type, enabled=True):
                compressed = self.model.forward_features(images, modality_ids)
            logits = self.model.forward_from_features(compressed)
            return logits, labels

        with torch.autocast(self.amp_device_type, enabled=self.use_amp):
            logits = self.model(images, modality_ids)
        return logits, labels

    def _run_epoch(self, optimizer, epoch: int, stage: str):
        self.model.train()
        if hasattr(self.model, "set_backbone_eval_mode") and stage == "stage_b":
            self.model.set_backbone_eval_mode(
                self.freeze_backbone_in_stage_b and self.backbone_eval_in_stage_b
            )
        if stage == "stage_a":
            if hasattr(self.model, "encoder") and epoch < self.freeze_encoder_epochs:
                self.model.encoder.freeze_early(num_blocks=5)
            elif hasattr(self.model, "encoder"):
                self.model.encoder.unfreeze_all()
        elif stage != "stage_b" or not self.freeze_backbone_in_stage_b:
            if hasattr(self.model, "encoder"):
                self.model.encoder.unfreeze_all()

        use_cache = "train" in self._feature_loaders
        loader = (
            self._feature_loaders["train"] if use_cache else self.train_loader
        )
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"{stage} epoch {epoch+1}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            logits, labels = self._compute_logits(batch, use_cache, stage)
            loss = self.criterion(logits, labels)

            if self.use_amp and not use_cache:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader=None, split: str | None = None) -> dict:
        if split and split in self._feature_loaders:
            loader = self._feature_loaders[split]
            return self._evaluate_cached(loader)

        loader = loader or self.val_loader
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        for batch in loader:
            images = batch["image"].to(self.classical_device, non_blocking=True)
            labels = batch["label"].to(self.classical_device, non_blocking=True)
            modality_ids = batch["modality_id"].to(
                self.classical_device, non_blocking=True
            )
            with torch.autocast(self.amp_device_type, enabled=self.use_amp):
                logits = self.model(images, modality_ids)
            probs = torch.softmax(logits.float(), dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        return compute_metrics(all_labels, all_preds, all_probs)

    @torch.no_grad()
    def _evaluate_cached(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        for features, labels, _mods in loader:
            labels = labels.to(self.classical_device, non_blocking=True)
            logits = self.model.forward_from_features(features)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        return compute_metrics(all_labels, all_preds, all_probs)

    def _score(self, metrics: dict) -> float:
        rate = metrics.get("pred_positive_rate", 0.5)
        if rate >= 0.95 or rate <= 0.05:
            return -1.0
        return float(metrics.get(self.selection_metric, metrics.get("f1", 0.0)))

    def _get_param_groups(self, stage: str):
        train_cfg = self.config.get("training", {})
        classical_params = []
        quantum_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ("quantum_layer", "angle_encoder", "feature_norm")):
                quantum_params.append(param)
            elif "head" in name and "classical_head" not in name:
                quantum_params.append(param)
            else:
                classical_params.append(param)

        if stage == "stage_a" or not quantum_params:
            return [
                {
                    "params": classical_params,
                    "lr": train_cfg.get("lr_classical", 1e-4),
                }
            ]
        if stage == "stage_b":
            groups = []
            if classical_params:
                groups.append(
                    {
                        "params": classical_params,
                        "lr": train_cfg.get("lr_classical", 1e-4),
                        "weight_decay": train_cfg.get("weight_decay", 1e-4),
                    }
                )
            groups.append(
                {
                    "params": quantum_params,
                    "lr": train_cfg.get("lr_quantum", 1e-4),
                    "weight_decay": self.quantum_weight_decay,
                }
            )
            return groups
        return [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": train_cfg.get("lr_joint", 1e-5),
            }
        ]

    def _latest_ckpt_path(self) -> Path:
        return self.ckpt_dir / f"{self.experiment_name}_latest.pt"

    def _best_ckpt_path(self) -> Path:
        return self.ckpt_dir / f"{self.experiment_name}.pt"

    def _progress_path(self) -> Path:
        return self.results_dir / f"{self.experiment_name}_progress.json"

    def _save_checkpoint(self, stage_name: str):
        payload = {
            "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "best_state_dict": self.best_state,
            "best_score": self.best_score,
            "total_epochs": self.total_epochs,
            "stage_epochs_done": dict(self.stage_epochs_done),
            "history": self.history,
            "experiment_name": self.experiment_name,
            "current_stage": stage_name,
            "best_stage": self.best_stage,
        }
        torch.save(payload, self._latest_ckpt_path())
        if self.best_state:
            torch.save(self.best_state, self._best_ckpt_path())

        progress = {
            "experiment": self.experiment_name,
            "total_epochs": self.total_epochs,
            "stage_epochs_done": self.stage_epochs_done,
            "stage_targets": {
                "stage_a": self.stage_a_epochs,
                "stage_b": self.stage_b_epochs,
                "stage_c": self.stage_c_epochs,
            },
            "best_score": self.best_score,
            "best_stage": self.best_stage,
            "latest_checkpoint": str(self._latest_ckpt_path()),
            "best_checkpoint": str(self._best_ckpt_path()),
        }
        with open(self._progress_path(), "w") as f:
            json.dump(progress, f, indent=2)

    def _load_state_dict_lenient(
        self, state_dict: dict[str, torch.Tensor], label: str = "checkpoint"
    ) -> dict[str, torch.Tensor]:
        filtered, skipped = filter_compatible_state_dict(self.model, state_dict)
        incompat = self.model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"  {label}: skipped {len(skipped)} shape-mismatch keys")
            for key in skipped[:6]:
                print(f"    - {key}")
            if len(skipped) > 6:
                print(f"    ... and {len(skipped) - 6} more")
        if incompat.missing_keys:
            print(
                f"  {label}: {len(incompat.missing_keys)} keys kept at init "
                f"(new or incompatible layers)"
            )
        if incompat.unexpected_keys:
            print(f"  {label}: ignored {len(incompat.unexpected_keys)} unexpected keys")
        return filtered

    def _merge_best_state(self, state_dict: dict[str, torch.Tensor] | None) -> dict:
        """Build a full best_state after partial load (e.g. new VQC head)."""
        merged = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        if state_dict:
            filtered, _ = filter_compatible_state_dict(self.model, state_dict)
            merged.update({k: v.cpu().clone() for k, v in filtered.items()})
        return merged

    def _load_checkpoint(self, resume_path: Path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self._load_state_dict_lenient(ckpt["model_state_dict"], "model")
            self.best_state = self._merge_best_state(ckpt.get("best_state_dict"))
            self.best_score = ckpt.get("best_score", -1.0)
            self.best_stage = ckpt.get("best_stage", "stage_a")
            self.total_epochs = ckpt.get("total_epochs", 0)
            self.stage_epochs_done.update(ckpt.get("stage_epochs_done", {}))
            self.history = ckpt.get("history", self.history)
            print(f"Resumed from {resume_path}")
            print(f"  total_epochs={self.total_epochs}, stage progress={self.stage_epochs_done}")
            return

        if isinstance(ckpt, dict):
            self._load_state_dict_lenient(ckpt, "legacy weights")
        else:
            raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")
        self.best_state = self._merge_best_state(ckpt if isinstance(ckpt, dict) else None)
        print(f"Loaded legacy weights from {resume_path}")

    def train(
        self,
        stages_filter: list[str] | None = None,
        resume_path: str | Path | None = None,
    ) -> dict:
        if resume_path:
            self._load_checkpoint(Path(resume_path))

        stages = self._build_stage_plan(stages_filter)
        if not stages:
            raise ValueError("No training stages selected.")

        t0 = time.time()
        for stage_name, n_epochs in stages:
            already_done = self.stage_epochs_done.get(stage_name, 0)
            if already_done >= n_epochs:
                print(f"Skipping {stage_name} ({already_done}/{n_epochs} epochs done)")
                continue

            self._configure_stage(stage_name)
            self._prepare_feature_loaders(stage_name)
            param_groups = self._get_param_groups(stage_name)
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.get("training", {}).get("weight_decay", 1e-4),
            )

            for epoch in range(already_done, n_epochs):
                loss = self._run_epoch(optimizer, self.total_epochs, stage_name)
                epoch_num = epoch + 1
                val_metrics = None

                if epoch_num % self.val_interval == 0 or epoch_num == n_epochs:
                    val_metrics = self.evaluate(split="val" if self._feature_loaders else None)
                    self.history["train_loss"].append(loss)
                    self.history["val_metrics"].append(
                        {
                            k: v
                            for k, v in val_metrics.items()
                            if k not in ("labels", "preds", "probs", "roc")
                        }
                    )

                    score = self._score(val_metrics)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_stage = stage_name
                        self.best_state = {
                            k: v.cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }

                self.total_epochs += 1
                self.stage_epochs_done[stage_name] = epoch_num

                if epoch_num % self.checkpoint_interval == 0 or epoch_num == n_epochs:
                    self._save_checkpoint(stage_name)

                metric_str = ""
                if val_metrics:
                    metric_str = (
                        f"val {self.selection_metric}="
                        f"{val_metrics.get(self.selection_metric, 0):.3f} | "
                    )
                print(
                    f"{stage_name} epoch {epoch_num}/{n_epochs} | "
                    f"loss={loss:.4f} | {metric_str}best={self.best_score:.3f}"
                )

        if self.best_state:
            filtered, _ = filter_compatible_state_dict(self.model, self.best_state)
            self.model.load_state_dict(filtered, strict=False)
            if hasattr(self.model, "set_training_stage"):
                self.model.set_training_stage(self.best_stage)

        if self._feature_loaders.get("test"):
            final_metrics = self.evaluate(split="test")
            split_name = "test"
        elif self.test_loader is not None:
            final_metrics = self.evaluate(self.test_loader)
            split_name = "test"
        else:
            final_metrics = self.evaluate(split="val" if self._feature_loaders else None)
            split_name = "val"

        final_metrics["train_time_s"] = time.time() - t0
        final_metrics["eval_split"] = split_name
        final_metrics["selection_metric"] = self.selection_metric
        final_metrics["best_val_score"] = self.best_score
        final_metrics["best_stage"] = self.best_stage
        final_metrics["stage_epochs_done"] = dict(self.stage_epochs_done)

        serializable = {
            k: v
            for k, v in final_metrics.items()
            if k not in ("labels", "preds", "probs", "roc")
        }
        out_path = self.results_dir / f"{self.experiment_name}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)

        if self.best_state:
            torch.save(self.best_state, self._best_ckpt_path())
        self._save_checkpoint(stages[-1][0])

        history_path = self.results_dir / f"{self.experiment_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return serializable

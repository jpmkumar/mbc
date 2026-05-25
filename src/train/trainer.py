"""Two-stage training: classical head first, then VQC head."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.metrics import compute_metrics

from .losses import compute_class_weights
from .seed import set_seed


class HybridTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        test_loader=None,
        device: str | None = None,
        experiment_name: str = "default",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device or self._default_device()
        if getattr(model, "use_quantum", False):
            self.device = "cpu"  # PennyLane default.qubit requires CPU tensors
        self.model.to(self.device)
        self.experiment_name = experiment_name
        self.results_dir = Path(config.get("paths", {}).get("results", "results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config.get("training", {})
        self.stage_a_epochs = train_cfg.get("stage_a_epochs", 20)
        self.stage_b_epochs = train_cfg.get("stage_b_epochs", 30)
        self.stage_c_epochs = train_cfg.get("stage_c_epochs", 10)
        self.freeze_encoder_epochs = train_cfg.get("freeze_encoder_epochs", 5)
        self.freeze_backbone_in_stage_b = train_cfg.get(
            "freeze_backbone_in_stage_b", True
        )
        self.selection_metric = train_cfg.get(
            "selection_metric", "balanced_accuracy"
        )

        use_weights = train_cfg.get("use_class_weights", True)
        if use_weights:
            weights = compute_class_weights(train_loader).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_metrics": []}

    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _configure_stage(self, stage: str):
        if hasattr(self.model, "set_training_stage"):
            self.model.set_training_stage(stage)

        if not getattr(self.model, "use_quantum", False):
            return

        if stage == "stage_a":
            self.model.set_backbone_trainable(True)
            self.model.set_classical_head_trainable(True)
            self.model.set_vqc_head_trainable(False)
        elif stage == "stage_b":
            freeze_backbone = self.freeze_backbone_in_stage_b
            self.model.set_backbone_trainable(not freeze_backbone)
            self.model.set_classical_head_trainable(False)
            self.model.set_vqc_head_trainable(True)
        else:
            self.model.set_backbone_trainable(True)
            self.model.set_classical_head_trainable(False)
            self.model.set_vqc_head_trainable(True)

    def _run_epoch(self, optimizer, epoch: int, stage: str):
        self.model.train()
        if hasattr(self.model, "encoder") and epoch < self.freeze_encoder_epochs:
            self.model.encoder.freeze_early(num_blocks=5)
        elif hasattr(self.model, "encoder"):
            self.model.encoder.unfreeze_all()

        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"{stage} epoch {epoch+1}", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            modality_ids = batch["modality_id"].to(self.device)

            optimizer.zero_grad()
            logits = self.model(images, modality_ids)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def evaluate(self, loader=None) -> dict:
        loader = loader or self.val_loader
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            modality_ids = batch["modality_id"].to(self.device)
            logits = self.model(images, modality_ids)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        return compute_metrics(all_labels, all_preds, all_probs)

    def _score(self, metrics: dict) -> float:
        rate = metrics.get("pred_positive_rate", 0.5)
        # Reject all-one-class predictions (collapse)
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
            if "quantum_layer" in name or "angle_encoder" in name:
                quantum_params.append(param)
            elif "head" in name and "classical_head" not in name:
                if getattr(self.model, "use_quantum", False):
                    quantum_params.append(param)
                else:
                    classical_params.append(param)
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
                    }
                )
            groups.append(
                {
                    "params": quantum_params,
                    "lr": train_cfg.get("lr_quantum", 1e-4),
                }
            )
            return groups
        return [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": train_cfg.get("lr_joint", 1e-5),
            }
        ]

    def train(self) -> dict:
        best_score = -1.0
        best_state = None
        total_epochs = 0

        stages = [
            ("stage_a", self.stage_a_epochs),
            ("stage_b", self.stage_b_epochs),
            ("stage_c", self.stage_c_epochs),
        ]

        if not getattr(self.model, "use_quantum", False):
            stages = [("stage_a", self.stage_a_epochs + self.stage_b_epochs)]

        t0 = time.time()
        for stage_name, n_epochs in stages:
            self._configure_stage(stage_name)
            param_groups = self._get_param_groups(stage_name)
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.get("training", {}).get("weight_decay", 1e-4),
            )

            for epoch in range(n_epochs):
                loss = self._run_epoch(optimizer, total_epochs, stage_name)
                val_metrics = self.evaluate()
                self.history["train_loss"].append(loss)
                self.history["val_metrics"].append(val_metrics)

                score = self._score(val_metrics)
                if score > best_score:
                    best_score = score
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }

                total_epochs += 1

        if best_state:
            self.model.load_state_dict(best_state)

        eval_loader = self.test_loader or self.val_loader
        split_name = "test" if self.test_loader is not None else "val"
        final_metrics = self.evaluate(eval_loader)
        final_metrics["train_time_s"] = time.time() - t0
        final_metrics["eval_split"] = split_name
        final_metrics["selection_metric"] = self.selection_metric
        final_metrics["best_val_score"] = best_score

        serializable = {
            k: v
            for k, v in final_metrics.items()
            if k not in ("labels", "preds", "probs", "roc")
        }
        out_path = self.results_dir / f"{self.experiment_name}_metrics.json"
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)

        ckpt_dir = Path(
            self.config.get("paths", {}).get("checkpoints", "results/checkpoints")
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_dir / f"{self.experiment_name}.pt")

        history_path = self.results_dir / f"{self.experiment_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        return serializable

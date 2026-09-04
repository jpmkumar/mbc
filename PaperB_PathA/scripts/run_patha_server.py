#!/usr/bin/env python3
"""Run one Path A cell inside the qualified container, capturing provenance.

Mirrors scripts/run_histopath_width_server.py so that Path A cells carry the
same audit trail as the width matrix. The difference is the split manifest:
Path A uses the five-fold paper partition, whose digest is computed over the
git-tracked manifest files and is therefore reproducible by anyone with the
repository, rather than from a server-local checksum file.

Refuses to run unless the environment, config invariants, split manifest, fold
disjointness and GPU all match the declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# pandas, torch and yaml are imported lazily inside the functions that need
# them, so --print-manifest-sha runs on a bare host with only the standard
# library. The digest is the auditable part and must be reproducible by anyone
# holding the repository, not only inside the container.

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "preregistration/staged_hybrid_fair_warmup_protocol.md"

# Digest over the git-tracked five-fold manifest, in sorted path order, hashing
# "<relative path>\n" then the file bytes with line endings normalised to LF.
# Normalisation matters: git stores these blobs with LF, but a checkout under
# core.autocrlf can materialise CRLF, which would otherwise make the digest
# depend on the client platform rather than on the partition.
# Reproduce with --print-manifest-sha.
EXPECTED_SPLIT_SHA = "4a0a72fa3c89250cd012b943374be2301c0eb8ea2f4dd7d968b66c04b76bdf83"
MANIFEST_PREFIX = "data/splits/histopath_kaggle"
EXPECTED_GPU = "NVIDIA RTX A4000"
EXPECTED_CASE_IDS = 279

ARMS = {"control": "termwarm", "fair": "fairwarm"}


def manifest_sha(splits_dir: Path) -> str:
    """Digest the five-fold manifest exactly as EXPECTED_SPLIT_SHA was built."""
    members = sorted(
        [splits_dir / "patient_stats.csv", splits_dir / "split_stats.json"]
        + [
            splits_dir / "folds" / f"fold_{fold}" / name
            for fold in range(5)
            for name in ("test_patients.csv", "train_patients.csv")
        ]
    )
    digest = hashlib.sha256()
    for path in members:
        if not path.exists():
            raise RuntimeError(f"Manifest member missing: {path}")
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(f"{MANIFEST_PREFIX}/{path.relative_to(splits_dir)}".encode())
        digest.update(b"\n")
        digest.update(data)
    return digest.hexdigest()


def case_ids(path: Path) -> set[str]:
    import pandas as pd

    frame = pd.read_csv(path)
    column = "patient_id" if "patient_id" in frame.columns else frame.columns[0]
    return {str(value) for value in frame[column].unique()}


def verify_patch_manifests(splits_dir: Path, fold: int) -> None:
    """The loader needs patch-level manifests, which are untracked by design."""
    fold_dir = splits_dir / "folds" / f"fold_{fold}"
    missing = [name for name in ("train.csv", "test.csv") if not (fold_dir / name).exists()]
    if missing:
        raise RuntimeError(
            f"Patch-level manifests missing in {fold_dir}: {', '.join(missing)}.\n"
            "Only the case-ID lists ship in git. Rebuild the patch manifests "
            "from them plus the archive, once:\n"
            "  PaperB_PathA/scripts/rebuild_patch_manifests.sh"
        )


def verify_fold(splits_dir: Path, fold: int) -> dict[str, int]:
    verify_patch_manifests(splits_dir, fold)
    fold_dir = splits_dir / "folds" / f"fold_{fold}"
    train = case_ids(fold_dir / "train_patients.csv")
    test = case_ids(fold_dir / "test_patients.csv")
    if train & test:
        raise RuntimeError("Case-ID leakage between outer train and test.")
    total = len(train | test)
    if total != EXPECTED_CASE_IDS:
        raise RuntimeError(f"Expected {EXPECTED_CASE_IDS} case IDs, found {total}.")
    return {"outer_train": len(train), "outer_test": len(test)}


def verify_config(expect_fair: bool) -> None:
    import torch
    import yaml

    config = yaml.safe_load((ROOT / "configs/histopath.yaml").read_text())
    quantum = config["model"]["quantum"]
    expected = {
        "n_qubits": 8,
        "n_layers": 2,
        "entanglement": "linear",
        "encoding": "angle_y",
        "data_reuploading": False,
    }
    observed = {key: quantum[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"Quantum config changed: {observed} != {expected}")

    training = config["training"]
    if training["loss"] != "focal" or training["tta"] is not True:
        raise RuntimeError("The declared focal-loss/TTA bundle has changed.")
    if training["classical_device"] != "auto":
        raise RuntimeError("classical_device must remain auto.")
    for key, value in (("stage_a_epochs", 25), ("stage_b_epochs", 15), ("stage_c_epochs", 3)):
        if training[key] != value:
            raise RuntimeError(f"{key} is {training[key]}, declaration says {value}.")
    if training.get("early_stopping_patience") != 5:
        raise RuntimeError("early_stopping_patience must remain 5 for arms A0/A1.")
    # The arm is set by the CLI flag, so the config default must stay published.
    if training.get("stage_init_from_best", False) is not False:
        raise RuntimeError(
            "configs/histopath.yaml must keep stage_init_from_best false; the "
            "arm is selected by the CLI flag so both arms stay auditable."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    del expect_fair  # arm is recorded in provenance, not enforced against config


def required_environment() -> dict[str, str]:
    names = ("MBC_GIT_COMMIT", "MBC_IMAGE_ID", "MBC_ENV_LOCK_SHA", "MBC_DATASET_SHA")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing provenance environment: {', '.join(missing)}")
    return {name: os.environ[name] for name in names}


def nvidia_smi_metadata() -> list[dict[str, str]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
    )
    gpus = []
    for line in output.strip().splitlines():
        name, driver = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "driver_version": driver})
    return gpus


def capture_provenance(fold: int, arm: str, splits_dir: Path) -> dict:
    import torch

    environment = required_environment()
    commit = environment["MBC_GIT_COMMIT"]
    if len(commit) != 40:
        raise RuntimeError("MBC_GIT_COMMIT must be a full 40-character commit.")
    for name in ("MBC_ENV_LOCK_SHA", "MBC_DATASET_SHA"):
        value = environment[name]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise RuntimeError(f"{name} must be a lowercase SHA-256 digest.")

    observed_split = manifest_sha(splits_dir)
    if observed_split != EXPECTED_SPLIT_SHA:
        raise RuntimeError(
            "The five-fold manifest differs from the declared partition:\n"
            f"  observed {observed_split}\n  expected {EXPECTED_SPLIT_SHA}"
        )

    gpus = nvidia_smi_metadata()
    if len(gpus) != 1 or gpus[0]["name"] != EXPECTED_GPU:
        raise RuntimeError(f"Expected one {EXPECTED_GPU}; observed {gpus}.")

    return {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "paper_b_path_a_fair_warmup",
        "fold": fold,
        "arm": arm,
        "arm_tag": ARMS[arm],
        "stage_init_from_best": arm == "fair",
        "n_qubits": 8,
        "seed": 42,
        "effective_seed": 42 + fold,
        "source_commit": commit,
        "image_id": environment["MBC_IMAGE_ID"],
        "environment_lock_sha256": environment["MBC_ENV_LOCK_SHA"],
        "dataset_sha256": environment["MBC_DATASET_SHA"],
        "split_manifest_sha256": observed_split,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": gpus,
    }


def build_train_command(fold: int, arm: str, archive: Path | None, splits_dir: Path) -> list[str]:
    flag = "--stage-init-from-best" if arm == "fair" else "--no-stage-init-from-best"
    command = [
        sys.executable,
        "scripts/train_histopath_cv.py",
        "--fold",
        str(fold),
        "--experiment",
        "E3",
        "--seed",
        "42",
        "--n-qubits",
        "8",
        "--splits-dir",
        str(splits_dir),
        flag,
    ]
    if archive is not None:
        command += ["--archive-path", str(archive)]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--splits-dir", default=str(ROOT / MANIFEST_PREFIX))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--print-manifest-sha",
        action="store_true",
        help="Print the digest of --splits-dir and exit (no GPU needed)",
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)

    if args.print_manifest_sha:
        print(manifest_sha(splits_dir))
        return

    if args.fold is None or args.arm is None:
        parser.error("--fold and --arm are required unless --print-manifest-sha")

    archive = Path(args.archive_path) if args.archive_path else None
    command = build_train_command(args.fold, args.arm, archive, splits_dir)

    if args.dry_run:
        print(" ".join(map(str, command)))
        return

    if not PROTOCOL.exists():
        raise RuntimeError("The fair-warmup declaration is missing.")

    verify_config(expect_fair=args.arm == "fair")
    fold_counts = verify_fold(splits_dir, args.fold)
    provenance = capture_provenance(args.fold, args.arm, splits_dir)
    provenance["outer_case_id_counts"] = fold_counts

    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)
        target = args.output_root / f"provenance_fold{args.fold}_{ARMS[args.arm]}.json"
        target.write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Provenance written to {target}")

    print(json.dumps(provenance, indent=2), flush=True)
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Completed fold {args.fold} / {args.arm}.")


if __name__ == "__main__":
    main()

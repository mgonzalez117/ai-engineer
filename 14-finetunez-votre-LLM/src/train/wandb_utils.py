from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

DEFAULT_WANDB_PROJECT = "chsa-finetuning"
DEFAULT_WANDB_LOG_MODEL = "checkpoint"


def configure_wandb_env(
    default_project: str = DEFAULT_WANDB_PROJECT,
    default_log_model: str = DEFAULT_WANDB_LOG_MODEL,
) -> tuple[str, str]:
    project = os.getenv("WANDB_PROJECT", default_project).strip() or default_project
    log_model = os.getenv("WANDB_LOG_MODEL", default_log_model).strip() or default_log_model

    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_LOG_MODEL"] = log_model

    return project, log_model


def print_wandb_env(project: str, log_model: str) -> None:
    print("W&B project:", project)
    print("W&B model artifact logging:", log_model)


def artifact_version(value: str | None) -> int:
    if not value:
        return -1
    match = re.match(r"v(\d+)", value)
    return int(match.group(1)) if match else -1


def resolve_wandb_artifact_ref(
    *,
    explicit_artifact: str = "",
    run_path: str = "",
) -> str:
    explicit_artifact = (explicit_artifact or "").strip()
    run_path = (run_path or "").strip()

    if explicit_artifact:
        return explicit_artifact

    if not run_path:
        raise ValueError(
            "Set SFT_WANDB_RUN_PATH (entity/project/run_id) or SFT_WANDB_ARTIFACT (entity/project/artifact:alias)."
        )

    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    candidates = [artifact for artifact in run.logged_artifacts() if getattr(artifact, "type", "") == "model"]
    if not candidates:
        raise ValueError(f"No model artifact found for W&B run: {run_path}")

    checkpoint_candidates = [a for a in candidates if "checkpoint" in str(getattr(a, "name", "")).lower()]
    pool = checkpoint_candidates or candidates

    latest = max(pool, key=lambda a: artifact_version(getattr(a, "version", None)))
    name = str(getattr(latest, "name", "")).strip()
    version = str(getattr(latest, "version", "")).strip()

    if not name:
        raise ValueError(f"Could not resolve artifact reference from run: {run_path}")

    ref = name if ":" in name else f"{name}:{version}"
    print("Resolved SFT artifact from run:", ref)
    return ref


def download_wandb_artifact_dir(*, artifact_ref: str, download_root: Path) -> Path:
    import wandb

    api = wandb.Api()
    artifact = api.artifact(artifact_ref)

    if download_root.exists():
        shutil.rmtree(download_root)
    download_root.mkdir(parents=True, exist_ok=True)

    downloaded_dir = Path(artifact.download(root=str(download_root)))
    print("Downloaded W&B artifact to:", downloaded_dir)
    return downloaded_dir


from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import wandb
from huggingface_hub import HfApi

DEFAULT_WANDB_PROJECT = os.getenv("WANDB_PROJECT", "chsa-finetuning")
DEFAULT_WANDB_ENTITY = os.getenv("WANDB_ENTITY", "").strip()
DOWNLOAD_DIR = Path("tmp/wandb_deploy_lora")


def artifact_version(value: str | None) -> int:
    if not value:
        return -1
    match = re.match(r"v(\d+)", value)
    return int(match.group(1)) if match else -1


def detect_run_stage(run: Any) -> str | None:
    config = getattr(run, "config", {}) or {}
    text = " ".join(
        [
            str(getattr(run, "name", "")),
            str(config.get("run_name", "")),
            str(config.get("output_dir", "")),
            str(config.get("program", "")),
        ]
    ).lower()

    if "dpo" in text:
        return "dpo"
    return None


def pick_model_artifact(run: Any) -> Any | None:
    candidates = [artifact for artifact in run.logged_artifacts() if str(getattr(artifact, "type", "")) == "model"]
    if not candidates:
        return None

    def artifact_key(artifact: Any) -> tuple[int, int]:
        aliases = {str(alias).lower() for alias in (getattr(artifact, "aliases", []) or [])}
        has_latest_alias = 1 if "latest" in aliases else 0
        version = artifact_version(getattr(artifact, "version", None))
        return has_latest_alias, version

    return max(candidates, key=artifact_key)


def resolve_project_path(api: Any) -> str:
    if DEFAULT_WANDB_ENTITY:
        return f"{DEFAULT_WANDB_ENTITY}/{DEFAULT_WANDB_PROJECT}"

    default_entity = str(getattr(api, "default_entity", "")).strip()
    if default_entity:
        return f"{default_entity}/{DEFAULT_WANDB_PROJECT}"

    raise ValueError("Unable to resolve W&B entity. Set WANDB_ENTITY in CI variables.")


def latest_dpo_artifact_ref(api: Any) -> str:
    project_path = resolve_project_path(api)
    runs = api.runs(path=project_path, order="-created_at")

    for run in runs:
        if detect_run_stage(run) != "dpo":
            continue

        artifact = pick_model_artifact(run)
        if artifact is None:
            continue

        name = str(getattr(artifact, "name", "")).strip()
        version = str(getattr(artifact, "version", "")).strip()
        if not name:
            continue

        artifact_ref = name if ":" in name else f"{name}:{version or 'latest'}"
        print(f"Resolved latest DPO artifact: {artifact_ref}")
        return artifact_ref

    raise RuntimeError("No DPO model artifact found on W&B.")


def is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_adapter_dir(root: Path) -> Path:
    if is_adapter_dir(root):
        return root

    checkpoints = sorted(
        [path for path in root.rglob("checkpoint-*") if path.is_dir() and is_adapter_dir(path)],
        key=checkpoint_step,
    )
    if checkpoints:
        return checkpoints[-1]

    for config_path in root.rglob("adapter_config.json"):
        candidate = config_path.parent
        if is_adapter_dir(candidate):
            return candidate

    raise FileNotFoundError(f"No LoRA adapter found under: {root}")


def download_latest_dpo_adapter(api: Any) -> Path:
    artifact_ref = latest_dpo_artifact_ref(api)

    if DOWNLOAD_DIR.exists():
        shutil.rmtree(DOWNLOAD_DIR)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    artifact = api.artifact(artifact_ref)
    downloaded_root = Path(artifact.download(root=str(DOWNLOAD_DIR)))
    adapter_dir = find_adapter_dir(downloaded_root)

    print(f"Downloaded adapter dir: {adapter_dir}")
    return adapter_dir


def deploy_lora() -> None:
    hf_repo_id = os.getenv("HF_REPO_ID", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()

    if not hf_repo_id:
        raise ValueError("HF_REPO_ID is required.")
    if not hf_token:
        raise ValueError("HF_TOKEN is required.")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY is required.")

    os.environ["WANDB_API_KEY"] = wandb_api_key

    # On telecharge le dernier LoRA DPO depuis W&B puis on le pousse sur HF.
    wandb_api = wandb.Api()
    adapter_dir = download_latest_dpo_adapter(wandb_api)

    hf_api = HfApi(token=hf_token)
    hf_api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
    hf_api.upload_folder(
        repo_id=hf_repo_id,
        repo_type="model",
        folder_path=str(adapter_dir),
    )

    print(f"LoRA deployed to: {hf_repo_id}")


if __name__ == "__main__":
    deploy_lora()
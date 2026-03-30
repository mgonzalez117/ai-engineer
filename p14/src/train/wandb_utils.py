from __future__ import annotations

import os

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


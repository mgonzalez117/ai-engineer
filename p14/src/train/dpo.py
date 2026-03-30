from __future__ import annotations

import ast
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
try:
    from trl import DPOConfig, DPOTrainer
except Exception as exc:
    raise RuntimeError(
        "Failed to import DPOTrainer. Install compatible versions, for example: "
        "transformers==4.51.3 and trl==0.11.4."
    ) from exc

from src.dataset.io import PROCESSED_DIR
from src.train.wandb_utils import configure_wandb_env, print_wandb_env

class DPOTrainerCompat(DPOTrainer):
    """
    Compatibility shim for mismatched TRL/Transformers Trainer APIs.
    """

    def get_batch_samples(self, *args, **kwargs):
        # transformers.Trainer training-loop call signature
        if len(args) >= 2 and isinstance(args[1], int):
            return Trainer.get_batch_samples(self, *args, **kwargs)

        # TRL DPO internal call signature
        return DPOTrainer.get_batch_samples(self, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # transformers may pass num_items_in_batch (and potentially other kwargs)
        # that older TRL DPOTrainer.compute_loss does not accept.
        return DPOTrainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
        )

    def log(self, logs, start_time=None, **kwargs):
        # transformers.Trainer may call log(logs, start_time=...)
        # while TRL DPOTrainer expects only log(logs).
        return DPOTrainer.log(self, logs)


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B-Base")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "artifacts/dpo")

TRAIN_FILE = str(PROCESSED_DIR / "dpo_train.jsonl")
VAL_FILE = str(PROCESSED_DIR / "dpo_val.jsonl")
TEST_FILE = str(PROCESSED_DIR / "dpo_test.jsonl")

MAX_LENGTH = int(os.getenv("DPO_MAX_LENGTH", "512"))
MAX_PROMPT_LENGTH = int(os.getenv("DPO_MAX_PROMPT_LENGTH", "384"))
BETA = float(os.getenv("DPO_BETA", "0.1"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACC_STEPS = int(os.getenv("GRAD_ACC_STEPS", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-6"))
NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "1"))
SEED = int(os.getenv("SEED", "42"))

WANDB_PROJECT, WANDB_LOG_MODEL = configure_wandb_env()

# DPO start point:
# - preferred: fetch last SFT checkpoint from W&B run via SFT_WANDB_RUN_PATH
# - alternative: explicit artifact via SFT_WANDB_ARTIFACT
# - fallback: local adapter in artifacts/sft
USE_SFT_ADAPTER = os.getenv("DPO_USE_SFT_ADAPTER", "1") == "1"
SFT_WANDB_RUN_PATH = os.getenv("SFT_WANDB_RUN_PATH", "").strip()
SFT_WANDB_ARTIFACT = os.getenv("SFT_WANDB_ARTIFACT", "").strip()
SFT_LOCAL_DIR = os.getenv("SFT_LOCAL_DIR", "artifacts/sft")
SFT_DOWNLOAD_DIR = os.getenv("SFT_DOWNLOAD_DIR", "tmp/wandb_sft")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def try_parse_messages(value: str) -> list[dict[str, Any]] | None:
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(value)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def extract_assistant_text(raw_value: Any) -> str:
    text = clean_text(raw_value)
    if not text:
        return ""

    messages = try_parse_messages(text)
    if messages is None:
        return text

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() == "assistant":
            assistant_text = clean_text(message.get("content"))
            if assistant_text:
                return assistant_text

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        fallback_text = clean_text(message.get("content"))
        if fallback_text:
            return fallback_text

    return text


def format_dpo_example(example: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": clean_text(example.get("prompt")),
        "chosen": extract_assistant_text(example.get("chosen")),
        "rejected": extract_assistant_text(example.get("rejected")),
    }


def has_valid_preference(example: dict[str, str]) -> bool:
    prompt = clean_text(example.get("prompt"))
    chosen = clean_text(example.get("chosen"))
    rejected = clean_text(example.get("rejected"))
    return bool(prompt and chosen and rejected and chosen != rejected)


def save_eval_report(report: dict[str, Any]) -> None:
    report_path = Path(OUTPUT_DIR) / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation report: {report_path}")


def build_eval_only_trainer(
    trained_model: Any,
    eval_dataset: Any,
    training_args: DPOConfig,
    tokenizer: Any,
) -> DPOTrainerCompat:
    """
    Build a DPO trainer for standalone evaluation on an external split.
    Some TRL versions accept train_dataset=None, others expect a dataset.
    """
    try:
        return DPOTrainerCompat(
            model=trained_model,
            ref_model=None,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=None,
        )
    except Exception:
        fallback_train = eval_dataset.select(range(min(1, len(eval_dataset))))
        return DPOTrainerCompat(
            model=trained_model,
            ref_model=None,
            args=training_args,
            train_dataset=fallback_train,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=None,
        )


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def artifact_version(value: str | None) -> int:
    if not value:
        return -1
    match = re.match(r"v(\d+)", value)
    return int(match.group(1)) if match else -1


def is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()


def find_adapter_dir(root: Path) -> Path:
    if is_adapter_dir(root):
        return root

    checkpoints = sorted(
        [p for p in root.rglob("checkpoint-*") if p.is_dir() and is_adapter_dir(p)],
        key=checkpoint_step,
    )
    if checkpoints:
        return checkpoints[-1]

    for cfg in root.rglob("adapter_config.json"):
        candidate = cfg.parent
        if is_adapter_dir(candidate):
            return candidate

    raise FileNotFoundError(f"No LoRA adapter found under: {root}")


def resolve_wandb_artifact_ref() -> str:
    if SFT_WANDB_ARTIFACT:
        return SFT_WANDB_ARTIFACT

    if not SFT_WANDB_RUN_PATH:
        raise ValueError(
            "Set SFT_WANDB_RUN_PATH (entity/project/run_id) or SFT_WANDB_ARTIFACT (entity/project/artifact:alias)."
        )

    import wandb

    api = wandb.Api()
    run = api.run(SFT_WANDB_RUN_PATH)

    candidates = [artifact for artifact in run.logged_artifacts() if getattr(artifact, "type", "") == "model"]
    if not candidates:
        raise ValueError(f"No model artifact found for W&B run: {SFT_WANDB_RUN_PATH}")

    checkpoint_candidates = [a for a in candidates if "checkpoint" in str(getattr(a, "name", "")).lower()]
    pool = checkpoint_candidates or candidates

    latest = max(pool, key=lambda a: artifact_version(getattr(a, "version", None)))
    name = str(getattr(latest, "name", "")).strip()
    version = str(getattr(latest, "version", "")).strip()

    if not name:
        raise ValueError(f"Could not resolve artifact reference from run: {SFT_WANDB_RUN_PATH}")

    ref = name if ":" in name else f"{name}:{version}"

    print("Resolved SFT artifact from run:", ref)
    return ref


def download_adapter_from_wandb() -> Path:
    import wandb

    artifact_ref = resolve_wandb_artifact_ref()
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)

    download_root = Path(SFT_DOWNLOAD_DIR)
    if download_root.exists():
        shutil.rmtree(download_root)
    download_root.mkdir(parents=True, exist_ok=True)

    downloaded_dir = Path(artifact.download(root=str(download_root)))
    print("Downloaded W&B artifact to:", downloaded_dir)

    adapter_dir = find_adapter_dir(downloaded_dir)
    print("Resolved adapter dir:", adapter_dir)
    return adapter_dir


def resolve_sft_adapter_dir() -> Path:
    if SFT_WANDB_ARTIFACT or SFT_WANDB_RUN_PATH:
        return download_adapter_from_wandb()

    local_dir = Path(SFT_LOCAL_DIR)
    if not local_dir.exists():
        raise FileNotFoundError(
            f"SFT local dir not found: {local_dir}. Set SFT_WANDB_RUN_PATH/SFT_WANDB_ARTIFACT or SFT_LOCAL_DIR."
        )
    adapter_dir = find_adapter_dir(local_dir)
    print("Resolved local adapter dir:", adapter_dir)
    return adapter_dir


def main() -> None:
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        print(f"Removing existing output dir: {output_path}")
        shutil.rmtree(output_path)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {MODEL_NAME}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.config.use_cache = False

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print_wandb_env(WANDB_PROJECT, WANDB_LOG_MODEL)

    loaded_from_sft_adapter = False
    if USE_SFT_ADAPTER:
        adapter_dir = resolve_sft_adapter_dir()
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
        loaded_from_sft_adapter = True
        print("Loaded trainable SFT adapter:", adapter_dir)

    data_files = {
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    }
    has_test = Path(TEST_FILE).exists()
    if has_test:
        data_files["test"] = TEST_FILE

    dataset = load_dataset("json", data_files=data_files)
    print("Raw train columns:", dataset["train"].column_names)

    formatted_dataset = dataset.map(
        format_dpo_example,
        remove_columns=dataset["train"].column_names,
    )

    train_raw = formatted_dataset["train"]
    val_raw = formatted_dataset["validation"]
    test_raw = formatted_dataset["test"] if has_test else None

    train_dataset = train_raw.filter(has_valid_preference)
    val_dataset = val_raw.filter(has_valid_preference)
    test_dataset = test_raw.filter(has_valid_preference) if test_raw is not None else None

    print(f"Train rows kept: {len(train_dataset)}/{len(train_raw)}")
    print(f"Validation rows kept: {len(val_dataset)}/{len(val_raw)}")
    if test_raw is not None and test_dataset is not None:
        print(f"Test rows kept: {len(test_dataset)}/{len(test_raw)}")

    if len(train_dataset) == 0:
        raise ValueError("No valid DPO training rows found after preprocessing.")
    if len(val_dataset) == 0:
        raise ValueError("No valid DPO validation rows found after preprocessing.")

    print("First preprocessed train row:", train_dataset[0])

    peft_config = None
    if not loaded_from_sft_adapter:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
        )

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        run_name="dpo-qwen3",
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = DPOTrainerCompat(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()

    # Save right after training so final-eval issues never lose trained weights.
    print(f"Saving adapter/model to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Running final validation evaluation...")
    # Important: do not pass eval_dataset here. DPOTrainer has already prepared
    # its internal eval dataset, while passing the raw one can trigger KeyError
    # on missing tokenized columns (e.g., chosen_input_ids).
    val_metrics = trainer.evaluate(metric_key_prefix="val_final")
    trainer.log(val_metrics)

    test_metrics = None
    if test_dataset is not None:
        if len(test_dataset) == 0:
            print("Skipping final test evaluation: no valid test rows after preprocessing.")
        else:
            print("Running final test evaluation...")
            test_trainer = build_eval_only_trainer(
                trained_model=trainer.model,
                eval_dataset=test_dataset,
                training_args=training_args,
                tokenizer=tokenizer,
            )
            test_metrics = test_trainer.evaluate(metric_key_prefix="test_final")
            trainer.log(test_metrics)

    report = {
        "evaluated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "model_name": MODEL_NAME,
        "output_dir": OUTPUT_DIR,
        "beta": BETA,
        "max_length": MAX_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "sft_adapter_source": "wandb" if (SFT_WANDB_ARTIFACT or SFT_WANDB_RUN_PATH) else "local",
        "sft_wandb_run_path": SFT_WANDB_RUN_PATH or None,
        "sft_wandb_artifact": SFT_WANDB_ARTIFACT or None,
        "val_final": val_metrics,
        "test_final": test_metrics,
    }
    save_eval_report(report)

    print("Done.")


if __name__ == "__main__":
    main()


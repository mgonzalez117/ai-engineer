from __future__ import annotations

import json
import math
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.dataset.io import PROCESSED_DIR

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B-Base")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "artifacts/sft")

TRAIN_FILE = str(PROCESSED_DIR / "train.jsonl")
VAL_FILE = str(PROCESSED_DIR / "val.jsonl")
TEST_FILE = str(PROCESSED_DIR / "test.jsonl")

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACC_STEPS = int(os.getenv("GRAD_ACC_STEPS", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "1"))
SEED = int(os.getenv("SEED", "42"))

WANDB_LOG_MODEL = os.getenv("WANDB_LOG_MODEL", "checkpoint")
os.environ["WANDB_LOG_MODEL"] = WANDB_LOG_MODEL

def build_prompt(example: dict) -> tuple[str, str]:
    instruction = example["instruction"].strip()
    user_input = example["input"].strip()
    output = example["output"].strip()

    prompt = (
        f"Instruction:\n{instruction}\n\n"
        f"Input:\n{user_input}\n\n"
        f"Response:\n"
    )
    return prompt, output


def tokenize_example(example: dict, tokenizer) -> dict:
    prompt, answer = build_prompt(example)
    full_text = prompt + answer

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    )
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    prompt_len = len(prompt_tokens["input_ids"])
    labels = input_ids.copy()

    # On ignore la loss sur le prompt, on apprend seulement la reponse
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def has_supervised_target(example: dict, tokenizer) -> bool:
    """
    True when at least one answer token remains after truncation.
    This prevents eval/train NaN loss when all labels are -100.
    """
    prompt, answer = build_prompt(example)
    if not answer.strip():
        return False

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    )
    full_tokens = tokenizer(
        prompt + answer,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    )
    return len(full_tokens["input_ids"]) > len(prompt_tokens["input_ids"])


def with_perplexity(metrics: dict, loss_key: str, perplexity_key: str) -> dict:
    output = dict(metrics)
    loss_value = output.get(loss_key)
    if loss_value is not None:
        try:
            output[perplexity_key] = math.exp(float(loss_value))
        except (OverflowError, ValueError):
            output[perplexity_key] = float("inf")
    return output


def save_eval_report(report: dict) -> None:
    report_path = Path(OUTPUT_DIR) / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation report: {report_path}")


def main() -> None:
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        print(f"Removing existing output dir: {output_path}")
        shutil.rmtree(output_path)

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_NAME}")
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
    print("W&B model artifact logging:", os.environ.get("WANDB_LOG_MODEL"))
    data_files = {
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    }
    has_test = Path(TEST_FILE).exists()
    if has_test:
        data_files["test"] = TEST_FILE

    dataset = load_dataset("json", data_files=data_files)

    print("Train columns:", dataset["train"].column_names)
    print("First train row:", dataset["train"][0])

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

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_raw = dataset["train"]
    val_raw = dataset["validation"]
    test_raw = dataset["test"] if has_test else None

    train_filtered = train_raw.filter(lambda x: has_supervised_target(x, tokenizer))
    val_filtered = val_raw.filter(lambda x: has_supervised_target(x, tokenizer))
    test_filtered = (
        test_raw.filter(lambda x: has_supervised_target(x, tokenizer))
        if test_raw is not None
        else None
    )

    print(f"Train rows kept for supervision: {len(train_filtered)}/{len(train_raw)}")
    print(f"Validation rows kept for supervision: {len(val_filtered)}/{len(val_raw)}")
    if test_filtered is not None and test_raw is not None:
        print(f"Test rows kept for supervision: {len(test_filtered)}/{len(test_raw)}")

    train_dataset = train_filtered.map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=train_filtered.column_names,
    )
    val_dataset = val_filtered.map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=val_filtered.column_names,
    )

    test_dataset = None
    if test_filtered is not None:
        test_dataset = test_filtered.map(
            lambda x: tokenize_example(x, tokenizer),
            remove_columns=test_filtered.column_names,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        run_name="sft-qwen3",
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting SFT training...")
    trainer.train()

    print("Running final validation evaluation...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="val_final")
    val_metrics = with_perplexity(val_metrics, "val_final_loss", "val_final_perplexity")
    trainer.log(val_metrics)

    test_metrics = None
    if test_dataset is not None:
        print("Running final test evaluation...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test_final")
        test_metrics = with_perplexity(test_metrics, "test_final_loss", "test_final_perplexity")
        trainer.log(test_metrics)

    report = {
        "evaluated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "model_name": MODEL_NAME,
        "output_dir": OUTPUT_DIR,
        "val_final": val_metrics,
        "test_final": test_metrics,
    }
    save_eval_report(report)

    print(f"Saving adapter to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()



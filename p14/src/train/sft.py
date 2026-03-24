from __future__ import annotations

import os
import shutil
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

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACC_STEPS = int(os.getenv("GRAD_ACC_STEPS", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "1"))
SEED = int(os.getenv("SEED", "42"))


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

    # On ignore la loss sur le prompt, on apprend seulement la réponse
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


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
        dtype=dtype,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.config.use_cache = False

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
        },
    )

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

    train_dataset = dataset["train"].map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    val_dataset = dataset["validation"].map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=dataset["validation"].column_names,
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

    print(f"Saving adapter to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
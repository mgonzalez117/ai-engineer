from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def build_clinical_prompt(example: dict[str, Any]) -> tuple[str, str]:
    instruction = str(example.get("instruction", "")).strip()
    user_input = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()

    prompt = (
        f"Instruction:\n{instruction}\n\n"
        f"Input:\n{user_input}\n\n"
        f"Response:\n"
    )
    return prompt, output


def has_clinical_supervised_target(
    example: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> bool:
    prompt, answer = build_clinical_prompt(example)
    if not answer:
        return False

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    full_tokens = tokenizer(
        prompt + answer,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    return len(full_tokens["input_ids"]) > len(prompt_tokens["input_ids"])


def tokenize_clinical_example(
    example: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt, answer = build_clinical_prompt(example)
    full_text = prompt + answer

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    prompt_len = len(prompt_tokens["input_ids"])
    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _collate_supervised_batch(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(feature["input_ids"]) for feature in features)

    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []

    for feature in features:
        cur_len = len(feature["input_ids"])
        pad_len = max_len - cur_len

        input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(feature["attention_mask"] + [0] * pad_len)
        labels.append(feature["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def evaluate_clinical_holdout(
    *,
    model: Any,
    tokenizer: Any,
    clinical_eval_file: str | Path,
    max_seq_length: int,
    batch_size: int,
) -> dict[str, Any]:
    clinical_eval_path = Path(clinical_eval_file)

    report: dict[str, Any] = {
        "path": str(clinical_eval_path),
        "available": clinical_eval_path.exists(),
        "rows_total": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "token_count": 0,
        "loss": None,
        "perplexity": None,
    }

    if not clinical_eval_path.exists():
        return report

    raw_dataset = load_dataset("json", data_files={"clinical": str(clinical_eval_path)})["clinical"]
    report["rows_total"] = len(raw_dataset)

    filtered_dataset = raw_dataset.filter(
        lambda row: has_clinical_supervised_target(row, tokenizer, max_seq_length),
    )
    report["rows_used"] = len(filtered_dataset)
    report["rows_skipped"] = max(report["rows_total"] - report["rows_used"], 0)

    if report["rows_used"] == 0:
        return report

    tokenized_dataset = filtered_dataset.map(
        lambda row: tokenize_clinical_example(row, tokenizer, max_seq_length),
        remove_columns=filtered_dataset.column_names,
    )

    eval_batch_size = max(int(batch_size), 1)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id for clinical evaluation.")

    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=lambda features: _collate_supervised_batch(features, pad_token_id),
    )

    was_training = model.training
    model.eval()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_weighted_loss = 0.0
    total_target_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)

            token_count = int((batch["labels"] != -100).sum().item())
            if token_count == 0:
                continue

            batch_loss = float(outputs.loss.detach().cpu())
            total_weighted_loss += batch_loss * token_count
            total_target_tokens += token_count

    if was_training:
        model.train()

    report["token_count"] = total_target_tokens

    if total_target_tokens == 0:
        return report

    avg_loss = total_weighted_loss / total_target_tokens
    report["loss"] = avg_loss

    try:
        report["perplexity"] = math.exp(avg_loss)
    except OverflowError:
        report["perplexity"] = float("inf")

    return report


def clinical_metrics_for_logging(
    clinical_report: dict[str, Any],
    *,
    prefix: str = "clinical_eval",
) -> dict[str, float]:
    metrics = {
        f"{prefix}/available": 1.0 if clinical_report.get("available") else 0.0,
        f"{prefix}/rows_total": float(clinical_report.get("rows_total", 0)),
        f"{prefix}/rows_used": float(clinical_report.get("rows_used", 0)),
        f"{prefix}/rows_skipped": float(clinical_report.get("rows_skipped", 0)),
        f"{prefix}/token_count": float(clinical_report.get("token_count", 0)),
    }

    loss = clinical_report.get("loss")
    perplexity = clinical_report.get("perplexity")
    if loss is not None:
        metrics[f"{prefix}/loss"] = float(loss)
    if perplexity is not None:
        metrics[f"{prefix}/perplexity"] = float(perplexity)

    return metrics

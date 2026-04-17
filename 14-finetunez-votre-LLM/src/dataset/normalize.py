from __future__ import annotations

import re
from typing import Any

from .io import RAW_DIR, find_all_tables_with_columns, find_first_table_with_columns
from .metadata import build_metadata


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_id(prefix: str, index: int) -> str:
    return f"{prefix}-{index:08d}"


def get_options(row: dict[str, Any]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for label in ["A", "B", "C", "D", "E"]:
        key = f"answer_{label.lower()}"
        value = clean_text(row.get(key, ""))
        if value:
            options.append((label, value))
    return options


def parse_correct_answers(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        results = []
        for item in value:
            results.extend(parse_correct_answers(item))
        return results

    if isinstance(value, int):
        if 0 <= value <= 4:
            return [chr(ord("A") + value)]
        return []

    text = clean_text(value).strip().upper()
    if not text:
        return []

    if text in {"A", "B", "C", "D", "E"}:
        return [text]

    if text.isdigit():
        idx = int(text)
        if 0 <= idx <= 4:
            return [chr(ord("A") + idx)]

    matches = re.findall(r"[A-E]", text)
    if matches:
        return matches

    return []


def resolve_correct_option_texts(
    options: list[tuple[str, str]],
    correct_labels: list[str],
) -> list[str]:
    option_map = {label: text for label, text in options}
    return [option_map[label] for label in correct_labels if label in option_map]


def build_sft_row(
    *,
    row_id: str,
    dataset: str,
    language: str,
    instruction: str,
    input_text: str,
    output_text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": row_id,
        "dataset": dataset,
        "language": language,
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "metadata": metadata,
    }


def medquad_to_sft() -> list[dict[str, Any]]:
    dataset_dir = RAW_DIR / "MedQuad-MedicalQnADataset"
    df = find_first_table_with_columns(dataset_dir, {"Question", "Answer"})
    if df is None:
        return []

    rows: list[dict[str, Any]] = []

    for idx, record in enumerate(df.to_dict(orient="records")):
        question = clean_text(record.get("Question"))
        answer = clean_text(record.get("Answer"))
        qtype = clean_text(record.get("qtype"))

        if not question or not answer:
            continue

        input_text = f"Question : {question}"

        metadata = {
            **build_metadata(
                source="medquad",
                language="en",
                task_type="qa_open",
                text_for_clinical_case=question,
            ),
            "question_type": qtype or None,
            "medical_subject": None,
            "has_clinical_case": False,
            "source_row_id": None,
        }

        rows.append(
            build_sft_row(
                row_id=make_id("medquad", idx),
                dataset="medquad",
                language="en",
                instruction="Answer the following medical question clearly, concisely, and factually.",
                input_text=input_text,
                output_text=answer,
                metadata=metadata,
            )
        )

    return rows


def mediqal_to_sft() -> list[dict[str, Any]]:
    dataset_dir = RAW_DIR / "MediQAl"
    dfs = find_all_tables_with_columns(dataset_dir, {"question"})
    if not dfs:
        return []

    rows: list[dict[str, Any]] = []
    oeq_idx = 0
    mcq_idx = 0

    for df in dfs:
        records = df.to_dict(orient="records")
        columns = set(df.columns)

        # Open-ended QA
        if {"question", "answer"}.issubset(columns):
            for record in records:
                question = clean_text(record.get("question"))
                answer = clean_text(record.get("answer"))
                clinical_case = clean_text(record.get("clinical_case"))
                medical_subject = clean_text(record.get("medical_subject"))
                question_type = clean_text(record.get("question_type"))
                source_row_id = clean_text(record.get("id"))

                if not question or not answer:
                    continue

                parts = []
                if clinical_case:
                    parts.append(f"Cas clinique : {clinical_case}")
                parts.append(f"Question : {question}")
                input_text = "\n\n".join(parts)

                metadata = {
                    **build_metadata(
                        source="mediqal",
                        language="fr",
                        task_type="qa_open",
                        text_for_clinical_case=input_text,
                        medical_subject=medical_subject,
                        question_type=question_type,
                        source_row_id=source_row_id,
                    ),
                    "medical_subject": medical_subject or None,
                    "question_type": question_type or None,
                    "has_clinical_case": bool(clinical_case),
                    "source_row_id": source_row_id or None,
                }

                rows.append(
                    build_sft_row(
                        row_id=make_id("mediqal-oeq", oeq_idx),
                        dataset="mediqal",
                        language="fr",
                        instruction="Réponds de manière claire, concise et médicale à la question suivante.",
                        input_text=input_text,
                        output_text=answer,
                        metadata=metadata,
                    )
                )
                oeq_idx += 1

        # MCQ
        elif {"question", "correct_answers"}.issubset(columns):
            for record in records:
                question = clean_text(record.get("question"))
                clinical_case = clean_text(record.get("clinical_case"))
                medical_subject = clean_text(record.get("medical_subject"))
                question_type = clean_text(record.get("question_type"))
                source_row_id = clean_text(record.get("id"))

                options = get_options(record)
                correct_labels = parse_correct_answers(record.get("correct_answers"))
                correct_texts = resolve_correct_option_texts(options, correct_labels)

                if not question or not options or not correct_labels or not correct_texts:
                    continue

                parts = []
                if clinical_case:
                    parts.append(f"Cas clinique : {clinical_case}")
                parts.append(f"Question : {question}")
                parts.append("Options :\n" + "\n".join(f"{label}. {text}" for label, text in options))
                input_text = "\n\n".join(parts)

                # On met dans output le texte des bonnes réponses, pas seulement la lettre.
                output_text = "\n".join(correct_texts)

                metadata = {
                    **build_metadata(
                        source="mediqal",
                        language="fr",
                        task_type="mcq",
                        text_for_clinical_case=input_text,
                        medical_subject=medical_subject,
                        question_type=question_type,
                        source_row_id=source_row_id,
                    ),
                    "medical_subject": medical_subject or None,
                    "question_type": question_type or None,
                    "has_clinical_case": bool(clinical_case),
                    "source_row_id": source_row_id or None,
                    "correct_answer_labels": correct_labels,
                    "correct_answer_texts": correct_texts,
                    "options": [{"label": label, "text": text} for label, text in options],
                }

                rows.append(
                    build_sft_row(
                        row_id=make_id("mediqal-mcq", mcq_idx),
                        dataset="mediqal",
                        language="fr",
                        instruction="Choisis la ou les bonnes réponses parmi les options proposées.",
                        input_text=input_text,
                        output_text=output_text,
                        metadata=metadata,
                    )
                )
                mcq_idx += 1

    return rows


def frenchmedmcqa_to_sft() -> list[dict[str, Any]]:
    dataset_dir = RAW_DIR / "frenchmedmcqa"
    dfs = find_all_tables_with_columns(dataset_dir, {"question", "correct_answers"})
    if not dfs:
        return []

    rows: list[dict[str, Any]] = []
    idx = 0

    for df in dfs:
        for record in df.to_dict(orient="records"):
            question = clean_text(record.get("question"))
            options = get_options(record)
            correct_labels = parse_correct_answers(record.get("correct_answers"))
            correct_texts = resolve_correct_option_texts(options, correct_labels)

            if not question or not options or not correct_labels or not correct_texts:
                continue

            input_text = "\n\n".join([
                f"Question : {question}",
                "Options :\n" + "\n".join(f"{label}. {text}" for label, text in options),
            ])

            metadata = {
                **build_metadata(
                    source="frenchmedmcqa",
                    language="fr",
                    task_type="mcq",
                    text_for_clinical_case=input_text,
                ),
                "medical_subject": None,
                "question_type": "mcq",
                "has_clinical_case": False,
                "source_row_id": clean_text(record.get("id")) or None,
                "number_correct_answers": record.get("number_correct_answers"),
                "correct_answer_labels": correct_labels,
                "correct_answer_texts": correct_texts,
                "options": [{"label": label, "text": text} for label, text in options],
            }

            rows.append(
                build_sft_row(
                    row_id=make_id("frenchmedmcqa", idx),
                    dataset="frenchmedmcqa",
                    language="fr",
                    instruction="Choisis la ou les bonnes réponses parmi les options proposées.",
                    input_text=input_text,
                    output_text="\n".join(correct_texts),
                    metadata=metadata,
                )
            )
            idx += 1

    return rows


def ultramedical_to_dpo() -> list[dict[str, Any]]:
    dataset_dir = RAW_DIR / "UltraMedical-Preference"
    df = find_first_table_with_columns(dataset_dir, {"prompt", "chosen", "rejected"})
    if df is None:
        return []

    rows: list[dict[str, Any]] = []

    for idx, record in enumerate(df.to_dict(orient="records")):
        prompt = clean_text(record.get("prompt"))
        chosen = clean_text(record.get("chosen"))
        rejected = clean_text(record.get("rejected"))

        if not prompt or not chosen or not rejected:
            continue

        rows.append({
            "id": make_id("ultramedical", idx),
            "dataset": "ultramedical_preference",
            "language": "en",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": build_metadata(
                source="ultramedical_preference",
                language="en",
                task_type="preference",
                text_for_clinical_case=prompt,
            ),
        })

    return rows
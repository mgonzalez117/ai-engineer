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

        rows.append({
            "id": make_id("medquad", idx),
            "dataset": "medquad",
            "language": "en",
            "instruction": "Answer the following medical question clearly and factually.",
            "input": question,
            "output": answer,
            "metadata": {
                **build_metadata(
                    source="medquad",
                    language="en",
                    task_type="qa_open",
                    text_for_clinical_case=question,
                ),
                "question_type": qtype,
            },
        })

    return rows


def mediqal_to_sft() -> list[dict[str, Any]]:
    dataset_dir = RAW_DIR / "MediQAl"
    dfs = find_all_tables_with_columns(dataset_dir, {"question"})
    if not dfs:
        return []

    rows: list[dict[str, Any]] = []
    idx = 0

    for df in dfs:
        records = df.to_dict(orient="records")
        columns = set(df.columns)

        # Open-ended
        if {"question", "answer"}.issubset(columns):
            for record in records:
                question = clean_text(record.get("question"))
                answer = clean_text(record.get("answer"))
                clinical_case = clean_text(record.get("clinical_case"))
                medical_subject = clean_text(record.get("medical_subject"))
                question_type = clean_text(record.get("question_type"))
                task = clean_text(record.get("task"))
                source_row_id = clean_text(record.get("id"))

                if not question or not answer:
                    continue

                parts = []
                if medical_subject:
                    parts.append(f"[Spécialité médicale : {medical_subject}]")
                if question_type:
                    parts.append(f"[Type de question : {question_type}]")
                if clinical_case:
                    parts.append(f"Cas clinique :\n{clinical_case}")
                parts.append(f"Question :\n{question}")

                user_input = "\n\n".join(parts)

                rows.append({
                    "id": make_id("mediqal-oeq", idx),
                    "dataset": "mediqal",
                    "language": "fr",
                    "instruction": "Réponds de manière claire, concise et médicale à la question suivante.",
                    "input": user_input,
                    "output": answer,
                    "metadata": build_metadata(
                        source="mediqal",
                        language="fr",
                        task_type="qa_open",
                        text_for_clinical_case=user_input,
                        medical_subject=medical_subject,
                        question_type=question_type,
                        task=task,
                        source_row_id=source_row_id,
                    ),
                })
                idx += 1

        # MCQ
        elif {"question", "correct_answers"}.issubset(columns):
            for record in records:
                question = clean_text(record.get("question"))
                clinical_case = clean_text(record.get("clinical_case"))
                medical_subject = clean_text(record.get("medical_subject"))
                question_type = clean_text(record.get("question_type"))
                task = clean_text(record.get("task"))
                source_row_id = clean_text(record.get("id"))

                options = get_options(record)
                correct = parse_correct_answers(record.get("correct_answers"))

                if not question or not options or not correct:
                    continue

                parts = []
                if medical_subject:
                    parts.append(f"[Spécialité médicale : {medical_subject}]")
                if question_type:
                    parts.append(f"[Type de question : {question_type}]")
                if clinical_case:
                    parts.append(f"Cas clinique :\n{clinical_case}")
                parts.append(f"Question :\n{question}")
                parts.append("Options :\n" + "\n".join(f"{k}. {v}" for k, v in options))

                user_input = "\n\n".join(parts)

                # On garde les lettres/références correctes dans l'output,
                # comme dans ton implémentation actuelle.
                prefix = "Réponse correcte : " if len(correct) == 1 else "Réponses correctes : "
                output = prefix + ", ".join(correct)

                rows.append({
                    "id": make_id("mediqal-mcq", idx),
                    "dataset": "mediqal",
                    "language": "fr",
                    "instruction": "Choisis la ou les bonnes réponses parmi les options proposées.",
                    "input": user_input,
                    "output": output,
                    "metadata": build_metadata(
                        source="mediqal",
                        language="fr",
                        task_type="mcq",
                        text_for_clinical_case=user_input,
                        medical_subject=medical_subject,
                        question_type=question_type,
                        task=task,
                        source_row_id=source_row_id,
                        correct_answers=correct,
                    ),
                })
                idx += 1

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
            correct = parse_correct_answers(record.get("correct_answers"))

            if not question or not options or not correct:
                continue

            user_input = "\n\n".join([
                f"Question :\n{question}",
                "Options :\n" + "\n".join(f"{k}. {v}" for k, v in options),
            ])

            rows.append({
                "id": make_id("frenchmedmcqa", idx),
                "dataset": "frenchmedmcqa",
                "language": "fr",
                "instruction": "Choisis la bonne réponse.",
                "input": user_input,
                "output": "Réponse correcte : " + ", ".join(correct),
                "metadata": {
                    **build_metadata(
                        source="frenchmedmcqa",
                        language="fr",
                        task_type="mcq",
                        text_for_clinical_case=user_input,
                    ),
                    "number_correct_answers": record.get("number_correct_answers"),
                    "source_id": record.get("id"),
                },
            })
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
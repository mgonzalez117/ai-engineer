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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
try:
    from trl import DPOConfig, DPOTrainer
except Exception as exc:
    raise RuntimeError(
        "Impossible d'importer DPOTrainer. Installez des versions compatibles, par exemple : "
        "transformers==4.51.3 and trl==0.11.4."
    ) from exc

from src.dataset.io import PROCESSED_DIR
from src.train.eval_utils import clinical_metrics_for_logging, evaluate_clinical_holdout
from src.train.wandb_utils import configure_wandb_env, download_wandb_artifact_dir, print_wandb_env, resolve_wandb_artifact_ref

class DPOTrainerCompat(DPOTrainer):
    """
    Couche de compatibilité pour des API Trainer TRL/Transformers non alignées.
    """

    def get_batch_samples(self, *args, **kwargs):
        # Signature d'appel de la boucle d'entraînement de transformers.Trainer
        if len(args) >= 2 and isinstance(args[1], int):
            return Trainer.get_batch_samples(self, *args, **kwargs)

        # Signature d'appel interne de TRL DPO
        return DPOTrainer.get_batch_samples(self, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # transformers peut passer num_items_in_batch (et potentiellement d'autres kwargs)
        # que les anciennes versions de TRL DPOTrainer.compute_loss n'acceptent pas.
        return DPOTrainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
        )

    def log(self, logs, start_time=None, **kwargs):
        # transformers.Trainer peut appeler log(logs, start_time=...)
        # alors que TRL DPOTrainer attend uniquement log(logs).
        return DPOTrainer.log(self, logs)


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B-Base")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "artifacts/dpo")

TRAIN_FILE = str(PROCESSED_DIR / "dpo_train.jsonl")
VAL_FILE = str(PROCESSED_DIR / "dpo_val.jsonl")
TEST_FILE = str(PROCESSED_DIR / "dpo_test.jsonl")
CLINICAL_EVAL_FILE = str(PROCESSED_DIR / "clinical_eval.jsonl")

MAX_LENGTH = int(os.getenv("DPO_MAX_LENGTH", "512"))
MAX_PROMPT_LENGTH = int(os.getenv("DPO_MAX_PROMPT_LENGTH", "384"))
BETA = float(os.getenv("DPO_BETA", "0.1"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACC_STEPS = int(os.getenv("GRAD_ACC_STEPS", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-6"))
NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "1"))
SEED = int(os.getenv("SEED", "42"))
CLINICAL_EVAL_ENABLED = os.getenv("CLINICAL_EVAL_ENABLED", "1") == "1"
CLINICAL_EVAL_MAX_LENGTH = int(os.getenv("CLINICAL_EVAL_MAX_LENGTH", str(MAX_LENGTH)))
CLINICAL_EVAL_BATCH_SIZE = int(os.getenv("CLINICAL_EVAL_BATCH_SIZE", str(BATCH_SIZE)))

WANDB_PROJECT, WANDB_LOG_MODEL = configure_wandb_env()

# Point de départ DPO (utilise toujours un adaptateur SFT précédent) :
# - recommandé : récupérer le dernier checkpoint SFT depuis un run W&B via SFT_WANDB_RUN_PATH
# - alternative : artifact explicite via SFT_WANDB_ARTIFACT
# - secours : adaptateur local dans artifacts/sft
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
    print(f"Rapport d'évaluation sauvegardé : {report_path}")


def build_eval_only_trainer(
    trained_model: Any,
    eval_dataset: Any,
    training_args: DPOConfig,
    tokenizer: Any,
) -> DPOTrainerCompat:
    """
    Construit un trainer DPO pour une évaluation autonome sur un split externe.
    Certaines versions de TRL acceptent train_dataset=None, d'autres exigent un dataset.
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

    raise FileNotFoundError(f"Aucun adaptateur LoRA trouvé sous : {root}")



def download_adapter_from_wandb() -> Path:
    artifact_ref = resolve_wandb_artifact_ref(
        explicit_artifact=SFT_WANDB_ARTIFACT,
        run_path=SFT_WANDB_RUN_PATH,
    )
    downloaded_dir = download_wandb_artifact_dir(
        artifact_ref=artifact_ref,
        download_root=Path(SFT_DOWNLOAD_DIR),
    )

    adapter_dir = find_adapter_dir(downloaded_dir)
    print("Dossier d'adaptateur résolu :", adapter_dir)
    return adapter_dir


def resolve_sft_adapter_dir() -> Path:
    if SFT_WANDB_ARTIFACT or SFT_WANDB_RUN_PATH:
        return download_adapter_from_wandb()

    local_dir = Path(SFT_LOCAL_DIR)
    if not local_dir.exists():
        raise FileNotFoundError(
            f"Dossier local SFT introuvable : {local_dir}. Définissez SFT_WANDB_RUN_PATH/SFT_WANDB_ARTIFACT ou SFT_LOCAL_DIR."
        )
    adapter_dir = find_adapter_dir(local_dir)
    print("Dossier d'adaptateur local résolu :", adapter_dir)
    return adapter_dir


def main() -> None:
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        print(f"Suppression du dossier de sortie existant : {output_path}")
        shutil.rmtree(output_path)

    print(f"Chargement du tokenizer : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Chargement du modèle de base : {MODEL_NAME}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.config.use_cache = False

    print("CUDA disponible :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))
    print_wandb_env(WANDB_PROJECT, WANDB_LOG_MODEL)

    adapter_dir = resolve_sft_adapter_dir()
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    print("Adaptateur SFT entraînable chargé :", adapter_dir)

    data_files = {
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    }
    has_test = Path(TEST_FILE).exists()
    if has_test:
        data_files["test"] = TEST_FILE

    dataset = load_dataset("json", data_files=data_files)
    print("Colonnes brutes train :", dataset["train"].column_names)

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

    print(f"Lignes train conservées : {len(train_dataset)}/{len(train_raw)}")
    print(f"Lignes validation conservées : {len(val_dataset)}/{len(val_raw)}")
    if test_raw is not None and test_dataset is not None:
        print(f"Lignes test conservées : {len(test_dataset)}/{len(test_raw)}")

    if len(train_dataset) == 0:
        raise ValueError("Aucune ligne DPO d'entraînement valide trouvée après prétraitement.")
    if len(val_dataset) == 0:
        raise ValueError("Aucune ligne DPO de validation valide trouvée après prétraitement.")

    print("Première ligne train prétraitée :", train_dataset[0])


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
        peft_config=None,
    )

    print("Démarrage de l'entraînement DPO...")
    trainer.train()

    # Sauvegarde juste après l'entraînement pour que des soucis d'éval finale ne fassent jamais perdre les poids entraînés.
    print(f"Sauvegarde de l'adaptateur/modèle vers : {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Lancement de l'évaluation finale validation...")
    # Important : ne pas passer eval_dataset ici. DPOTrainer a déjà préparé
    # son dataset d'évaluation interne, alors que passer le brut peut provoquer un KeyError
    # sur des colonnes tokenisées absentes (ex. : chosen_input_ids).
    val_metrics = trainer.evaluate(metric_key_prefix="val_final")
    trainer.log(val_metrics)

    test_metrics = None
    if test_dataset is not None:
        if len(test_dataset) == 0:
            print("Évaluation finale test ignorée : aucune ligne test valide après prétraitement.")
        else:
            print("Lancement de l'évaluation finale test...")
            test_trainer = build_eval_only_trainer(
                trained_model=trainer.model,
                eval_dataset=test_dataset,
                training_args=training_args,
                tokenizer=tokenizer,
            )
            test_metrics = test_trainer.evaluate(metric_key_prefix="test_final")
            trainer.log(test_metrics)

    clinical_eval_metrics = None
    if CLINICAL_EVAL_ENABLED:
        print("Lancement de l'évaluation sur jeu clinique indépendant...")
        clinical_eval_metrics = evaluate_clinical_holdout(
            model=trainer.model,
            tokenizer=tokenizer,
            clinical_eval_file=CLINICAL_EVAL_FILE,
            max_seq_length=CLINICAL_EVAL_MAX_LENGTH,
            batch_size=CLINICAL_EVAL_BATCH_SIZE,
        )
        trainer.log(clinical_metrics_for_logging(clinical_eval_metrics))
    else:
        print("Évaluation clinique indépendante désactivée (CLINICAL_EVAL_ENABLED=0).")

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
        "clinical_eval": clinical_eval_metrics,
    }
    save_eval_report(report)

    print("Terminé.")


if __name__ == "__main__":
    main()

import json
import time
import uuid
import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
import statistics

from .models import ExperimentIn, ExperimentOut

# Création du router
router = APIRouter()

# Chemin du fichier de stockage
EXPERIMENTS_PATH = Path(os.getenv("EXPERIMENTS_PATH", "experiments/experiments.json"))


def _load_experiments() -> List[Dict[str, Any]]:
    """Charge les expérimentations depuis le fichier JSON"""
    if not EXPERIMENTS_PATH.exists():
        return []
    try:
        with EXPERIMENTS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement des expérimentations : {e}")
        return []


def _save_experiments(exps: List[Dict[str, Any]]) -> None:
    """Sauvegarde les expérimentations dans le fichier JSON"""
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(exps, f, ensure_ascii=False, indent=2)


@router.post("/", response_model=ExperimentOut)
async def create_experiment(exp: ExperimentIn):
    """Enregistre une nouvelle expérimentation"""
    try:
        record = exp.model_dump()
        record["experiment_id"] = str(uuid.uuid4())
        record["created_at"] = time.time()

        exps = _load_experiments()
        exps.append(record)
        _save_experiments(exps)

        return ExperimentOut(**record)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement : {str(e)}")


@router.get("/", response_model=dict)
async def list_experiments(limit: int = 200):
    """Liste les expérimentations enregistrées"""
    try:
        exps = _load_experiments()
        exps_sorted = sorted(exps, key=lambda x: x.get("created_at", 0), reverse=True)[:limit]
        return {"experiments": exps_sorted, "count": len(exps_sorted)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération : {str(e)}")


@router.get("/stats", response_model=dict)
async def get_experiments_stats():
    """
    Retourne les statistiques : moyenne, écart-type, min, max, taux de succès
    """
    try:
        exps = _load_experiments()

        if not exps:
            return {
                "total_experiments": 0,
                "avg_reward": None,
                "std_reward": None,
                "max_reward": None,
                "min_reward": None,
                "success_rate": None
            }

        rewards = [x.get("total_reward", 0) for x in exps if x.get("total_reward") is not None]
        successes = [x for x in exps if x.get("terminated", False) and x.get("total_reward", 0) > 200]

        avg_reward = sum(rewards) / len(rewards) if rewards else None
        std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0

        return {
            "total_experiments": len(exps),
            "avg_reward": round(avg_reward, 2) if avg_reward else None,
            "std_reward": round(std_reward, 2) if std_reward else None,
            "max_reward": round(max(rewards), 2) if rewards else None,
            "min_reward": round(min(rewards), 2) if rewards else None,
            "success_rate": round(len(successes) / len(exps) * 100, 2) if exps else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul des stats : {str(e)}")
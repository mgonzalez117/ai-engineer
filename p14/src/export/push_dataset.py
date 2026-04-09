from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi

try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    from huggingface_hub.utils import HfHubHTTPError

from src.dataset.io import PROCESSED_DIR


DEFAULT_DATASET_REPO_ID = "MGonzalez117/chsa-triage-medical"
DEFAULT_REVISION = "dev"
DEFAULT_PATH_IN_REPO = "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push .data/dataset/processed vers HF dataset repo (branche dev)."
    )
    parser.add_argument("--repo-id", default=os.getenv("DATASET_REPO_ID", DEFAULT_DATASET_REPO_ID))
    parser.add_argument("--local-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--path-in-repo", default=DEFAULT_PATH_IN_REPO)
    parser.add_argument("--commit-message", default="")
    return parser.parse_args()


def ensure_branch(api: HfApi, repo_id: str, revision: str) -> None:
    if not revision or revision in {"main", "master"}:
        return

    try:
        api.create_branch(repo_id=repo_id, repo_type="dataset", branch=revision)
        print(f"Branch created: {revision}")
    except HfHubHTTPError as exc:
        error_text = str(exc).lower()
        if "already exists" in error_text or "reference already exists" in error_text:
            print(f"Branch already exists: {revision}")
            return
        raise


def push_dataset(
    *,
    repo_id: str,
    local_dir: Path,
    revision: str,
    path_in_repo: str,
    token: str,
    commit_message: str,
) -> None:
    if not token:
        raise ValueError("HF_TOKEN is required.")

    if not local_dir.exists():
        raise FileNotFoundError(f"Local dataset directory not found: {local_dir}")
    if not local_dir.is_dir():
        raise ValueError(f"Local dataset path is not a directory: {local_dir}")

    files = [p for p in local_dir.iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in local dataset directory: {local_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    ensure_branch(api, repo_id=repo_id, revision=revision)

    final_commit_message = commit_message or (
        "Update processed dataset - "
        + datetime.now(UTC).replace(microsecond=0).isoformat()
    )

    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        path_in_repo=path_in_repo.strip("/"),
        revision=revision,
        commit_message=final_commit_message,
    )

    print(f"Dataset pushed to: {repo_id}")
    print(f"Revision: {revision}")
    print(f"Path in repo: {path_in_repo.strip('/') or '/'}")
    if getattr(commit_info, "commit_url", None):
        print(f"Commit URL: {commit_info.commit_url}")
    print(f"Local copy kept at: {local_dir.resolve()}")


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN", "").strip()

    push_dataset(
        repo_id=args.repo_id.strip(),
        local_dir=Path(args.local_dir),
        revision=args.revision.strip(),
        path_in_repo=args.path_in_repo,
        token=token,
        commit_message=args.commit_message.strip(),
    )


if __name__ == "__main__":
    main()


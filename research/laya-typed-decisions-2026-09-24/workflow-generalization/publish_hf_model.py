#!/usr/bin/env python3
"""Stage, hash-verify, then make a new LightJev checkpoint public on HF."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "rongxinzy/LightJev-0.6B-typed-decisions"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN must be provided through the environment")
    model_dir = args.model_dir.resolve()
    for relative in ("README.md", "LICENSE", "NOTICE", "manifest.json", "model.safetensors",
                     "backbone/config.json", "tokenizer/tokenizer.json", "SHA256SUMS"):
        if not (model_dir / relative).is_file():
            raise SystemExit(f"missing required model file: {relative}")

    api = HfApi(token=token)
    if api.repo_exists(REPO_ID, repo_type="model"):
        # A prior upload attempt may have staged this exact new release privately
        # before a network failure during verification. Resume that private repo.
        info = api.model_info(REPO_ID, files_metadata=True)
        if not info.private:
            raise SystemExit(f"refusing to overwrite public repository: {REPO_ID}")
    else:
        api.create_repo(REPO_ID, repo_type="model", private=True)
    api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=str(model_dir),
                      commit_message="Publish LightJev typed-decisions checkpoint")

    local_hash = sha256(model_dir / "model.safetensors")
    info = api.model_info(REPO_ID, files_metadata=True)
    siblings = {file.rfilename: file for file in info.siblings}
    remote = siblings.get("model.safetensors")
    remote_hash = getattr(getattr(remote, "lfs", None), "sha256", None) if remote else None
    if remote_hash is None:
        cached = hf_hub_download(REPO_ID, "model.safetensors", token=token)
        remote_hash = sha256(Path(cached))
    if remote_hash.lower() != local_hash.lower():
        raise SystemExit(f"uploaded weights failed SHA256 verification: local={local_hash} remote={remote_hash}")

    api.update_repo_settings(REPO_ID, private=False)
    public_info = api.model_info(REPO_ID, files_metadata=True)
    if public_info.private:
        raise SystemExit("model repository is still private after publication")
    print(f"PUBLIC https://huggingface.co/{REPO_ID}")
    print(f"MODEL_SHA256 {local_hash}")
    print(f"FILES {len(public_info.siblings)}")


if __name__ == "__main__":
    main()

"""Download every runtime model at a tested, immutable Hugging Face revision."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    from tools.model_manifest import COMMON_MODELS, MUSETALK_MODELS, SEED_MODELS, TRANSLATION_MODELS
except ModuleNotFoundError:  # Direct execution: python tools/download_models.py
    from model_manifest import COMMON_MODELS, MUSETALK_MODELS, SEED_MODELS, TRANSLATION_MODELS


def download_local(repo: str, revision: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n[{repo}] -> {target}", flush=True)
    snapshot_download(repo_id=repo, revision=revision, local_dir=target)


def download_seed(seed_root: Path) -> None:
    checkpoints = seed_root / "checkpoints"
    hf_cache = checkpoints / "hf_cache"
    checkpoints.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)
    jobs = (
        (*SEED_MODELS[0], checkpoints,
         ["DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"]),
        (*SEED_MODELS[1], checkpoints, ["campplus_cn_common.bin"]),
        (*SEED_MODELS[2], hf_cache, None),
        (*SEED_MODELS[3], hf_cache, None),
    )
    for repo, revision, cache, patterns in jobs:
        print(f"\n[{repo}] -> cache {cache}", flush=True)
        snapshot_download(repo_id=repo, revision=revision, cache_dir=cache, allow_patterns=patterns)


def download_musetalk(root: Path) -> None:
    models = root / "models"
    jobs = (
        (*MUSETALK_MODELS[0], models,
         ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"]),
        (*MUSETALK_MODELS[1], models / "sd-vae",
         ["config.json", "diffusion_pytorch_model.bin"]),
        (*MUSETALK_MODELS[2], models / "whisper",
         ["config.json", "pytorch_model.bin", "preprocessor_config.json"]),
        (*MUSETALK_MODELS[3], models / "dwpose",
         ["dw-ll_ucoco_384.pth"]),
        (*MUSETALK_MODELS[4], models / "face-parse-bisent",
         ["79999_iter.pth", "resnet18-5c106cde.pth"]),
    )
    for repo, revision, target, patterns in jobs:
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n[{repo}] -> {target}", flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=target, allow_patterns=patterns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--profile", choices=("cuda", "apple_silicon"), default="cuda")
    parser.add_argument("--translation-model", choices=("7b", "1.8b"))
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    translation_key = args.translation_model or ("7b" if args.profile == "apple_silicon" else "cuda")
    for repo, revision, relative in (*COMMON_MODELS, TRANSLATION_MODELS[translation_key]):
        download_local(repo, revision, runtime / relative)
    download_seed(runtime / "seed-vc")
    if args.profile == "cuda":
        download_musetalk(runtime / "musetalk")
    print("\nAll models downloaded.")


if __name__ == "__main__":
    main()

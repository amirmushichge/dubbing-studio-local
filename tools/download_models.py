"""Download every runtime model at a tested, immutable Hugging Face revision."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = (
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "81b57e8e790c07e8fa7f82d8bdfd7574d485c396", "qwen3-tts/models/Qwen3-TTS-12Hz-1.7B-Base"),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "2c2c027c574a85c0d5a5d48e3cfed1a8a051ff9d", "qwen3-tts/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    ("tencent/Hy-MT2-7B-FP8", "0b585e94c179bf49ee34a5cc903f6a88104f8c8e", "hymt/models/Hy-MT2-7B-FP8"),
    ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478", "asr/models/faster-whisper-large-v3"),
)


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
        ("Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e", checkpoints,
         ["DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"]),
        ("funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede", checkpoints, ["campplus_cn_common.bin"]),
        ("nvidia/bigvgan_v2_22khz_80band_256x", "633ff708ed5b74903e86ff1298cf4a98e921c513", hf_cache, None),
        ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", hf_cache, None),
    )
    for repo, revision, cache, patterns in jobs:
        print(f"\n[{repo}] -> cache {cache}", flush=True)
        snapshot_download(repo_id=repo, revision=revision, cache_dir=cache, allow_patterns=patterns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for repo, revision, relative in MODELS:
        download_local(repo, revision, runtime / relative)
    download_seed(runtime / "seed-vc")
    print("\nВсе модели загружены.")


if __name__ == "__main__":
    main()

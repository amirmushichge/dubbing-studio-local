# Apple Silicon implementation status

The macOS backend is implemented for M1 and newer Macs. Intel Macs are outside the target platform.

## Implemented platform substitutions

| Pipeline stage | Windows NVIDIA | Apple Silicon |
|---|---|---|
| Speech recognition | faster-whisper CUDA FP16 | faster-whisper CPU INT8 |
| Source separation | Demucs CUDA | Demucs MPS with PyTorch fallback |
| Translation | Hy-MT2 7B FP8 on CUDA | Hy-MT2 7B FP16 on 32 GB+, Hy-MT2 1.8B FP16 below 32 GB |
| Native speech | Qwen3-TTS CUDA BF16 | Qwen3-TTS MPS FP16 |
| Voice conversion | Seed-VC CUDA FP16 | Seed-VC MPS FP32 |
| H.264 export | NVENC | VideoToolbox |

The orchestration layer is shared. Runtime paths, devices, dtypes, ASR compute type and video encoder are selected by the `apple_silicon` backend rather than duplicated in a separate application. Interface behavior, error recovery, line-level progress, timing fallbacks, project storage and tests are therefore identical on Windows and macOS; product changes must never be maintained as a Windows-only fork.

## Installer and diagnostics

- `setup.command` verifies Apple Silicon, macOS 14+, unified memory and free storage; installs prerequisites through Homebrew; creates isolated Python environments; pins external repositories and model revisions; downloads the appropriate translator; prefetches Demucs; runs diagnostics; and creates a Desktop launcher.
- `start.command` validates the runtime, prevents an unrelated service from being mistaken for the studio, opens the local interface and stops with the Terminal window.
- `doctor.command` verifies FFmpeg, VideoToolbox, Apple architecture, PyTorch MPS, every environment and every required model.

## Validation boundary

Cross-platform path selection, encoder selection, worker arguments, compilation and unit/JavaScript tests are validated on both Windows and macOS runners. The mock interface flow is also checked on the Windows development host. A physical Apple Silicon machine is still required for the final real-model reference suite and performance measurements. Until that suite passes, the Mac edition is labelled experimental alpha rather than silently claiming identical throughput or quality.

The physical-device release suite is:

1. one- and multi-speaker videos;
2. adult, child, quiet and expressive voices;
3. English, Russian and Simplified Chinese in both short and long lines;
4. punctuation, pauses and visible-speech coverage;
5. voice stability, clicks, loudness and peak checks;
6. complete H.264/AAC decode and caption placement;
7. cancel, delete, restart recovery and repeated exports.

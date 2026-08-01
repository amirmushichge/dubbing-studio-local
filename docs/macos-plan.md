# macOS Apple Silicon port plan

## Goal

Preserve the same workflow and quality gates on M1/M2/M3/M4 Macs without presenting a slow CPU build as a production port. Intel Macs are outside the target platform.

## Components that cannot move unchanged

- CUDA devices used by every AI worker;
- FP8 loading for Hy-MT2;
- `h264_nvenc` in the final FFmpeg render;
- CUDA-specific Seed-VC and Demucs paths;
- current speed, memory and voice-QA thresholds.

## Work plan

1. Introduce `cuda` and `apple_silicon` backend profiles without duplicating orchestration logic.
2. Replace ASR with an Apple Silicon-optimized backend and compare punctuation and timestamps against faster-whisper Large v3.
3. Select a local translation model that runs efficiently through MLX/Metal and repeat name, number, length and punctuation tests.
4. Validate Qwen3-TTS and Seed-VC on Metal. If stability is insufficient, select a separate TTS and voice-conversion pair while preserving the two-stage principle: native pronunciation first, identity transfer second.
5. Use `h264_videotoolbox` and validate captions, AAC, duration and complete decoding.
6. Create `setup.command`, `start.command` and Homebrew / Xcode Command Line Tools diagnostics without manual config edits.
7. Run the reference set: one and multiple speakers, adult and child voices, Russian ↔ English and Russian → Chinese.

## Release gates

- one-command setup on a clean Apple Silicon Mac;
- fully local inference after the first download;
- no strong source accent in the target language;
- stable identity for every speaker;
- punctuation-aware delivery;
- no boundary clicks;
- line speed remains below the defined limit;
- the final MP4 decodes completely;
- documented unified-memory requirements and expected speed.

Until every gate passes, macOS remains a roadmap item rather than a supported platform.

# MVP architecture

## Principle

Dubbing Studio is a local orchestrator for specialized models. The web server does not retain heavy models in its own process; it starts each worker inside that model’s isolated environment and stores artifacts in the project directory.

## Project state

`uploaded → queued → analyzing → review → queued → rendering → complete | quality_review`

Any stage may transition to `failed`. A project can be corrected and run again without deleting earlier artifacts. New finished renders use `complete`; legacy `quality_review` projects remain previewable and downloadable, with their notices shown as advisory information.

Long-running model workers write atomic progress snapshots. The server maps model loading, generated lines and timing-adaptation passes into monotonic project percentages; the client animates each confirmed change in one-percent steps.

Caption adjustments are a lightweight export operation. The app reuses `work/dub.wav`, `work/background.wav` and `work/translation.json`, writes positioned ASS captions, and rebuilds only the final MP4 while preserving the previous export.

## Project directory

```text
data/projects/<id>/
  input/       source video
  preview/     thumbnail
  work/        WAV, JSON, profiles and intermediate synthesis
  output/      finished versions
  logs/        stdout/stderr for every worker
  project.json state, settings, QA and event history
```

## Accelerator queue

The backend runs a single heavy-task queue. This prevents Hy-MT2, Qwen3-TTS and Seed-VC from competing for NVIDIA VRAM or Apple unified memory. Voice previews reserve the same queue so they cannot race a dub. Queued payloads are persisted in `project.json` and restored on startup. Users can stop a queued or active task; active cancellation terminates its complete child-process tree. Deleting a busy project performs that cancellation first and waits for its files to be released. A task interrupted by an application shutdown is marked Failed rather than left permanently stuck, allowing a safe manual retry.

## Export safety

Each render uses a unique run ID for its MP4 and SRT. Earlier files remain in `output/` and their metadata is retained in `project.json → exports`. Transcript changes archive and invalidate the current result. QA notices are advisory: any current, fully written export remains downloadable in both `complete` and legacy `quality_review` states.

## Pipeline

1. `ffprobe` reads metadata and FFmpeg creates a thumbnail.
2. FFmpeg extracts PCM audio.
3. Demucs `htdemucs` separates speech and background.
4. faster-whisper Large v3 produces lines, words and timestamps.
5. Voice embeddings plus agglomerative clustering identify speakers and collect 10–25 seconds of reference audio per person.
6. The user reviews transcript, punctuation and speaker assignment.
7. Hy-MT2 translates ordered JSON and adapts line length.
8. Qwen3-TTS VoiceDesign creates native articulation profiles.
9. Qwen3-TTS Base synthesizes deterministic lines.
10. In source-match mode, Seed-VC V1 transfers vocal characteristics and the QA report shows the measured similarity.
11. Lines fit the original timeline with a 1.35× maximum speed and boundary fades.
12. FFmpeg mixes background, burns optional captions and encodes MP4.
13. QA re-transcribes the dub, compares speaker embeddings and checks boundaries, LUFS, true peak, streams and full decoding.

## MVP limits

- Production TTS is exposed for ten validated output languages.
- Catalog voice mode supports one speaker. Multi-speaker video uses a separate clone for each person.
- Automatic speaker count is heuristic. Human review remains a required quality gate.

## Compute profiles

- `cuda`: faster-whisper FP16, Demucs CUDA, Hy-MT2 7B FP8, Qwen3-TTS BF16, Seed-VC FP16 and NVENC.
- `apple_silicon`: faster-whisper CPU INT8, Demucs MPS, non-FP8 Hy-MT2 FP16, Qwen3-TTS MPS FP16, Seed-VC MPS FP32 and VideoToolbox.
- `cpu`: a development/fallback profile using CPU workers and libx264; it is not presented as a practical production configuration for the full model stack.

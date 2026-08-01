# MVP architecture

## Principle

Dubbing Studio is a local orchestrator for specialized models. The web server does not retain heavy models in its own process; it starts each worker inside that model’s isolated environment and stores artifacts in the project directory.

## Project state

`uploaded → queued → analyzing → review → queued → rendering → complete | quality_review`

Any stage may transition to `failed`. A project can be corrected and run again without deleting earlier artifacts. `quality_review` keeps the preview available but blocks download until a later version passes every automated gate.

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

## GPU queue

The backend runs a single heavy-task queue. This prevents Hy-MT2, Qwen3-TTS and Seed-VC from competing for RTX 4080 Super VRAM. Queued payloads are persisted in `project.json` and restored on startup. A task interrupted while a model was active is marked Failed rather than left permanently stuck, allowing a safe manual retry.

## Export safety

Each render uses a unique run ID for its MP4 and SRT. Earlier files remain in `output/` and their metadata is retained in `project.json → exports`. Transcript changes archive and invalidate the current result. Preview media is available during quality review, while `/download/*` endpoints require the project to be `complete`.

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
10. In clone mode, Seed-VC V1 transfers only the original voice identity.
11. Lines fit the original timeline with a 1.35× maximum speed and boundary fades.
12. FFmpeg mixes background, burns optional captions and encodes MP4.
13. QA re-transcribes the dub, compares speaker embeddings and checks boundaries, LUFS, true peak, streams and full decoding.

## MVP limits

- Production TTS is exposed for ten validated output languages.
- Catalog voice mode supports one speaker. Multi-speaker video uses a separate clone for each person.
- Automatic speaker count is heuristic. Human review remains a required quality gate.

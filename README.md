# Dubbing Studio Local

A fully local workspace for translating and professionally dubbing video. Dubbing Studio transcribes speech, separates voices from the original mix, translates each line, generates native pronunciation, restores the identity of every speaker, fits the performance to the original timing and exports a finished video.

After the first setup, inference runs on your computer. Source videos and voice samples are not sent to a cloud API.

> Clone only your own voice or a voice you have explicit permission to use. Never use this software for deception, impersonation, fraud or identity-verification bypass.

![Dubbing Studio home screen](docs/images/dashboard.png)

## What it does

- Imports MP4, MOV, MKV, WebM and M4V files up to 12 GB.
- Detects the source language and number of speakers automatically or accepts manual values.
- Lets you correct the transcript and punctuation before translation.
- Translates locally with Tencent Hy-MT2.
- Preserves a separate, stable voice identity for every speaker.
- Offers previewable catalog voices for single-speaker videos when cloning is not needed.
- Controls delivery expression and background-audio level.
- Disables captions completely or applies one of six compact, live-preview caption treatments in Manrope, with independent Small/Medium/Large sizing and White/Yellow/Black color choices.
- Checks duration, recognition coverage, voice similarity, boundaries, peaks, loudness and output decoding.
- Blocks downloads when a mandatory quality gate reports a warning.
- Keeps versioned MP4 and SRT exports instead of overwriting an earlier render.
- Recovers queued work after a restart and marks an interrupted active stage for a safe retry.
- Works offline after the complete first installation.

## System requirements

Automatic setup currently targets **Windows 10/11 with an NVIDIA GPU**.

| Component | Minimum | Recommended |
|---|---:|---:|
| GPU | NVIDIA with 12 GB VRAM | RTX 4080 / 4080 Super or better, 16 GB+ |
| System memory | 24 GB | 32 GB+ |
| Free storage | 45 GB | 55 GB+ plus project storage |
| Internet | Required for initial setup | Stable broadband connection |
| Driver | Recent NVIDIA driver | NVIDIA Studio Driver |

AMD GPUs, Intel GPUs, macOS and Linux are not supported by the production installer yet. CPU-only processing is not practical for this model stack. The models occupy roughly 30 GB, and a finished dub normally takes longer to process than the source video duration.

### macOS

An Apple Silicon edition is planned for M1 and newer Macs. This is a real runtime port, not a renamed setup script: CUDA, FP8 and NVENC components must be replaced and voice quality must be calibrated again on Metal/MLX. Intel Macs are not planned. See the [Apple Silicon port plan](docs/macos-plan.md).

## Installation for complete beginners

### 1. Download the project

On GitHub, select **Code → Download ZIP**, then extract the archive to a normal folder such as:

```text
F:\Dubbing-Studio
```

Do not run the application from inside the ZIP and do not place it in `Program Files`.

Git users can clone it instead:

```powershell
git clone https://github.com/amirmushichge/dubbing-studio-local.git
cd dubbing-studio-local
```

### 2. Run automatic setup

Double-click **`setup.bat`** and wait for the green success message. Setup will:

1. check Windows, disk capacity and NVIDIA support;
2. install missing Git, Python 3.10 and FFmpeg through `winget` or a pinned fallback installer;
3. create isolated Python environments;
4. download tested revisions of Qwen3-TTS, Hy-MT2, faster-whisper, Demucs and Seed-VC;
5. run diagnostics;
6. create a desktop shortcut.

Windows may ask for permission to install prerequisites. If the connection is interrupted, run `setup.bat` again: model downloads resume instead of starting over.

### 3. Start the studio

Double-click **`start.bat`** or the **Dubbing Studio** desktop shortcut. Your browser opens:

```text
http://127.0.0.1:8765
```

This is a private address on your own computer, not a public website. Keep the server window open while a project is processing. Close it or press `Ctrl+C` to stop the studio.

## Creating a dub

1. Drop a video onto the home screen or choose a file.
2. Select the source language or leave automatic detection enabled.
3. Select the number of speakers or leave it on automatic.
4. Start the analysis.
5. Review the transcript, punctuation and speaker assignments. These corrections directly affect translation and delivery.
6. Select the output language.
7. Choose a voice mode:
   - **Preserve original voices** retains a separate timbre and manner for each person;
   - **Choose a new voice** creates a catalog voice for a single-speaker video.
8. Set expression and background-audio level. A background value of `70%` means 30% quieter than the source mix.
9. Enable or disable captions. If enabled, select a visual treatment, size and color from the live previews, then choose whether captions should be burned into the video.
10. Select render quality and create the dub.
11. Review the quality report. MP4 and SRT downloads unlock only after every automated gate passes.

Recommended starting point: original voice preservation, moderate expression, 42% background audio, Clean Medium White captions and High quality.

![Dub configuration](docs/images/delivery.png)

## Output languages

| Code | Language |
|---|---|
| `zh` | Chinese, Simplified |
| `en` | English |
| `ru` | Russian |
| `de` | German |
| `fr` | French |
| `es` | Spanish |
| `it` | Italian |
| `pt` | Portuguese |
| `ja` | Japanese |
| `ko` | Korean |

Source transcription supports more languages, but production output is restricted to the tested intersection of the translator and speech synthesizer. Bosnian is not in the validated set yet; it requires separate TTS, pronunciation and quality calibration.

## How the pipeline works

```text
video
  → FFmpeg extracts audio
  → Demucs separates speech from the original mix
  → faster-whisper transcribes words and timing
  → voice embeddings group individual speakers
  → Hy-MT2 translates while preserving punctuation
  → Qwen3-TTS creates native target-language pronunciation
  → Seed-VC restores each original voice identity
  → lines are fitted to the original time slots
  → FFmpeg mixes background, voices and captions
  → automated QA validates the result
```

The two-stage voice design is intentional. Direct cross-language voice conversion can preserve the source accent and destabilize identity. Qwen first creates native pronunciation; Seed-VC then transfers the original timbre onto that performance.

## Project files

```text
Dubbing-Studio/
├─ app/                 web server and orchestration
├─ static/              English AmirStyle interface
├─ workers/             ASR, translation, TTS, voice conversion and QA
├─ runtime/             environments and models created by setup.bat
├─ data/projects/       sources, intermediate files and exports
├─ logs/                setup logs
└─ tests/               lightweight automated tests
```

`runtime/`, `data/` and `logs/` are excluded from Git. To keep projects on another drive, copy `.env.example` to `.env` and set `DUBBING_STUDIO_DATA`. To place models elsewhere, set `DUBBING_STUDIO_RUNTIME` **before the first setup**.

Every successful render receives a unique ID and remains in the project’s `output/` directory. Editing a completed transcript archives the current result and returns the project to Review so stale media cannot be presented as current. The `exports` array in `project.json` records the version history.

## Updating and uninstalling

Git installations can run **`update.bat`**. It pulls the latest commit and safely reapplies setup. ZIP users can download the latest archive and run `setup.bat` again.

To uninstall, save any exports you need from `data/projects`, then delete the Dubbing Studio folder and desktop shortcut. Dubbing Studio does not install a background service. Git, Python and FFmpeg remain installed because other applications may use them.

## Diagnostics and troubleshooting

Run `doctor.ps1`. Every required runtime, model, FFmpeg and NVIDIA check should report `OK`.

| Problem | Resolution |
|---|---|
| SmartScreen blocks BAT/PS1 | Select **More info → Run anyway** only if the project came from the trusted repository. |
| `winget` is unavailable | Setup uses pinned direct installers for Python, Git and FFmpeg. |
| NVIDIA is not detected | Install the latest NVIDIA Studio Driver, restart Windows and verify `nvidia-smi`. |
| Not enough storage | Free at least 45 GB on the runtime drive or set `DUBBING_STUDIO_RUNTIME`. |
| A download was interrupted | Run `setup.bat` again; Hugging Face continues incomplete files. |
| Port 8765 is in use | If Dubbing Studio is already running, the script opens it. Otherwise stop the process using that port. |
| A task was queued when the app closed | Start Dubbing Studio again; queued tasks are restored automatically. A task interrupted mid-model is marked Failed so it can be retried safely. |
| Download is blocked by QA | Read the listed warning, correct the transcript, speaker assignment or settings, then render a new version. The preview remains available for diagnosis. |
| The dub has a source accent | Verify the target language, preserve the original voice, correct punctuation and avoid maximum expression. |
| Voice identity changes | Verify speaker count and assignments. Longer clean speech references produce more stable identities. |
| Clicks or noise are audible | Reduce the background level and review QA warnings. Clean source dialogue produces the best result. |
| A line is cut off | Shorten its translation. Automatic fitting limits speed so delivery remains natural. |
| Caption characters are wrong | Use the generated UTF-8 BOM SRT and a modern video player. |

Project logs are stored in `data/projects/<ID>/logs`. When reporting an issue, attach `doctor.ps1` output and the relevant text log, never private video or voice samples.

## UI development without models

Use mock mode to test the interface and workflow without downloading AI models:

```powershell
$env:DUBBING_STUDIO_MOCK='1'
.\start.ps1
```

Mock mode validates uploads, navigation and test exports. It does not measure real AI quality.

## Development

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
node --check static\app.js
```

The API is built with FastAPI. Main routes include `/api/health`, `/api/catalog`, `/api/projects`, `/analyze`, `/render` and `/ws/projects/{id}`. One heavy GPU task runs at a time so translation, synthesis and conversion models do not compete for VRAM.

All external repositories and model weights are pinned to immutable revisions in `setup.ps1` and `tools/download_models.py`. GitHub Actions does not download large models; CI compiles Python, runs unit tests and validates JavaScript and PowerShell.

## Privacy and limitations

- Inference is local after setup; GitHub and Hugging Face are contacted only to download dependencies and models.
- Project data is not encrypted at rest. Any Windows user with access to `data/projects` can read it.
- Quality depends on source cleanliness, overlapping speech, line duration, language and available VRAM.
- The application does not guarantee perfect identity, emotion or lip synchronization.
- Always review the complete result and respect rights to the video, music, translation and voice.

## Licenses and credits

The original interface and orchestration code use the MIT License. Setup downloads independent third-party components under their own terms:

- Qwen3-TTS — Apache-2.0;
- Tencent Hy-MT2 — Apache-2.0;
- faster-whisper — MIT;
- Demucs — MIT;
- Seed-VC — GPL-3.0;
- Linly-Dubbing — Apache-2.0, used as a workflow research reference rather than a required runtime component;
- Manrope — SIL Open Font License 1.1.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source links and pinned revisions.

## FAQ

**Is Codex required after installation?** No. The complete workflow runs from the local web interface.

**Is the internet required for every project?** No. Models work offline after complete setup.

**Can multiple videos process at once?** Projects may be imported in advance, but heavy GPU work runs sequentially.

**Can captions be disabled?** Yes. You can also save SRT without burning captions into the video.

**Which voice mode should I use?** Preserve original voices for identity and multi-speaker video. Use a catalog voice when identity is not required and the video has one speaker.

**Why is setup so large?** Recognition, translation, native speech generation, voice conversion and music separation use specialized local models.

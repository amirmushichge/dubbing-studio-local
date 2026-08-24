from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel

try:
    from .device import empty_device_cache, model_load_options
except ImportError:  # Executed as a standalone worker script.
    from device import empty_device_cache, model_load_options


def command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def report_progress(path_value: str | None, current: int, total: int, label: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump({"current": current, "total": total, "label": label}, handle)
        temporary = Path(handle.name)
    temporary.replace(path)


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def atempo(speed: float) -> str:
    parts = []
    remaining = speed
    while remaining > 2:
        parts.append("atempo=2")
        remaining /= 2
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def fit_line(source: Path, target: Path, seconds: float, timing_fallback: bool = False) -> float:
    trimmed = target.with_suffix(".trim.wav")
    command([
        "ffmpeg", "-v", "error", "-y", "-i", str(source), "-af",
        "silenceremove=start_periods=1:start_silence=0.03:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_silence=0.12:start_threshold=-50dB,areverse",
        "-ar", "24000", "-ac", "1", str(trimmed),
    ])
    source_seconds = duration(trimmed)
    fill_ratio = source_seconds / max(seconds, 0.25)
    minimum_fill = .70 if timing_fallback else .82
    maximum_speed = 1.50 if timing_fallback else 1.35
    if seconds >= 2.0 and fill_ratio < minimum_fill:
        raise RuntimeError(f"Line fills {fill_ratio:.3f} of its time slot; expand its translation: {source.name}")
    speed = max(.78 if timing_fallback else .9, fill_ratio)
    if speed > maximum_speed:
        raise RuntimeError(f"Line requires {speed:.3f}x speed; shorten its translation: {source.name}")
    fade_out = max(0.0, seconds - 0.025)
    command([
        "ffmpeg", "-v", "error", "-y", "-i", str(trimmed), "-af",
        f"aresample=24000,{atempo(speed)},apad=pad_dur={seconds:.6f},atrim=duration={seconds:.6f},highpass=f=60,lowpass=f=11500,afade=t=in:st=0:d=0.015,afade=t=out:st={fade_out:.6f}:d=0.025,loudnorm=I=-18:TP=-2:LRA=7",
        "-ar", "24000", "-ac", "1", str(target),
    ])
    return speed


def make_profiles(config: dict, speakers: list[dict], profile_dir: Path) -> dict[str, dict]:
    profile_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(config["qwen_root"]) / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    missing = [speaker for speaker in speakers if not (profile_dir / f"{speaker['id']}.wav").exists()]
    if missing:
        model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            **model_load_options(config.get("device", "cuda:0"), config.get("dtype", "bfloat16")),
            attn_implementation="sdpa",
        )
        descriptions = []
        for speaker in missing:
            base = config["voice_description"] if config["voice_mode"] == "catalog" else speaker.get("profile", config["voice_description"])
            energy = "restrained and realistic" if config["expression"] < 0.35 else "natural expressive conversational delivery" if config["expression"] < 0.7 else "lively expressive delivery without theatrical exaggeration"
            descriptions.append(f"{base} Native {config['tts_language']} pronunciation with no foreign accent; {energy}.")
        wavs, sample_rate = model.generate_voice_design(
            text=[config["sample_text"]] * len(missing),
            language=[config["tts_language"]] * len(missing),
            instruct=descriptions,
            do_sample=True, temperature=0.42, top_p=0.72,
        )
        for speaker, description, wav in zip(missing, descriptions, wavs):
            sf.write(profile_dir / f"{speaker['id']}.wav", wav, sample_rate)
            (profile_dir / f"{speaker['id']}.txt").write_text(config["sample_text"], encoding="utf-8")
            (profile_dir / f"{speaker['id']}.description.txt").write_text(description, encoding="utf-8")
        del model
        empty_device_cache(config.get("device", "cuda:0"))
    return {
        speaker["id"]: {"audio": profile_dir / f"{speaker['id']}.wav", "text": config["sample_text"]}
        for speaker in speakers
    }


def synthesize(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    job = Path(config["job_dir"])
    lines = json.loads(Path(config["translation_path"]).read_text(encoding="utf-8"))
    speakers = config["speakers"]
    work = job / "work" / "synthesis"
    raw_dir, fitted_dir = work / "raw", work / "fitted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fitted_dir.mkdir(parents=True, exist_ok=True)
    progress_total = len(lines) + 3
    report_progress(config.get("progress_path"), 0, progress_total, "Preparing target voices")
    profiles = make_profiles(config, speakers, work / "profiles")
    report_progress(config.get("progress_path"), 1, progress_total, "Loading the speech model")

    model_path = Path(config["qwen_root"]) / "models" / "Qwen3-TTS-12Hz-1.7B-Base"
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        **model_load_options(config.get("device", "cuda:0"), config.get("dtype", "bfloat16")),
        attn_implementation="sdpa",
    )
    prompts = {
        speaker_id: model.create_voice_clone_prompt(ref_audio=str(profile["audio"]), ref_text=profile["text"], x_vector_only_mode=False)
        for speaker_id, profile in profiles.items()
    }
    report_progress(config.get("progress_path"), 2, progress_total, "Generating speech · 0 of %d lines" % len(lines))
    report = []
    timing_issues = []
    timeline = np.zeros(round(config["duration"] * 24000), dtype=np.float32)
    for index, item in enumerate(lines):
        speaker = item["speaker"]
        raw = raw_dir / f"{index:04d}_{speaker}.wav"
        raw_text = raw.with_suffix(".txt")
        current_text = item["translation"].strip()
        cached_text = raw_text.read_text(encoding="utf-8").strip() if raw_text.exists() else None
        if not raw.exists() or cached_text != current_text:
            wavs, sample_rate = model.generate_voice_clone(
                text=current_text, language=config["tts_language"],
                voice_clone_prompt=prompts[speaker], do_sample=False,
            )
            sf.write(raw, wavs[0], sample_rate)
            raw_text.write_text(current_text, encoding="utf-8")
        report_progress(
            config.get("progress_path"), index + 3, progress_total,
            f"Generating speech · {index + 1} of {len(lines)} lines",
        )
        fitted = fitted_dir / f"{index:04d}_{speaker}.wav"
        try:
            speed = fit_line(raw, fitted, item["end"] - item["start"], bool(config.get("timing_fallback")))
        except RuntimeError as exc:
            timing_issues.append({"index": index, "speaker": speaker, "error": str(exc)})
            continue
        audio, sample_rate = sf.read(fitted, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        start = round(item["start"] * sample_rate)
        end = min(start + len(audio), len(timeline))
        timeline[start:end] += audio[:end - start]
        report.append({"index": index, "speaker": speaker, "speed": round(speed, 3)})

    timing_issues_path = work / "timing_issues.json"
    if timing_issues:
        timing_issues_path.write_text(json.dumps(timing_issues, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"Detected {len(timing_issues)} dubbing timing issues")
    timing_issues_path.unlink(missing_ok=True)

    native_timeline = work / "native_timeline.wav"
    sf.write(native_timeline, timeline, 24000, subtype="PCM_16")
    manifest = {"native_timeline": str(native_timeline), "report": report, "roles": []}
    if config["voice_mode"] == "catalog":
        final = job / "work" / "dub.wav"
        sf.write(final, timeline, 24000, subtype="PCM_16")
        manifest["final"] = str(final)
    else:
        role_dir = work / "roles"
        role_dir.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(round(0.35 * 24000), dtype=np.float32)
        for speaker in speakers:
            speaker_id = speaker["id"]
            chunks, mapping, cursor = [], [], 0
            for index, item in enumerate(lines):
                if item["speaker"] != speaker_id:
                    continue
                path = fitted_dir / f"{index:04d}_{speaker_id}.trim.wav"
                audio, sample_rate = sf.read(path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                start = cursor / sample_rate
                chunks.append(audio)
                cursor += len(audio)
                end = cursor / sample_rate
                mapping.append({"index": index, "comp_start": start, "comp_end": end, "target_start": item["start"], "target_end": item["end"]})
                chunks.append(gap)
                cursor += len(gap)
            # Diarization can expose a transient/false-positive speaker that has
            # no translated lines. Do not create a one-sample role: Seed-VC
            # cannot process it and there is nothing to restore for that role.
            if not mapping:
                continue
            stream = np.concatenate(chunks)
            stream_path = role_dir / f"{speaker_id}_native.wav"
            map_path = role_dir / f"{speaker_id}_mapping.json"
            sf.write(stream_path, stream, 24000, subtype="PCM_16")
            map_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["roles"].append({
                "speaker": speaker_id, "source": str(stream_path), "mapping": str(map_path),
                "reference": speaker["reference"],
            })
    (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_progress(config.get("progress_path"), progress_total, progress_total, "Native speech ready")
    print(work / "manifest.json")


def preview(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = Path(config["output"])
    target.parent.mkdir(parents=True, exist_ok=True)
    model_path = Path(config["qwen_root"]) / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        **model_load_options(config.get("device", "cuda:0"), config.get("dtype", "bfloat16")),
        attn_implementation="sdpa",
    )
    wavs, sample_rate = model.generate_voice_design(
        text=config["sample_text"], language=config["tts_language"], instruct=config["voice_description"],
        do_sample=True, temperature=0.42, top_p=0.72,
    )
    sf.write(target, wavs[0], sample_rate)
    print(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    synth = sub.add_parser("synthesize")
    synth.add_argument("config", type=Path)
    preview_parser = sub.add_parser("preview")
    preview_parser.add_argument("config", type=Path)
    args = parser.parse_args()
    if args.command == "synthesize":
        synthesize(args.config)
    else:
        preview(args.config)


if __name__ == "__main__":
    main()

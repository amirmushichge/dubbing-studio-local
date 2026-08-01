from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def atempo(speed: float) -> str:
    return f"atempo={speed:.6f}"


def fit(source: Path, target: Path, seconds: float) -> float:
    speed = max(1.0, duration(source) / max(seconds, 0.25))
    if speed > 1.35:
        raise RuntimeError(f"Converted line requires {speed:.3f}x")
    fade_out = max(0.0, seconds - 0.025)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(source), "-af",
        f"aresample=24000,{atempo(speed)},apad=pad_dur={seconds:.6f},atrim=duration={seconds:.6f},highpass=f=60,lowpass=f=10500,afade=t=in:st=0:d=0.015,afade=t=out:st={fade_out:.6f}:d=0.025,loudnorm=I=-18:TP=-2:LRA=7",
        "-ar", "24000", "-ac", "1", str(target),
    ], check=True)
    return speed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("translation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--duration", type=float, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lines = json.loads(args.translation.read_text(encoding="utf-8"))
    work = args.manifest.parent / "converted_lines"
    work.mkdir(parents=True, exist_ok=True)
    timeline = np.zeros(round(args.duration * 24000), dtype=np.float32)
    report = []
    for role in manifest["roles"]:
        converted = Path(role["converted"])
        audio, sample_rate = sf.read(converted, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        mapping = json.loads(Path(role["mapping"]).read_text(encoding="utf-8"))
        ratio = duration(converted) / duration(Path(role["source"]))
        for entry in mapping:
            start = round(entry["comp_start"] * ratio * sample_rate)
            end = round(entry["comp_end"] * ratio * sample_rate)
            clip = audio[max(0, start):min(len(audio), end)]
            active = np.flatnonzero(np.abs(clip) > 0.003)
            if len(active):
                clip = clip[max(0, int(active[0] - .03 * sample_rate)):min(len(clip), int(active[-1] + .12 * sample_rate))]
            raw = work / f"{entry['index']:04d}_{role['speaker']}.raw.wav"
            fitted = work / f"{entry['index']:04d}_{role['speaker']}.wav"
            sf.write(raw, clip, sample_rate, subtype="PCM_16")
            item = lines[entry["index"]]
            speed = fit(raw, fitted, item["end"] - item["start"])
            line, sr = sf.read(fitted, dtype="float32")
            if line.ndim > 1:
                line = line.mean(axis=1)
            target_start = round(item["start"] * sr)
            target_end = min(target_start + len(line), len(timeline))
            timeline[target_start:target_end] += line[:target_end - target_start]
            report.append({"index": entry["index"], "speaker": role["speaker"], "speed": round(speed, 3)})
    peak = float(np.max(np.abs(timeline)))
    if peak > .95:
        timeline *= .95 / peak
    sf.write(args.output, timeline, 24000, subtype="PCM_16")
    (args.manifest.parent / "clone_timing_report.json").write_text(json.dumps(sorted(report, key=lambda x: x["index"]), indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()


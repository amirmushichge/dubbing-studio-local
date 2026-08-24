from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def loudness(path: Path) -> tuple[float | None, float | None]:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", os.devnull],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    summary = result.stderr[result.stderr.rfind("Summary:"):]
    integrated = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    return (float(integrated.group(1)) if integrated else None, float(peak.group(1)) if peak else None)


def activity_mask(signal: np.ndarray, sample_rate: int, frame_seconds: float = .04) -> tuple[np.ndarray, float]:
    """Return a conservative speech/activity mask at a fixed frame rate."""
    frame_size = max(1, round(frame_seconds * sample_rate))
    usable = len(signal) - len(signal) % frame_size
    if not usable:
        return np.asarray([], dtype=bool), frame_seconds
    frames = signal[:usable].reshape(-1, frame_size)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    floor = float(np.quantile(rms, .25))
    peak = float(np.quantile(rms, .9))
    threshold = max(.0025, floor * 2.5, peak * .08)
    return rms > threshold, frame_size / sample_rate


def uncovered_source_regions(
    source_signal: np.ndarray,
    dub_signal: np.ndarray,
    source_sample_rate: int,
    dub_sample_rate: int | None = None,
    minimum_seconds: float = .72,
) -> list[dict[str, float]]:
    """Find sustained source speech/activity that has no corresponding dubbed delivery."""
    dub_sample_rate = dub_sample_rate or source_sample_rate
    source_mask, frame_seconds = activity_mask(source_signal, source_sample_rate)
    dub_mask, dub_frame_seconds = activity_mask(dub_signal, dub_sample_rate)
    if abs(frame_seconds - dub_frame_seconds) > .001:
        raise RuntimeError("Source and dub activity frames are not aligned")
    count = min(len(source_mask), len(dub_mask))
    if not count:
        return []
    # Permit normal cross-language timing drift, breaths and phrase-final pauses.
    gap_frames = max(1, round(.45 / frame_seconds))
    expanded_dub = np.convolve(dub_mask[:count].astype(np.int8), np.ones(gap_frames * 2 + 1), mode="same") > 0
    missing = source_mask[:count] & ~expanded_dub
    minimum_frames = max(1, round(minimum_seconds / frame_seconds))
    regions: list[dict[str, float]] = []
    start = None
    for index, value in enumerate(np.append(missing, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum_frames:
                regions.append({
                    "start": round(start * frame_seconds, 3),
                    "end": round(index * frame_seconds, 3),
                    "duration": round((index - start) * frame_seconds, 3),
                })
            start = None
    return regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dub", type=Path)
    parser.add_argument("segments", type=Path)
    parser.add_argument("final_video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-vocals", type=Path)
    args = parser.parse_args()

    lines = json.loads(args.segments.read_text(encoding="utf-8"))
    audio, sample_rate = sf.read(args.dub, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    derivative = np.abs(np.diff(mono))
    threshold = max(.08, float(np.quantile(derivative, .999)) * 2) if len(derivative) else .08
    boundary_flags = []
    coverage_flags = []
    for index, item in enumerate(lines):
        for edge in ("start", "end"):
            position = max(1, min(len(mono) - 2, round(item[edge] * sample_rate)))
            jump = abs(float(mono[position] - mono[position - 1]))
            if jump > threshold:
                boundary_flags.append({"index": index, "edge": edge, "jump": round(jump, 4)})
        start = max(0, round(item["start"] * sample_rate))
        end = min(len(mono), round(item["end"] * sample_rate))
        clip = mono[start:end]
        line_seconds = (end - start) / sample_rate
        if line_seconds >= 2 and len(clip):
            frame_size = max(1, round(.02 * sample_rate))
            usable = len(clip) - len(clip) % frame_size
            frames = clip[:usable].reshape(-1, frame_size) if usable else np.empty((0, frame_size))
            rms = np.sqrt(np.mean(frames * frames, axis=1)) if len(frames) else np.asarray([])
            activity_threshold = max(.0025, float(np.quantile(rms, .9)) * .08) if len(rms) else .0025
            active = np.flatnonzero(rms > activity_threshold)
            if not len(active):
                coverage_flags.append({"index": index, "reason": "silent"})
            else:
                leading = float(active[0] * frame_size / sample_rate)
                trailing = float(line_seconds - (active[-1] + 1) * frame_size / sample_rate)
                if leading > max(.55, line_seconds * .18) or trailing > max(.75, line_seconds * .22):
                    coverage_flags.append({
                        "index": index, "leading_silence": round(leading, 3),
                        "trailing_silence": round(max(0, trailing), 3), "duration": round(line_seconds, 3),
                    })

    similarities = {}
    if args.manifest and args.manifest.exists():
        from resemblyzer import VoiceEncoder, preprocess_wav

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        encoder = VoiceEncoder()
        for role in manifest.get("roles", []):
            if not role.get("converted") or not Path(role["reference"]).exists():
                continue
            source_embedding = encoder.embed_utterance(preprocess_wav(Path(role["reference"])))
            converted_embedding = encoder.embed_utterance(preprocess_wav(Path(role["converted"])))
            similarities[role["speaker"]] = round(cosine(source_embedding, converted_embedding), 4)

    source_gap_flags = []
    if args.source_vocals and args.source_vocals.exists():
        source_audio, source_rate = sf.read(args.source_vocals, always_2d=True, dtype="float32")
        source_mono = source_audio.mean(axis=1)
        source_gap_flags = uncovered_source_regions(source_mono, mono, source_rate, sample_rate)

    integrated, peak = loudness(args.final_video)
    warnings = []
    if boundary_flags:
        warnings.append(f"Detected {len(boundary_flags)} suspicious audio boundaries")
    if coverage_flags:
        warnings.append(f"Detected {len(coverage_flags)} speech regions with uncovered visible delivery")
    # Demucs' vocal stem can retain sung stingers and music transients, so only
    # source-only regions that overlap a known transcript interval are a hard
    # gate. This catches phrase tails where lips/source speech continue after
    # the translated delivery has already stopped.
    aligned_source_gaps = []
    for region in source_gap_flags:
        for index, item in enumerate(lines):
            overlap = min(float(region["end"]), float(item["end"])) - max(float(region["start"]), float(item["start"]))
            if overlap >= .5:
                aligned_source_gaps.append({**region, "index": index, "overlap": round(overlap, 3)})
                break
    if aligned_source_gaps:
        warnings.append(f"Detected {len(aligned_source_gaps)} transcript-aligned source speech tails without dub")
    if integrated is not None and not (-16 <= integrated <= -13.5):
        warnings.append(f"Integrated loudness outside target range: {integrated} LUFS")
    if peak is not None and peak > -.8:
        warnings.append(f"True peak is too high: {peak} dBFS")
    result = {
        "boundary_flags": boundary_flags, "coverage_flags": coverage_flags,
        "source_gap_flags": source_gap_flags, "aligned_source_gap_flags": aligned_source_gaps,
        "speaker_similarity": similarities,
        "integrated_lufs": integrated, "true_peak_dbfs": peak, "warnings": warnings,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

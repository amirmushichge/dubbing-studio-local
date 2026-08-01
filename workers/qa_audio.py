from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def loudness(path: Path) -> tuple[float | None, float | None]:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "NUL"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    summary = result.stderr[result.stderr.rfind("Summary:"):]
    integrated = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    return (float(integrated.group(1)) if integrated else None, float(peak.group(1)) if peak else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dub", type=Path)
    parser.add_argument("segments", type=Path)
    parser.add_argument("final_video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    lines = json.loads(args.segments.read_text(encoding="utf-8"))
    audio, sample_rate = sf.read(args.dub, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    derivative = np.abs(np.diff(mono))
    threshold = max(.08, float(np.quantile(derivative, .999)) * 2) if len(derivative) else .08
    boundary_flags = []
    for index, item in enumerate(lines):
        for edge in ("start", "end"):
            position = max(1, min(len(mono) - 2, round(item[edge] * sample_rate)))
            jump = abs(float(mono[position] - mono[position - 1]))
            if jump > threshold:
                boundary_flags.append({"index": index, "edge": edge, "jump": round(jump, 4)})

    similarities = {}
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        encoder = VoiceEncoder()
        for role in manifest.get("roles", []):
            if not role.get("converted") or not Path(role["reference"]).exists():
                continue
            source_embedding = encoder.embed_utterance(preprocess_wav(Path(role["reference"])))
            converted_embedding = encoder.embed_utterance(preprocess_wav(Path(role["converted"])))
            similarities[role["speaker"]] = round(cosine(source_embedding, converted_embedding), 4)

    integrated, peak = loudness(args.final_video)
    warnings = []
    if boundary_flags:
        warnings.append(f"Detected {len(boundary_flags)} suspicious audio boundaries")
    for speaker, score in similarities.items():
        if score < .76:
            warnings.append(f"Low voice similarity for {speaker}: {score}")
    if integrated is not None and not (-16 <= integrated <= -13.5):
        warnings.append(f"Integrated loudness outside target range: {integrated} LUFS")
    if peak is not None and peak > -.8:
        warnings.append(f"True peak is too high: {peak} dBFS")
    result = {
        "boundary_flags": boundary_flags, "speaker_similarity": similarities,
        "integrated_lufs": integrated, "true_peak_dbfs": peak, "warnings": warnings,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

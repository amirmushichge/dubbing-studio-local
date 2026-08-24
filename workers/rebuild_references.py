from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MAX_REFERENCE_SECONDS = 12.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("references", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    waveform, sample_rate = librosa.load(args.audio, sr=16000, mono=True)
    args.references.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(round(0.12 * sample_rate), dtype=np.float32)
    report = {}
    for speaker in payload["speakers"]:
        clips = []
        seconds = 0.0
        for segment in payload["segments"]:
            if segment["speaker"] != speaker["id"] or seconds >= MAX_REFERENCE_SECONDS:
                continue
            start = max(0, round((float(segment["start"]) - 0.08) * sample_rate))
            end = min(len(waveform), round((float(segment["end"]) + 0.12) * sample_rate))
            remaining = round((MAX_REFERENCE_SECONDS - seconds) * sample_rate)
            clip = waveform[start:end][:remaining]
            if len(clip):
                clips.extend((clip, gap))
                seconds += (len(clip) + len(gap)) / sample_rate
        reference = np.concatenate(clips) if clips else np.zeros(sample_rate, dtype=np.float32)
        target = args.references / f"{speaker['id']}.wav"
        sf.write(target, reference, sample_rate, subtype="PCM_16")
        speaker["reference"] = str(target)
        report[speaker["id"]] = round(len(reference) / sample_rate, 3)
    print(json.dumps(report))


if __name__ == "__main__":
    main()

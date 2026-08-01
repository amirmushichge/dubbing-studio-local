from __future__ import annotations

import argparse
import json
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    language = None if args.language == "auto" else args.language
    segments, info = model.transcribe(
        str(args.audio), language=language, beam_size=5, best_of=5,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 350},
        condition_on_previous_text=True, word_timestamps=True,
    )
    result = []
    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if not text:
            continue
        result.append({
            "id": index,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": text,
            "speaker": "SPEAKER_00",
            "words": [
                {"start": round(word.start, 3), "end": round(word.end, 3), "word": word.word}
                for word in (segment.words or []) if word.start is not None and word.end is not None
            ],
        })
    payload = {"language": info.language, "language_probability": info.language_probability, "segments": result}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"language": info.language, "segments": len(result)}))


if __name__ == "__main__":
    main()


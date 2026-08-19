from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.diarization import merge_tiny_clusters


def normalized_labels(labels: np.ndarray) -> list[int]:
    mapping: dict[int, int] = {}
    output = []
    for value in labels.tolist():
        mapping.setdefault(value, len(mapping))
        output.append(mapping[value])
    return output


def select_cluster_count(embeddings: np.ndarray) -> int:
    if len(embeddings) < 5:
        return 1
    best_count, best_score = 1, -1.0
    for count in range(2, min(4, len(embeddings) - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=count, metric="cosine", linkage="average").fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_count, best_score = count, score
    return best_count if best_score >= 0.12 else 1


def main() -> None:
    from resemblyzer import VoiceEncoder, preprocess_wav

    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("--count", default="auto")
    args = parser.parse_args()

    payload = json.loads(args.transcript.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"language": "unknown", "language_probability": None, "segments": payload}
    segments = payload["segments"]
    waveform, sample_rate = librosa.load(args.audio, sr=16000, mono=True)
    encoder = VoiceEncoder()
    embeddings = []
    clips = []
    for item in segments:
        start = max(0, round((item["start"] - 0.08) * sample_rate))
        end = min(len(waveform), round((item["end"] + 0.12) * sample_rate))
        clip = waveform[start:end]
        if len(clip) < sample_rate * 0.45:
            clip = np.pad(clip, (0, max(0, round(sample_rate * 0.45) - len(clip))))
        clips.append(clip)
        embeddings.append(encoder.embed_utterance(preprocess_wav(clip, source_sr=sample_rate)))
    matrix = np.stack(embeddings) if embeddings else np.zeros((0, 256), dtype=np.float32)
    count = select_cluster_count(matrix) if args.count == "auto" else max(1, int(args.count))
    if count == 1 or len(matrix) < count:
        labels = [0] * len(segments)
    else:
        raw = AgglomerativeClustering(n_clusters=count, metric="cosine", linkage="average").fit_predict(matrix)
        labels = normalized_labels(raw)
        if args.count == "auto":
            labels = merge_tiny_clusters(labels, matrix.tolist(), segments)
    for item, label in zip(segments, labels):
        item["speaker"] = f"SPEAKER_{label:02d}"

    args.references.mkdir(parents=True, exist_ok=True)
    speakers = []
    gap = np.zeros(round(0.12 * sample_rate), dtype=np.float32)
    for label in sorted(set(labels)):
        selected = [clip for clip, current in zip(clips, labels) if current == label]
        total = []
        length = 0
        for clip in selected:
            remaining = round(25 * sample_rate) - length
            if remaining <= 0:
                break
            piece = clip[:remaining]
            total.extend((piece, gap))
            length += len(piece) + len(gap)
        reference = np.concatenate(total) if total else np.zeros(sample_rate, dtype=np.float32)
        path = args.references / f"SPEAKER_{label:02d}.wav"
        sf.write(path, reference, sample_rate, subtype="PCM_16")
        speakers.append({
            "id": f"SPEAKER_{label:02d}",
            "label": f"Speaker {label + 1}",
            "reference": str(path),
            "profile": "A natural native speaker matching the source age and conversational energy, realistic and not theatrical.",
        })
    result = {"language": payload["language"], "language_probability": payload["language_probability"], "speakers": speakers, "segments": segments}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"speakers": len(speakers), "segments": len(segments)}))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_json(text: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        raise ValueError(f"Translation model did not return JSON: {text[-1000:]}")
    return json.loads(match.group(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    args = parser.parse_args()

    transcript = json.loads(args.input.read_text(encoding="utf-8"))
    segments = transcript.get("segments", transcript)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", local_files_only=True, trust_remote_code=True).eval()
    by_id = {}
    for offset in range(0, len(segments), 20):
        chunk = segments[offset:offset + 20]
        source = [
            {"id": offset + index, "text": item["text"], "seconds": round(item["end"] - item["start"], 2)}
            for index, item in enumerate(chunk)
        ]
        prompt = f"""Translate the following spoken-video segments from {args.source_language} into {args.target_language}.

Requirements:
1. Use natural conversational spoken language suitable for professional dubbing.
2. Preserve meaning, names, numbers, questions, punctuation, emotion and line order.
3. Keep every translation concise enough for the provided number of seconds.
4. Return ONLY valid JSON with exactly {len(source)} objects: [{{\"id\": {offset}, \"translation\": \"...\"}}].
5. Never merge, split, omit or reorder lines.

Source JSON:
{json.dumps(source, ensure_ascii=False)}"""
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max(768, len(source) * 80), do_sample=False, repetition_penalty=1.05)
        generated = output[0, inputs["input_ids"].shape[-1]:]
        translated = extract_json(tokenizer.decode(generated, skip_special_tokens=True))
        by_id.update({int(item["id"]): str(item["translation"]).strip() for item in translated})
    if set(by_id) != set(range(len(segments))):
        raise ValueError("Translation IDs do not match transcript")
    result = []
    for index, item in enumerate(segments):
        line = dict(item)
        line["translation"] = by_id[index]
        result.append(line)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

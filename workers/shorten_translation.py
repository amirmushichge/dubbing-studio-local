from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("translation", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    args = parser.parse_args()

    lines = json.loads(args.translation.read_text(encoding="utf-8"))
    line = lines[args.index]
    original = line["translation"]
    target_percent = max(45, min(90, round(args.ratio * 100)))
    prompt = f"""Rewrite this spoken {args.language} dubbing line to approximately {target_percent}% of its current spoken length.

Keep the exact meaning, names, numbers, question/statement intent and natural punctuation. Use concise native conversational wording. Do not add information. Return ONLY the rewritten line without quotes or explanation.

Line: {original}"""
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", local_files_only=True, trust_remote_code=True).eval()
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=160, do_sample=False, repetition_penalty=1.05)
    generated = output[0, inputs["input_ids"].shape[-1]:]
    shortened = tokenizer.decode(generated, skip_special_tokens=True).strip().strip('"').strip()
    if not shortened or shortened == original:
        raise RuntimeError("Translation shortener did not change the line")
    line["translation"] = shortened
    line.setdefault("revision_history", []).append({"reason": "timing", "previous": original})
    args.translation.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"index": args.index, "before": original, "after": shortened}, ensure_ascii=False))


if __name__ == "__main__":
    main()


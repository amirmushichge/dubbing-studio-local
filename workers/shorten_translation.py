from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .device import model_load_options
except ImportError:  # Executed as a standalone worker script.
    from device import model_load_options


def report_progress(path: Path | None, current: int, total: int, label: str) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump({"current": current, "total": total, "label": label}, handle)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("translation", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--mode", choices=("shorten", "expand"), default="shorten")
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()

    lines = json.loads(args.translation.read_text(encoding="utf-8"))
    if args.requests:
        requests = json.loads(args.requests.read_text(encoding="utf-8"))
    elif args.index is not None and args.ratio is not None:
        requests = [{"index": args.index, "ratio": args.ratio, "mode": args.mode}]
    else:
        raise RuntimeError("Provide --requests or both --index and --ratio")
    total = len(requests) * 3 + 1
    report_progress(args.progress, 0, total, f"Loading timing adapter · {len(requests)} lines")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_load_options(args.device, args.dtype),
        local_files_only=True,
        trust_remote_code=True,
    ).eval()
    report_progress(args.progress, 1, total, f"Adapting speech timing · 0 of {len(requests)} lines")
    results = []
    for request_index, request in enumerate(requests):
        index = int(request["index"])
        mode = request.get("mode", "shorten")
        ratio = float(request["ratio"])
        line = lines[index]
        original = line["translation"]
        source = line.get("text", "")
        if mode == "expand":
            target_percent = max(115, min(195, round(ratio * 100)))
            direction = "Retranslate and adapt"
            wording = "Use only meaning explicitly present in the source. Do not add examples, consequences, motivations, people, features, opinions, parenthetical explanations or new facts. Natural emphasis and discourse wording are allowed."
        else:
            target_percent = max(45, min(90, round(ratio * 100)))
            direction = "Rewrite"
            wording = "Use concise native conversational wording. Do not add information."
        original_tokens = len(tokenizer.encode(original, add_special_tokens=False))
        target_tokens = max(2, round(original_tokens * ratio))
        lower_tokens = max(1, round(target_tokens * .78))
        upper_tokens = max(lower_tokens + 1, round(target_tokens * 1.22))
        prompt = f"""{direction} this spoken {args.language} dubbing line to approximately {target_percent}% of its current spoken length.

Keep the exact meaning, names, numbers, question/statement intent and natural punctuation. {wording} Return ONLY the rewritten line without quotes or explanation.
The result must be roughly {target_tokens} tokenizer tokens long (acceptable range {lower_tokens}-{upper_tokens}).

Original source: {source}
Current {args.language}: {original}"""
        revised = ""
        best_revised = ""
        best_distance = float("inf")
        for attempt in range(3):
            report_progress(
                args.progress, 1 + request_index * 3 + attempt, total,
                f"Adapting timing · line {request_index + 1} of {len(requests)} · pass {attempt + 1}",
            )
            attempt_prompt = prompt
            if attempt:
                actual_tokens = len(tokenizer.encode(revised, add_special_tokens=False))
                attempt_prompt += f"\n\nYour previous attempt was {actual_tokens} tokens and outside the required range. Rewrite it again at the requested length: {revised}"
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": attempt_prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max(20, min(96, upper_tokens + 12)),
                    do_sample=False,
                    repetition_penalty=1.05,
                )
            generated = output[0, inputs["input_ids"].shape[-1]:]
            revised = tokenizer.decode(generated, skip_special_tokens=True).strip().strip('"').strip()
            actual_tokens = len(tokenizer.encode(revised, add_special_tokens=False))
            distance = 0 if lower_tokens <= actual_tokens <= upper_tokens else min(
                abs(actual_tokens - lower_tokens), abs(actual_tokens - upper_tokens)
            )
            if revised and revised != original and distance < best_distance:
                best_revised, best_distance = revised, distance
            if lower_tokens <= actual_tokens <= upper_tokens:
                break
        if not revised or revised == original:
            raise RuntimeError(f"Translation timing adapter did not change line {index + 1}")
        if not lower_tokens <= len(tokenizer.encode(revised, add_special_tokens=False)) <= upper_tokens:
            revised = best_revised
        if not revised:
            raise RuntimeError(f"Translation timing adapter produced no usable wording for line {index + 1}")
        line["translation"] = revised
        line.setdefault("revision_history", []).append({"reason": f"timing_{mode}", "previous": original})
        results.append({"index": index, "before": original, "after": revised})
        report_progress(
            args.progress, min(total, 1 + len(results) * 3), total,
            f"Adapting speech timing · {len(results)} of {len(requests)} lines",
        )
    args.translation.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

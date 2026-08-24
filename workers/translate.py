from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from translation_output import TranslationOutputError, extract_translation_payload

try:
    from .device import model_load_options
except ImportError:  # Executed as a standalone worker script.
    from device import model_load_options

MAX_GENERATION_ATTEMPTS = 3


def report_progress(path: Path | None, current: int, total: int, label: str) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump({"current": current, "total": total, "label": label}, handle)
        temporary = Path(handle.name)
    temporary.replace(path)


class CompleteTranslationCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int, expected_ids: list[int], progress_callback=None) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.expected_ids = expected_ids
        self._last_checked_length = 0
        self._last_reported_length = 0
        self.progress_callback = progress_callback

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated_length = input_ids.shape[-1] - self.prompt_length
        if self.progress_callback and generated_length - self._last_reported_length >= 4:
            self._last_reported_length = generated_length
            self.progress_callback(generated_length)
        if generated_length < 12 or generated_length - self._last_checked_length < 4:
            return False
        self._last_checked_length = generated_length
        response = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
        try:
            extract_translation_payload(response, self.expected_ids)
        except TranslationOutputError:
            return False
        return True


def generate_translation(model, tokenizer, prompt: str, expected_ids: list[int], progress_callback=None) -> list[dict]:
    messages = [{"role": "user", "content": prompt}]
    last_error = "unknown structured-output error"
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.inference_mode():
            callback = (lambda generated: progress_callback(attempt, generated)) if progress_callback else None
            stopping = StoppingCriteriaList([
                CompleteTranslationCriteria(tokenizer, inputs["input_ids"].shape[-1], expected_ids, callback)
            ])
            output = model.generate(
                **inputs,
                max_new_tokens=max(320, len(expected_ids) * 64),
                do_sample=False,
                repetition_penalty=1.05,
                stopping_criteria=stopping,
            )
        generated = output[0, inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            return extract_translation_payload(response, expected_ids)
        except TranslationOutputError as exc:
            last_error = str(exc)
            print(
                json.dumps({
                    "event": "translation_output_rejected",
                    "attempt": attempt,
                    "max_attempts": MAX_GENERATION_ATTEMPTS,
                    "reason": last_error,
                }),
                file=sys.stderr,
                flush=True,
            )
            if attempt < MAX_GENERATION_ATTEMPTS:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response[-6000:]},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was rejected because it was not exactly one valid translation JSON array "
                            f"({last_error}). Return the corrected JSON array only. Do not add Markdown, commentary, or a second array."
                        ),
                    },
                ]
    raise RuntimeError(
        f"Translation model returned invalid structured output after {MAX_GENERATION_ATTEMPTS} attempts: {last_error}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()

    transcript = json.loads(args.input.read_text(encoding="utf-8"))
    segments = transcript.get("segments", transcript)
    chunk_count = max(1, (len(segments) + 19) // 20)
    progress_total = 1000
    report_progress(args.progress, 0, progress_total, "Loading the translation model")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_load_options(args.device, args.dtype),
        local_files_only=True,
        trust_remote_code=True,
    ).eval()
    report_progress(args.progress, 100, progress_total, f"Translating · 0 of {len(segments)} lines")
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
3. Adapt each line to naturally fill roughly 75-95% of its provided speaking time. Do not make it needlessly terse.
4. Preserve all meaningful clauses and discourse cues; never summarize or omit a thought merely to shorten it.
5. Return ONLY valid JSON with exactly {len(source)} objects: [{{\"id\": {offset}, \"translation\": \"...\"}}].
6. Never merge, split, omit or reorder lines.

Source JSON:
{json.dumps(source, ensure_ascii=False)}"""
        expected_generation = max(48, len(source) * 18)

        def generation_progress(attempt: int, generated: int) -> None:
            attempt_fraction = ((attempt - 1) + min(.98, generated / expected_generation)) / MAX_GENERATION_ATTEMPTS
            chunk_fraction = (offset // 20 + attempt_fraction) / chunk_count
            current = 100 + round(850 * chunk_fraction)
            report_progress(
                args.progress, current, progress_total,
                f"Translating · {min(offset + len(chunk), len(segments))} of {len(segments)} lines",
            )

        translated = generate_translation(
            model, tokenizer, prompt, [item["id"] for item in source], generation_progress
        )
        by_id.update({int(item["id"]): str(item["translation"]).strip() for item in translated})
        report_progress(
            args.progress, 100 + round(850 * ((offset // 20 + 1) / chunk_count)), progress_total,
            f"Translating · {min(offset + len(chunk), len(segments))} of {len(segments)} lines",
        )
    if set(by_id) != set(range(len(segments))):
        raise ValueError("Translation IDs do not match transcript")
    result = []
    for index, item in enumerate(segments):
        line = dict(item)
        line["translation"] = by_id[index]
        result.append(line)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_progress(args.progress, progress_total, progress_total, "Translation ready")
    print(args.output.resolve())


if __name__ == "__main__":
    main()

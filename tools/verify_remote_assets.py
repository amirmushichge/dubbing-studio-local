"""Fail when a source commit or pinned model revision is no longer available."""
from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from download_models import COMMON_MODELS, MUSETALK_MODELS, SEED_MODELS, TRANSLATION_MODELS

GITHUB_REVISIONS = (
    ("QwenLM/Qwen3-TTS", "022e286b98fbec7e1e916cb940cdf532cd9f488e"),
    ("Plachtaa/seed-vc", "51383efd921027683c89e5348211d93ff12ac2a8"),
    ("TMElyralab/MuseTalk", "0a89dec45a0192b824e3cf4daf96c239440c5ed8"),
)


def request_json(url: str, *, github: bool = False) -> dict:
    headers = {"Accept": "application/json", "User-Agent": "dubbing-studio-release-check"}
    if github and (token := os.environ.get("GITHUB_TOKEN")):
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Remote asset check failed for {url}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    failures: list[str] = []
    for repository, revision in GITHUB_REVISIONS:
        data = request_json(f"https://api.github.com/repos/{repository}/commits/{revision}", github=True)
        if data.get("sha") != revision:
            failures.append(f"GitHub revision mismatch: {repository}@{revision}")
        else:
            print(f"OK GitHub {repository}@{revision}")

    model_revisions = [
        *((repository, revision) for repository, revision, _ in COMMON_MODELS),
        *((repository, revision) for repository, revision, _ in TRANSLATION_MODELS.values()),
        *SEED_MODELS,
        *MUSETALK_MODELS,
    ]
    for repository, revision in model_revisions:
        data = request_json(f"https://huggingface.co/api/models/{repository}/revision/{revision}")
        if data.get("sha") != revision:
            failures.append(f"Hugging Face revision mismatch: {repository}@{revision}")
        else:
            print(f"OK Hugging Face {repository}@{revision}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(GITHUB_REVISIONS) + len(model_revisions)} pinned remote assets")


if __name__ == "__main__":
    main()

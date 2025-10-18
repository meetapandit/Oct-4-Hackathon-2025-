"""CLI helper to exercise sentence post-processing without calling OpenAI."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("AAC_DISABLE_SPACY", "1")

import app  # noqa: E402


def _stub_client(responses: Iterable[str]):
    responses = list(responses)
    if not responses:
        responses = [""]

    def factory(_: str):
        iterator = iter(responses)

        def create(**kwargs):
            try:
                content = next(iterator)
            except StopIteration:
                content = responses[-1]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    return factory


def run_cases(cases: List[dict]) -> List[dict]:
    client = app.app.test_client()
    original_get_client = app.get_openai_client

    try:
        results = []
        for case in cases:
            app.get_openai_client = _stub_client(case.get("responses", case.get("response", [""])))
            payload = {
                "words": case["words"],
                "api_key": "local-test"
            }
            response = client.post("/generate-sentences", json=payload)
            try:
                data = response.get_json()
            except Exception:  # pragma: no cover
                data = {"raw": response.data.decode("utf-8", errors="replace")}
            results.append({
                "label": case.get("label"),
                "status": response.status_code,
                "payload": payload,
                "response": data,
            })
        return results
    finally:
        app.get_openai_client = original_get_client


def main(argv: List[str]) -> int:
    if "--practical" in argv:
        cases = [
            {
                "label": "pronoun case",
                "words": ["I", "like", "she"],
                "responses": ["i like her"],
            },
            {
                "label": "lemma agreement",
                "words": ["She", "know", "I"],
                "responses": ["she knows me"],
            },
            {
                "label": "invalid synonym",
                "words": ["I", "want", "food"],
                "responses": ["I crave snacks"],
            },
            {
                "label": "bad reordering",
                "words": ["want", "food", "now"],
                "responses": ["food now want"],
            },
        ]
    else:
        cases = []
        if sys.stdin.isatty():
            print("Reading cases from stdin (JSON lines)")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
        if not cases:
            print("No cases provided. Use --practical or pipe JSON lines.", file=sys.stderr)
            return 1

    results = run_cases(cases)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

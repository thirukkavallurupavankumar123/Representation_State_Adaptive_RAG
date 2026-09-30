"""Run the fresh, separately authorized 48-pair smoke-generation v1.1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sar_rag.final_evaluation import file_sha256  # noqa: E402
from sar_rag.generation_v1_1 import build_chat_payload_v1_1  # noqa: E402
from sar_rag.smoke_recovery import AttemptOutcome, load_jsonl  # noqa: E402
from sar_rag.smoke_v1_1 import V11SmokeRunner  # noqa: E402


def sanitize(value: str) -> str:
    value = re.sub(r"organization `[^`]+`", "organization `[redacted]`", value)
    return re.sub(r"gsk_[A-Za-z0-9]+", "[redacted-api-key]", value)[:4000]


def retry_after_seconds(value: str) -> float | None:
    match = re.search(r"try again in\s+([0-9.]+)\s*(ms|s)", value, re.IGNORECASE)
    if not match:
        return None
    value_float = float(match.group(1))
    return value_float / 1000 if match.group(2).lower() == "ms" else value_float


class GroqV11Answerer:
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, timeout_seconds: int = 90,
                 minimum_interval_seconds: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.minimum_interval_seconds = minimum_interval_seconds
        self.last_started: float | None = None
        self.not_before = 0.0

    def _wait(self) -> None:
        while True:
            interval = (
                self.last_started + self.minimum_interval_seconds
                if self.last_started is not None else 0.0
            )
            delay = max(interval, self.not_before) - perf_counter()
            if delay <= 0:
                return
            sleep(min(delay, 30.0))

    def attempt(self, item: dict) -> AttemptOutcome:
        self._wait()
        self.last_started = perf_counter()
        payload = build_chat_payload_v1_1(item["question"], item["evidence"])
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "sar-rag-research/0.1",
            },
        )
        started = perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8"))
                response_headers = {key.lower(): value for key, value in response.headers.items() if key.lower().startswith("x-ratelimit-") or key.lower() == "retry-after"}
            raw = parsed["choices"][0]["message"].get("content") or ""
            usage = parsed.get("usage", {})
            try:
                response_payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                return AttemptOutcome(
                    status="json_parse_error", raw_response=raw,
                    input_tokens=int(usage.get("prompt_tokens", 0)),
                    output_tokens=int(usage.get("completion_tokens", 0)),
                    latency_ms=(perf_counter() - started) * 1000,
                    error_type=type(exc).__name__, error=sanitize(str(exc)),
                )
            return AttemptOutcome(
                status="success", payload=response_payload, raw_response=raw,
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                latency_ms=(perf_counter() - started) * 1000,
                http_status=200, rate_limit_headers=response_headers,
            )
        except HTTPError as exc:
            body = sanitize(exc.read().decode("utf-8", "replace"))
            response_headers = {key.lower(): value for key, value in exc.headers.items() if key.lower().startswith("x-ratelimit-") or key.lower() == "retry-after"}
            delay = retry_after_seconds(body)
            if delay is None and response_headers.get("retry-after"):
                try:
                    delay = float(response_headers["retry-after"])
                except ValueError:
                    pass
            if delay is not None:
                self.not_before = max(self.not_before, perf_counter() + delay + 0.25)
            retryable = (
                exc.code == 429 or 500 <= exc.code <= 599
                or (exc.code == 400 and "json_validate_failed" in body)
            )
            return AttemptOutcome(
                status="api_error" if retryable else "fatal_error",
                latency_ms=(perf_counter() - started) * 1000,
                error_type="HTTPError", error=f"Groq HTTP {exc.code}: {body}",
                http_status=exc.code, rate_limit_headers=response_headers,
            )
        except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            return AttemptOutcome(
                status="transport_error", latency_ms=(perf_counter() - started) * 1000,
                error_type=type(exc).__name__, error=sanitize(str(exc)),
            )


def verify_authorization(path: Path | None, protocol: dict, manifest: dict) -> None:
    if path is None or not path.is_file():
        raise SystemExit("PAID V1.1 BLOCKED: separately matching authorization file is absent")
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "EXPLICIT_USER_AUTHORIZATION_RECORDED",
        "authorized_stage": "paid_smoke_generation_v1_1_only",
        "user_authorized": True,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "run_manifest_fingerprint": manifest["manifest_fingerprint"],
        "maximum_total_api_attempts_including_retries": manifest[
            "maximum_total_api_attempts_including_retries"
        ],
        "maximum_attempts_per_pair_including_first": manifest[
            "maximum_attempts_per_pair_including_first"
        ],
        "maximum_cost_usd": manifest["maximum_cost_usd"],
    }
    if actual != expected:
        raise SystemExit("PAID V1.1 BLOCKED: authorization content/fingerprint mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--execute-paid", action="store_true")
    args = parser.parse_args()
    if not args.execute_paid:
        raise SystemExit("PAID V1.1 BLOCKED: --execute-paid was not provided")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_authorization(args.authorization_file, protocol, manifest)

    output = ROOT / "output/final_evaluation"
    for name, expected in protocol["preserved_artifact_sha256"].items():
        if file_sha256(output / name) != expected:
            raise SystemExit(f"PAID V1.1 BLOCKED: preserved artifact changed: {name}")
    if args.checkpoint_dir.resolve() in {
        (output / "paid_smoke_checkpoint").resolve(),
        (output / "recovery_v1/checkpoint").resolve(),
    }:
        raise SystemExit("PAID V1.1 BLOCKED: checkpoint target would overwrite prior evidence")

    from sar_rag.environment import load_project_env
    load_project_env()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("PAID V1.1 BLOCKED: GROQ_API_KEY unavailable")
    requests = load_jsonl(output / "SMOKE_TEST_REQUESTS.jsonl")
    runner = V11SmokeRunner(
        args.checkpoint_dir, protocol, manifest,
        protocol_path=args.protocol, manifest_path=args.manifest,
    )
    summary = runner.run(
        requests,
        GroqV11Answerer(
            api_key,
            timeout_seconds=90,
            minimum_interval_seconds=float(protocol["rate_limit_policy"]["minimum_seconds_between_attempt_starts"]),
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

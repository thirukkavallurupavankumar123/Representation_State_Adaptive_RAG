"""Local Qwen3-8B (bitsandbytes 4-bit NF4) answer backend for the reduced local study.

This module is isolated from the Groq v1.1 plumbing. It reuses, unchanged, the
frozen v1.1 user/system prompt text (``build_messages_v1_1``), the frozen answer
schema (``ANSWER_SCHEMA``) and the frozen semantic validator
(``validate_answer``). Only the transport/rendering layer is new:

* Groq strict mode delivered the schema through ``response_format``; plain
  Transformers has no such channel, so the *same* schema is rendered as text in
  the system message. No grammar/constrained decoding is used, so schema
  conformance is NOT guaranteed by decoding and is checked after generation.
* Outputs are parsed strictly (``json.loads`` on the raw decoded text). Nothing
  is repaired, coerced, inserted or rewritten; invalid outputs are recorded as
  invalid generations.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from .final_evaluation import (
    ANSWER_SCHEMA,
    ANSWERABILITY,
    evidence_hash,
    file_sha256,
    logical_key,
    object_sha256,
    validate_answer,
)
from .generation_v1_1 import USER_TEMPLATE_VERSION, build_messages_v1_1


EXPERIMENT_VERSION = "local_qwen_4bit_reduced_v1"
BACKEND_VERSION = "sar-rag-local-qwen-4bit-backend-1.0.0"
TRANSPORT_VERSION = "qwen-chat-template-schema-in-system-v1"
MODEL_REPO = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_MODEL_PATH = "/mnt/f/research/models/Qwen3-8B"
MASTER_SEED = 20260930
SYSTEMS = ("traditional_k10", "sar_rag_v1", "adaptive_single_view_v2")
MAX_NEW_TOKENS = 1600
# Frozen v1.1 supplied-evidence cap. The frozen packer counts it with the MiniLM
# retrieval tokenizer (``SentenceTransformerEncoder.token_count``), so it does not
# depend on the answer model. Qwen token counts are logged in addition.
SUPPLIED_CONTEXT_CAP_TOKENS = 1600
REPEATABILITY_NOTE = (
    "Fixed seeds improve repeatability but do not guarantee identical outputs "
    "across hardware or library versions."
)
SEED_RULE = (
    "question_seed = int.from_bytes(SHA-256(UTF-8('{master_seed}|{dataset}|{question_id}'))[:4], 'big'); "
    "random.seed(question_seed) and torch.manual_seed(question_seed) immediately before every "
    "system's generate() call for that question"
)

GENERATION = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_beams": 1,
    "batch_size": 1,
    "enable_thinking": False,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "source": "Qwen/Qwen3-8B README (pinned revision) non-thinking recommendation: "
              "Temperature=0.7, TopP=0.8, TopK=20, MinP=0",
}
QUANTIZATION = {
    "backend": "bitsandbytes",
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16 if torch.cuda.is_bf16_supported() else float16",
    "device_map": {"": 0},
    "cpu_offload_allowed": False,
}

SCHEMA_TRANSPORT_INSTRUCTION = (
    "Respond with exactly one JSON object and nothing else: no markdown code fences, "
    "no commentary before or after it. The JSON object must conform to this JSON Schema:\n"
)
# Same compact serialization the repository used to count Groq schema tokens.
SERIALIZED_SCHEMA = json.dumps(ANSWER_SCHEMA, separators=(",", ":"))
EVIDENCE_ID_RE = re.compile(r"^E[1-9][0-9]*$")
NON_THINKING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Prompt transport
# ---------------------------------------------------------------------------

def build_qwen_messages(question: str, evidence: list[dict]) -> tuple[list[dict], str]:
    """Frozen v1.1 messages plus the schema text that Groq sent via response_format."""
    messages, v1_1_prompt_hash = build_messages_v1_1(question, evidence)
    system = messages[0]["content"] + "\n\n" + SCHEMA_TRANSPORT_INSTRUCTION + SERIALIZED_SCHEMA
    qwen_messages = [{"role": "system", "content": system}, dict(messages[1])]
    transport_spec = {
        "transport_version": TRANSPORT_VERSION,
        "v1_1_prompt_version": USER_TEMPLATE_VERSION,
        "v1_1_prompt_hash": v1_1_prompt_hash,
        "system_suffix": "\n\n" + SCHEMA_TRANSPORT_INSTRUCTION + SERIALIZED_SCHEMA,
        "user_message": "byte-identical to build_messages_v1_1",
    }
    return qwen_messages, object_sha256(transport_spec)


# ---------------------------------------------------------------------------
# Seeds and ordering
# ---------------------------------------------------------------------------

def question_seed(dataset: str, question_id: str, master_seed: int = MASTER_SEED) -> int:
    digest = sha256(f"{master_seed}|{dataset}|{question_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def stable_rank(*parts: object) -> str:
    return sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def system_order(dataset: str, question_id: str, order_seed: int) -> list[str]:
    return sorted(SYSTEMS, key=lambda system: stable_rank(order_seed, "system-order", dataset, question_id, system))


# ---------------------------------------------------------------------------
# Strict parsing and validation (no repair)
# ---------------------------------------------------------------------------

class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKeyError(f"duplicate JSON object keys: {sorted({k for k in keys if keys.count(k) > 1})}")
    return dict(pairs)


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def schema_errors(payload: object) -> list[str]:
    """Hand-written check of ANSWER_SCHEMA (types, required, enum, pattern, closed objects)."""
    if not isinstance(payload, dict):
        return [f"top-level value is {type(payload).__name__}, not an object"]
    errors: list[str] = []
    required = ("answerability", "claims", "missing_information")
    errors += [f"missing required field: {key}" for key in required if key not in payload]
    extra = sorted(set(payload) - set(required))
    if extra:
        errors.append(f"additional top-level fields: {extra}")
    if "answerability" in payload:
        value = payload["answerability"]
        if not isinstance(value, str):
            errors.append(f"answerability is {type(value).__name__}, not string")
        elif value not in ANSWERABILITY:
            errors.append(f"answerability not in enum: {value!r}")
    if "claims" in payload:
        claims = payload["claims"]
        if not isinstance(claims, list):
            errors.append(f"claims is {type(claims).__name__}, not array")
        else:
            for index, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    errors.append(f"claims[{index}] is {type(claim).__name__}, not object")
                    continue
                errors += [f"claims[{index}] missing {key}" for key in ("text", "evidence_ids") if key not in claim]
                extra_claim = sorted(set(claim) - {"text", "evidence_ids"})
                if extra_claim:
                    errors.append(f"claims[{index}] additional fields: {extra_claim}")
                if "text" in claim and not isinstance(claim["text"], str):
                    errors.append(f"claims[{index}].text is {type(claim['text']).__name__}, not string")
                if "evidence_ids" in claim:
                    ids = claim["evidence_ids"]
                    if not isinstance(ids, list):
                        errors.append(f"claims[{index}].evidence_ids is {type(ids).__name__}, not array")
                    else:
                        for position, item in enumerate(ids):
                            if not isinstance(item, str) or not EVIDENCE_ID_RE.fullmatch(item):
                                errors.append(f"claims[{index}].evidence_ids[{position}] violates ^E[1-9][0-9]*$: {item!r}")
    if "missing_information" in payload and not isinstance(payload["missing_information"], str):
        errors.append(f"missing_information is {type(payload['missing_information']).__name__}, not string")
    return errors


def evaluate_output(raw_output: str, evidence_count: int) -> dict:
    """Classify one immutable raw output. Never modifies or repairs it."""
    result = {
        "parse_status": None, "parse_error": None, "parsed_output": None,
        "schema_valid": False, "schema_errors": [],
        "semantic_valid": False, "semantic_error": None,
        "validation_status": None, "validated_response": None,
        "thinking_markers_present": "<think>" in raw_output or "</think>" in raw_output,
    }
    try:
        parsed = json.loads(raw_output, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        result.update(parse_status="json_parse_failure", parse_error=f"{type(exc).__name__}: {exc}",
                      validation_status="parse_failure")
        return result
    result["parse_status"] = "parsed"
    result["parsed_output"] = parsed
    errors = schema_errors(parsed)
    result["schema_errors"] = errors
    if errors:
        result["validation_status"] = "schema_invalid"
        return result
    result["schema_valid"] = True
    try:
        # Frozen v1.1 semantic validator; returns the payload plus a rendered answer.
        result["validated_response"] = validate_answer(parsed, evidence_count)
    except Exception as exc:  # validator raises ValueError on semantic violations
        result.update(semantic_error=f"{type(exc).__name__}: {exc}", validation_status="semantic_invalid")
        return result
    result.update(semantic_valid=True, validation_status="valid")
    return result


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def generation_config_record() -> dict:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "backend_version": BACKEND_VERSION,
        "transport_version": TRANSPORT_VERSION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "quantization": QUANTIZATION,
        "generation": GENERATION,
        "master_seed": MASTER_SEED,
        "seed_rule": SEED_RULE,
        "attempts_per_pair": 1,
    }


def qwen_request_fingerprint(request: dict, rendered_prompt_sha256: str, seed: int) -> str:
    """New local-Qwen fingerprint. Deliberately unrelated to any Groq fingerprint."""
    _, transport_hash = build_qwen_messages(request["question"], request["evidence"])
    return object_sha256({
        "experiment_version": EXPERIMENT_VERSION,
        "dataset": request["dataset"],
        "question_id": request["question_id"],
        "system": request["system"],
        "question": request["question"],
        "evidence_hash": evidence_hash(request["evidence"]),
        "transport_hash": transport_hash,
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "question_seed": seed,
        "generation_config": generation_config_record(),
    })


# ---------------------------------------------------------------------------
# Local generator
# ---------------------------------------------------------------------------

class ContextCapacityError(RuntimeError):
    """Prompt plus output allowance would exceed the model context; truncation is prohibited."""


def local_revision(model_path: Path) -> str | None:
    """Commit hash recorded by huggingface_hub in the local download metadata."""
    metadata = model_path / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if not metadata.is_file():
        return None
    return metadata.read_text(encoding="utf-8").splitlines()[0].strip()


class LocalQwenGenerator:
    """Loads the pinned local Qwen3-8B once, fully on GPU 0, and generates one answer at a time."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, *, device_index: int = 0) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

        self.torch = torch
        self.model_path = Path(model_path)
        self.device_index = device_index
        revision = local_revision(self.model_path)
        if revision != MODEL_REVISION:
            raise RuntimeError(f"local model revision {revision!r} != pinned {MODEL_REVISION}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU execution is not permitted by the protocol")
        bf16 = torch.cuda.is_bf16_supported()
        self.compute_dtype = torch.bfloat16 if bf16 else torch.float16
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self.compute_dtype,
        )
        load_started = perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            quantization_config=quant,
            device_map={"": device_index},
            dtype=self.compute_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        torch.cuda.synchronize(device_index)
        self.load_seconds = perf_counter() - load_started
        self.load_peak_allocated_gib = torch.cuda.max_memory_allocated(device_index) / 1024**3
        self.load_peak_reserved_gib = torch.cuda.max_memory_reserved(device_index) / 1024**3
        self.placement = self._verify_placement()
        self.context_capacity = int(self.model.config.max_position_embeddings)
        self.generation_config = GenerationConfig(
            do_sample=GENERATION["do_sample"],
            temperature=GENERATION["temperature"],
            top_p=GENERATION["top_p"],
            top_k=GENERATION["top_k"],
            min_p=GENERATION["min_p"],
            repetition_penalty=GENERATION["repetition_penalty"],
            max_new_tokens=GENERATION["max_new_tokens"],
            num_beams=GENERATION["num_beams"],
            eos_token_id=GENERATION["eos_token_id"],
            pad_token_id=GENERATION["pad_token_id"],
            bos_token_id=151643,
        )
        # Replace, rather than merge with, the checkpoint's thinking-mode defaults.
        self.model.generation_config = self.generation_config
        self.warmup_seconds: float | None = None

    def _verify_placement(self) -> dict:
        devices = {str(p.device) for p in self.model.parameters()} | {str(b.device) for b in self.model.buffers()}
        hf_map = {str(k): str(v) for k, v in (getattr(self.model, "hf_device_map", None) or {}).items()}
        bad_map = {k: v for k, v in hf_map.items() if v not in {str(self.device_index), f"cuda:{self.device_index}"}}
        expected = {f"cuda:{self.device_index}"}
        import bitsandbytes as bnb
        linear4bit = sum(isinstance(m, bnb.nn.Linear4bit) for m in self.model.modules())
        report = {
            "parameter_and_buffer_devices": sorted(devices),
            "hf_device_map": hf_map,
            "linear4bit_modules": linear4bit,
            "cpu_or_disk_offload": bool(bad_map) or devices != expected,
            "quantization_config": self.model.config.quantization_config.to_dict()
            if hasattr(self.model.config.quantization_config, "to_dict") else str(self.model.config.quantization_config),
        }
        if report["cpu_or_disk_offload"]:
            raise RuntimeError(f"model not fully on GPU {self.device_index}: {report}")
        if linear4bit == 0:
            raise RuntimeError("no bitsandbytes Linear4bit modules found; model is not 4-bit quantized")
        return report

    def environment(self) -> dict:
        import bitsandbytes, transformers, accelerate, tokenizers
        torch = self.torch
        props = torch.cuda.get_device_properties(self.device_index)
        return {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "bitsandbytes": bitsandbytes.__version__,
            "tokenizers": tokenizers.__version__,
            "gpu": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "vram_total_gib": props.total_memory / 1024**3,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "compute_dtype": str(self.compute_dtype).replace("torch.", ""),
            "model_path": str(self.model_path),
            "model_revision_local_metadata": local_revision(self.model_path),
            "tokenizer_json_sha256": file_sha256(self.model_path / "tokenizer.json"),
            "tokenizer_config_sha256": file_sha256(self.model_path / "tokenizer_config.json"),
            "context_capacity": self.context_capacity,
            "load_seconds": self.load_seconds,
            "load_peak_allocated_gib": self.load_peak_allocated_gib,
            "load_peak_reserved_gib": self.load_peak_reserved_gib,
            "warmup_seconds": self.warmup_seconds,
            "placement": self.placement,
            "effective_generation_config": self.generation_config.to_dict(),
        }

    def render(self, request: dict) -> dict:
        messages, transport_hash = build_qwen_messages(request["question"], request["evidence"])
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        if not rendered.endswith(NON_THINKING_SUFFIX):
            raise RuntimeError("chat template did not render the empty non-thinking block")
        encoded = self.tokenizer(rendered, add_special_tokens=False, truncation=False, return_attention_mask=True)
        input_ids = encoded["input_ids"]
        if self.tokenizer.decode(input_ids) != rendered:
            raise RuntimeError("rendered prompt does not round-trip through the tokenizer")
        if len(input_ids) + MAX_NEW_TOKENS > self.context_capacity:
            raise ContextCapacityError(
                f"prompt {len(input_ids)} + allowance {MAX_NEW_TOKENS} > capacity {self.context_capacity}"
            )
        evidence_tokens = [
            len(self.tokenizer(item["text"], add_special_tokens=False, truncation=False)["input_ids"])
            for item in request["evidence"]
        ]
        return {
            "messages": messages,
            "transport_hash": transport_hash,
            "rendered_prompt": rendered,
            "rendered_prompt_sha256": sha256(rendered.encode("utf-8")).hexdigest(),
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"],
            "rendered_prompt_tokens": len(input_ids),
            "qwen_supplied_evidence_tokens_per_item": evidence_tokens,
            "qwen_supplied_evidence_tokens": sum(evidence_tokens),
            "prompt_plus_allowance_tokens": len(input_ids) + MAX_NEW_TOKENS,
        }

    def warmup(self) -> float:
        """Kernel/allocator warm-up on a fixed non-research prompt; output is discarded."""
        request = {"question": "Warm-up.", "evidence": [{"evidence_id": "E1", "text": "Warm-up text."}]}
        rendered = self.render(request)
        started = perf_counter()
        self._generate_ids(rendered, seed=0, max_new_tokens=16)
        self.warmup_seconds = perf_counter() - started
        return self.warmup_seconds

    def _generate_ids(self, rendered: dict, seed: int, max_new_tokens: int | None = None) -> dict:
        torch = self.torch
        from transformers import LogitsProcessor, LogitsProcessorList

        marks: dict[str, float] = {}

        class FirstLogitsTimer(LogitsProcessor):
            def __call__(self, input_ids, scores):  # identity; records prefill completion time
                if "first" not in marks:
                    torch.cuda.synchronize()
                    marks["first"] = perf_counter()
                return scores

        input_ids = torch.tensor([rendered["input_ids"]], device=f"cuda:{self.device_index}")
        attention_mask = torch.tensor([rendered["attention_mask"]], device=f"cuda:{self.device_index}")
        torch.cuda.reset_peak_memory_stats(self.device_index)
        random.seed(seed)
        torch.manual_seed(seed)  # seeds the CPU and every CUDA generator
        torch.cuda.synchronize(self.device_index)
        started = perf_counter()
        kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                logits_processor=LogitsProcessorList([FirstLogitsTimer()]),
                **kwargs,
            )
        torch.cuda.synchronize(self.device_index)
        finished = perf_counter()
        new_ids = output[0, input_ids.shape[1]:].tolist()
        return {
            "new_ids": new_ids,
            "generate_seconds": finished - started,
            "prefill_seconds": (marks["first"] - started) if "first" in marks else None,
            "decode_seconds": (finished - marks["first"]) if "first" in marks else None,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device_index) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device_index) / 1024**3,
        }

    def generate(self, request: dict, seed: int) -> dict:
        started = perf_counter()
        rendered = self.render(request)
        rendered_at = perf_counter()
        generated = self._generate_ids(rendered, seed)
        new_ids = generated["new_ids"]
        eos = set(GENERATION["eos_token_id"])
        hit_eos = bool(new_ids) and new_ids[-1] in eos
        raw_output = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        raw_output_with_special = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        decoded_at = perf_counter()
        evaluation = evaluate_output(raw_output, len(request["evidence"]))
        validated_at = perf_counter()
        output_tokens = len(new_ids)
        decode_seconds = generated["decode_seconds"]
        return {
            "rendered": rendered,
            "raw_output": raw_output,
            "raw_output_with_special_tokens": raw_output_with_special,
            "output_token_ids_sha256": object_sha256(new_ids),
            "output_tokens": output_tokens,
            "finish_reason": "eos" if hit_eos else "max_new_tokens",
            "output_truncated_at_max_new_tokens": not hit_eos,
            "evaluation": evaluation,
            "timing": {
                "render_and_tokenize_seconds": rendered_at - started,
                "generate_seconds": generated["generate_seconds"],
                "prefill_seconds": generated["prefill_seconds"],
                "decode_seconds": decode_seconds,
                "detokenize_seconds": decoded_at - rendered_at - generated["generate_seconds"],
                "parse_and_validate_seconds": validated_at - decoded_at,
                "output_tokens_per_second_generate": output_tokens / generated["generate_seconds"] if generated["generate_seconds"] else None,
                "decode_tokens_per_second": (output_tokens - 1) / decode_seconds if decode_seconds and output_tokens > 1 else None,
            },
            "peak_allocated_gib": generated["peak_allocated_gib"],
            "peak_reserved_gib": generated["peak_reserved_gib"],
        }


# ---------------------------------------------------------------------------
# Checkpointed one-attempt runner
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, row: dict) -> None:
    """Append one JSON line durably. A torn final line from a crash is tolerated by the loader."""
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_newline = existing.read(1) != b"\n"
    payload = ("\n" if needs_newline else "") + json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def load_jsonl_tolerant(path: Path) -> tuple[list[dict], list[int]]:
    """Return parseable rows and the line numbers of unparseable (torn) lines."""
    rows, bad = [], []
    if not path.exists():
        return rows, bad
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(number)
    return rows, bad


def write_json_atomic(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True)
class PlannedPair:
    dataset: str
    question_id: str
    system: str
    dependency_group: str
    unit_index: int

    @property
    def key(self) -> str:
        return "|".join((self.dataset, self.question_id, self.system))


class LocalQwenRunner:
    """One generation attempt per pair; infrastructure failures are logged separately and resumed."""

    def __init__(self, directory: Path, meta: dict) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.predictions_path = directory / "PREDICTIONS.jsonl"
        self.validation_errors_path = directory / "VALIDATION_ERRORS.jsonl"
        self.infrastructure_path = directory / "INFRASTRUCTURE_FAILURES.jsonl"
        self.timings_path = directory / "REQUEST_TIMINGS.jsonl"
        self.run_log_path = directory / "RUN_LOG.jsonl"
        self.meta_path = directory / "CHECKPOINT_META.json"
        if self.meta_path.exists():
            saved = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if saved != meta:
                raise ValueError("checkpoint configuration mismatch; refusing to resume under a different configuration")
        else:
            write_json_atomic(self.meta_path, meta)
        self.meta = meta
        rows, torn = load_jsonl_tolerant(self.predictions_path)
        self.completed = {row["logical_key"]: row for row in rows}
        if len(self.completed) != len(rows):
            raise ValueError("duplicate completed pair records in checkpoint")
        self.torn_lines = torn

    def log(self, event: str, **fields) -> None:
        append_jsonl(self.run_log_path, {"event": event, "utc": utc_now(), **fields})

    def run(self, plan: list[PlannedPair], requests: dict[str, dict], generator: LocalQwenGenerator,
            *, run_id: str, deadline_epoch: float | None = None,
            fallback_seconds_per_pair: float = 60.0, reserve_factor: float = 1.5,
            base_record: dict | None = None, pair_extras: dict[str, dict] | None = None) -> dict:
        import time
        base_record = base_record or {}
        pair_extras = pair_extras or {}
        if self.torn_lines:
            append_jsonl(self.infrastructure_path, {
                "run_id": run_id, "utc": utc_now(), "type": "torn_checkpoint_lines_ignored",
                "line_numbers": self.torn_lines,
            })
        self.log("run_started", run_id=run_id, planned_pairs=len(plan), already_completed=len(self.completed))
        units: dict[int, list[PlannedPair]] = {}
        for pair in plan:
            units.setdefault(pair.unit_index, []).append(pair)
        durations: list[float] = []
        new = 0
        stop_reason = "plan_completed"
        for unit_index in sorted(units):
            pending = [pair for pair in units[unit_index] if pair.key not in self.completed]
            for pair in units[unit_index]:
                if pair.key in self.completed:
                    self._verify_completed(pair, requests[pair.key], generator)
            if not pending:
                continue
            if deadline_epoch is not None:
                per_pair = (sum(durations) / len(durations)) if durations else fallback_seconds_per_pair
                if time.time() + reserve_factor * per_pair * len(pending) > deadline_epoch:
                    stop_reason = "execution_budget_deadline_before_unit"
                    self.log("deadline_stop", run_id=run_id, unit_index=unit_index, pending_pairs=len(pending))
                    break
            for pair in pending:
                started_wall = perf_counter()
                request = requests[pair.key]
                try:
                    record = self._generate_record(pair, request, generator, run_id,
                                                   {**base_record, **pair_extras.get(pair.key, {})})
                except Exception as exc:
                    append_jsonl(self.infrastructure_path, {
                        "run_id": run_id, "utc": utc_now(), "type": "exception_during_generation",
                        "logical_key": pair.key, "error_type": type(exc).__name__, "error": str(exc)[:4000],
                        "traceback": traceback.format_exc()[-8000:],
                        "note": "infrastructure failure: not a model output; pair remains pending and resumes with the same seed",
                    })
                    self.log("run_aborted_infrastructure", run_id=run_id, logical_key=pair.key)
                    raise
                record["latency_end_to_end_before_save_seconds"] = perf_counter() - started_wall
                append_jsonl(self.predictions_path, record)
                if record["validation_status"] != "valid":
                    append_jsonl(self.validation_errors_path, {
                        key: record[key] for key in (
                            "logical_key", "request_fingerprint", "parse_status", "parse_error",
                            "schema_errors", "semantic_error", "validation_status", "finish_reason",
                            "output_tokens", "raw_output",
                        )
                    })
                total = perf_counter() - started_wall
                append_jsonl(self.timings_path, {
                    "logical_key": pair.key, "request_fingerprint": record["request_fingerprint"],
                    "run_id": run_id, "latency_end_to_end_including_save_seconds": total,
                })
                self.completed[pair.key] = record
                durations.append(total)
                new += 1
        self.log("run_finished", run_id=run_id, new_pairs=new, stop_reason=stop_reason,
                 completed_pairs=len(self.completed))
        return {"new_pairs": new, "completed_pairs": len(self.completed), "stop_reason": stop_reason,
                "mean_seconds_per_new_pair": (sum(durations) / len(durations)) if durations else None}

    def _verify_completed(self, pair: PlannedPair, request: dict, generator: LocalQwenGenerator) -> None:
        saved = self.completed[pair.key]
        seed = question_seed(pair.dataset, pair.question_id)
        rendered = generator.render(request)
        expected = qwen_request_fingerprint(request, rendered["rendered_prompt_sha256"], seed)
        if saved["request_fingerprint"] != expected:
            raise ValueError(f"resume fingerprint mismatch for {pair.key}")

    def _generate_record(self, pair: PlannedPair, request: dict, generator: LocalQwenGenerator,
                         run_id: str, base_record: dict) -> dict:
        if (request["dataset"], request["question_id"], request["system"]) != (pair.dataset, pair.question_id, pair.system):
            raise ValueError(f"request/plan mismatch for {pair.key}")
        seed = question_seed(pair.dataset, pair.question_id)
        started_utc = utc_now()
        result = generator.generate(request, seed)
        rendered = result["rendered"]
        evaluation = result["evaluation"]
        fingerprint = qwen_request_fingerprint(request, rendered["rendered_prompt_sha256"], seed)
        return {
            **base_record,
            "experiment_version": EXPERIMENT_VERSION,
            "run_id": run_id,
            "logical_key": pair.key,
            "dataset": pair.dataset,
            "question_id": pair.question_id,
            "source_group_id": pair.dependency_group,
            "system": pair.system,
            "unit_index": pair.unit_index,
            "request_fingerprint": fingerprint,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "quantization": QUANTIZATION,
            "compute_dtype": str(generator.compute_dtype).replace("torch.", ""),
            "generation_config": GENERATION,
            "master_seed": MASTER_SEED,
            "question_seed": seed,
            "attempt": 1,
            "question": request["question"],
            "supplied_evidence": request["evidence"],
            "supplied_evidence_ids": [item["evidence_id"] for item in request["evidence"]],
            "supplied_evidence_hash": evidence_hash(request["evidence"]),
            "transport_hash": rendered["transport_hash"],
            "rendered_prompt_sha256": rendered["rendered_prompt_sha256"],
            "rendered_prompt_tokens": rendered["rendered_prompt_tokens"],
            "input_tokens": rendered["rendered_prompt_tokens"],
            "qwen_supplied_evidence_tokens": rendered["qwen_supplied_evidence_tokens"],
            "qwen_supplied_evidence_tokens_per_item": rendered["qwen_supplied_evidence_tokens_per_item"],
            "prompt_plus_allowance_tokens": rendered["prompt_plus_allowance_tokens"],
            "context_capacity": generator.context_capacity,
            "max_new_tokens": MAX_NEW_TOKENS,
            "raw_output": result["raw_output"],
            "raw_output_with_special_tokens": result["raw_output_with_special_tokens"],
            "output_token_ids_sha256": result["output_token_ids_sha256"],
            "output_tokens": result["output_tokens"],
            "total_tokens": rendered["rendered_prompt_tokens"] + result["output_tokens"],
            "finish_reason": result["finish_reason"],
            "output_truncated_at_max_new_tokens": result["output_truncated_at_max_new_tokens"],
            "parse_status": evaluation["parse_status"],
            "parse_error": evaluation["parse_error"],
            "parsed_output": evaluation["parsed_output"],
            "schema_valid": evaluation["schema_valid"],
            "schema_errors": evaluation["schema_errors"],
            "semantic_valid": evaluation["semantic_valid"],
            "semantic_error": evaluation["semantic_error"],
            "validation_status": evaluation["validation_status"],
            "validated_response": evaluation["validated_response"],
            "thinking_markers_present": evaluation["thinking_markers_present"],
            "timing": result["timing"],
            "latency_generation_seconds": result["timing"]["generate_seconds"],
            "throughput_output_tokens_per_second": result["timing"]["output_tokens_per_second_generate"],
            "peak_allocated_vram_gib": result["peak_allocated_gib"],
            "peak_reserved_vram_gib": result["peak_reserved_gib"],
            "started_utc": started_utc,
            "finished_utc": utc_now(),
        }

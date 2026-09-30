"""Create protocol amendment 001 separating exposed-train and held-out ID registries."""
from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "adaptive_single_view_v2"


def digest(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def main() -> int:
    amendment = OUT / "PROTOCOL_AMENDMENT_001.json"
    if amendment.exists():
        raise SystemExit(f"refusing to overwrite amendment: {amendment}")
    broad = json.loads((OUT / "DESIGN_EXPOSED_IDS.json").read_text(encoding="utf-8"))
    train = sorted(example_id for example_id in broad["ids"] if ":train:" in example_id)
    heldout = sorted(example_id for example_id in broad["ids"] if ":train:" not in example_id)
    train_payload = {
        "status": "ACTIVE_EXCLUSION_REGISTRY", "scope": "previously exposed training IDs",
        "count": len(train), "ids_sha256": digest(train), "ids": train,
    }
    heldout_payload = {
        "status": "ID_ONLY_OVERLAP_REGISTRY_DO_NOT_USE_FOR_DEVELOPMENT",
        "scope": "identifiers extracted from pre-existing normalized validation/test artifacts; no text, labels, or outcomes inspected by the freeze step",
        "count": len(heldout), "ids_sha256": digest(heldout), "ids": heldout,
    }
    payload = {
        "status": "FROZEN_BEFORE_IMPLEMENTATION",
        "amendment": "001", "date": "2026-09-05",
        "reason": (
            "The initial artifact scanner conservatively placed validation/test identifiers in a file named "
            "DESIGN_EXPOSED_IDS. Identifier-only overlap checking is required, but those identifiers are not "
            "development examples and no held-out text, labels, or outcomes were inspected. This amendment "
            "separates the active training exclusion registry from the locked held-out ID-only registry."
        ),
        "supersedes": {
            "PROTOCOL.json.partitions.design_exposed.file": "DESIGN_EXPOSED_TRAIN_IDS.json",
            "PROTOCOL.json.partitions.design_exposed.count": len(train),
            "PROTOCOL.json.partitions.design_exposed.ids_sha256": train_payload["ids_sha256"],
        },
        "heldout_overlap_registry": {
            "file": "HELDOUT_ID_REGISTRY.json", "count": len(heldout),
            "ids_sha256": heldout_payload["ids_sha256"],
            "permitted_use": "set-intersection of example IDs only",
            "forbidden_use": "loading documents, questions, labels, evidence, answers, or metrics during development",
        },
        "original_files_preserved": ["PROTOCOL.json", "PROTOCOL.md", "DESIGN_EXPOSED_IDS.json"],
        "method_or_selection_rule_change": False,
    }
    (OUT / "DESIGN_EXPOSED_TRAIN_IDS.json").write_text(json.dumps(train_payload, indent=2), encoding="utf-8")
    (OUT / "HELDOUT_ID_REGISTRY.json").write_text(json.dumps(heldout_payload, indent=2), encoding="utf-8")
    amendment.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "PROTOCOL_AMENDMENT_001.md").write_text(f"""# Adaptive Single-View V2 protocol amendment 001

**Status:** FROZEN BEFORE IMPLEMENTATION  
**Date:** 2026-09-05

The initial conservative scan combined {len(train)} previously exposed training IDs with {len(heldout)} identifiers from pre-existing normalized validation/test files. The freeze step read identifiers only; it did not load held-out text, labels, evidence, answers, or outcomes.

This amendment makes `DESIGN_EXPOSED_TRAIN_IDS.json` the active development exclusion registry. `HELDOUT_ID_REGISTRY.json` is locked and may be used only for ID-set overlap checks. The original three frozen files remain preserved. No method, hypothesis, metric, margin, or selection rule changes.
""", encoding="utf-8")
    print(json.dumps({"training_exclusions": len(train), "heldout_id_only": len(heldout)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

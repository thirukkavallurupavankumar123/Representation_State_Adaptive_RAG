# Adaptive Single-View V2 protocol amendment 001

**Status:** FROZEN BEFORE IMPLEMENTATION  
**Date:** 2026-09-05

The initial conservative scan combined 475 previously exposed training IDs with 7291 identifiers from pre-existing normalized validation/test files. The freeze step read identifiers only; it did not load held-out text, labels, evidence, answers, or outcomes.

This amendment makes `DESIGN_EXPOSED_TRAIN_IDS.json` the active development exclusion registry. `HELDOUT_ID_REGISTRY.json` is locked and may be used only for ID-set overlap checks. The original three frozen files remain preserved. No method, hypothesis, metric, margin, or selection rule changes.

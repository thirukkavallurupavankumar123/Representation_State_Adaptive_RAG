# SAR-RAG Detailed Implementation Blueprint

> **Superseded primary online algorithm (4 August 2026):** The authoritative primary
> implementation is `ONE_CALL_RESEARCH_PIPELINE.md`. The multi-call evidence judge in
> this document is a legacy ablation, not the main research method.

## Probe-Based Representation Selection, Adaptive Retrieval Without Fixed Final K, and Post-Retrieval Deduplication

**Version:** 2.0  
**Date:** 2 August 2026  
**Status:** corrected implementation specification

---

## 1. Purpose

This document specifies how to implement Source-Aligned Representation-State Adaptive RAG (SAR-RAG) as an executable and testable software system.

The corrected online pipeline is:

```text
Query
  -> requirements
  -> probe Top-p from every candidate representation
  -> source alignment without deleting candidates
  -> evidence-group RRF
  -> representation profiles
  -> select useful representations
  -> adaptive retrieval from selected indexes with no fixed final k
  -> stop by evidence sufficiency or safety budget
  -> final qualifier-preserving deduplication
  -> compact final evidence
  -> cited answer or abstention
```

The fixed probe depth `p` is not the final retrieval size. The fixed adaptive batch size is also not the final retrieval size. The final evidence count is discovered only after adaptive retrieval and final deduplication.

---

## 2. Final implementation decisions

### 2.1 Candidate representations

The implementation supports five registered candidate representations because the research question evaluates whether all five contribute useful evidence:

1. `sentence_window`
2. `sliding_window`
3. `paragraph`
4. `semantic`
5. `hierarchical`

The registry is configurable. Three-view experiments can disable `sliding_window` and `semantic`, but the software does not hard-code either three or five.

### 2.2 Probe and adaptive retrieval

```yaml
probe_depth: 3               # primary; run 5 as sensitivity experiment
adaptive_batch_size: 3       # computational batch, not final k
rrf_k0: 60
max_candidates_examined: 60
max_retrieval_rounds: 8
max_final_context_tokens: 2000
max_reformulations_per_requirement: 1
confirmation_rounds: 1
```

The primary experiment uses `probe_depth = 3`; `probe_depth = 5` is a pre-declared sensitivity condition. They must not be chosen per test query.

### 2.3 Calibration and fusion

- Raw similarity scores are not calibrated across representations.
- Robust z-score normalization is not used in the primary implementation.
- Score-knee filtering is not used in the primary implementation.
- Raw cosine scores order candidates only inside their own representation.
- Source-aligned RRF ranks probe evidence groups.
- RRF helps representation selection but is never the only selection signal.

### 2.4 Duplicate handling

- Source alignment occurs during the probe and adaptive retrieval.
- No semantic or containment-based evidence deletion occurs before representation selection.
- Exact duplicates may share a cached judgment to save computation, but all retrieval provenance remains in the trace.
- Final exact, containment, and semantic deduplication occurs only after adaptive retrieval stops.

### 2.5 No fixed final k

There is no configured final `top_k`.

```text
k_final = number of evidence items remaining after final deduplication and compaction
```

Safety budgets bound execution but do not specify final evidence size.

---

## 3. Recommended repository structure

```text
research_projectt/
  pyproject.toml
  README.md
  configs/
    base.yaml
    representations.yaml
    experiments/
      probe3_five_views.yaml
      probe5_five_views.yaml
      probe3_three_views.yaml
  src/sar_rag/
    __init__.py
    config.py
    domain/
      models.py
      enums.py
      errors.py
    ingestion/
      registry.py
      parsers/
        base.py
        text.py
        html.py
        markdown.py
        pdf.py
      canonicalize.py
      validate.py
      versioning.py
    representations/
      base.py
      sentence_window.py
      sliding_window.py
      paragraph.py
      semantic.py
      hierarchical.py
      registry.py
    embeddings/
      encoder.py
      cache.py
    indexes/
      base.py
      faiss_exact.py
      manifest.py
    query/
      requirements.py
      validation.py
      reformulation.py
    retrieval/
      probe.py
      alignment.py
      rrf.py
      profiles.py
      representation_selection.py
      adaptive.py
      stopping.py
    evidence/
      judge.py
      applicability.py
      states.py
      accumulator.py
      deduplicate.py
      compact.py
    generation/
      context_builder.py
      answer.py
    tracing/
      events.py
      run_store.py
    evaluation/
      retrieval_metrics.py
      state_metrics.py
      answer_metrics.py
      efficiency_metrics.py
    api/
      schemas.py
      service.py
    cli.py
  tests/
    unit/
    integration/
    fixtures/
    golden/
  data/
    raw/
    canonical/
    indexes/
    manifests/
  runs/
  output/
```

The implementation should keep ingestion, representation construction, retrieval control, evidence judgment, and answer generation separate. Otherwise, an experiment cannot identify which module caused an improvement or failure.

---

## 4. Technology stack

Recommended controlled implementation:

```text
Language: Python 3.11 or later
Data validation: Pydantic v2
Configuration: YAML validated into Pydantic settings
Embedding: sentence-transformers or FlagEmbedding with BAAI/bge-m3
Experimental vector index: FAISS IndexFlatIP
Sparse baseline: rank_bm25 or Pyserini
Model inference: transformers/vLLM or a recorded model API
PDF parsing: PyMuPDF plus layout metadata
HTML parsing: BeautifulSoup/lxml
Sentence segmentation: spaCy or Stanza with pinned model
Token counting: embedding tokenizer for chunking; generator tokenizer for context
Persistent metadata: SQLite for prototype, PostgreSQL for multi-user deployment
Trace files: JSON Lines
Testing: pytest
Metrics: numpy, scipy, scikit-learn
```

Every package and model revision must be locked. Model repository names alone are insufficient for final reproducibility.

---

## 5. Core domain models

The following schemas are the minimum required data contracts. Field names should remain stable because experiment traces depend on them.

### 5.1 Document record

```python
class DocumentRecord(BaseModel):
    document_id: str
    version_id: str
    title: str
    source_uri: str | None
    publisher: str | None
    effective_from: date | None
    effective_to: date | None
    authority_level: str | None
    supersedes_version_id: str | None
    content_hash: str
    language: str
    license: str | None
    parser_name: str
    parser_version: str
```

`authority_level` is descriptive unless a corpus-specific authority ordering is explicitly configured. The software must not infer authority from the string.

### 5.2 Canonical source unit

```python
class CanonicalUnit(BaseModel):
    unit_id: str
    document_id: str
    version_id: str
    unit_type: Literal[
        "heading", "sentence", "list_item", "table_row",
        "caption", "footnote", "equation", "code_block", "form_field"
    ]
    parent_id: str | None
    previous_unit_id: str | None
    next_unit_id: str | None
    section_path: list[str]
    page_number: int | None
    anchor: str | None
    character_start: int
    character_end: int
    raw_text: str
    normalized_hash: str
    metadata: dict[str, Any]
```

For a table row, `metadata` contains column headers and table identity. For a footnote, it contains the marker and referenced unit IDs.

### 5.3 Representation chunk

```python
class RepresentationChunk(BaseModel):
    chunk_id: str
    representation_id: str
    document_id: str
    version_id: str
    canonical_unit_ids: list[str]
    parent_chunk_id: str | None
    child_chunk_ids: list[str]
    embedding_text: str
    evidence_text: str
    token_count: int
    metadata: dict[str, Any]
```

`embedding_text` may include headings for context. `evidence_text` must preserve the original citable source.

### 5.4 Probe candidate

```python
class RetrievedCandidate(BaseModel):
    candidate_id: str
    query_id: str
    representation_id: str
    chunk_id: str
    rank: int
    raw_score: float
    retrieval_query: str
    retrieval_stage: Literal["probe", "adaptive", "reformulation"]
    canonical_unit_ids: list[str]
    document_id: str
    version_id: str
    token_count: int
```

The raw score is stored for analysis but never compared directly across representations in the primary algorithm.

### 5.5 Evidence requirement

```python
class EvidenceRequirement(BaseModel):
    requirement_id: str
    text: str
    critical: bool = True
    entities: list[str]
    applicability_variables: list[str]
```

### 5.6 Query analysis

```python
class QueryFact(BaseModel):
    fact_id: str
    text: str
    source: Literal["explicit_query"]

class QueryAnalysis(BaseModel):
    query_id: str
    original_query: str
    requirements: list[EvidenceRequirement]
    query_facts: list[QueryFact]
    decomposition_confident: bool
```

### 5.7 Candidate evidence judgment

```python
class ApplicabilityFrame(BaseModel):
    entity: list[str] = []
    population: list[str] = []
    department: list[str] = []
    jurisdiction: list[str] = []
    time_scope: list[str] = []
    version_scope: list[str] = []
    exception_state: list[str] = []

class Qualifier(BaseModel):
    qualifier_type: Literal[
        "condition", "exception", "negation", "threshold", "date",
        "scope", "authority", "version", "modality", "prerequisite"
    ]
    text: str
    cited_unit_ids: list[str]

class RequirementFinding(BaseModel):
    requirement_id: str
    status: Literal["SUPPORTED", "REFUTED", "NOT_APPLICABLE", "UNRESOLVED"]
    claim: str | None
    cited_unit_ids: list[str]
    applicability: ApplicabilityFrame
    qualifiers: list[Qualifier]

class CandidateJudgment(BaseModel):
    candidate_id: str
    findings: list[RequirementFinding]
    context_restoration_needed: bool
    missing_context_types: list[str]
    valid: bool
```

### 5.8 Evidence group

```python
class EvidenceGroupMember(BaseModel):
    candidate_id: str
    representation_id: str
    rank: int
    canonical_unit_ids: list[str]

class EvidenceGroup(BaseModel):
    group_id: str
    anchor_candidate_id: str
    document_id: str
    version_id: str
    members: list[EvidenceGroupMember]
    rrf_score: float
    group_unit_ids: list[str]
```

### 5.9 Representation profile

```python
class RepresentationProfile(BaseModel):
    representation_id: str
    requirements_supported: set[str]
    requirements_refuted: set[str]
    exclusive_obligations: set[str]
    unique_qualifier_keys: set[str]
    context_restoration_keys: set[str]
    contradiction_investigation_keys: set[str]
    independent_source_keys: set[str]
    useful_group_ids: set[str]
    redundant_group_ids: set[str]
    rrf_contribution: float
    mean_probe_tokens: float
    selected: bool = False
    selection_reasons: list[str] = []
```

### 5.10 Accumulator state

```python
class RequirementState(BaseModel):
    requirement_id: str
    status: Literal["SUPPORTED", "REFUTED", "NOT_APPLICABLE", "UNRESOLVED"]
    supporting_candidate_ids: list[str]
    refuting_candidate_ids: list[str]
    applicability_frames: list[ApplicabilityFrame]
    qualifier_keys: set[str]
    unresolved_conflicts: list[str]

class RepresentationRuntimeState(BaseModel):
    representation_id: str
    selected: bool
    active: bool
    next_rank: int
    consecutive_empty_batches: int
    accepted_candidate_ids: list[str]
    pause_reason: str | None

class AccumulatorState(BaseModel):
    requirement_states: dict[str, RequirementState]
    accepted_candidate_ids: list[str]
    provisional_candidate_ids: list[str]
    detected_states: set[str]
    representation_states: dict[str, RepresentationRuntimeState]
    candidates_examined: int
    retrieval_round: int
    final_context_tokens: int
```

---

## 6. Offline ingestion implementation

### 6.1 Document acquisition

Each input file is registered before parsing:

```python
def register_document(path: Path, supplied_metadata: dict) -> DocumentRecord:
    raw_bytes = path.read_bytes()
    content_hash = sha256(raw_bytes).hexdigest()
    return DocumentRecord(
        document_id=stable_document_id(supplied_metadata, content_hash),
        version_id=stable_version_id(supplied_metadata, content_hash),
        content_hash=content_hash,
        **supplied_metadata,
    )
```

Document identity and version identity must be separate. A changed version of the same policy keeps `document_id` but receives a new `version_id`.

### 6.2 Parser interface

```python
class DocumentParser(Protocol):
    def supports(self, path: Path) -> bool: ...
    def parse(self, document: DocumentRecord, path: Path) -> ParsedDocument: ...
```

`ParsedDocument` contains structural blocks with exact offsets and page or anchor metadata.

### 6.3 Canonicalization

Canonicalization converts parser blocks into typed source units without changing their original text.

Rules:

1. Headings become units and structural parents.
2. Paragraph sentences become sentence units.
3. Each list item becomes a unit.
4. Each table row becomes a unit containing its headers in metadata.
5. Captions and footnotes become separate linked units.
6. Equations remain linked to their definition paragraph.
7. Code is split only at structurally complete elements when possible.

Stable unit ID:

```python
unit_id = sha256(
    f"{version_id}|{unit_type}|{character_start}|{character_end}|{normalized_hash}"
).hexdigest()[:24]
```

### 6.4 Validation

The ingestion job fails if:

- Offsets fall outside the extracted source.
- A parent ID is missing.
- Parent-child links contain a cycle.
- The same unit ID maps to different text.
- Table rows lose required headers.
- A representation references a nonexistent unit.

Parser warnings are stored in the document manifest and included in answerability decisions when relevant evidence comes from an unreliable region.

---

## 7. Representation builders

Every builder implements:

```python
class RepresentationBuilder(Protocol):
    representation_id: str

    def build(
        self,
        document: DocumentRecord,
        units: list[CanonicalUnit],
    ) -> list[RepresentationChunk]: ...
```

### 7.1 Sentence-window builder

Primary parameters:

```yaml
sentence_window:
  previous_units: 1
  next_units: 1
  cross_parent_boundary: false
  max_embedding_tokens: 256
```

The target unit is marked inside the embedding text. Evidence lineage contains all window units, while metadata identifies the target unit.

```text
[SECTION PATH]
[PREVIOUS] ...
[TARGET] ...
[NEXT] ...
```

### 7.2 Sliding-window builder

```yaml
sliding_window:
  window_tokens: 256
  overlap_tokens: 64
  split_on_unit_boundaries: true
  cross_section_boundary: false
```

Windows are assembled from complete canonical units. If a single unit exceeds the limit, it is fragmented with a shared parent unit ID.

### 7.3 Paragraph-aware builder

```yaml
paragraph:
  merge_below_tokens: 96
  target_max_tokens: 384
  cross_subsection_boundary: false
  overlap_tokens: 0
```

Natural paragraphs, lists, and table groups are preferred. Small adjacent blocks may merge inside the same subsection.

### 7.4 Semantic builder

The semantic builder uses adjacent-unit embedding similarity, not an LLM-generated rewrite.

```yaml
semantic:
  min_tokens: 96
  max_tokens: 384
  breakpoint_method: percentile
  breakpoint_percentile: 20
  cross_subsection_boundary: false
```

For adjacent unit embeddings `u_i` and `u_(i+1)`:

```python
boundary_score_i = 1 - cosine(u_i, u_i_plus_1)
```

A boundary is proposed when the discontinuity is in the highest 20% within the document section, subject to minimum and maximum sizes. This parameter is frozen before test evaluation.

### 7.5 Hierarchical builder

Hierarchical nodes exist at document, section, subsection, and block-parent levels.

The primary implementation uses no generated summaries:

```python
child_centroid = normalize(mean(normalize(v) for v in child_vectors))
heading_vector = normalize(embed(heading_path_text))
node_vector = normalize((child_centroid + heading_vector) / 2)
```

Retrieved hierarchical nodes guide scope restoration and child retrieval. Their original headings may be evidence for scope, but their vector centroid is never treated as source text.

### 7.6 Representation registry

```python
BUILDERS = {
    "sentence_window": SentenceWindowBuilder,
    "sliding_window": SlidingWindowBuilder,
    "paragraph": ParagraphBuilder,
    "semantic": SemanticBuilder,
    "hierarchical": HierarchicalBuilder,
}
```

An experiment config declares enabled representations. Index names include corpus release, representation ID, embedding model revision, and builder configuration hash.

---

## 8. Embedding and indexing

### 8.1 Encoder

```python
class EmbeddingEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
```

Cache key:

```text
sha256(model_revision + tokenizer_revision + embedding_text)
```

### 8.2 Exact index interface

```python
class RepresentationIndex(Protocol):
    representation_id: str

    def search(
        self,
        query_vector: np.ndarray,
        start_rank: int,
        limit: int,
        filters: dict | None = None,
    ) -> list[RetrievedCandidate]: ...
```

FAISS does not directly return rank ranges. The exact implementation retrieves `start_rank + limit - 1` results and slices the requested range.

```python
scores, ids = index.search(query_vector[None, :], start_rank + limit - 1)
selected = zip(scores[0][start_rank-1:], ids[0][start_rank-1:])
```

This allows adaptive retrieval to request ranks 4-6 after a Top-3 probe without pretending that the final `k` is six.

### 8.3 Index manifest

Every index stores:

```json
{
  "corpus_release": "...",
  "representation_id": "paragraph",
  "builder_config_hash": "...",
  "embedding_model": "BAAI/bge-m3",
  "embedding_revision": "...",
  "dimension": 1024,
  "metric": "inner_product_on_normalized_vectors",
  "chunk_count": 12345,
  "created_at": "..."
}
```

The service refuses to combine indexes built from different corpus releases or embedding revisions.

---

## 9. Query analysis implementation

### 9.1 Requirement extraction

The extractor uses a frozen instruction model with temperature zero and schema-constrained JSON.

```python
def analyze_query(query: str) -> QueryAnalysis:
    raw = judge_model.generate(REQUIREMENT_PROMPT.format(query=query))
    try:
        result = QueryAnalysis.model_validate_json(raw)
        validate_query_traceability(result, query)
        return result
    except ValidationError as exc:
        repaired = judge_model.generate(REPAIR_PROMPT.format(error=exc, output=raw))
        try:
            result = QueryAnalysis.model_validate_json(repaired)
            validate_query_traceability(result, query)
            return result
        except ValidationError:
            return whole_query_fallback(query)
```

Traceability validation requires every requirement to correspond to an explicit requested aspect of the query. Requirements unsupported by the wording are rejected.

### 9.2 Query embedding

The original query is embedded once for the probe. Requirements do not replace it.

Reformulated requirement queries receive separate query vectors later and are tagged with their originating requirement ID.

---

## 10. Phase A: probe retrieval for representation selection

### 10.1 Probe operation

```python
def run_probe(
    query: QueryAnalysis,
    indexes: dict[str, RepresentationIndex],
    probe_depth: int,
) -> list[RetrievedCandidate]:
    vector = encoder.encode([query.original_query])[0]
    results = []
    for representation_id, index in indexes.items():
        candidates = index.search(vector, start_rank=1, limit=probe_depth)
        results.extend(mark_stage(candidates, "probe"))
    return results
```

The probe fetches `probe_depth` from every enabled representation. It does not accept them as final evidence and does not establish final `k`.

### 10.2 Probe candidate judgment

Representation selection needs observable evidence value, not ranks alone. Each probe candidate is judged against the requirements before building representation profiles.

For hierarchical candidates, the probe judgment sees:

- Heading path.
- Node metadata.
- At most the highest query-ranked child passage.

This prevents a node centroid from being judged as if it were textual evidence.

### 10.3 Source alignment without deletion

Alignment function:

```python
def compatible_for_group(a: RetrievedCandidate, b: RetrievedCandidate) -> bool:
    if (a.document_id, a.version_id) != (b.document_id, b.version_id):
        return False
    ua, ub = set(a.canonical_unit_ids), set(b.canonical_unit_ids)
    if ua == ub:
        return True
    if ua <= ub or ub <= ua:
        return True
    return len(ua & ub) / len(ua | ub) >= 0.5
```

Group construction is anchor-based, not transitive:

```python
def build_groups(candidates):
    ordered = sorted(candidates, key=global_stable_probe_order)
    groups = []
    for candidate in ordered:
        group = first_group_with_compatible_anchor(candidate, groups)
        if group is None:
            groups.append(new_group(candidate))
        else:
            group.members.append(candidate)
    return groups
```

`global_stable_probe_order` is exactly `(rank ascending, representation_id ascending, candidate_id ascending)`. It does not use raw similarity scores from different representations. This makes anchor selection deterministic across repeated runs.

No candidate is deleted. All representation memberships and ranks are preserved.

### 10.4 RRF calculation

For evidence group `g`:

```python
def group_rrf(group: EvidenceGroup, k0: int = 60) -> float:
    best_rank_by_representation = {}
    for member in group.members:
        best_rank_by_representation[member.representation_id] = min(
            member.rank,
            best_rank_by_representation.get(member.representation_id, math.inf),
        )
    return sum(1.0 / (k0 + rank) for rank in best_rank_by_representation.values())
```

One representation contributes at most once to a group. Multiple overlapping chunks from the same index cannot inflate the score.

### 10.5 Representation RRF contribution

Only useful groups contribute:

```python
def representation_rrf_contribution(rep_id, useful_groups, k0=60):
    total = 0.0
    for group in useful_groups:
        ranks = [m.rank for m in group.members if m.representation_id == rep_id]
        if ranks:
            total += 1.0 / (k0 + min(ranks))
    return total
```

An evidence group is useful if at least one validated probe judgment:

- Resolves a requirement.
- Adds a qualifier.
- Establishes applicability.
- Restores context.
- Exposes or resolves a disagreement.

RRF contribution is not a relevance probability.

---

## 11. Representation profile construction

### 11.1 Obligation keys

Representation selection operates on discrete obligations, not a weighted score.

```text
REQ:R1:SUPPORTED
REQ:R2:REFUTED
APPLICABILITY:department:Customer Support
QUALIFIER:R2:exception:official emergency
CONTEXT:R1:heading_scope
CONFLICT:R3:version_resolution
SOURCE:R1:independent:document-27
```

Each validated probe finding yields zero or more obligation keys.

### 11.2 Profile algorithm

```python
def build_profiles(representations, groups, judgments):
    profiles = {r: empty_profile(r) for r in representations}

    for group in groups:
        group_obligations = obligations_from_group(group, judgments)
        useful = bool(group_obligations)

        for member in group.members:
            profile = profiles[member.representation_id]
            if useful:
                profile.useful_group_ids.add(group.group_id)
                add_obligations(profile, group_obligations)
            else:
                profile.redundant_group_ids.add(group.group_id)

    mark_exclusive_obligations(profiles)

    for profile in profiles.values():
        profile.rrf_contribution = representation_rrf_contribution(
            profile.representation_id,
            useful_groups_for(profile),
        )

    return profiles
```

### 11.3 Exclusive contribution

An obligation is exclusive when exactly one representation's probe candidates establish it.

Exclusive contribution is not automatically assumed correct; it must come from a valid cited judgment. Representations with valid exclusive critical obligations are mandatory selections.

---

## 12. Representation-selection algorithm

RRF alone cannot select representations because unique evidence may receive a lower RRF score than repeated evidence.

The final selector uses mandatory unique contribution followed by deterministic set cover.

### 12.1 Selection input

```text
Profiles for every enabled representation
Probe-discovered obligation universe
Critical requirement IDs
Probe evidence groups and RRF scores
Mean candidate token cost
```

### 12.2 Mandatory selections

Select a representation immediately when it is the only probe view that supplies:

- A critical requirement finding.
- A query-relevant qualifier.
- Required scope restoration.
- Evidence needed to investigate a contradiction.

### 12.3 Greedy set cover

For obligations not yet covered:

```python
def select_representations(profiles, obligations):
    selected = set()
    uncovered = set(obligations)

    for profile in profiles.values():
        if profile.exclusive_obligations:
            selected.add(profile.representation_id)
            uncovered -= all_obligations(profile)
            profile.selection_reasons.append("exclusive_probe_obligation")

    while uncovered:
        candidates = [p for p in profiles.values() if p.representation_id not in selected]
        if not candidates:
            break

        ranked = sorted(
            candidates,
            key=lambda p: (
                -len(all_obligations(p) & uncovered),
                -p.rrf_contribution,
                p.mean_probe_tokens,
                p.representation_id,
            ),
        )
        best = ranked[0]
        gain = all_obligations(best) & uncovered
        if not gain:
            break

        selected.add(best.representation_id)
        uncovered -= gain
        best.selection_reasons.append(f"set_cover:{sorted(gain)}")

    return selected, uncovered
```

Tie-breaking order is:

1. More uncovered obligations.
2. Higher useful-group RRF contribution.
3. Lower mean probe token cost.
4. Stable representation ID ordering for reproducibility.

There are no tuned selection weights.

### 12.4 Shared-insufficiency fallback

If the probe discovers no valid critical requirement evidence:

```text
state = SHARED_INSUFFICIENCY
selected representations = all enabled representations
action = retrieve one adaptive batch from each, then rebuild profiles
```

This prevents the selector from making an arbitrary decision when the probe contains no useful signal.

### 12.5 Minimum safety inclusion

If the set-cover result contains only one direct representation and no structural representation, select the highest-ranked context-restoring representation when any probe candidate signals missing scope. This is based on an explicit context obligation, not a permanent rule that hierarchy must always be selected.

### 12.6 Selector output

```json
{
  "selected_representations": ["paragraph", "hierarchical"],
  "paused_representations": ["sentence_window", "sliding_window", "semantic"],
  "selection_reasons": {
    "paragraph": ["set_cover: R1, R2"],
    "hierarchical": ["exclusive context: department scope"]
  },
  "uncovered_probe_obligations": [],
  "probe_depth": 3
}
```

---

## 13. Phase B: adaptive retrieval with no fixed final k

### 13.1 Candidate streams

Every selected representation receives runtime state:

```python
runtime_state = RepresentationRuntimeState(
    representation_id=rep_id,
    selected=True,
    active=True,
    next_rank=1,
    consecutive_empty_batches=0,
    accepted_candidate_ids=[],
)
```

Probe candidates from selected representations are reused as the first candidate batch. They are not automatically accepted.

After the probe candidates are processed:

```python
next_rank = probe_depth + 1
```

### 13.2 Adaptive loop

```python
def adaptive_retrieve(ctx: QueryContext) -> AccumulatorState:
    state = initialize_accumulator(ctx)

    process_selected_probe_candidates(ctx, state)

    while True:
        decision = evaluate_stopping(ctx, state)
        if decision.stop:
            state.stop_reason = decision.reason
            break

        active = choose_active_representations(state)
        if not active:
            if can_reactivate_unselected(ctx, state):
                reactivate_best_unselected(ctx, state)
                active = choose_active_representations(state)
            else:
                state.stop_reason = "unresolved_no_active_representation"
                break

        state.retrieval_round += 1
        for rep_id in active:
            batch = fetch_next_batch(ctx, state, rep_id)
            accepted_before = len(state.accepted_candidate_ids)
            process_batch(ctx, state, batch)
            update_representation_activity(
                state,
                rep_id,
                accepted_delta=len(state.accepted_candidate_ids) - accepted_before,
            )

        if all_critical_resolved(state):
            run_confirmation_round_if_needed(ctx, state)

    return state
```

### 13.3 Fetching the next batch

```python
def fetch_next_batch(ctx, state, rep_id):
    runtime = state.representation_states[rep_id]
    batch = ctx.indexes[rep_id].search(
        ctx.original_query_vector,
        start_rank=runtime.next_rank,
        limit=ctx.config.adaptive_batch_size,
    )
    runtime.next_rank += len(batch)
    return mark_stage(batch, "adaptive")
```

`adaptive_batch_size` controls the size of an index request. It does not set final `k`.

### 13.4 Candidate processing

```python
def process_candidate(ctx, state, candidate):
    state.candidates_examined += 1
    alignment = align_candidate_to_existing(candidate, state)

    judgment = get_or_create_judgment(ctx, candidate, alignment)
    validate_judgment_citations(judgment, candidate, ctx.canonical_store)

    contribution = marginal_contribution(judgment, state)
    if contribution.is_useful:
        state.provisional_candidate_ids.append(candidate.candidate_id)
        apply_findings_to_state(state, candidate, judgment)

    record_trace(candidate, alignment, judgment, contribution)
```

Candidates are provisional until final deduplication.

### 13.5 Marginal-contribution keys

A candidate is useful when it adds at least one previously unsecured key:

```text
REQ:<id>:SUPPORTED
REQ:<id>:REFUTED
REQ:<id>:NOT_APPLICABLE
APPLICABILITY:<field>:<value>
QUALIFIER:<requirement>:<type>:<normalized value>
CONTEXT:<requirement>:<missing context type>
CONFLICT:<requirement>:<resolution or disclosure>
INDEPENDENT_SOURCE:<requirement>:<lineage ID>
```

Semantic novelty without one of these keys is not enough for acceptance.

### 13.6 Representation activity update

A selected representation remains active when its latest batch:

- Adds a new obligation.
- Adds a valid new qualifier.
- Restores context.
- Advances a contradiction investigation.
- Finds a relevant independent source.

Pause rule:

```python
if batch_added_no_obligation and not representation_needed_for_active_state:
    consecutive_empty_batches += 1
else:
    consecutive_empty_batches = 0

if consecutive_empty_batches >= 1:
    active = False
    pause_reason = "latest_batch_added_no_value"
```

The primary method uses one empty batch to pause because the initial selection already used a probe. A sensitivity ablation can use patience two.

### 13.7 Reactivation

An unselected or paused representation can be activated when:

1. A critical requirement remains unresolved and all active direct views are paused.
2. A context-loss state identifies a representation capable of parent or adjacent restoration.
3. A reformulated requirement has strong probe similarity in that representation.
4. A genuine contradiction requires a separate source or document version.

Reactivation choice:

```text
Most matching unresolved obligation from stored profile
then higher RRF contribution
then lower probe token cost
```

If stored profiles provide no signal, reactivate all unselected representations for one batch and rebuild profiles.

### 13.8 Requirement reformulation

After all representations reach an empty batch for an unresolved requirement, reformulate that requirement once:

```python
reformulated = reformulator.generate(
    original_query=ctx.query.original_query,
    unresolved_requirement=requirement.text,
    known_query_facts=ctx.query.query_facts,
    prohibited_content="proposed answer",
)
```

The new query searches active and reactivated representations in adaptive batches. Results are tagged with the requirement ID and do not replace the original query globally.

---

## 14. Evidence sufficiency and stopping

### 14.1 Requirement resolution

A critical requirement is resolved only when the accumulator contains valid cited evidence for one of:

```text
SUPPORTED
REFUTED
NOT_APPLICABLE
```

Conflicting supported and refuted findings keep the requirement unresolved until the conflict is resolved or explicitly classified for disclosure.

### 14.2 Applicability completeness

Every accepted rule must have values for the applicability variables declared in its requirement. Unknown required scope variables prevent successful stopping.

### 14.3 Successful stopping

```python
def success_ready(state):
    return (
        all_critical_requirements_resolved(state)
        and all_required_applicability_complete(state)
        and no_unresolved_context_loss(state)
        and all_contradictions_resolved_or_disclosable(state)
        and state.confirmation_rounds_completed >= 1
        and state.confirmation_added_obligations == 0
    )
```

### 14.4 Confirmation round

After apparent sufficiency, request one additional batch from each still-active direct representation. Structural expansion occurs only if the current state still signals missing context.

If the confirmation batch adds a qualifier or changes applicability, update the state and continue retrieval. Otherwise stop successfully.

### 14.5 Safety stopping

Stop when any limit is reached:

```text
maximum candidates examined
maximum retrieval rounds
maximum judge calls
maximum elapsed retrieval time
maximum estimated monetary cost
```

Return `INSUFFICIENT_UNDER_BUDGET` when critical requirements remain unresolved.

### 14.6 Final k

No stopping function checks `accepted_count >= k`. Final `k` is calculated only after final deduplication:

```python
k_final = len(final_evidence)
```

---

## 15. Context-loss and contradiction handling

### 15.1 Context expansion service

```python
class ContextExpansionService:
    def expand(self, candidate, missing_context_type):
        match missing_context_type:
            case "heading_scope":
                return canonical_store.heading_path(candidate)
            case "containing_passage":
                return canonical_store.parent_passage(candidate)
            case "adjacent_units":
                return canonical_store.adjacent(candidate, before=1, after=1)
            case "table_header":
                return canonical_store.table_headers(candidate)
            case "footnote":
                return canonical_store.linked_footnotes(candidate)
            case "subsection":
                return canonical_store.subsection(candidate)
            case "version_metadata":
                return version_store.related_versions(candidate)
```

Expansion follows the smallest-to-largest order and stops as soon as interpretation becomes complete.

### 15.2 Applicability overlap

Two claims can contradict only when their applicability frames overlap on every query-relevant variable.

```python
def frames_can_contradict(a, b, required_fields):
    for field in required_fields:
        av, bv = set(getattr(a, field)), set(getattr(b, field))
        if av and bv and av.isdisjoint(bv):
            return False
    return True
```

An unknown value does not prove overlap; it triggers context expansion or leaves the contradiction unresolved.

### 15.3 Version resolution

Version precedence is applied only if:

```text
same document family
same applicable scope
explicit supersedes relationship
effective date covers the query time
```

Authority precedence is applied only through a corpus-specific configured ordering. Otherwise, retain both claims for disclosure.

---

## 16. Phase C: final post-retrieval deduplication

Adaptive retrieval finishes with a provisional evidence pool. Deduplication now removes repetition while preserving obligations and provenance.

### 16.1 Required candidate signature

```python
class EvidenceSignature(BaseModel):
    candidate_id: str
    document_id: str
    version_id: str
    canonical_unit_ids: frozenset[str]
    normalized_claim_keys: frozenset[str]
    requirement_keys: frozenset[str]
    qualifier_keys: frozenset[str]
    applicability_key: str
    source_lineage_id: str
    token_count: int
```

### 16.2 Stage 1: exact same-source duplicates

Candidates are exact duplicates when:

```text
same document version
same canonical-unit set
```

Keep one evidence item and attach every representation/rank provenance to it.

```python
exact_key = (version_id, tuple(sorted(canonical_unit_ids)))
```

### 16.3 Stage 2: containment dominance

Candidate `a` is dominated by `b` when:

```text
a units are contained in b units
a requirements are contained in b requirements
a qualifiers are contained in b qualifiers
a applicability frame equals b applicability frame
a source lineage equals b source lineage
removing a loses no obligation
```

Do not remove the longer candidate merely because it contains the shorter one. Prefer the candidate that preserves all required obligations at lower token cost.

### 16.4 Stage 3: semantic duplicate judgment

Only compare candidates that cover the same requirements and have compatible applicability frames.

They are semantic duplicates only if the judge returns:

```text
mutual entailment = true
same qualifiers = true
same modality = true
no unique condition = true
no unique source obligation = true
```

Different independent sources are retained when corroboration is an obligation, even if their claims are equivalent.

### 16.5 Deduplication algorithm

```python
def deduplicate(provisional, obligation_universe):
    exact_merged = merge_exact_same_source(provisional)

    retained = list(exact_merged)
    for a, b in candidate_pairs_same_lineage(retained):
        if containment_dominates(a, b, obligation_universe):
            retained.remove(a)
        elif containment_dominates(b, a, obligation_universe):
            retained.remove(b)

    for a, b in semantic_comparison_pairs(retained):
        if semantic_duplicates(a, b):
            remove_preferred_duplicate(retained, a, b, obligation_universe)

    assert obligations_covered(retained, obligation_universe)
    return retained
```

### 16.6 Invariants

After every removal:

- All resolved critical requirements remain covered.
- All necessary qualifiers remain present.
- All applicability facts remain present.
- Every disclosable contradiction remains represented.
- Required independent corroboration remains present.

If an invariant fails, undo the removal.

---

## 17. Final evidence compaction

Deduplication removes equivalent evidence. Compaction removes evidence that is useful but no longer necessary after the full pool is known.

### 17.1 Obligation universe

```python
obligations = (
    resolved_requirement_keys
    | required_qualifier_keys
    | required_applicability_keys
    | contradiction_disclosure_keys
    | required_independent_source_keys
)
```

### 17.2 Greedy cover

```python
while uncovered:
    best = min(
        candidates,
        key=lambda c: (
            -len(c.obligations & uncovered) / max(c.token_count, 1),
            authority_order(c),
            applicable_version_order(c),
            c.token_count,
            c.candidate_id,
        ),
    )
    selected.append(best)
    uncovered -= best.obligations
```

Authority and version tie-breaks are used only when explicitly declared and applicable.

### 17.3 Deletion pass

Iterate from longest selected item to shortest. Remove an item if all obligations remain covered without it.

The resulting evidence is locally irreducible. The implementation must not label it globally minimal.

---

## 18. Context construction and answer generation

### 18.1 Evidence order

```text
1. Direct applicable findings
2. Scope and applicability
3. Conditions, exceptions, thresholds, and negations
4. Complementary findings
5. Unresolved contradictory findings
6. Version, date, and authority metadata
```

### 18.2 Context item format

```text
[EVIDENCE E3]
Document: Employee Policy
Version: 2.0
Effective date: 2026-01-01
Section: Customer Support Operations
Canonical units: U4, U5
Requirements: R1, R2
Applicability: Customer Support; weekdays
Selection reason: direct rule plus emergency exception
Text:
Customer Support employees must work from the office on weekdays.
Remote work is allowed only during officially declared emergencies.
```

### 18.3 Generator contract

The generator returns:

```json
{
  "answerability": "PARTIALLY_ANSWERABLE",
  "answer": "...",
  "claim_citations": [
    {"claim": "...", "evidence_ids": ["E3"]}
  ],
  "unresolved_requirements": ["R3"],
  "disclosed_conflicts": []
}
```

### 18.4 Output validation

Validate that:

- Every cited evidence ID exists.
- Every factual claim has at least one citation.
- Every unresolved critical requirement is mentioned.
- The answerability label matches accumulator state.
- The answer does not cite representation vectors or generated summaries as source evidence.

Failed output is repaired once. A second failure returns an extractive evidence response with the answerability label.

---

## 19. End-to-end orchestrator

```python
def answer_query(query_text: str, config: SarRagConfig) -> AnswerResult:
    run = trace_store.start_run(query_text, config)

    # Query analysis
    query = requirement_extractor.analyze(query_text)
    trace_store.write(run, "query_analysis", query)

    # Phase A: diagnostic probe for representation selection
    probe = run_probe(query, enabled_indexes(config), config.probe_depth)
    probe_judgments = judge_probe_candidates(query, probe)
    groups = build_groups(probe)
    score_groups_with_rrf(groups, k0=config.rrf_k0)
    profiles = build_profiles(config.enabled_representations, groups, probe_judgments)
    selected, uncovered_probe = select_representations(
        profiles,
        obligation_universe_from_probe(probe_judgments),
    )

    if probe_has_shared_insufficiency(probe_judgments):
        selected = set(config.enabled_representations)

    trace_store.write(run, "probe_candidates", probe)
    trace_store.write(run, "evidence_groups", groups)
    trace_store.write(run, "representation_profiles", profiles)
    trace_store.write(run, "selected_representations", selected)

    # Phase B: no-fixed-k adaptive retrieval
    adaptive_context = AdaptiveContext(
        query=query,
        selected_representations=selected,
        probe_candidates=probe,
        probe_judgments=probe_judgments,
        profiles=profiles,
        config=config,
    )
    state = adaptive_retrieve(adaptive_context)
    trace_store.write(run, "adaptive_final_state", state)

    # Phase C: post-retrieval evidence processing
    provisional = load_provisional_candidates(state)
    obligations = final_obligation_universe(state)
    deduplicated = deduplicate(provisional, obligations)
    final_evidence = compact(deduplicated, obligations, config.max_final_context_tokens)

    assert obligations_covered(final_evidence, obligations)

    context = context_builder.build(query, state, final_evidence)
    answer = answer_generator.generate(query, state, context)
    validated = validate_or_repair_answer(answer, query, state, final_evidence)

    result = AnswerResult(
        query_id=query.query_id,
        answer=validated,
        final_k=len(final_evidence),
        selected_representations=selected,
        stop_reason=state.stop_reason,
        evidence=final_evidence,
    )
    trace_store.finish_run(run, result)
    return result
```

No line in this orchestrator assigns a fixed final `k`.

---

## 20. Practical execution trace

### 20.1 Source

```text
U1 heading: Remote Work Pilot - Engineering and Design
U2: Engineering and Design employees may work remotely on Fridays
    after 90 days and with manager approval.

U3 heading: Customer Support Operations
U4: Customer Support employees must work from the office on weekdays.
U5: Remote work is allowed only during officially declared emergencies.
```

### 20.2 Query

```text
Can Priya, a newly joined Customer Support employee,
work remotely this Friday?
```

### 20.3 Requirements

```text
R1: Normal weekday rule for Customer Support
R2: Customer Support remote-work exception
R3: Whether the emergency condition is established for this Friday
```

Query facts:

```text
Priya is in Customer Support
Priya is newly joined
The requested day is Friday
```

### 20.4 Top-3 probe

```text
sentence_window:
  rank 1 -> U2
  rank 2 -> U5
  rank 3 -> U4

sliding_window:
  rank 1 -> U1,U2
  rank 2 -> U3,U4,U5
  rank 3 -> unrelated

paragraph:
  rank 1 -> U4,U5
  rank 2 -> U2
  rank 3 -> unrelated

semantic:
  rank 1 -> U2
  rank 2 -> U4,U5
  rank 3 -> unrelated

hierarchical:
  rank 1 -> section U1,U2
  rank 2 -> section U3,U4,U5
  rank 3 -> unrelated section
```

### 20.5 Aligned groups

```text
G1 Engineering policy:
  sentence rank 1
  sliding rank 1
  paragraph rank 2
  semantic rank 1
  hierarchical rank 1

G2 Customer Support policy:
  sentence ranks 2 and 3
  sliding rank 2
  paragraph rank 1
  semantic rank 2
  hierarchical rank 2
```

Both groups receive strong RRF scores. RRF alone cannot choose the representation because G1 is strongly repeated but not applicable to Priya.

### 20.6 Probe judgments

```text
G1:
  Engineering Friday policy is NOT_APPLICABLE to Priya.
  Hierarchical view uniquely confirms department heading scope.

G2:
  R1 SUPPORTED by U4.
  R2 SUPPORTED by U5.
  R3 UNRESOLVED.
```

### 20.7 Profiles

```text
paragraph:
  covers R1 and R2 in one complete candidate
  high RRF contribution

hierarchical:
  supplies department applicability and heading scope
  exclusive context-restoration obligation

sentence_window:
  precise R1 and R2 evidence but no unique obligation

sliding_window:
  repeats paragraph evidence at higher token cost

semantic:
  repeats paragraph and Engineering evidence
```

### 20.8 Selected representations

```text
paragraph: selected by set cover for R1 and R2
hierarchical: mandatory because it restores exclusive department scope
sentence_window: paused after probe
sliding_window: paused after probe
semantic: paused after probe
```

If precise evidence for R3 is likely to be atomic, the runtime can reactivate `sentence_window` when R3 remains unresolved.

### 20.9 Adaptive retrieval

The selected probe candidates are processed first. They do not establish final `k`.

```text
paragraph probe candidates -> U4,U5 accepted provisionally
hierarchical heading U1 -> scope evidence accepted provisionally
```

R3 remains unresolved. The next paragraph batch returns ranks 4-6 and adds nothing. Paragraph pauses.

The selector reactivates sentence-window for precise emergency-declaration retrieval. Its next batch adds nothing. R3 is reformulated once:

```text
officially declared emergency Customer Support remote work Friday
```

No supporting declaration is found. The system stops with `PARTIALLY_ANSWERABLE`, not a guessed `No emergency` claim.

### 20.10 Final deduplication

The provisional pool contains:

```text
P1 -> U4,U5
H1 -> U1 scope heading
S3 -> U4
S2 -> U5
```

Exact and containment analysis removes S3 and S2 because P1 preserves both requirements and the emergency qualifier. H1 is retained only if needed to explain why the Engineering policy is not applicable.

Possible final evidence:

```text
E1 -> U4,U5
E2 -> U1,U2, only for non-applicability explanation
```

Therefore:

```text
k_final = 2
```

The value two was discovered after retrieval and deduplication. It was never configured.

### 20.11 Final answer

```text
Priya cannot normally work remotely this Friday. Customer Support
employees must work from the office on weekdays and may work remotely
only during an officially declared emergency. The available evidence
does not establish that an emergency has been declared. The 90-day and
manager-approval conditions belong to Engineering and Design and do not
apply to Priya.
```

---

## 21. Configuration schema

```yaml
run:
  seed: 42
  trace_dir: runs/
  fail_on_manifest_mismatch: true

models:
  embedding:
    name: BAAI/bge-m3
    revision: REQUIRED_EXACT_HASH
    normalize: true
  requirement_judge:
    name: Qwen/Qwen2.5-7B-Instruct
    revision: REQUIRED_EXACT_HASH
    temperature: 0
    top_p: 1
  answer_generator:
    name: Qwen/Qwen2.5-14B-Instruct
    revision: REQUIRED_EXACT_HASH
    temperature: 0
    top_p: 1

retrieval:
  probe_depth: 3
  adaptive_batch_size: 3
  rrf_k0: 60
  max_candidates_examined: 60
  max_retrieval_rounds: 8
  max_judge_calls: 70
  confirmation_rounds: 1
  max_reformulations_per_requirement: 1
  enabled_representations:
    - sentence_window
    - sliding_window
    - paragraph
    - semantic
    - hierarchical

selection:
  mandatory_exclusive_obligations: true
  set_cover: true
  rrf_as_tie_breaker: true
  token_cost_as_second_tie_breaker: true
  shared_insufficiency_select_all: true

stopping:
  require_all_critical_resolved: true
  require_applicability_complete: true
  require_context_loss_resolved: true
  allow_disclosable_contradiction: true
  max_final_context_tokens: 2000

deduplication:
  run_after_adaptive_retrieval: true
  exact_same_source: true
  containment: true
  semantic_mutual_entailment: true
  preserve_qualifiers: true
  preserve_applicability: true
  preserve_independent_sources: true
```

`REQUIRED_EXACT_HASH` must be replaced with resolved model revision hashes before any official experiment. The experiment launcher refuses placeholder hashes.

---

## 22. API design

### 22.1 Ingestion

```http
POST /v1/corpora/{corpus_id}/documents
POST /v1/corpora/{corpus_id}/build-representations
POST /v1/corpora/{corpus_id}/build-indexes
GET  /v1/corpora/{corpus_id}/manifest
```

### 22.2 Query

```http
POST /v1/query
```

Request:

```json
{
  "corpus_id": "policy-corpus-v1",
  "query": "Can Priya work remotely this Friday?",
  "config_id": "probe3-five-views"
}
```

Response:

```json
{
  "query_id": "...",
  "answerability": "PARTIALLY_ANSWERABLE",
  "answer": "...",
  "selected_representations": ["paragraph", "hierarchical"],
  "probe_depth": 3,
  "final_k": 2,
  "final_evidence_tokens": 94,
  "stop_reason": "unresolved_after_reformulation",
  "unresolved_requirements": ["R3"],
  "evidence": [...],
  "trace_id": "..."
}
```

### 22.3 Trace inspection

```http
GET /v1/traces/{trace_id}
```

The trace exposes probe candidates, groups, RRF scores, profiles, selection reasons, adaptive rounds, provisional evidence, deduplication decisions, final evidence, and answer validation.

---

## 23. Trace event schema

Every event is appended to JSON Lines:

```json
{
  "run_id": "...",
  "query_id": "...",
  "sequence": 17,
  "timestamp": "...",
  "event_type": "representation_paused",
  "payload": {
    "representation_id": "semantic",
    "reason": "latest_batch_added_no_value",
    "next_rank": 7
  }
}
```

Required event types:

```text
query_analyzed
probe_candidate_retrieved
probe_candidate_judged
evidence_group_created
rrf_calculated
representation_profile_created
representation_selected
representation_paused
representation_reactivated
adaptive_batch_retrieved
candidate_provisionally_accepted
candidate_rejected
context_expanded
requirement_state_changed
contradiction_detected
reformulation_created
stopping_evaluated
deduplication_removed
evidence_compacted
answer_generated
answer_validated
run_completed
```

---

## 24. Unit tests

### 24.1 Canonical source tests

- Stable unit IDs for unchanged documents.
- New version IDs when content changes.
- Correct table header attachment.
- Correct heading parent relationships.
- No cross-version alignment.

### 24.2 RRF tests

- One contribution per representation per group.
- Best rank used when a representation has multiple group members.
- Unique evidence receives one contribution without being deleted.
- Raw scores never affect cross-representation RRF.

### 24.3 Representation selection tests

- Exclusive qualifier representation is always selected.
- High-RRF redundant representation can be paused.
- Set cover selects the deterministic minimum under tie rules.
- Shared insufficiency selects all enabled representations for one batch.
- RRF breaks only equal-coverage ties.

### 24.4 Adaptive retrieval tests

- Probe candidates are reused and not re-retrieved.
- Next adaptive rank starts at `probe_depth + 1`.
- Batch size does not become final `k`.
- Empty batch pauses the representation.
- Unresolved requirement can reactivate a paused representation.
- No successful stop before confirmation round.
- Budget exhaustion produces `INSUFFICIENT_UNDER_BUDGET`.

### 24.5 Deduplication tests

- Final dedup runs after adaptive retrieval.
- Same-source exact duplicates merge provenance.
- Longer evidence survives when it adds a qualifier.
- Shorter evidence survives when the longer candidate adds only noise.
- Different document lineages remain when corroboration is required.
- Contradictory evidence is never deduplicated.
- Removal never loses an obligation.

### 24.6 Applicability tests

- Engineering conditions are not applied to Customer Support.
- Different effective periods are labeled different scope.
- Unknown scope does not become matching scope.
- Explicit supersession resolves versions correctly.
- No authority order is invented.

---

## 25. Integration tests

Create small deterministic corpora for these scenarios:

1. Simple definition requiring one evidence item.
2. Multi-part comparison requiring complementary representations.
3. Short sentence missing a section heading.
4. Same claim repeated across all representations.
5. Longer passage adding an exception.
6. Different departments with similar rules.
7. Superseded document version.
8. Genuine same-scope contradiction.
9. No answer in the corpus.
10. Answer found only after reactivating an unselected representation.

Every golden test asserts:

```text
selected representations
probe RRF groups
adaptive rank ranges requested
requirement state transitions
deduplication removals
final k
answerability
final citations
```

---

## 26. Experiment runner

The authoritative dataset bundle and dataset-specific protocols are defined in `DATASET_SELECTION_DECISION.md`. The core tracks are ConditionalQA, QASPER, MuSiQue-Answerable/MuSiQue-Full, and the custom SAR-RAG Representation-State Benchmark. HotpotQA Distractor is optional rather than core.

Each experiment config is immutable and hashed.

```python
def run_experiment(config_path, dataset_path):
    config = load_and_validate_config(config_path)
    assert_no_placeholders(config)
    run_manifest = create_run_manifest(config, dataset_path)

    for item in load_locked_dataset(dataset_path):
        result = answer_query(item.query, config)
        evaluator.score(item, result)

    aggregate_metrics(run_manifest)
    paired_bootstrap(run_manifest, resamples=10_000)
```

Required primary configurations:

```text
probe3_five_views
probe5_five_views
probe3_three_views
fixed_top3_each_view
fixed_top5_each_view
deep_all_rrf
full_sar_rag
```

All comparisons store:

- Candidates retrieved.
- Candidates judged.
- Adaptive rounds.
- Selected representations.
- Final `k` distribution.
- Final context tokens.
- Retrieval, judgment, and generation latency.
- Estimated cost.
- Evidence and answer metrics.

---

## 27. Implementation sequence

Build in this order:

### Milestone 1: canonical corpus

1. Document registry and version model.
2. Text/Markdown/HTML parser.
3. Canonical unit generator.
4. Validation and manifest.

Exit condition: canonical units reconstruct source text and pass lineage tests.

### Milestone 2: representations and exact indexes

1. Sentence-window builder.
2. Sliding-window builder.
3. Paragraph builder.
4. Semantic builder.
5. Hierarchical builder.
6. Embedding cache and FAISS indexes.

Exit condition: every chunk maps to valid canonical units and all indexes share a manifest.

### Milestone 3: query and probe selection

1. Requirement extraction with fallback.
2. Top-p probe retrieval.
3. Probe judgment.
4. Source grouping.
5. RRF.
6. Profiles and deterministic representation selector.

Exit condition: golden tests select unique and complementary representations correctly.

### Milestone 4: adaptive no-fixed-k retrieval

1. Runtime representation states.
2. Incremental rank-range retrieval.
3. Evidence accumulator.
4. Pause/reactivate logic.
5. Context expansion.
6. Reformulation.
7. Stopping and budgets.

Exit condition: no code path uses a fixed final evidence `k`.

### Milestone 5: final evidence and answer

1. Post-retrieval deduplication.
2. Obligation-preserving compaction.
3. Context builder.
4. Answer generator and validator.
5. Complete trace.

Exit condition: qualifier and applicability tests pass and final `k` is derived.

### Milestone 6: evaluation

1. Dataset adapters.
2. Retrieval and evidence metrics.
3. State and applicability metrics.
4. Answer and citation metrics.
5. Efficiency measurement.
6. Bootstrap and paired tests.

---

## 28. Critical implementation invariants

These must be enforced in code and tests:

1. Probe depth is never reported as final `k`.
2. Adaptive batch size is never reported as final `k`.
3. Probe candidates are not automatically accepted as evidence.
4. Raw scores are never directly compared across representations.
5. RRF is calculated only after source alignment.
6. RRF is not the sole representation-selection rule.
7. Unique applicable evidence can keep a low-RRF representation selected.
8. Source alignment before selection does not delete candidates.
9. Final semantic and containment deduplication occurs after adaptive retrieval.
10. Duplicate removal never removes a unique qualifier or applicability fact.
11. Two representations of the same source are not independent corroboration.
12. Unknown applicability never becomes applicable by default.
13. Absence of evidence never becomes refutation.
14. A newer document does not win without explicit version logic.
15. A successful stop requires an empty confirmation round.
16. Budget exhaustion is not successful evidence sufficiency.
17. Final `k` is calculated from final evidence after deduplication and compaction.

---

## 29. Final implementation statement

The implementation has three separate online responsibilities:

1. **Probe and select representations.** Top-3 or Top-5 from every view is used only to observe representation behavior. Source-aligned RRF measures agreement and supplies a tie-breaker, while unique requirement, qualifier, applicability, and context contributions protect complementary views.

2. **Retrieve adaptively from selected indexes.** Selected indexes are consumed as ranked streams in small computational batches. There is no fixed final `k`. Views pause, reactivate, or receive requirement reformulations according to observable evidence state.

3. **Deduplicate after retrieval.** The complete provisional evidence pool is deduplicated only after adaptive stopping. Exact, contained, and semantic duplicates are removed only when all requirements, qualifiers, applicability facts, contradictions, and required independent sources remain represented.

This corrected design preserves the original research idea while making representation selection, adaptive retrieval, stopping, and final evidence construction executable and reproducible.

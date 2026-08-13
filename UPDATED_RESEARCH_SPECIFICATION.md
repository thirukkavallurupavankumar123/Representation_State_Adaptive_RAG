# Source-Aligned Representation-State Adaptive RAG

> **Superseded primary online algorithm (4 August 2026):** The authoritative primary
> method is now `ONE_CALL_RESEARCH_PIPELINE.md`. This older document describes the
> per-candidate LLM-judged controller, which is retained only as an ablation.

> **Implementation correction:** The authoritative implementation sequence is now defined in `DETAILED_IMPLEMENTATION_BLUEPRINT.md`. Probe Top-3/Top-5 is used for representation selection, selected indexes are then retrieved adaptively without a fixed final `k`, and final qualifier-preserving deduplication runs after adaptive retrieval. Where this earlier specification differs, the detailed implementation blueprint takes precedence.

## A Task-Training-Free Framework for Multi-Granularity Retrieval, Scope-Safe Evidence Accumulation, and Abstaining Question Answering

**Specification status:** implementation-ready research protocol  
**Version:** 1.0  
**Date:** 2 August 2026

---

## 1. Final research definition

Source-Aligned Representation-State Adaptive RAG (SAR-RAG) is a document-grounded question-answering framework that indexes the same source corpus at three complementary granularities, probes all three views at a small fixed depth, aligns retrieved material to shared canonical source units, and selectively examines additional candidates only where evidence obligations remain unresolved.

The method does not train a query router, retrieval classifier, confidence model, fusion model, or task-specific language model. It uses frozen pretrained embedding and instruction models. For that reason, the precise claim is **task-training-free**, not model-free and not learning-free.

The final context is not claimed to be the globally smallest possible context. The method constructs a **compact, sufficient, locally irreducible evidence set under declared candidate, token, and retrieval-round budgets**. “Locally irreducible” means that no selected evidence item can be removed without losing a resolved requirement, a necessary qualifier, an applicability decision, or a contradiction that must be disclosed.

The central contribution is not the number of chunking methods. It is the combination of:

1. Canonical source lineage shared by heterogeneous retrieval granularities.
2. Query-specific diagnosis of direct evidence, complementarity, redundancy, context loss, source contradiction, and insufficiency.
3. Explicit tracking of evidence requirements and their applicability.
4. Selective deeper candidate examination without a trained router.
5. Scope-, condition-, version-, and qualifier-preserving evidence reduction.
6. A reproducible abstention policy when the corpus does not support an answer.

---

## 2. Research questions and hypotheses

### 2.1 Primary research question

Can shallow probing of source-aligned atomic, passage, and structural document representations identify which retrieval views should remain active and produce more complete, less redundant, and more scope-correct evidence than fixed-representation and fixed-Top-k RAG under equal resource budgets?

### 2.2 Secondary research questions

1. Can canonical source alignment distinguish a short chunk that lost a condition or heading from a genuinely contradictory source?
2. Does explicit applicability tracking reduce the transfer of conditions between unrelated entities, departments, sections, dates, or policy versions?
3. Does adaptive evidence accumulation improve evidence coverage per 1,000 context tokens?
4. When the corpus is insufficient, does the method abstain more reliably than fixed-Top-k systems?
5. Are all three representation families necessary, and does each contribute unique evidence after cost is considered?

### 2.3 Pre-registered hypotheses

- **H1:** SAR-RAG has higher gold evidence F1 than the strongest cost-matched static RAG baseline.
- **H2:** SAR-RAG has higher condition-and-scope preservation accuracy than cost-matched RRF over all representations.
- **H3:** SAR-RAG uses fewer final context tokens than deep search-and-concatenate while achieving non-inferior answer correctness.
- **H4:** Canonical source alignment reduces false contradiction classifications relative to text-only contradiction detection.
- **H5:** Applicability tracking reduces cross-scope qualifier leakage.
- **H6:** At least two representation families make statistically significant unique contributions, while the value of the third is an empirical question rather than an assumption.

---

## 3. Fixed design decisions

The primary system uses exactly three representation families:

1. **Atomic contextualized units** for precise facts and definitions.
2. **Structure-preserving passages** for conditions, explanations, and local multi-sentence evidence.
3. **Hierarchical structural nodes** for headings, scope, parent context, and long-range organization.

Sliding-window and semantic-breakpoint chunking are not part of the primary system. They are retained as baselines and optional ablations. This avoids hard-coding five substantially overlapping indexes before their value has been demonstrated.

The primary retrieval experiment uses:

- Frozen embedding model: `BAAI/bge-m3`.
- Similarity: cosine similarity over L2-normalized embeddings.
- Experimental index: exact inner-product search using FAISS `IndexFlatIP`, eliminating approximate-nearest-neighbor recall as a confound.
- Probe depth: five candidates per representation family.
- Maximum search depth: twenty ranked candidates per direct-evidence representation.
- Cross-representation fusion: rank-only Reciprocal Rank Fusion with `k0 = 60`.
- Raw-score calibration: none.
- Robust z-score normalization: removed from the primary method.
- Score-knee filtering: removed from the primary method and evaluated only as an ablation.
- Task-specific model training: none.

Every model repository name, model revision hash, tokenizer revision, library version, prompt, and decoding parameter must be written to the experiment manifest. If a named model becomes unavailable, the replacement constitutes a new experimental condition and must not be mixed into the original results.

---

## 4. Terminology and operational definitions

### 4.1 Canonical source unit

The smallest addressable piece of the original document that can be cited without changing its meaning. A unit may be:

- A sentence.
- A list item.
- A table row paired with its column headers.
- A figure or table caption.
- A footnote.
- An equation paired with its defining text.
- A code block or structurally complete code element.
- A form field and its label.

Canonical units are not restricted to sentences.

### 4.2 Evidence requirement

An independently verifiable obligation that must be resolved to answer the user’s explicit question. Requirements do not replace the original query.

### 4.3 Applicability variable

A fact needed to determine whether a rule or claim applies, such as entity, department, jurisdiction, product version, date, employment tenure, document version, or declared exception.

### 4.4 Qualifier

A condition that changes the validity or scope of a claim, including negation, exception, threshold, date, authority, population, location, version, prerequisite, or modality such as “may” versus “must.”

### 4.5 Resolved requirement

A requirement is resolved only when it is labeled one of:

- `SUPPORTED`: sufficient evidence supports it.
- `REFUTED`: sufficient evidence supports its negation.
- `NOT_APPLICABLE`: evidence shows that it does not apply to the queried case.

`UNRESOLVED` is not a resolved state. Absence of evidence is never treated as refutation.

### 4.6 Context-loss disagreement

Two retrieved items appear inconsistent, but the apparent disagreement disappears when the shorter item is restored to its canonical parent, adjacent source units, heading path, table header, footnote, or applicable version.

### 4.7 Source-level contradiction

Two complete, applicable source contexts assert logically incompatible claims about the same entity, relation, time, jurisdiction, and version scope.

### 4.8 Sufficient evidence set

An evidence set is sufficient when every critical requirement is resolved, every accepted claim has an applicable scope, all detected context-loss disagreements are resolved, and all genuine contradictions are either resolved by authority/version rules or retained for disclosure.

### 4.9 Independent corroboration

Evidence is independently corroborating only when it comes from a distinct provenance lineage. Different chunks or representations of the same canonical source are not independent. Copies or derivatives of the same upstream document are also not independent.

---

## 5. Corpus and document model

### 5.1 Supported primary document types

The primary experiment supports digitally extractable HTML, Markdown, plain text, and tagged PDF documents. Scanned PDFs and visually complex documents may be included only if layout extraction quality is manually verified. Fully multimodal retrieval is outside the primary claim and must be reported as future work rather than silently treated as solved.

### 5.2 Document identity

Every document record contains:

```text
document_id
source_uri
publisher_or_author
document_title
document_version
effective_date
retrieval_date
authority_level
content_hash
language
license
parser_version
```

Documents with different versions receive different `document_version` values and retain explicit supersession links.

### 5.3 Canonical unit identity

Every canonical unit contains:

```text
unit_id
document_id
document_version
unit_type
section_path
page_number_or_html_anchor
character_start
character_end
ordinal_within_parent
parent_id
previous_unit_id
next_unit_id
raw_text
normalized_text_hash
```

The canonical ID is immutable within a fixed corpus release. Any parser change that changes unit boundaries creates a new corpus release.

### 5.4 Parsing validation

Before indexing, the following checks must pass:

1. Every non-empty extracted text span belongs to at least one canonical unit.
2. Character offsets reconstruct the original extracted text.
3. Parent-child links are acyclic.
4. Section paths are preserved.
5. Table rows include the correct headers.
6. Footnotes link to their markers when detectable.
7. Document and unit hashes are reproducible.

A random 10% document sample, with at least 30 documents, is manually inspected. Parsing accuracy is reported rather than assumed.

---

## 6. Representation construction

All token counts use the frozen `BAAI/bge-m3` tokenizer.

### 6.1 Representation A: atomic contextualized units

Each canonical unit is independently retrievable. Its embedding input is:

```text
[DOCUMENT TITLE]
[SECTION PATH]
[UNIT TYPE]
[RAW CANONICAL UNIT TEXT]
```

The title and section path provide retrieval context but are not presented as quoted source evidence unless they are themselves canonical heading units. The evidence payload retains the original unit and its source lineage.

If a unit exceeds 256 embedding tokens, it is split only at internal structural boundaries. Each fragment retains the same parent canonical unit ID plus a fragment ordinal. Fragments are recombined before final citation.

Purpose:

- Precise facts.
- Definitions.
- Atomic conditions.
- Low-token evidence.

Expected weakness:

- Distant qualifiers may be absent.
- A heading prefix can improve retrieval without fully proving applicability.

### 6.2 Representation B: structure-preserving passages

Passages are constructed deterministically from adjacent canonical units within the same structural parent.

Rules:

1. A natural paragraph, list, table, code block, or caption group is the initial block.
2. Adjacent blocks shorter than 96 tokens are merged while the result remains at most 384 tokens.
3. Blocks between 96 and 384 tokens remain unchanged.
4. Blocks longer than 384 tokens are split at canonical-unit boundaries into the largest prefixes not exceeding 384 tokens.
5. No overlap is used in the primary passage representation.
6. Units from different subsection parents are never merged.
7. A passage embedding input contains document title, section path, and passage text.

Purpose:

- Conditions and exceptions.
- Local explanations.
- Procedures and arguments.
- Evidence distributed over adjacent units.

### 6.3 Representation C: hierarchical structural nodes

The document tree contains document, section, subsection, and block parents. A structural node is indexed when it has at least two descendant passages.

Its vector is constructed without generative summarization:

1. L2-normalize every direct descendant passage embedding.
2. Calculate their arithmetic mean.
3. Embed the node’s heading path separately.
4. Calculate the arithmetic mean of the descendant centroid and heading embedding.
5. L2-normalize the result.

The structural node stores no invented summary. Retrieval of a structural node triggers inspection of its heading, applicable metadata, and query-ranked descendant passages. It is not treated as direct answer evidence unless cited canonical descendants are selected.

Purpose:

- Recover section scope.
- Identify applicable parent policies.
- Navigate long documents.
- Restore headings and structural qualifiers.

### 6.4 Representations excluded from the primary system

- Fixed sliding windows are evaluated as a baseline because they provide uniform coverage but create boundary-crossing duplication.
- Semantic-breakpoint chunks are evaluated as an ablation because their boundaries depend on an additional algorithm and may overlap conceptually with paragraph passages.
- Generative hierarchical summaries are excluded because they introduce unsupported-summary and model-dependence risks.

---

## 7. Query analysis

### 7.1 Original query preservation

The exact original query is used for the initial retrieval in all representations. Requirement extraction never replaces it.

### 7.2 Requirement extraction model

The primary open, frozen judge is `Qwen/Qwen2.5-7B-Instruct`, run with:

```text
temperature = 0
top_p = 1
max_new_tokens = 768
response format = JSON
```

Its exact model revision is frozen in the experiment manifest. A second judge-model condition is run as a robustness experiment; it is not used to choose favorable primary results.

### 7.3 Requirement schema

```json
{
  "original_query": "...",
  "requirements": [
    {
      "requirement_id": "R1",
      "text": "...",
      "critical": true,
      "entities": ["..."],
      "applicability_variables": ["department", "date"]
    }
  ],
  "query_facts": [
    {
      "fact": "Priya is a Customer Support employee",
      "source": "explicitly stated in query"
    }
  ],
  "decomposition_confident": true
}
```

The extractor returns at most eight requirements. Query facts are kept separate from corpus-supported requirements. It must not invent missing facts.

### 7.4 Requirement validation

The following deterministic validator is applied:

1. IDs must be unique.
2. Requirement count must be between one and eight.
3. Every named entity must occur in the query or be a normalized coreference of a query entity.
4. Every requirement must be traceable to an explicit interrogative, comparison dimension, condition, or requested output in the query.
5. Duplicate requirements are merged using normalized text identity followed by a judge check.

If JSON validation fails, the model is retried once with the validation errors. If it fails again or returns `decomposition_confident = false`, the system uses a single requirement containing the complete original query.

### 7.5 Requirement prompt

```text
Extract the smallest set of independently verifiable evidence requirements needed
to answer the query. Do not answer the query. Do not add facts or requirements that
are not requested. Separate facts explicitly supplied by the query from facts that
must be supported by the corpus. Identify variables needed to decide whether a rule
or claim applies. Return only JSON matching the supplied schema. Use one requirement
for a simple question and no more than eight requirements.
```

---

## 8. Retrieval and rank fusion

### 8.1 Probe retrieval

For every query, exact cosine search retrieves:

```text
Atomic representation: top 5
Passage representation: top 5
Structural representation: top 5
```

The probe therefore exposes at most fifteen candidates. Only these five per view are evaluated initially. Ranks 6–20 are not retrieved until that view remains active.

### 8.2 Calibration decision

The primary system performs **no raw similarity-score calibration**.

Reasons:

- Top-list robust z-scores are based on a truncated and query-conditional distribution.
- MAD may be zero or unstable.
- A z-score is not a relevance probability.
- Different representation score scales need not be comparable when rank fusion is used.

Within a representation, raw cosine similarity determines rank only. Across atomic and passage representations, ranks are combined using Reciprocal Rank Fusion:

\[
RRF(g)=\sum_{r\in\{atomic,passage\}}\frac{1}{60+rank_r(g)}
\]

Here, `g` is a source-aligned overlap group. The structural representation is not fused as direct answer evidence; it supplies parent context and descendant expansion.

If an overlap group contains more than one candidate from the same representation, `rank_r(g)` is the best (lowest) rank of any member from that representation. A representation contributes at most once to a group, preventing overlapping chunks from one index from artificially increasing the fusion score.

The fixed value 60 follows the conventional stability setting and is not tuned on the test set. Sensitivity values 10, 30, and 100 are evaluated on the validation split and reported separately; the pre-registered primary result remains 60.

### 8.3 Source-aligned overlap groups

Each direct candidate maps to its canonical-unit set `U(c)`. Two candidates enter the same overlap group only when they belong to the same document version and meet at least one condition:

1. Their canonical-unit sets are identical.
2. One canonical-unit set contains the other.
3. Their canonical-unit Jaccard overlap is at least 0.5.

Connected components are not used because transitive overlap can incorrectly merge a long chain of different evidence. Grouping is anchored to the highest-ranked candidate: each later candidate joins the first compatible existing anchor; otherwise it starts a new group.

The group retains every member’s text, representation, rank, and unit set. RRF ranks groups for examination but does not itself establish relevance.

### 8.4 Exact duplicate removal

Before judging, candidates with identical `(document_version, canonical_unit_set)` are combined into one candidate record with multiple retrieval provenances. Text-hash duplication is recorded but does not merge evidence from different documents, because independent sources must remain distinguishable.

Containment and semantic similarity do not cause removal at this stage; the longer candidate may contain a critical qualifier.

---

## 9. Evidence judgment

### 9.1 Candidate judgment input

For each candidate, the judge receives:

- Original query.
- Extracted requirements.
- Query facts.
- Candidate canonical text.
- Section path.
- Immediate parent heading.
- Document version and effective date.
- Adjacent canonical units only when the candidate is being evaluated for context restoration.

The judge is prohibited from using parametric knowledge as evidence. Every support, refutation, qualifier, and applicability decision must cite canonical unit IDs.

### 9.2 Candidate judgment output

```json
{
  "candidate_id": "C1",
  "directly_answer_bearing": true,
  "requirement_findings": [
    {
      "requirement_id": "R1",
      "status": "SUPPORTED",
      "cited_unit_ids": ["D1-V2-S4-P2-U3"],
      "claim": "...",
      "applies_to": {
        "entity": "...",
        "time": "...",
        "jurisdiction": "...",
        "version": "..."
      },
      "qualifiers": [
        {
          "type": "exception",
          "text": "only during officially declared emergencies",
          "cited_unit_ids": ["..."]
        }
      ]
    }
  ],
  "missing_context_signals": ["heading_scope"],
  "context_restoration_needed": true
}
```

Allowed requirement statuses at candidate level are `SUPPORTED`, `REFUTED`, `NOT_APPLICABLE`, and `UNRESOLVED`.

### 9.3 Deterministic validation of judge output

1. Every cited unit ID must exist in the supplied candidate/context payload.
2. At least one citation is required for any non-`UNRESOLVED` status.
3. Qualifier text must be traceable to cited source text.
4. The document version in `applies_to` must match supplied metadata.
5. Invalid output is retried once; a second failure makes the candidate unresolved.

The system does not convert judge confidence into a probability and does not accept unsupported confidence values.

---

## 10. Role and state diagnosis

Roles are assigned first to candidates and then aggregated descriptively by representation. A representation label alone never overrides candidate-level evidence.

### 10.1 Candidate roles

- **Direct:** resolves at least one currently unresolved critical requirement with cited evidence.
- **Complementary:** resolves a different requirement from the current highest-ranked direct evidence.
- **Context-restoring:** supplies a heading, parent scope, exception, date, version, or adjacent unit that changes or validates interpretation of another candidate.
- **Corroborating:** makes the same applicable claim from an independent provenance lineage.
- **Redundant:** adds no unresolved requirement, qualifier, applicability fact, contradiction information, or independent provenance.
- **Potentially misleading alone:** appears answer-bearing but is missing context that changes its applicability or meaning.
- **Insufficient:** resolves no requirement and restores no relevant context.

### 10.2 Active representation rule

A direct-evidence representation remains active for deeper retrieval if any of the following is true:

1. It resolved at least one requirement in the latest batch and critical requirements remain unresolved.
2. Its latest batch added a unique applicable qualifier.
3. Its latest batch exposed a context-loss or source-contradiction state requiring more evidence.
4. It contains an independently sourced document relevant to an unresolved contradiction.

It is paused if its latest batch contains no accepted candidate and it is not needed for an active context or contradiction investigation.

The structural representation is active only when parent context, version, authority, heading scope, or descendant navigation is required.

### 10.3 State definitions

- **Agreement:** applicable claims resolve the same requirement consistently with matching qualifiers.
- **Complementarity:** candidates resolve different requirements without incompatibility.
- **Redundancy:** candidates add the same claim and qualifiers from the same provenance lineage.
- **Context-loss disagreement:** incompatibility disappears after canonical parent/adjacent restoration.
- **Source-level contradiction:** complete applicable contexts remain logically incompatible.
- **Shared insufficiency:** no examined representation resolves a critical requirement.

States are multi-label. For example, a query may involve complementarity for one requirement and contradiction for another.

---

## 11. Context-loss and contradiction procedure

Potential disagreement is evaluated only between findings attached to the same requirement or the same normalized subject-relation-object claim.

### 11.1 Restoration sequence

For an apparent disagreement, inspect in this order:

1. Heading path.
2. Complete containing passage.
3. Previous and next canonical units.
4. Table header, footnote, caption, or list parent.
5. Subsection parent.
6. Document version, effective date, and authority metadata.

Expansion stops as soon as the necessary scope is recovered or the full subsection has been examined.

### 11.2 Applicability matching

Two claims can contradict only if their applicability frames overlap on all variables relevant to the query:

```text
entity or population
relation or policy
time or effective period
location or jurisdiction
document version
exception state
```

If frames do not overlap, label the pair `DIFFERENT_SCOPE`, not contradiction.

### 11.3 Version and authority handling

- A newer version supersedes an older version only when the document metadata explicitly establishes succession and both address the same scope.
- A higher-authority source controls only when the evaluation corpus supplies a declared authority ordering.
- The system never invents an authority hierarchy.
- If incompatible complete sources remain and no declared resolution rule applies, retain both and require disclosure in the answer.

---

## 12. Selective deeper retrieval

### 12.1 Retrieval rounds

```text
Round 1: ranks 1–5 from all three representations
Round 2: ranks 6–10 from active direct-evidence representations
Round 3: ranks 11–15 from active direct-evidence representations
Round 4: ranks 16–20 from active direct-evidence representations
```

Structural retrieval after Round 1 is expansion-based: for each active structural node, examine its heading metadata and the three highest-ranked descendant passages for the unresolved requirement. At most two structural nodes are expanded per requirement per round.

### 12.2 Unresolved-requirement reformulation

After Round 4, each unresolved critical requirement may generate exactly one retrieval reformulation. The reformulation prompt receives the original query and the unresolved requirement, and returns a short search query containing no proposed answer.

The reformulated query retrieves Top-5 from atomic and passage representations and Top-3 structural nodes. These results count toward all candidate and token budgets.

If the requirement remains unresolved, the final answer abstains from that part rather than generating a likely answer.

### 12.3 Candidate examination priority

Within each new batch:

1. Context-restoration candidates.
2. Candidates retrieved for unresolved requirements.
3. Candidates relevant to an active source contradiction.
4. Remaining source-aligned groups by RRF.
5. Shorter candidate first when two candidates have the same provenance, claims, qualifiers, and applicability frame.

---

## 13. Adaptive evidence accumulation

### 13.1 Acceptance rule

A candidate is accepted only if it has valid source citations and adds at least one item not already secured by accepted evidence:

- Resolves a critical requirement.
- Establishes that a requirement is not applicable.
- Adds a query-relevant qualifier.
- Establishes an applicability variable.
- Restores context that changes or validates interpretation.
- Resolves or documents a genuine contradiction.
- Provides independent corroboration required by the task or evaluation protocol.

### 13.2 Rejection rule

A candidate is rejected from the final set when:

- It resolves nothing.
- It is an exact or same-provenance semantic duplicate with no unique qualifier.
- It belongs to a different, irrelevant applicability scope.
- Its judge citations are invalid.
- It exceeds the remaining token budget and no shorter sufficient substitute can be formed.

An irrelevant-scope candidate may still be retained temporarily if it is necessary to demonstrate why another candidate was misleading, but it is removed from final answer evidence unless that distinction must be explained.

### 13.3 Global evidence state

For each requirement, the accumulator maintains:

```text
status
supporting candidate IDs
refuting candidate IDs
applicability frame
qualifiers
unresolved conflicts
source lineages
```

A requirement is never marked resolved solely because a candidate has high embedding similarity.

### 13.4 Budgets

Primary experimental budgets are fixed before test evaluation:

```text
maximum direct candidates judged: 50
maximum structural descendants judged: 18
maximum retrieval rounds: 4 plus one reformulation round
maximum final evidence context: 2,000 tokens
maximum requirements: 8
maximum reformulations: 1 per unresolved critical requirement
```

If a budget is exhausted before sufficiency, the outcome is `INSUFFICIENT_UNDER_BUDGET`, not success.

### 13.5 Successful stopping rule

Stop successfully only when:

1. Every critical requirement is `SUPPORTED`, `REFUTED`, or `NOT_APPLICABLE`.
2. Every accepted finding has a compatible applicability frame.
3. Every context-loss disagreement is resolved.
4. Every source contradiction is resolved or flagged for disclosure.
5. One confirmation batch from every still-active direct representation yields no newly accepted evidence.

The confirmation-batch condition prevents immediate stopping after the first apparently complete set.

### 13.6 Unsuccessful stopping rule

Stop without full answerability when:

- The candidate, round, or token budget is reached.
- All representations are paused while critical requirements remain unresolved.
- Reformulation fails to resolve the remaining requirements.
- Parsing or provenance validation makes the relevant source unreliable.

The answer generator receives the exact unresolved requirements and must explicitly abstain from them.

---

## 14. Final evidence reduction

The accepted set may still contain removable evidence. Final reduction has two stages.

### 14.1 Obligation universe

Construct a set containing:

- Every resolved critical requirement.
- Every qualifier required for correct interpretation.
- Every applicability fact required for a selected rule.
- Every unresolved contradiction that must be disclosed.
- Every explicitly requested independent corroboration obligation.

### 14.2 Greedy compacting and deletion pass

1. Select the candidate covering the largest number of uncovered obligations per evidence token.
2. Break ties by: direct source authority declared in metadata, applicable newer version, more precise canonical citation, then fewer tokens.
3. Repeat until all obligations are covered.
4. Iterate through the selected set from longest to shortest and remove any candidate whose removal leaves every obligation covered.

The result is locally irreducible, not guaranteed globally minimum. Report:

```text
final evidence item count
final canonical unit count
final token count
requirements covered
qualifiers preserved
```

The primary adaptive size measure is final evidence tokens and canonical units, not chunk count, because chunk counts are incomparable across granularities.

---

## 15. Final context and answer generation

### 15.1 Context order

```text
1. Direct applicable evidence
2. Applicability and scope evidence
3. Conditions, exceptions, and negations
4. Complementary requirements
5. Unresolved contradictory evidence
6. Version, date, and authority metadata
```

Each evidence item includes document title, version, date, section path, canonical unit IDs, representation provenance, requirements covered, and selection reason.

### 15.2 Generator

The primary answer generator is `Qwen/Qwen2.5-14B-Instruct`, frozen at a recorded revision, with:

```text
temperature = 0
top_p = 1
max_new_tokens = 1,024
```

The generator is distinct from the primary evidence judge to reduce same-model self-confirmation. A same-model condition is reported as a cost ablation.

### 15.3 Generation rules

```text
Answer only from the supplied evidence.
Apply every claim only within its recorded applicability frame.
Preserve negations, conditions, exceptions, dates, versions, and modality.
Do not transfer a condition from one policy, entity, section, or version to another.
Cite the canonical evidence supporting each factual claim.
Disclose unresolved source contradictions.
For every unresolved critical requirement, explicitly state that the supplied corpus
does not establish the answer. Do not fill the gap from model knowledge.
```

### 15.4 Answerability labels

- `ANSWERABLE`: all critical requirements resolved without undisclosed contradiction.
- `ANSWERABLE_WITH_CONFLICT`: requirements have evidence, but complete applicable sources conflict.
- `PARTIALLY_ANSWERABLE`: some critical requirements resolved and others unresolved.
- `INSUFFICIENT`: no safe answer to the main request.
- `INSUFFICIENT_UNDER_BUDGET`: further corpus evidence may exist, but the declared budget was exhausted.

---

## 16. Corrected worked example

### 16.1 Source

**Remote Work Pilot — Engineering and Design**

> Engineering and Design employees may work remotely on Fridays after completing 90 days of employment and receiving manager approval.

**Customer Support Operations**

> Customer Support employees must work from the office on weekdays. Remote work is allowed only during officially declared emergencies.

### 16.2 Query

> Can Priya, a newly joined Customer Support employee, work remotely this Friday?

### 16.3 Query facts

- Priya is newly joined.
- Priya belongs to Customer Support.
- The requested day is Friday.

These facts are explicitly supplied by the query and do not need corpus retrieval.

### 16.4 Evidence requirements

- `R1`: What is the normal weekday work-location rule for Customer Support?
- `R2`: Is there an exception permitting Customer Support remote work?
- `R3`: Is the exception’s triggering condition established for this Friday?

The Engineering and Design 90-day and manager-approval requirements are not requirements for Priya because that policy has a different department scope.

### 16.5 Diagnosis

- The atomic sentence “employees may work remotely on Fridays” is potentially misleading alone because its heading scope is missing.
- Structural restoration shows that the Friday policy applies to Engineering and Design.
- The Customer Support passage directly resolves R1 and R2.
- R3 remains unresolved unless the corpus states that an emergency has been officially declared.

### 16.6 Correct answer

> Priya cannot normally work remotely this Friday. The Customer Support policy requires employees to work from the office on weekdays and permits remote work only during an officially declared emergency. The supplied evidence does not say that such an emergency has been declared. The 90-day tenure and manager-approval conditions belong to the separate Engineering and Design policy and do not apply to Priya.

This example demonstrates why applicability tracking is mandatory: retrieving a condition is not enough; the system must establish that the condition belongs to the queried entity and policy.

---

## 17. Evaluation datasets

Evaluation has two complementary tracks.

### 17.1 Track A: established QA benchmarks

Use document-grounded subsets of:

- HotpotQA distractor setting.
- 2WikiMultiHopQA.
- MuSiQue.

These evaluate retrieval coverage and multi-hop answering but are not treated as sufficient tests of scope loss or document chunking.

### 17.2 Track B: SAR-RAG Scope and Evidence benchmark

Create a manually verified benchmark from public, redistributable policy manuals, technical documentation, regulations, and versioned procedural documents.

Target composition:

```text
minimum documents: 150
minimum document families: 50
minimum questions: 900
minimum questions per principal state: 150
```

Principal states are:

- Agreement.
- Complementarity.
- Redundancy.
- Context-loss disagreement.
- Source-level contradiction or version conflict.
- Shared insufficiency.

States may co-occur; “principal state” is used only for sampling balance.

Each question is annotated with:

- Original question.
- Query facts.
- Minimal gold requirements.
- Critical/noncritical flags.
- Gold canonical evidence units.
- Applicable entity/time/version frame.
- Qualifiers.
- State labels.
- Answerability label.
- Reference answer.
- Acceptable abstention behavior.

### 17.3 Annotation protocol

1. Two trained annotators independently label every item.
2. A third annotator adjudicates disagreements.
3. Annotators see canonical source context, not system outputs.
4. Cohen’s kappa is reported for categorical labels.
5. Span-level F1 is reported for evidence annotations.
6. Items with unresolved adjudication are excluded before model evaluation, with counts reported.

### 17.4 Leakage control

Split by document family, not by question:

```text
training/development of prompts: 20% of document families
validation and sensitivity analysis: 20%
locked test: 60%
```

No test document or alternate version of a test document may appear in development or validation. Because the method is task-training-free, the development split is used only for prompt debugging and schema validation. Test results remain locked until the protocol is frozen.

Any synthetic questions must be human-verified and reported separately from human-authored questions. The primary result is computed on human-authored or human-rewritten items to avoid favoring the language model used for generation.

---

## 18. Baselines

All baselines use the same corpus release, parser output where applicable, embedding model, and answer generator unless the baseline definition requires otherwise.

### 18.1 Retrieval baselines

1. BM25 atomic units.
2. Dense atomic retrieval.
3. Dense structure-preserving passage retrieval.
4. Fixed 256-token sliding windows with 64-token overlap.
5. Semantic-breakpoint chunks.
6. Hierarchical child retrieval with parent expansion.
7. Hybrid BM25 plus dense retrieval.
8. Dense retrieval plus frozen cross-encoder reranking.
9. Proposition-level retrieval.
10. Late-chunking retrieval.
11. Full-document or longest-context feasible baseline.

### 18.2 Adaptive and fusion baselines

12. Search all representations and concatenate.
13. Search all representations with RRF.
14. RRF plus ordinary text deduplication.
15. Adaptive-RAG-style query-complexity routing.
16. Mix-of-Granularity-style trained routing, reported separately because it is not task-training-free.
17. Random active-representation selection under the same budget.
18. Fixed Top-3, Top-5, Top-10, and Top-20.
19. Adaptive accumulation without source-state diagnosis.
20. Full SAR-RAG.

### 18.3 Oracle baselines

21. Oracle best single representation per query.
22. Oracle useful-representation set.
23. Oracle gold evidence context.

Oracle baselines estimate headroom and are never presented as deployable systems.

### 18.4 Fair budget conditions

Every major comparison is reported under four separate controls:

- Equal candidates judged.
- Equal final context tokens.
- Equal end-to-end latency.
- Equal estimated monetary cost.

Quality-only unconstrained results may be reported additionally but cannot support an efficiency claim.

---

## 19. Ablations

Run the following ablations against the complete method:

1. Atomic plus passage only.
2. Atomic plus structural only.
3. Passage plus structural only.
4. Replace passage chunks with sliding windows.
5. Add semantic-breakpoint chunks as a fourth view.
6. Remove requirement decomposition.
7. Remove applicability frames.
8. Remove canonical source alignment.
9. Remove context restoration.
10. Remove contradiction/version handling.
11. Search all views to depth 20 instead of selective continuation.
12. Remove the confirmation batch.
13. Fixed Top-k instead of adaptive accumulation.
14. Remove final evidence reduction.
15. Same model for judging and generation.
16. Add robust z-score tail filtering.
17. Add deterministic score-gap knee filtering.
18. Remove abstention and force an answer.

Offline representation analysis is not called an ablation because it does not affect online system behavior.

---

## 20. Metrics

### 20.1 Primary endpoints

1. Gold canonical evidence F1.
2. Condition-and-scope preservation accuracy.
3. Answer correctness.
4. Final evidence tokens.
5. Unsupported claim rate.

### 20.2 Retrieval and representation metrics

- Recall@5, Recall@10, Recall@20.
- Mean reciprocal rank.
- Unique gold-evidence recall by representation.
- Pairwise canonical-unit overlap.
- Useful-representation recall.
- Missed-useful-representation rate.
- Structural restoration success.
- Index storage and construction time.

### 20.3 State metrics

- Multi-label precision, recall, and F1 for every state.
- Context-loss detection F1.
- Genuine contradiction detection F1.
- Different-scope versus contradiction confusion rate.
- Insufficiency detection F1.

### 20.4 Applicability and evidence metrics

- Requirement resolution accuracy.
- Applicability-frame accuracy.
- Cross-scope qualifier leakage rate.
- Critical-qualifier preservation.
- Premature stopping rate.
- Unnecessary continuation rate.
- Evidence coverage per 1,000 tokens.
- Final-context canonical redundancy.
- Locally removable evidence rate after reduction; this should be zero by construction and is verified independently.

### 20.5 Answer metrics

- Exact Match and token F1 where appropriate.
- Claim-level precision and recall.
- Faithfulness to supplied evidence.
- Citation entailment.
- Citation completeness.
- Citation source correctness.
- Abstention precision and recall.
- Human preference for long-form answers where reference strings are inadequate.

### 20.6 Efficiency metrics

- Probe retrieval latency.
- Judge latency.
- Deep retrieval latency.
- Generator latency.
- Total candidates retrieved, judged, accepted, and presented.
- Number of active representations by round.
- Final context tokens.
- Index storage.
- Embedding construction cost.
- End-to-end monetary cost using recorded provider or hardware rates.

Chunk count is secondary because chunks differ in granularity.

---

## 21. Statistical analysis

1. Freeze primary endpoints and hypotheses before opening the test set.
2. Report means with 95% paired bootstrap confidence intervals using 10,000 resamples by question.
3. Use a paired randomization test for the primary comparison between SAR-RAG and the strongest cost-matched baseline.
4. Correct the five primary-endpoint p-values using Holm’s method.
5. Report effect sizes, not only statistical significance.
6. Report results separately by dataset, question type, principal state, document length, and answerability.
7. Temperature-zero runs are repeated three times on a 10% test sample to measure residual nondeterminism. If outputs differ materially, all primary test queries are run three times and majority or median results are reported using the pre-declared aggregation rule.
8. No method is selected based on test performance.

Non-inferiority for H3 is declared only if the answer-correctness lower confidence bound is no more than two absolute percentage points below the strongest deep-search baseline while final context tokens are significantly lower. The two-point margin must be pre-registered and may not be changed after test inspection.

---

## 22. Offline representation analysis

Representation analysis validates diversity and cost; it does not route online queries.

For each representation, report separately:

- Gold evidence recall.
- Unique evidence recall.
- Pairwise canonical overlap.
- Boundary/context-loss error rate.
- Condition-preservation rate.
- Mean evidence tokens.
- Index entries and storage.
- Embedding time.
- Retrieval latency.

Do not combine these into an arbitrary weighted quality score.

A representation is considered removable when its 95% confidence interval shows no meaningful unique-evidence improvement and its removal does not worsen primary end-to-end outcomes under the pre-registered non-inferiority margin. Until that experiment is complete, the paper must not claim that all three views are universally necessary.

---

## 23. Reproducibility requirements

The released artifact must contain:

```text
corpus manifest and licenses
document hashes and parser version
canonical-unit data schema
representation-building code
model and tokenizer revision hashes
all prompts
all fixed constants and budgets
experiment configuration files
random seeds
hardware and software environment
raw retrieval rankings
judge JSON outputs
final evidence selections
answers and citations
evaluation scripts
statistical-analysis scripts
error-analysis labels
```

Secrets, proprietary documents, and personal data must not be committed. If some source documents cannot be redistributed, release hashes, acquisition instructions, and derived annotations only where licensing permits.

Every experiment run receives an immutable run ID and stores the Git commit, configuration hash, corpus release, and timestamp.

---

## 24. Failure analysis protocol

Manually analyze at least 100 locked-test failures, stratified across:

- Parser failure.
- Retrieval miss.
- Requirement extraction error.
- Applicability error.
- Context restoration error.
- False contradiction.
- Missed contradiction.
- Judge citation error.
- Premature stopping.
- Excess retrieval.
- Evidence reduction error.
- Generator misuse of correct evidence.

Two annotators assign the primary failure cause. Report examples from both successes and failures. Do not attribute every error to retrieval when the judge or generator is responsible.

---

## 25. Claims the research may and may not make

### 25.1 Permitted if supported by results

- No task-specific router or classifier was trained.
- Source alignment improved context-loss diagnosis.
- Applicability tracking reduced qualifier leakage.
- Selective candidate examination reduced cost at matched quality.
- The final evidence set was sufficient and locally irreducible under the declared procedure.

### 25.2 Prohibited without additional proof

- The method is completely training-free in the sense of using no pretrained models.
- The final context is globally minimal.
- Raw similarity scores are calibrated probabilities.
- All three representations are always necessary.
- The method solves multimodal document understanding.
- A newer document is automatically more authoritative.
- Two representations of the same text provide independent corroboration.
- Absence of retrieved evidence proves that a claim is false.

---

## 26. End-to-end algorithm

```text
OFFLINE

1. Acquire and version documents.
2. Parse documents into typed canonical source units.
3. Validate offsets, hierarchy, tables, captions, and provenance.
4. Build atomic contextualized units.
5. Build deterministic structure-preserving passages.
6. Build non-generative hierarchical structural vectors.
7. Embed all representations using frozen BGE-M3.
8. Build exact cosine indexes for the controlled experiment.
9. Store every representation-to-canonical lineage mapping.
10. Freeze the corpus release and experiment manifest.

ONLINE

1. Preserve the original query exactly.
2. Extract at most eight evidence requirements and separate query facts.
3. Validate extraction; fall back to one whole-query requirement if needed.
4. Probe Top-5 atomic, passage, and structural candidates.
5. Perform no raw-score calibration.
6. Align atomic and passage candidates by canonical overlap.
7. Merge exact same-source duplicates.
8. Rank direct evidence groups using RRF with k0=60.
9. Judge candidates with canonical citations and applicability frames.
10. Assign candidate roles and multi-label evidence states.
11. Restore headings, parents, adjacent units, tables, and versions where needed.
12. Accept only evidence adding a requirement, applicability fact, qualifier,
    context restoration, contradiction resolution, or independent corroboration.
13. Keep only representations meeting the explicit active-view rule.
14. Retrieve ranks 6–10, 11–15, and 16–20 from active direct views.
15. Expand at most two structural nodes per unresolved requirement per round.
16. If still unresolved, reformulate each unresolved requirement once.
17. Stop successfully only after sufficiency plus an empty confirmation batch.
18. Otherwise stop at declared budgets and abstain from unsupported parts.
19. Compact accepted evidence using obligation coverage and a deletion pass.
20. Construct the final scope-ordered context with canonical citations.
21. Generate a grounded answer, disclose conflicts, or abstain.
22. Record every retrieval, judgment, state change, and stopping reason.
```

---

## 27. Positioning against related work

The literature comparison is part of the research claim, not optional background.

- [Adaptive-RAG](https://arxiv.org/abs/2403.14403) trains a classifier to choose among no-retrieval, single-step, and iterative strategies based on question complexity. SAR-RAG instead probes every retained granularity and uses cited evidence state without a task-trained router.
- [Self-RAG](https://arxiv.org/abs/2310.11511) trains a language model to retrieve and critique using reflection tokens. SAR-RAG does not train the generator and operates on canonical document provenance before generation.
- [Corrective RAG](https://arxiv.org/abs/2401.15884) evaluates retrieval and triggers corrective actions, including external search. SAR-RAG focuses on representation disagreement, source scope, and selective examination inside a frozen corpus.
- [FLARE](https://arxiv.org/abs/2305.06983) performs active retrieval during generation based on anticipated content and token confidence. SAR-RAG completes evidence sufficiency checks before answer generation.
- [Dense X Retrieval](https://arxiv.org/abs/2312.06648) demonstrates the importance of retrieval granularity and proposition-level units. It motivates the fine-grained baseline and the atomic view.
- [RAPTOR](https://arxiv.org/abs/2401.18059) uses recursively clustered abstractive summaries in a retrieval tree. SAR-RAG’s primary structural view is non-generative and exists mainly for provenance-safe scope restoration.
- [Mix-of-Granularity](https://aclanthology.org/2025.coling-main.384/) learns a router for query-dependent granularity. It is a required trained-routing comparison for the task-training-free claim.
- [Late Chunking](https://arxiv.org/abs/2409.04701) preserves document context inside chunk embeddings. It is a required baseline for testing whether multiple explicit views are worth their storage and retrieval cost.
- [RAGChecker](https://arxiv.org/abs/2408.08067) provides fine-grained retrieval and generation diagnostics. Its metrics supplement, but do not replace, SAR-RAG’s gold applicability and qualifier annotations.
- [HiChunk](https://aclanthology.org/2026.acl-long.1372/) emphasizes that ordinary QA benchmarks can be too evidence-sparse for chunking evaluation. This supports the separate scope-and-evidence benchmark, but it also raises the novelty bar for claims about hierarchical chunking.

The paper must include a systematic search protocol with databases, dates, search strings, inclusion criteria, and a table comparing supervision, representations, source alignment, applicability tracking, contradiction handling, stopping, abstention, and cost controls. Novelty is finalized only after that review is updated immediately before submission.

---

## 28. Final novelty statement

> SAR-RAG is a task-training-free, source-aligned multi-granularity retrieval framework that uses observable, cited evidence states rather than a trained query router to control deeper candidate examination. Its distinctive objective is to construct a compact and locally irreducible evidence context while preventing conditions, exceptions, headings, dates, and versions from being transferred across incompatible scopes.

This novelty must be compared directly with adaptive retrieval, multi-granularity routing, proposition retrieval, hierarchical retrieval, late chunking, corrective RAG, and diagnostic RAG evaluation. The final paper should claim novelty only for components or combinations that remain distinct after this comparison and are supported by ablations.

---

## 29. Final assessment of research readiness

This specification is internally complete enough to begin implementation. It fixes the algorithmic choices that were previously ambiguous:

- Three primary representations, not five assumed-useful views.
- Exact experimental retrieval rather than an unspecified ANN index.
- Rank-only RRF instead of unstable raw-score normalization.
- No knee filter in the primary method.
- Explicit evidence and applicability JSON schemas.
- Exact retrieval rounds and hard budgets.
- A confirmation batch before successful stopping.
- Local irreducibility instead of an unsupported global-minimality claim.
- Separate handling of context loss, different scope, and genuine contradiction.
- Correct abstention and version/authority behavior.
- Equal-budget baselines, locked document-family splits, and pre-registered statistics.

The remaining unknowns are empirical results, not unspecified design assumptions. They must be answered by executing the frozen protocol; they must not be decided after viewing the test results.

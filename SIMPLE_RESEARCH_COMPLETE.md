# Source-Aligned Representation-State Adaptive RAG (SAR-RAG)

> **Superseded primary online algorithm (4 August 2026):** See
> `ONE_CALL_RESEARCH_PIPELINE.md`. Candidate-level LLM judgment is no longer used by
> the primary method; Groq is called once after final evidence construction.

> **Implementation correction:** `DETAILED_IMPLEMENTATION_BLUEPRINT.md` is the current source of truth for the online algorithm. Probe Top-3/Top-5 selects representations; selected indexes are searched adaptively without a fixed final `k`; final deduplication occurs after adaptive retrieval.

## Complete Research Plan in Simple Words

Version 1.0 - 2 August 2026

## 1. Research in one paragraph

This research builds a Retrieval-Augmented Generation system that reads the same document in three different ways: small fact-sized units, normal structure-preserving passages, and document hierarchy such as sections and headings. For every question, the system first checks a few results from all three views. It then identifies which results directly answer the question, which add missing information, which restore a lost condition or heading, and which are repeated or misleading. It searches deeper only where useful evidence is still missing. Finally, it builds a compact evidence set, answers only from that evidence, preserves conditions and exceptions, shows citations, and refuses to guess when the documents are insufficient.

## 2. Why this research is needed

Normal RAG systems divide documents into chunks and retrieve a fixed number such as Top-5. This creates several problems:

- A small chunk can retrieve the exact sentence but lose its heading or condition.
- A large chunk can preserve meaning but add too much irrelevant text.
- Fixed-size chunks can split a rule, table, or explanation at the wrong point.
- Different chunks can repeat the same source text.
- A fixed Top-k can stop too early for a complex question or retrieve too much for a simple question.
- A language model can incorrectly apply a condition from one department, person, date, or policy version to another.
- Two chunks can appear contradictory even though they apply to different scopes.
- When no sufficient evidence exists, ordinary systems often guess instead of abstaining.

SAR-RAG is designed to solve these problems through multiple source-aligned views, explicit evidence requirements, applicability checking, selective deeper retrieval, and safe stopping.

## 3. What is genuinely new

Using different chunk sizes is not new by itself. Adaptive retrieval and hierarchical retrieval also already exist. The research contribution is the combined source-safe decision process:

1. Every retrieved item maps back to the exact original source units.
2. The system compares different representations of the same source.
3. It detects when disagreement is caused by lost context rather than a real contradiction.
4. It tracks what evidence is still needed for the question.
5. It checks whether each rule or condition actually applies to the queried case.
6. It searches deeper only in useful views.
7. It creates a compact evidence set without removing necessary conditions.
8. It abstains from unsupported parts of an answer.

The correct description is task-training-free. The system does not train a router, classifier, confidence model, or task-specific language model. It still uses frozen pretrained embedding and instruction models, so it must not be described as model-free.

## 4. Main research question

Can a shallow probe of three source-aligned document representations find the useful retrieval views for each question and produce more complete, less repetitive, and more scope-correct evidence than fixed-representation or fixed-Top-k RAG under the same resource budget?

Secondary questions are:

- Can source alignment separate lost-context disagreement from genuine source contradiction?
- Does applicability tracking prevent conditions from being applied to the wrong entity, department, date, or document version?
- Can adaptive evidence accumulation improve evidence coverage per 1,000 context tokens?
- Can the system abstain reliably when the corpus is insufficient?
- Do all three representations add enough unique value to justify their cost?

## 5. Final representation design

The primary method uses three representations. Five representations are not assumed to be necessary.

### 5.1 Atomic contextualized units

This is the fine-grained view. A unit may be a sentence, list item, table row with its headers, caption, footnote, equation with defining text, or complete code element.

The embedding input contains the document title, section path, unit type, and original unit text. The heading information helps retrieval, but the system still cites the original source unit.

Best for direct facts, definitions, numbers, and exact conditions. Its weakness is that distant context may still be missing.

### 5.2 Structure-preserving passages

This is the middle-sized view. It keeps natural paragraphs, lists, tables, and related adjacent units together.

Passage rules are fixed:

- Natural blocks are kept intact where possible.
- Blocks shorter than 96 embedding tokens are merged with adjacent blocks.
- The maximum passage length is 384 tokens.
- Long blocks are split only at canonical-unit boundaries.
- Passages never cross subsection boundaries.
- The primary method uses no overlap.

Best for explanations, conditions, exceptions, procedures, and evidence spread across nearby sentences.

### 5.3 Hierarchical structural nodes

This is the document-structure view. It represents documents, sections, subsections, and parent blocks.

The structural vector is built without an LLM summary. It combines the heading embedding with the average embedding of the node's child passages. This prevents a generated summary from inventing information.

Structural nodes are used to recover headings, policy scope, document versions, and relevant child passages. A structural node is not accepted as direct evidence unless its original descendant units are selected and cited.

### 5.4 Representations kept as comparisons

Sliding windows and semantic-breakpoint chunks remain important baselines and ablations. Proposition retrieval and late chunking are also required comparisons. They are not silently included in the primary method.

## 6. Canonical source alignment

Every original source unit gets a stable identifier. Each representation stores the identifiers of the source units it contains.

A unit record stores:

- Document ID and version.
- Unit ID and type.
- Section path.
- Page number or HTML anchor.
- Character offsets.
- Parent, previous, and next unit links.
- Original text and normalized hash.
- Source publisher, effective date, and authority metadata when available.

This makes it possible to find exact overlap, containment, parent-child relationships, adjacent context, document-version differences, and repeated text.

Canonical units are not limited to sentences. Tables, lists, captions, footnotes, equations, and code must be represented correctly.

## 7. Fixed models and retrieval settings

The controlled primary experiment uses:

- Embedding model: BAAI/bge-m3.
- Similarity: cosine similarity over normalized vectors.
- Search index: FAISS IndexFlatIP for exact search.
- Requirement and evidence judge: Qwen2.5-7B-Instruct.
- Answer generator: Qwen2.5-14B-Instruct.
- Temperature: 0 for both instruction models.
- Top-p: 1.
- Maximum requirements: 8.
- Probe depth: Top-5 from each representation.
- Maximum direct search depth: Top-20 per direct-evidence representation.
- Final evidence budget: 2,000 tokens.

The exact model revision hashes, tokenizer revisions, package versions, prompts, hardware, and random seeds must be saved in every experiment manifest.

## 8. Final calibration decision

The primary method does not calibrate raw similarity scores.

Robust z-score normalization was removed because it would be calculated from a small, truncated Top-20 list. Its median absolute deviation can be zero or unstable, and the resulting number is not a probability of relevance.

Score-knee or score-tail filtering was also removed from the primary method because it requires another algorithm and thresholds that may behave differently across queries.

The final rule is simple:

- Within each representation, cosine similarity decides rank.
- Across the atomic and passage views, source-aligned Reciprocal Rank Fusion combines ranks.
- RRF uses k0 = 60.
- Structural results are used for scope restoration and child navigation, not direct RRF evidence fusion.

For a source-aligned evidence group g:

RRF(g) = sum over direct representations of 1 / (60 + best rank of g in that representation).

A representation contributes only once to a group. This stops many overlapping chunks from one index from unfairly increasing the score.

Robust z-score and knee filtering are evaluated only as ablations. They are not part of the main claim.

## 9. Evidence requirements

The original query is always preserved and used for initial retrieval. The system also extracts the smallest set of evidence requirements needed to decide when enough evidence has been collected.

For a simple question, there is one requirement. For a comparison or multi-part question, there may be several. The system allows at most eight.

Facts explicitly provided by the user are stored separately as query facts. They do not need to be retrieved again.

Each requirement has four possible states:

- Supported.
- Refuted.
- Not applicable.
- Unresolved.

Only the first three count as resolved. Missing evidence never means that a claim is false.

If requirement extraction produces invalid JSON, it is retried once. If it fails again or the model is uncertain, the complete original query becomes one requirement. This provides a safe fallback.

## 10. Applicability tracking

Applicability tracking is a required part of the method.

For every accepted claim, the system records relevant scope variables such as:

- Person or population.
- Department or organization.
- Product or software version.
- Date or effective period.
- Jurisdiction or location.
- Document version.
- Exception state.

A retrieved condition is not accepted merely because it is relevant. The system must show that the condition applies to the queried entity and situation.

This directly prevents a condition from an Engineering policy from being applied to a Customer Support employee, or an old policy from being applied after a newer version became effective.

## 11. Probe retrieval

For every question, the system initially retrieves:

- Top-5 atomic candidates.
- Top-5 passage candidates.
- Top-5 structural nodes.

This produces at most 15 probe candidates. Ranks 6 to 20 are not requested until a direct representation remains active.

Atomic and passage candidates are grouped when they come from the same document version and have identical source units, containment, or at least 0.5 canonical-unit Jaccard overlap.

Exact same-source duplicates are merged before judging. Longer or semantically similar candidates are not removed yet because they may contain an important condition.

## 12. Evidence judgment

The evidence judge receives the original query, requirements, query facts, candidate text, section path, document version, and source-unit IDs.

For every finding, it must return:

- Requirement ID.
- Supported, refuted, not applicable, or unresolved.
- Exact cited source-unit IDs.
- The supported claim.
- Applicability frame.
- Conditions and qualifiers.
- Whether more parent or adjacent context is needed.

Every cited ID is validated. Any supported or refuted decision without a valid supplied citation is rejected. Invalid model output is retried once; a second failure leaves the candidate unresolved.

The judge's confidence is not treated as a calibrated probability.

## 13. Candidate roles

Candidate roles are descriptive and are based on cited findings:

- Primary or direct: resolves an unresolved critical requirement.
- Complementary: resolves a different requirement.
- Context-restoring: adds a heading, condition, exception, date, version, or parent scope needed to interpret another result.
- Corroborating: provides the same applicable claim from an independent source lineage.
- Redundant: adds no new requirement, condition, context, or independent source.
- Potentially misleading alone: looks relevant but is missing context that changes its meaning or applicability.
- Insufficient: resolves nothing and restores no relevant context.

Two representations of the same source sentence are not independent corroboration.

## 14. Representation states

The overall evidence can have more than one state:

- Agreement: complete applicable evidence says the same thing.
- Complementarity: different evidence resolves different parts of the question.
- Redundancy: evidence repeats the same claim and conditions from the same source lineage.
- Context-loss disagreement: a short result looks inconsistent because it lost a heading, condition, or parent scope.
- Source-level contradiction: complete evidence from overlapping scopes genuinely conflicts.
- Shared insufficiency: no examined representation resolves an important requirement.

States are multi-label because one part of a question may have agreement while another part has a contradiction.

## 15. Context loss versus real contradiction

When results appear inconsistent, the system checks context in this order:

1. Heading path.
2. Complete containing passage.
3. Previous and next source units.
4. Table header, footnote, caption, or list parent.
5. Subsection parent.
6. Document version, date, and authority metadata.

Two claims can contradict only when they refer to the same entity, issue, time, location, document scope, and exception state.

If scopes differ, the result is different scope, not contradiction. If a newer source supersedes an older source, that must be explicitly established in metadata. The system never assumes that newer automatically means more authoritative.

If complete sources still conflict and no valid authority rule resolves them, both are retained and the answer must disclose the conflict.

## 16. Selective deeper retrieval

Retrieval proceeds in fixed rounds:

- Round 1: ranks 1-5 from all three views.
- Round 2: ranks 6-10 from active direct views.
- Round 3: ranks 11-15 from active direct views.
- Round 4: ranks 16-20 from active direct views.

A direct view stays active when its latest batch resolves a requirement, adds an applicable qualifier, reveals context loss, or helps investigate a contradiction. It is paused when its latest batch adds nothing and is not needed for an active investigation.

The structural view stays active only when the system needs a heading, parent, version, authority record, or child passage. At most two structural nodes per unresolved requirement are expanded in a round, with three child passages examined per node.

After Round 4, every unresolved critical requirement may be reformulated once into a short search query. The reformulation cannot contain a proposed answer. If it still fails, the system abstains from that part.

## 17. Adaptive evidence accumulation

A candidate is accepted only when it has valid citations and adds something required:

- Resolves a critical requirement.
- Proves that a requirement is not applicable.
- Adds a condition or exception.
- Establishes an applicability fact.
- Restores missing context.
- Resolves or documents a source contradiction.
- Provides genuinely independent corroboration when needed.

A candidate is rejected from the final evidence when it resolves nothing, repeats same-source evidence without a unique qualifier, belongs to an irrelevant scope, has invalid citations, or cannot fit the budget and has a shorter sufficient replacement.

Primary hard limits are:

- Maximum 50 direct candidates judged.
- Maximum 18 structural child passages judged.
- Four retrieval rounds plus one reformulation round.
- Maximum 2,000 final evidence tokens.

If the system reaches a budget before finding sufficient evidence, it reports insufficient under budget. It does not call the search successful.

## 18. Stopping rule

The system stops successfully only when:

1. Every critical requirement is supported, refuted, or not applicable.
2. Every accepted claim has a matching applicability scope.
3. Every context-loss disagreement is resolved.
4. Every genuine contradiction is resolved or marked for disclosure.
5. One additional confirmation batch from every still-active direct view adds no useful evidence.

The confirmation batch prevents stopping immediately after the first apparently complete result.

The system stops unsuccessfully when budgets are reached, all views are inactive while important requirements remain unresolved, reformulation fails, or the source cannot be validated.

## 19. Final evidence reduction

The accepted evidence may still contain unnecessary items. The system creates a list of obligations:

- Resolved requirements.
- Necessary conditions and exceptions.
- Applicability facts.
- Contradictions that must be shown.
- Required independent support.

It selects evidence that covers the most uncovered obligations per token. Ties prefer declared authority, an explicitly applicable newer version, more exact citations, and then fewer tokens.

It then performs a deletion pass. Any selected item that can be removed without losing an obligation is removed.

The result is compact and locally irreducible. It is not claimed to be the globally smallest possible evidence set.

Evidence tokens and canonical-unit counts are the main size measures. Chunk count is secondary because a sentence chunk and section chunk are not comparable.

## 20. Answer generation and abstention

The final context is ordered as:

1. Direct applicable evidence.
2. Scope and applicability evidence.
3. Conditions, exceptions, and negations.
4. Complementary evidence.
5. Unresolved contradictory evidence.
6. Date, version, and authority information.

The answer generator must use only the supplied evidence, preserve every relevant condition, cite factual claims, disclose unresolved contradictions, and explicitly say when the corpus does not establish an answer.

Final answerability labels are:

- Answerable.
- Answerable with conflict.
- Partially answerable.
- Insufficient.
- Insufficient under budget.

## 21. Corrected example

Source policy 1 is under the heading Remote Work Pilot - Engineering and Design:

Engineering and Design employees may work remotely on Fridays after completing 90 days and receiving manager approval.

Source policy 2 is under Customer Support Operations:

Customer Support employees must work from the office on weekdays. Remote work is allowed only during officially declared emergencies.

Question: Can Priya, a newly joined Customer Support employee, work remotely this Friday?

The query already tells us that Priya is newly joined, belongs to Customer Support, and is asking about Friday.

The actual requirements are:

- What is the normal weekday rule for Customer Support?
- Is there a Customer Support exception?
- Is the exception's emergency condition established for this Friday?

The 90-day and manager-approval requirements do not apply to Priya because they belong to Engineering and Design.

Correct answer:

Priya cannot normally work remotely this Friday. Customer Support employees must work from the office on weekdays and may work remotely only during an officially declared emergency. The evidence does not say that an emergency has been declared. The 90-day and manager-approval conditions belong to the separate Engineering and Design policy and do not apply to Priya.

This correction is important because the earlier research draft incorrectly transferred Engineering conditions to Customer Support.

## 22. Evaluation datasets

Evaluation uses two tracks.

Track A uses established document-grounded subsets of HotpotQA, 2WikiMultiHopQA, and MuSiQue. These test retrieval and multi-hop answering but are not enough for scope-loss evaluation.

Track B creates the SAR-RAG Scope and Evidence benchmark from public, redistributable policy manuals, technical documentation, regulations, and versioned procedures.

The target is at least:

- 150 documents.
- 50 document families.
- 900 questions.
- 150 questions for each principal state: agreement, complementarity, redundancy, context loss, contradiction/version conflict, and insufficiency.

Every question records gold requirements, query facts, canonical evidence units, applicability frames, qualifiers, states, answerability, and a reference answer.

Two annotators label every item independently and a third adjudicates disagreements. Agreement scores and evidence-span F1 must be reported.

The split is by document family: 20% development, 20% validation, and 60% locked test. No alternate version of a test document can appear in another split.

## 23. Required baselines

The study compares against:

- BM25.
- Dense atomic retrieval.
- Dense passage retrieval.
- Sliding-window retrieval.
- Semantic-breakpoint retrieval.
- Hierarchical child-to-parent retrieval.
- Hybrid BM25 plus dense retrieval.
- Dense retrieval plus cross-encoder reranking.
- Proposition retrieval.
- Late chunking.
- Long-context or full-document input where feasible.
- Search all views and concatenate.
- Search all views with RRF.
- RRF with ordinary deduplication.
- Adaptive-RAG-style routing.
- Mix-of-Granularity-style trained routing.
- Random view selection under the same budget.
- Fixed Top-3, Top-5, Top-10, and Top-20.
- Oracle best representation.
- Oracle useful-view set.
- Oracle gold evidence.
- Full SAR-RAG.

Major comparisons must use equal candidates, equal final context tokens, equal latency, and equal estimated cost. Uncontrolled quality results cannot support an efficiency claim.

## 24. Required ablations

The study removes or changes one component at a time:

- Every pair of the three representations.
- Sliding windows instead of structure-preserving passages.
- Add semantic chunks as a fourth view.
- Remove requirement decomposition.
- Remove applicability frames.
- Remove source alignment.
- Remove context restoration.
- Remove contradiction and version handling.
- Search all views deeply instead of selective continuation.
- Remove the confirmation batch.
- Use fixed Top-k.
- Remove final evidence reduction.
- Use the same model for judging and answering.
- Add robust z-score filtering.
- Add score-knee filtering.
- Force an answer instead of abstaining.

Offline representation analysis is not an ablation because it does not change the online method.

## 25. Evaluation metrics

Primary outcomes are:

- Gold canonical evidence F1.
- Condition-and-scope preservation accuracy.
- Answer correctness.
- Final evidence tokens.
- Unsupported claim rate.

Additional retrieval measures include Recall@k, reciprocal rank, unique evidence recall, overlap, useful-view recall, structural restoration success, storage, and latency.

State measures include F1 for context loss, contradiction, different scope, and insufficiency.

Evidence measures include requirement accuracy, applicability accuracy, qualifier leakage, premature stopping, unnecessary continuation, coverage per 1,000 tokens, and final redundancy.

Answer measures include Exact Match, token F1, claim precision and recall, faithfulness, citation correctness and completeness, and abstention precision and recall.

Efficiency measures include retrieval, judging and generation latency, candidates retrieved and judged, active views by round, context tokens, storage, embedding cost, and total monetary cost.

## 26. Statistical testing

The primary hypotheses and metrics are frozen before opening the locked test set.

Results use 95% paired bootstrap confidence intervals with 10,000 question-level samples. The primary comparison uses a paired randomization test. Five primary p-values are corrected with Holm's method. Effect sizes are always reported.

Results are separated by dataset, question type, evidence state, document length, and answerability.

For the efficiency hypothesis, answer correctness may be no more than two absolute percentage points below the strongest deep-search baseline while using significantly fewer final tokens. This non-inferiority margin is fixed before test evaluation.

## 27. What the research can claim

If supported by results, the paper may claim:

- No task-specific router or classifier was trained.
- Source alignment improved context-loss diagnosis.
- Applicability tracking reduced qualifier leakage.
- Selective examination reduced cost at matched quality.
- The final evidence was sufficient and locally irreducible under the declared policy.

It must not claim:

- No pretrained models were used.
- The evidence set is globally minimal.
- Similarity scores are calibrated probabilities.
- All three representations are always necessary.
- Multimodal documents are completely solved.
- Newer sources are automatically more authoritative.
- Repeated chunks are independent support.
- Missing evidence proves a statement false.

## 28. Limitations

The main limitations are:

- Evidence judging still depends on a pretrained language model and can make mistakes.
- Building multiple representations increases indexing and storage cost.
- Exact search is appropriate for controlled research but may not scale to production-sized corpora.
- Complex tables, scanned pages, diagrams, and images need additional multimodal work.
- Requirement and applicability annotations are expensive to create.
- Public QA datasets may already be familiar to pretrained models, so retrieval-specific evidence evaluation is essential.
- The third representation may fail to add enough value; this must be decided by results.
- A compact greedy evidence set is not guaranteed to be the mathematical global minimum.

These limitations are reported openly rather than hidden.

## 29. End-to-end pipeline

Offline phase:

1. Collect, license, identify, and version documents.
2. Parse them into typed canonical units.
3. Validate text offsets, hierarchy, tables, footnotes, and provenance.
4. Build atomic, passage, and structural representations.
5. Embed them with frozen BGE-M3.
6. Build exact cosine-search indexes.
7. Save every representation-to-source mapping.
8. Freeze the corpus and experiment configuration.

Online phase:

1. Keep the original query unchanged.
2. Extract and validate evidence requirements and query facts.
3. Retrieve Top-5 from all three views.
4. Use no raw-score calibration.
5. Align direct candidates to canonical source units.
6. Merge exact same-source duplicates.
7. Rank direct source groups with RRF k0 = 60.
8. Judge evidence with citations and applicability frames.
9. Diagnose roles, context loss, different scope, contradiction, and insufficiency.
10. Restore required headings, parents, adjacent units, tables, and versions.
11. Accept only evidence that adds a required obligation.
12. Retrieve deeper from active views in batches of five up to rank 20.
13. Reformulate each unresolved critical requirement once if necessary.
14. Stop after sufficiency and an empty confirmation batch, or stop at a declared budget.
15. Compact evidence and remove locally unnecessary items.
16. Generate a cited answer, disclose conflict, or abstain.
17. Record every decision for evaluation and reproducibility.

## 30. Final research statement

SAR-RAG is a task-training-free, source-aligned, multi-granularity RAG framework. It uses evidence states rather than a trained query router to decide where deeper retrieval is useful. Its central goal is to build compact, sufficient, and scope-correct evidence while preventing conditions, exceptions, dates, headings, and versions from being moved across unrelated contexts.

The design is now fixed enough for implementation. The remaining questions are empirical: whether the method improves accuracy and efficiency, which representation families contribute unique evidence, and how reliably the frozen judge follows the source. These questions must be answered by running the locked protocol, not by changing rules after seeing test results.

## 31. Key related work

- Adaptive-RAG: adaptive retrieval based on a trained question-complexity classifier.
- Self-RAG: trained retrieval and self-critique using reflection tokens.
- Corrective RAG: retrieval evaluation followed by corrective actions.
- FLARE: active retrieval during generation.
- Dense X Retrieval: evidence that retrieval granularity matters and propositions can be effective.
- RAPTOR: hierarchical retrieval through recursive summaries.
- Mix-of-Granularity: learned routing between retrieval granularities.
- Late Chunking: context-aware chunk embeddings from long-context encoders.
- RAGChecker: detailed retrieval and generation evaluation metrics.
- HiChunk and HiCBench: hierarchical chunking and evidence-dense chunking evaluation.

The final paper must update the systematic literature search immediately before submission and compare supervision, representation design, source alignment, applicability, contradiction handling, stopping, abstention, and cost.

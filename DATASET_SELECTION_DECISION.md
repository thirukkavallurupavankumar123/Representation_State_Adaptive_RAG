# SAR-RAG Dataset Selection Decision

## Final Core Bundle, Exclusions, and Evaluation Protocol

**Date:** 2 August 2026  
**Status:** authoritative dataset decision

## 1. Final decision

The final core dataset bundle is:

1. **ConditionalQA** - primary public dataset for conditions, applicability, long policy documents, multi-hop evidence, and not-answerable questions.
2. **QASPER** - primary public dataset for long scientific documents, section hierarchy, multi-section evidence, and unanswerable questions.
3. **MuSiQue-Answerable and MuSiQue-Full** - primary public dataset for connected multi-hop evidence, adaptive evidence coverage, and insufficiency/abstention.
4. **SAR-RAG Representation-State Benchmark** - mandatory custom benchmark for context-loss disagreement, representation roles, redundancy, qualifier preservation, genuine contradiction, and shared insufficiency.

**HotpotQA Distractor** is a secondary external sanity benchmark, not a core dataset. Use it only after the four core tracks work.

**QuALITY, HotpotQA FullWiki, 2WikiMultiHopQA, and SQuAD 2.0 are excluded from the primary evaluation.**

This bundle is smaller and better aligned with the research contribution than the earlier recommendation of HotpotQA + MuSiQue + QASPER + custom benchmark.

## 2. Why ConditionalQA replaces HotpotQA as a core dataset

SAR-RAG's distinctive claims concern:

- Conditions and exceptions.
- Applicability to a person or scenario.
- Long-document structure.
- Evidence requirements.
- Context loss.
- Abstention when a condition is unknown.

ConditionalQA directly contains conditional answers, long and logically complex documents, multi-hop questions, extractive and yes/no answers, multiple answers, and not-answerable questions. It also annotates answer conditions at the HTML-element level.

HotpotQA is useful for multi-hop and comparison retrieval, but MuSiQue already provides a stronger connected multi-hop test and includes an explicit unanswerable contrast set. HotpotQA adds recognition and comparability, but it does not directly test policy applicability or qualifier transfer.

Therefore:

```text
ConditionalQA = core
MuSiQue = core
HotpotQA = optional sanity check
```

## 3. Capability-to-dataset mapping

| Research claim | Primary dataset | Secondary check |
|---|---|---|
| Conditional applicability | ConditionalQA | Custom benchmark |
| Qualifier and exception preservation | ConditionalQA | Custom benchmark |
| Long policy-document retrieval | ConditionalQA | - |
| Long scientific-document hierarchy | QASPER | - |
| Section and parent-context restoration | QASPER | Custom benchmark |
| Connected multi-hop evidence | MuSiQue-Answerable | HotpotQA optional |
| Evidence sufficiency | MuSiQue-Full | ConditionalQA, QASPER |
| Adaptive no-fixed-final-k behavior | MuSiQue, QASPER | ConditionalQA |
| Abstention | MuSiQue-Full | ConditionalQA, QASPER, custom |
| Representation selection accuracy | Custom benchmark | Public-dataset derived analysis |
| Context-loss disagreement | Custom benchmark | Natural QASPER/ConditionalQA subset |
| Redundancy detection | Custom benchmark | All public datasets |
| Genuine source contradiction | Custom benchmark | - |
| Cross-domain generalization | ConditionalQA + QASPER + MuSiQue | HotpotQA optional |

No public dataset alone evaluates the complete method.

## 4. Dataset-specific protocols

### 4.1 ConditionalQA

Use ConditionalQA as the main public applicability dataset.

Primary tasks:

- Retrieve the answer evidence.
- Retrieve all required answer conditions.
- Preserve conditions in the generated answer.
- Mark conditions that are unresolved from the scenario.
- Distinguish deterministic, conditional, and not-answerable cases.
- Compare representations on long structured HTML documents.

Gold labels used:

- Answer.
- Answer type.
- Condition HTML elements.
- Answerability.
- Document structure.

Important restriction:

Condition annotations are used only for evaluation. They must not be inserted into embedding text, retrieval metadata, representation profiles, or prompts at inference time.

Primary metrics:

- Answer EM/F1.
- Condition-element precision, recall, and F1.
- Joint answer-and-condition correctness.
- Applicability accuracy.
- Qualifier preservation.
- Abstention precision and recall.
- Final evidence tokens and final `k` distribution.

Retrieval setting:

- Primary: document-known, full-document internal retrieval. This isolates chunk representation and adaptive evidence accumulation.
- Secondary: pooled-document retrieval across the evaluation split. This adds document retrieval difficulty and must be reported separately.

### 4.2 QASPER

Use QASPER as the main hierarchy and long-scientific-document dataset.

Primary tasks:

- Retrieve evidence from complete papers.
- Compare sentence, paragraph, semantic, sliding, and hierarchical views.
- Recover section headings and parent context.
- Accumulate evidence across different sections.
- Identify unanswerable questions.

Gold labels used:

- Answer annotations.
- Supporting evidence annotations.
- Paper section and paragraph structure.
- Unanswerable annotations.

Important limitation:

QASPER does not directly label representation-induced context loss or genuine contradictions. It supports natural hierarchy analysis, not the complete state-classification claim.

Primary metrics:

- Gold evidence precision, recall, and F1.
- Section retrieval recall.
- Evidence coverage per 1,000 tokens.
- Parent-context restoration rate.
- Answer F1.
- Unanswerable detection.
- Representation contribution and final `k` distribution.

Retrieval setting:

- Primary: paper-known retrieval over the entire paper.
- Secondary: pooled retrieval over all papers in the split.

The paper-known setting does not evaluate document routing. It evaluates the research question of representation and evidence selection inside long structured documents.

### 4.3 MuSiQue-Answerable

Use MuSiQue-Answerable for connected 2-4 hop evidence retrieval.

Primary tasks:

- Determine whether selected representations recover every reasoning hop.
- Test complementary representation selection.
- Measure premature stopping.
- Measure adaptive final evidence size as hop count and evidence complexity change.

Gold labels used:

- Answer.
- Supporting paragraphs.
- Hop structure where available.

Primary metrics:

- Answer F1.
- Support F1.
- Complete-hop coverage.
- Premature stopping rate.
- Evidence tokens.
- Final `k` distribution by 2-hop, 3-hop, and 4-hop question.

### 4.4 MuSiQue-Full

Use MuSiQue-Full for evidence sufficiency and abstention.

Primary tasks:

- Distinguish answerable and unanswerable contrast questions.
- Test whether adaptive retrieval continues when a reasoning hop is absent.
- Test safe stopping after reformulation fails.
- Measure incorrect forced answers.

Primary metrics:

- Group answer sufficiency F1.
- Group support sufficiency F1.
- Answerability accuracy.
- Abstention precision and recall.
- Unnecessary continuation.
- Budget-exhaustion rate.

Leakage control:

MuSiQue is composed from earlier single-hop resources. The official repository publishes dev/test source-question IDs. These IDs must be checked against any data used to train or tune additional task-specific components. SAR-RAG does not task-train a router, but this issue must still be disclosed because frozen pretrained models may have seen public benchmarks.

Retrieval setting:

- Primary controlled setting: official question context, testing evidence selection and stopping.
- Secondary retrieval setting: pool unique paragraphs within the split and retrieve from the pooled corpus.

Do not combine results from the controlled and pooled settings into one score.

## 5. Custom SAR-RAG Representation-State Benchmark

The custom benchmark is mandatory because no selected public dataset provides all required state labels.

### 5.1 Final target size

Create **720 human-verified examples**:

| Principal state | Target |
|---|---:|
| Agreement | 100 |
| Complementarity | 120 |
| Redundancy | 100 |
| Context-loss disagreement | 180 |
| Genuine contradiction/version conflict | 100 |
| Shared insufficiency | 120 |
| **Total** | **720** |

Context loss receives the largest allocation because it is the main novel claim. Contradiction must have at least 100 examples; the earlier suggestion of 50 is too small for stable per-class evaluation.

States are multi-label. The principal state controls sampling only.

### 5.2 Source composition

Use three source groups:

1. Public government or organizational policy and eligibility documents.
2. Structured scientific papers, including a held-out subset compatible with QASPER-style structure.
3. Versioned technical or procedural documentation.

At least 60 document families should be represented. Alternate versions of the same document belong to the same family.

### 5.3 Gold annotations

Each example contains:

```text
question
query facts
critical evidence requirements
canonical supporting source units
canonical refuting source units
applicability frame
conditions and qualifiers
answerability
state labels
expected retrieval action
reference answer
acceptable abstention behavior
source version and authority metadata
```

### 5.4 Do not make representation chunks the primary gold labels

The earlier proposal stored pre-generated chunks under:

```text
representations.sentence_window
representations.sliding_window
representations.paragraph
representations.semantic
representations.hierarchical
```

This should not be the benchmark's primary truth. Representation chunks depend on chunk sizes, overlap, semantic thresholds, embedding models, and parser versions. Hard-coding them would make the benchmark favor one implementation.

The primary gold truth must be canonical source units, claims, qualifiers, applicability, and states.

Frozen representation chunks may be released as one reproducible benchmark configuration, but they must be versioned as derived artifacts:

```text
benchmark source release
parser version
representation builder version
builder configuration hash
embedding model revision
```

### 5.5 Representation-role labels

Representation roles are query- and configuration-dependent. Do not annotate a universal role such as “paragraph is primary” without first fixing the representation builders.

Use two evaluation levels:

1. **Configuration-independent:** Did the system retrieve the gold canonical obligations and diagnose the correct state?
2. **Frozen-configuration analysis:** Given the published five representation builders, did the probe select every representation that contained unique gold evidence or required context?

Useful-representation recall should be derived from gold obligations contained in frozen chunks, not from annotator preference for a chunking method.

### 5.6 Natural and controlled examples

Report two subsets separately:

- **Natural subset:** Conditions, scope loss, contradiction, or insufficiency already exist in original documents.
- **Controlled subset:** A source-preserving transformation removes a heading, table header, negation context, antecedent, condition, or parent scope from one derived view.

Controlled examples must never rewrite a claim into a stronger or weaker sentence. They may only change the visible source span. This ensures the benchmark tests chunk-induced context loss rather than synthetic claim generation.

### 5.7 Annotation and splitting

- Two independent annotators per example.
- Third-person adjudication.
- Report Cohen's kappa or Krippendorff's alpha for states and answerability.
- Report span-level F1 for canonical evidence.
- Split by document family, not by question.
- Suggested split: 20% development, 20% validation, 60% locked test.
- No alternate version or derived question from a locked-test family may enter development or validation.

## 6. HotpotQA decision

HotpotQA Distractor is optional and should be used only as a recognized external sanity check.

Reasons to retain it optionally:

- Sentence-level supporting facts.
- Comparison questions.
- Easy comparison with prior multi-hop work.

Reasons not to make it core:

- Its main role overlaps with MuSiQue.
- Distractor mode supplies only a small question-specific paragraph set.
- Wikipedia paragraph structure is weak for the hierarchy and applicability claims.
- It does not directly label conditions, scope loss, or unanswerable cases.

If included, use the full distractor development set after the core experiments. Do not begin with FullWiki.

## 7. Excluded datasets

### 7.1 QuALITY

Exclude from the primary study because it is multiple-choice and lacks precise supporting-evidence annotations comparable to ConditionalQA, QASPER, or MuSiQue. It adds long-context evaluation but does not isolate representation selection or qualifier preservation well enough to justify the cost.

### 7.2 HotpotQA FullWiki

Exclude initially because it adds large-scale open-domain document retrieval infrastructure. It would confound representation selection with first-stage corpus retrieval quality.

### 7.3 2WikiMultiHopQA

Exclude because MuSiQue already covers connected multi-hop reasoning and gives an insufficiency track. Add 2Wiki only if reviewers specifically require another reasoning-path dataset.

### 7.4 SQuAD 2.0

Exclude because its short passage format does not test heterogeneous long-document representations, source hierarchy, or complementary multi-hop evidence.

## 8. Development and final experiment sizes

### 8.1 Engineering pilot

Use only official training/development material:

| Dataset | Pilot amount |
|---|---:|
| ConditionalQA | 500 questions |
| QASPER | 150 papers or at least 400 questions |
| MuSiQue-Answerable | 500 questions, stratified by hop count |
| MuSiQue-Full | 500 questions, balanced answerability |
| Custom benchmark | 120 examples, 20 per principal state |

The pilot exists to verify schemas, parsers, prompts, budgets, and metrics. It must not be used to report final claims.

### 8.2 Validation

Use the complete official validation splits where labels are available, plus the custom validation split. Freeze:

- Prompts.
- Probe depth condition.
- Batch size.
- Retrieval budgets.
- Representation builders.
- RRF constant.
- Stopping rules.

### 8.3 Locked final evaluation

Use:

- ConditionalQA official test or a pre-registered held-out document-family split when public test labels are unavailable.
- QASPER official test split.
- MuSiQue official test split or leaderboard evaluation.
- Custom benchmark locked 60% document-family test split.

HotpotQA Distractor full development is reported separately as an optional sanity result.

Do not tune any threshold, prompt, representation parameter, or stopping rule on these final results.

## 9. Required reporting structure

Report separate result tables rather than one average across unrelated datasets.

### Table A: applicability and qualifiers

```text
ConditionalQA + custom applicability subset
```

### Table B: hierarchy and long-document evidence

```text
QASPER + custom natural context-loss subset
```

### Table C: multi-hop coverage and adaptive stopping

```text
MuSiQue-Answerable + MuSiQue-Full
```

### Table D: representation-state diagnosis

```text
Custom benchmark only
```

### Table E: optional external sanity check

```text
HotpotQA Distractor
```

Never average answer F1, condition F1, evidence F1, and state F1 into a single “overall quality” number.

## 10. Final selected bundle

```text
CORE PUBLIC DATASETS

ConditionalQA
  -> conditions, applicability, long policies, not-answerable

QASPER
  -> long scientific papers, hierarchy, evidence, unanswerable

MuSiQue-Answerable + MuSiQue-Full
  -> connected multi-hop evidence, sufficiency, abstention

CORE CUSTOM DATASET

SAR-RAG Representation-State Benchmark
  -> representation selection, context loss, redundancy,
     contradiction, qualifier preservation, insufficiency

OPTIONAL SANITY DATASET

HotpotQA Distractor
  -> standard multi-hop and comparison result
```

This is the smallest defensible bundle that tests every primary claim without adding a second dataset for the same purpose.

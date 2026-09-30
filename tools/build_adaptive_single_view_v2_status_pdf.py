"""Build the corrected SAR-RAG/v2 research-status PDF from verified artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Spacer

from build_audit_pdf import GREY, GREEN, H, MB, MT, MX, NAVY, RED, RULE, W, AMBER, BLUE, para, table

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "output" / "independent_verification"
V2 = ROOT / "output" / "adaptive_single_view_v2"
RUN = V2 / "recalibration_seed20260905"
OUTPUT = ROOT / "output" / "pdf" / "SAR_RAG_RESEARCH_STATUS_V2.pdf"
LABELS = {"RAW": GREEN, "STAT": NAVY, "INFER": GREY, "BLOCKED": RED,
          "LIMITATION": AMBER, "DECISION": BLUE, "CORRECTION": RED}


def tag(label: str) -> str:
    return f'<font color="#{LABELS[label].hexval()[2:]}" size="7"><b>[{label}]</b></font>'


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def image(path: Path, width: float, height: float) -> Image:
    item = Image(str(path), width=width, height=height)
    item.hAlign = "CENTER"
    return item


def build() -> None:
    avail = W - 2 * MX
    corrected = json.loads((VERIFY / "SCREENING_V3_VALIDATION_CORRECTED.json").read_text(encoding="utf-8"))
    v2_manifest = json.loads((RUN / "FROZEN_V2_MANIFEST.json").read_text(encoding="utf-8"))
    decision = json.loads((RUN / "PARETO_AND_DECISION.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((RUN / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    uncertainty = {(row["metric"], row["reference"]): row
                   for row in read_csv(RUN / "PAIRED_UNCERTAINTY.csv")}
    subgroups = read_csv(RUN / "SUBGROUP_RESULTS.csv")
    winner = v2_manifest["selection_metrics"]
    oracle = corrected["oracle_complementarity"]
    reach = corrected["failure_reachability"]["original_full_all_scored"]
    story = []

    story.append(para("SAR-RAG Research Status - Corrected Screening and V2 Recalibration", "title"))
    story.append(para(
        "Critical engineering audit | training-only retrieval evidence | 286-example corrected screening + "
        "225-example disjoint v2 recalibration | <b>zero LLM/Groq calls</b> | 5 September 2026", "sub"))
    story.append(para("Executive conclusion", "h1"))
    story.append(table([
        ["Question", "Verified answer"],
        ["Is RRF one-or-two-view routing supported?", "<b>No.</b> Candidate Oracle@2 adds 0 over Oracle@1; the attainable routing ceiling is too small."],
        ["Did adaptive retrieval survive?", "<b>Yes, in scope.</b> The selected v2 retriever strongly beats the shallow probe."],
        ["What did the frozen v2 rule choose?", "Paragraph, 1,600-token operating point; relevance percentile 0.45; patience 1."],
        ["Does v2 replace v1?", "<b>No.</b> Frozen v1 point-dominates v2 on mean recall and actual tokens; neither paired difference is significant."],
        ["Overall v2 category", "<b>UNSUPPORTED as a replacement claim.</b> Scoped probe/fixed-k trade-offs remain informative."],
        ["Held-out answer quality?", "<b>Unknown and blocked.</b> No paid or held-out answer-generation run occurred."],
    ], [55 * mm, avail - 55 * mm]))
    story.append(para(
        f'{tag("DECISION")} The paper-safe result is a negative routing finding plus a scoped adaptive-retrieval '
        "finding. Development data do not establish answer-quality superiority, and the v2 single-view method "
        "does not outperform frozen v1 at the observed recall-token point.", "body"))
    story.append(para("Evidence labels", "h1"))
    story.append(table([
        ["Label", "Meaning"], ["RAW", "Directly recomputed from per-example artifacts"],
        ["STAT", "Paired test under the frozen analysis"], ["INFER", "Interpretation beyond a direct measurement"],
        ["CORRECTION", "Repairs an earlier population, wording, or comparison error"],
        ["BLOCKED", "Not run; requires explicit authorization"],
    ], [30 * mm, avail - 30 * mm]))

    story.append(PageBreak())
    story.append(para("1. Original hypothesis and version-1 pipeline", "h1"))
    story.append(para(
        "The original hypothesis was that four structured representations - sentence window, paragraph, "
        "semantic, and hierarchical - contain complementary evidence, and that normalized reciprocal-rank "
        "fusion can select one or two useful views per query before adaptive retrieval.", "body"))
    story.append(table([
        ["V1 stage", "Operation", "LLM use"],
        ["1", "Parse documents into canonical structure", "None"],
        ["2", "Generate and embed four representations with all-MiniLM-L6-v2", "None"],
        ["3", "Probe top three per representation; normalized RRF chooses one/two views", "None"],
        ["4", "Adaptive retrieval with relevance, novelty, gap, utility, redundancy, patience and caps", "None"],
        ["5", "Post-retrieval deduplication", "None"],
        ["6", "One final Groq answer-generation call", "One - not used in these studies"],
    ], [16 * mm, avail - 54 * mm, 38 * mm]))
    story.append(para("2. Screening design and corrected audit", "h1"))
    story.append(table([
        ["Check", "Verified value"],
        ["Raw screening table", "9,724 rows = 286 questions x 34 conditions; no duplicate keys"],
        ["Partitions", "194 tuning / 92 selection"],
        ["Evidence-scored", "210 total / 143 tuning / 67 selection"],
        ["Expected undefined metric cells", "2,584 per evidence metric = 76 no-gold questions x 34 conditions"],
        ["Frozen v1 manifest", corrected["frozen_v1_manifest_sha256"]],
        ["Working tree", "Not clean; only the frozen-manifest hash is asserted unchanged"],
    ], [58 * mm, avail - 58 * mm]))
    story.append(para(
        f'{tag("CORRECTION")} Answerable CER now uses 46 scored selection questions. The 21 MuSiQue '
        "insufficient questions with residual support are reported separately. They are never pooled into "
        "the primary CER denominator.", "body"))
    story.append(para(
        f'{tag("CORRECTION")} The former budget Boolean was ambiguous. Absence of a significantly superior '
        "traditional arm does not establish a significant SAR advantage. Comparisons now report actual tokens, "
        "matched actual-token points, and a recall-token Pareto analysis.", "body"))

    story.append(PageBreak())
    story.append(para("3. Corrected screening result: routing rejected", "h1"))
    cand_sel = oracle["hierarchical_v2_candidate"]["selection|both"]
    cand_all = oracle["hierarchical_v2_candidate"]["all|both"]
    legacy_all = oracle["legacy"]["all|both"]
    story.append(table([
        ["Scope", "n", "Oracle@1", "Oracle@2", "Pair gain", "Positive pair-gain examples"],
        ["Candidate selection", cand_sel["n"], f'{cand_sel["oracle_at_1"]:.4f}', f'{cand_sel["oracle_at_2"]:.4f}', f'{cand_sel["mean_pair_complementarity_gain"]:.6f}', cand_sel["examples_with_positive_pair_gain"]],
        ["Candidate pooled", cand_all["n"], f'{cand_all["oracle_at_1"]:.4f}', f'{cand_all["oracle_at_2"]:.4f}', f'{cand_all["mean_pair_complementarity_gain"]:.6f}', cand_all["examples_with_positive_pair_gain"]],
        ["Legacy pooled", legacy_all["n"], f'{legacy_all["oracle_at_1"]:.4f}', f'{legacy_all["oracle_at_2"]:.4f}', f'{legacy_all["mean_pair_complementarity_gain"]:.6f}', legacy_all["examples_with_positive_pair_gain"]],
    ], [34 * mm, 12 * mm, 22 * mm, 22 * mm, 22 * mm, avail - 112 * mm], align=[1,2,3,4,5]))
    story.append(para(
        f'{tag("CORRECTION")} Pair complementarity is exactly zero for the hierarchical-v2 candidate family '
        f'across all {cand_all["n"]} scored examples and for legacy selection. Legacy pooled data contain one '
        f'tuning example with gain {legacy_all["maximum_example_pair_gain"]:.6f}; pooled mean gain is '
        f'{legacy_all["mean_pair_complementarity_gain"]:.6f}. "Zero in every scope" was too broad.', "body"))
    story.append(table([
        ["Component observation", "Verified result"],
        ["Novelty removal", "0/286 final outputs changed"],
        ["Similarity-gap removal", "0/286 final outputs changed; threshold unreachable in screening"],
        ["Marginal utility / redundancy", "0 candidates bound by either gate"],
        ["Deduplication", "Changed k/tokens on 56/286; recall changed on 0 - efficiency only"],
        ["Failure reachability", f'{reach["unreachable_by_any_candidate_single_or_pair"]} of {reach["complete_misses"]} full-system complete misses unreachable by tested candidate singles/pairs'],
    ], [58 * mm, avail - 58 * mm]))
    story.append(para(
        f'{tag("INFER")} Most complete misses are unreachable under the tested encoder, chunking, representation '
        "family, budget, and evidence matching. This does not prove the encoder is universally the dominant "
        "failure or that routing can never work elsewhere.", "body"))

    story.append(PageBreak())
    story.append(para("4. Adaptive Single-View V2: frozen design", "h1"))
    story.append(table([
        ["Design element", "Frozen specification"],
        ["Data", "Official QASPER train and MuSiQue-Full train only"],
        ["Fresh sample", "225 questions, nine evidence-bearing strata, seed 20260905"],
        ["Partitions", "162 tuning / 63 selection, assigned before retrieval outcomes"],
        ["Prior training IDs excluded", "475; overlap = 0"],
        ["Held-out ID-only registry", "7,291 validation/test identifiers; overlap = 0; content/labels not used"],
        ["Candidate representations", "semantic, paragraph, hierarchical_v2, sentence-window"],
        ["Token points", "800, 1,200, 1,600; 2,000 is a safety maximum, not a target"],
        ["Adaptive controls", "depth-3 probe, batch 3, percentile relevance, patience, token/round/candidate/chunk caps"],
        ["Removed from primary v2", "RRF/router/pairs, novelty, similarity gap, marginal utility, redundancy"],
        ["Selection", "Tuning picks one tuple per representation/token point; frozen selection rule chooses among 12"],
    ], [52 * mm, avail - 52 * mm]))
    story.append(para(
        f'{tag("RAW")} The run evaluated 162 x 1,152 = 186,624 tuning configuration-question rows and '
        "63 x 49 = 3,087 selection rows. The selected manifest reproduced identically on all substantive "
        "fields. Final k remained emergent. No Groq or other LLM call occurred.", "body"))
    story.append(para("5. Selected v2 operating point", "h1"))
    cfg = v2_manifest["selected_configuration"]
    story.append(table([
        ["Representation", "Token point", "Rel. percentile", "Patience", "Round cap", "Candidate cap", "Chunk cap"],
        [cfg["representation"], f'{cfg["max_context_tokens"]:,}', cfg["relevance_percentile_threshold"], cfg["empty_batch_patience"], cfg["max_retrieval_rounds"], cfg["max_candidates_examined"], cfg["max_final_chunks"]],
    ], [30 * mm, 23 * mm, 26 * mm, 20 * mm, 20 * mm, 24 * mm, avail - 143 * mm], align=[1,2,3,4,5,6]))
    story.append(table([
        ["Selection metric", "Value", "Population / caveat"],
        ["Unique gold-span recall", f'{winner["evidence_recall"]:.4f}', "63 evidence-scored training-selection questions"],
        ["Answerable CER", f'{winner["answerable_CER"]:.4f}', "42 answerable questions only"],
        ["Residual-support complete recovery", f'{winner["insufficient_residual_support_complete_recovery"]:.4f}', "21 insufficient questions; separate secondary metric"],
        ["Mean actual context", f'{winner["context_tokens"]:.1f} tokens', "75% of 1,600 point"],
        ["Mean final k", f'{winner["final_k"]:.2f}', f'emergent range {decision["final_k_range"][0]}-{decision["final_k_range"][1]}'],
        ["Candidates / rounds", f'{winner["candidates_examined"]:.2f} / {winner["retrieval_rounds"]:.2f}', "means"],
    ], [58 * mm, 35 * mm, avail - 93 * mm]))

    story.append(PageBreak())
    story.append(para("6. Quality-efficiency trade-off", "h1"))
    def u(metric, ref):
        row = uncertainty[(metric, ref)]
        return float(row["difference"]), float(row["ci_low"]), float(row["ci_high"]), float(row["holm_p"])
    probe_r = u("evidence_recall", "probe_only")
    k10_r, k10_t = u("evidence_recall", "fixed_k10"), u("context_tokens", "fixed_k10")
    k20_r, k20_t = u("evidence_recall", "fixed_k20"), u("context_tokens", "fixed_k20")
    story.append(table([
        ["Comparator", "Recall difference (v2-ref)", "95% CI", "Holm p", "Token difference", "Interpretation"],
        ["Same-view probe", f'{probe_r[0]:+.4f}', f'[{probe_r[1]:+.4f}, {probe_r[2]:+.4f}]', f'{probe_r[3]:.4g}', "+941.2", "Large retrieval gain; large context increase"],
        ["Same-view k=10", f'{k10_r[0]:+.4f}', f'[{k10_r[1]:+.4f}, {k10_r[2]:+.4f}]', f'{k10_r[3]:.4g}', f'{k10_t[0]:+.1f}', "Higher recall, but consumes more"],
        ["Same-view k=20", f'{k20_r[0]:+.4f}', f'[{k20_r[1]:+.4f}, {k20_r[2]:+.4f}]', f'{k20_r[3]:.4g}', f'{k20_t[0]:+.1f}', "Recall-noninferior; 17.6% fewer tokens"],
    ], [30 * mm, 33 * mm, 35 * mm, 19 * mm, 25 * mm, avail - 142 * mm]))
    story.append(image(RUN / "RECALL_TOKEN_PARETO.png", 150 * mm, 94 * mm))
    story.append(para(
        f'{tag("DECISION")} V2 is non-dominated when the descriptive v1 point is excluded. With v1 included, '
        "v1 is the sole point-estimate dominator: recall 0.7460 vs 0.7381 and actual tokens 1,147.6 vs "
        "1,199.2. Paired v2-minus-v1 recall is -0.0079, CI [-0.0675, +0.0476]; token difference is +51.6, "
        "CI [-2.9, +100.3]. Neither is significant. Different nominal maxima further limit a formal claim.", "body"))

    story.append(PageBreak())
    story.append(para("7. Behavior, subgroup limitations, and failures", "h1"))
    story.append(image(RUN / "FINAL_K_HISTOGRAM.png", 78 * mm, 50 * mm))
    story.append(image(RUN / "STOP_REASON_DISTRIBUTION.png", 92 * mm, 52 * mm))
    story.append(para("Selection subgroups", "h2"))
    dataset_rows = [row for row in subgroups if row["group_type"] in {"dataset", "answerability"}]
    story.append(table([["Group", "n", "Recall", "Complete recovery", "Tokens", "Final k"]] + [
        [row["group"], row["n"], f'{float(row["evidence_recall"]):.4f}',
         f'{float(row["complete_evidence"]):.4f}', f'{float(row["context_tokens"]):.1f}',
         f'{float(row["final_k"]):.2f}'] for row in dataset_rows],
        [42 * mm, 12 * mm, 24 * mm, 30 * mm, 25 * mm, avail - 133 * mm], align=[1,2,3,4,5]))
    story.append(para(
        f'{tag("LIMITATION")} QASPER recall is 0.4286 (n=21) versus MuSiQue 0.8929 (n=42). These small, '
        "stratified development cells are not standalone evidence. The contrast is a failure-analysis signal, "
        "not a dataset-general conclusion.", "body"))
    story.append(table([
        ["Failure class", "Count / 63"],
        ["Complete recovery", decision["failure_counts"]["complete"]],
        ["Partial recovery", decision["failure_counts"]["partial"]],
        ["Total miss", decision["failure_counts"]["total_miss"]],
    ], [70 * mm, avail - 70 * mm]))

    story.append(PageBreak())
    story.append(para("8. Limitations, blocked paid work, and paper-safe conclusion", "h1"))
    story.append(table([
        ["Limitation", "Consequence"],
        ["Training selection is design-exposed", "All v2 results are developmental; they cannot support an unbiased confirmatory claim."],
        ["Gold-span substring matching", "Paraphrased support may be missed; nDCG depends on representation granularity."],
        ["Selection among 12 candidates", "Selection estimates are optimistic; paired tests on the same partition do not remove selection bias."],
        ["Different v1/v2 nominal maxima", "Actual-token Pareto dominance is descriptive, not a controlled proof of v1 superiority."],
        ["No answer generation", "Retrieval gains may not translate to answer token F1 or user-visible quality."],
        ["Current encoder/representation family", "Negative routing evidence is scoped; it is not a universal impossibility result."],
    ], [54 * mm, avail - 54 * mm]))
    story.append(para("Paid work remains blocked", "h1"))
    story.append(table([
        ["Gate", "Status"],
        ["Retrieval-only v2 artifacts, tests, reproduction", "Complete"],
        ["Answer prompt/model fingerprints and verified run-date rates", "Pending"],
        ["Explicit paid smoke authorization with call/cost ceiling", "Not provided"],
        ["Separate final held-out authorization", "Not provided"],
        ["Paid smoke / held-out answer evaluation", "<b>NOT RUN</b>"],
    ], [72 * mm, avail - 72 * mm]))
    story.append(para(
        f'{tag("BLOCKED")} No paid smoke test and no held-out answer-generation evaluation may run without '
        "explicit authorization. The current artifacts contain zero LLM calls.", "body"))
    story.append(para("Paper-safe conclusion", "h1"))
    story.append(para(
        "The tested RRF-based one-or-two-representation routing mechanism is not supported under the current "
        "encoder, representation family, datasets, and retrieval configuration. A fresh training-only "
        "recalibration selected adaptive paragraph retrieval and confirmed that retrieving beyond a shallow "
        "probe materially improves evidence recall. It also found a quality-efficiency trade-off against "
        "same-view fixed-k comparators. However, frozen v1 point-dominates the selected v2 operating point on "
        "mean recall and actual context, with no significant paired difference. The v2 replacement claim is "
        "therefore unsupported. Whether either method improves held-out answer quality remains unproven.", "body"))
    story.append(Spacer(1, 5))
    story.append(para(
        f'Run fingerprint: {run_manifest["experiment_fingerprint"]}<br/>'
        f'V1 manifest SHA-256: {corrected["frozen_v1_manifest_sha256"]}<br/>'
        f'V2 selected configuration SHA-256: {v2_manifest["selected_configuration_fingerprint"]}', "note"))

    def decorate(canvas, doc):
        canvas.saveState(); canvas.setStrokeColor(RULE); canvas.setLineWidth(0.5)
        canvas.line(MX, H - MT + 5 * mm, W - MX, H - MT + 5 * mm)
        canvas.line(MX, MB - 4 * mm, W - MX, MB - 4 * mm)
        canvas.setFont("Helvetica", 7); canvas.setFillColor(GREY)
        canvas.drawString(MX, H - MT + 6.5 * mm, "SAR-RAG - corrected screening and v2 status")
        canvas.drawRightString(W - MX, H - MT + 6.5 * mm, "Training-only retrieval | zero LLM calls")
        canvas.drawString(MX, MB - 8 * mm, "Development evidence only. No held-out superiority claim.")
        canvas.drawRightString(W - MX, MB - 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=MX, rightMargin=MX,
                               topMargin=MT, bottomMargin=MB,
                               title="SAR-RAG Research Status - Corrected Screening and V2")
    frame = Frame(MX, MB, W - 2 * MX, H - MT - MB, id="main",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    document.addPageTemplates([PageTemplate(id="std", frames=[frame], onPage=decorate)])
    document.build(story)
    print(json.dumps({"output": str(OUTPUT), "bytes": OUTPUT.stat().st_size}))


if __name__ == "__main__":
    build()

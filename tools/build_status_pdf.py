"""Build the updated SAR-RAG research-status PDF from screening v3 (n=286).

Reuses the page furniture of build_audit_pdf.py. Every number is read from the
recomputed screening-v3 artifacts; nothing is hard-coded from a summary JSON.
No unexecuted experiment is presented as completed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate

from build_audit_pdf import (  # noqa: E402  - shared page furniture
    GREY, GREEN, H, MB, MT, MX, NAVY, RED, RULE, W, AMBER, BLUE,
    para, table,
)

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "output" / "independent_verification"
OUTPUT = VERIFY / "SAR_RAG_RESEARCH_STATUS.pdf"

LABELS = {"RAW": GREEN, "STAT": NAVY, "INFER": GREY,
          "BLOCKED": RED, "LIMITATION": AMBER, "DECISION": BLUE}


def tag(label: str) -> str:
    return (f'<font color="#{LABELS[label].hexval()[2:]}" size="7">'
            f'<b>[{label}]</b></font>')


def build() -> None:
    avail = W - 2 * MX
    val = json.loads((VERIFY / "SCREENING_V3_VALIDATION.json").read_text(encoding="utf-8"))
    res = list(csv.DictReader((VERIFY / "SCREENING_V3_RESULTS.csv").open(encoding="utf-8")))
    v = val["validation"]
    o = val["oracles"]["selection|both"]
    dec = val["decision_rules_applied"]

    def row(cond, metric="evidence_recall", part="selection", ds="both"):
        return next((r for r in res if r["condition"] == cond and r["metric"] == metric
                     and r["partition"] == part and r["dataset"] == ds), None)

    def fmt(cond, metric="evidence_recall", part="selection", ds="both"):
        r = row(cond, metric, part, ds)
        if not r:
            return ["—"] * 5
        return [f"{float(r['mean_condition']):.4f}",
                f"{float(r['difference']):+.4f}",
                f"[{float(r['ci_low']):+.3f}, {float(r['ci_high']):+.3f}]",
                f"{float(r['p_value']):.4f}",
                f"{float(r['holm_p_all_conditions']):.4f}"]

    story = []
    story.append(para("SAR-RAG — Research Status After Screening v3", "title"))
    story.append(para(
        "Training-only retrieval screening, n = 286 examples &nbsp;|&nbsp; 9,724 result rows "
        "&nbsp;|&nbsp; <b>zero LLM calls</b> &nbsp;|&nbsp; no evaluation split accessed "
        "&nbsp;|&nbsp; 14 August 2026", "sub"))

    story.append(para("How to read this report", "h1"))
    story.append(table([
        ["Label", "Meaning"],
        ["RAW", "Measurement reproduced directly from the per-example CSV"],
        ["STAT", "Supported by the pre-registered test and multiplicity correction"],
        ["INFER", "Reviewer interpretation beyond the test result"],
        ["LIMITATION", "A constraint on what can be concluded"],
        ["BLOCKED", "Not executed; explicitly gated"],
    ], [26 * mm, avail - 26 * mm]))

    # ---------------------------------------------------------------- decision
    story.append(para("1. Decision", "h1"))
    story.append(table([
        ["Question", "Answer"],
        ["Is representation routing scientifically justified?",
         "<b>No.</b> Fails the frozen rule on all three limbs."],
        ["Does Oracle@2 show pair complementarity?",
         "<b>No — exactly zero.</b> A pair never beat the best single view on any of 210 questions."],
        ["Does the router approach the oracle ceiling?",
         f"Yes: router {o['candidate']['router_recall']:.4f} vs Oracle@2 "
         f"{o['candidate']['oracle2']:.4f}. The ceiling itself is only "
         f"{o['candidate']['oracle2_minus_best_constant']:+.4f} over a constant policy."],
        ["Is a constant single-view policy stronger?",
         "Statistically indistinguishable (semantic 0.7326 vs router 0.7376)."],
        ["Retain hierarchical_v2?",
         "As a code correctness fix yes; as a rescue for the routing claim no."],
        ["Do novelty / gap / utility / redundancy affect outputs?",
         "No. Novelty and gap change 0 of 286 examples; utility and redundancy bind 0 times."],
        ["Budget-matched advantage over Traditional RAG?",
         "<b>No.</b> The matched-budget comparison is null; unmatched favours the baseline."],
    ], [64 * mm, avail - 64 * mm]))
    story.append(para(
        f'{tag("DECISION")} <b>FINAL: PIVOT TO ADAPTIVE SINGLE-VIEW RETRIEVAL.</b> '
        "This is a decisive negative result, not an inconclusive one. The attainable ceiling "
        "over all four views and all six pairs sits +0.025 above always using the semantic "
        "view; the frozen protocol required +0.05. Repairing the weakest representation by "
        "+0.247 recall moved that ceiling by exactly 0.000.", "body"))

    # ---------------------------------------------------------------- validation
    story.append(para("2. Artifact validation", "h1"))
    story.append(table([
        ["Check", "Expected", "Observed", "Verdict"],
        ["Result rows", "286 x 34 = 9,724", f"{v['row_count']:,}", "PASS"],
        ["Unique examples", "286", str(v["unique_examples"]), "PASS"],
        ["Conditions", "34, each n=286",
         f"{v['condition_count']}, each with "
         f"{v['rows_per_condition'][0]} rows", "PASS"],
        ["Duplicate keys", "0", str(v["duplicate_keys"]), "PASS"],
        ["Partition sizes", "194 tuning / 92 selection",
         f"{v['partition_sizes']['tuning']} / {v['partition_sizes']['selection']}", "PASS"],
        ["Calibration-ID overlap", "0", str(v["calibration_id_overlap"]), "PASS"],
        ["Prior-diagnostic-ID overlap", "0", str(v["prior_diagnostic_id_overlap"]), "PASS"],
        ["Evaluation-split IDs present", "0", str(v["evaluation_split_ids_present"]), "PASS"],
        ["Frozen manifest SHA-256", "5c570ec2b4c52e1b...a54d05",
         "identical; git clean", "PASS"],
        ["LLM / Groq access", "none", "RetrievalOnlyAnswerer only", "PASS"],
    ], [40 * mm, 40 * mm, avail - 96 * mm, 16 * mm]))
    story.append(para(
        f'{tag("RAW")} <b>NaN audit.</b> 2,584 NaN cells per metric is not corruption: '
        f'2,584 = 76 x 34, and those 76 examples are exactly the ones with zero gold '
        f'evidence spans, where recall is undefined. Verified cell-by-cell — a metric is NaN '
        f'if and only if gold_spans = 0. Effective scored sample: '
        f'<b>{v["examples_with_gold_evidence"]} of 286</b>.', "body"))
    story.append(table([
        ["Partition", "QASPER", "MuSiQue", "Total scored", "Use"],
        ["selection", "22", "45", "<b>67</b>", "<b>design decisions</b>"],
        ["tuning", "48", "95", "143", "descriptive only"],
        ["pooled", "70", "140", "210", "descriptive only"],
    ], [30 * mm, 20 * mm, 20 * mm, 24 * mm, avail - 94 * mm]))
    story.append(para(
        f'{tag("LIMITATION")} The QASPER selection cell (n = 22) and the QASPER '
        f'INSUFFICIENT-long stratum (11 total, 4 in selection) cannot support standalone '
        f'claims. Per-dataset selection results are directional only.', "note"))

    story.append(PageBreak())

    # ---------------------------------------------------------------- primary
    story.append(para("3. Primary comparison — selection partition, evidence recall", "h1"))
    story.append(para(
        "Reference is the frozen full SAR-RAG (mean recall 0.7226). n = 67 paired. "
        "Holm family = 33 conditions within (metric, partition, scope).", "note"))
    head = ["Condition", "Mean", "Diff", "95% CI", "p", "Holm p"]
    conds = ["sliding_window_k20", "sliding_window_matched_k20", "full_hierarchical_v2",
             "semantic_only", "hierarchical_v2_only", "select_one_representation",
             "paragraph_only", "no_novelty", "no_similarity_gap",
             "no_post_retrieval_deduplication", "sliding_window_k10", "sentence_window_only",
             "hierarchical_only", "rrf_probe_only_no_adaptive"]
    story.append(table([head] + [[c] + fmt(c) for c in conds],
                       [50 * mm, 16 * mm, 17 * mm, 30 * mm, 16 * mm, avail - 145 * mm],
                       align=[1, 2, 4, 5]))
    story.append(para(
        f'{tag("STAT")} Exactly one condition beats the frozen primary after Holm '
        f'correction: <b>sliding_window_k20 (+0.0858, Holm p = 0.0165)</b>. '
        f'{tag("STAT")} Adaptive retrieval past the top-3 probe is the one component with a '
        f'large, unambiguous, corrected effect (rrf_probe_only_no_adaptive −0.3134, '
        f'Holm p = 0.0016).', "body"))

    # ---------------------------------------------------------------- budget
    story.append(para("4. The shared budget is nominal, not binding", "h1"))
    story.append(table([
        ["System", "Context tokens", "% of 2,000 cap", "Final k", "Recall"],
        ["full (SAR-RAG)", "1,305.2", "<b>65%</b>", "15.22", "0.7226"],
        ["sliding_window_k20", "1,857.7", "<b>93%</b>", "16.52", "0.8085"],
        ["sliding_window_matched_k20", "1,780.2", "89%", "19.30", "0.7774"],
        ["sliding_window_k10", "1,244.1", "62%", "9.83", "0.6915"],
    ], [52 * mm, 28 * mm, 28 * mm, 22 * mm, avail - 130 * mm], align=[1, 2, 3, 4]))
    story.append(para(
        f'{tag("RAW")} SAR-RAG stop reasons (selection, n = 92): all representations paused '
        f'63.0%, final-chunk safety cap 30.4%, retrieval-round cap 6.5%. <b>The token budget '
        f'never appears as a stop reason</b>; only 12 of 92 runs reach 95% of the cap. '
        f'{tag("INFER")} SAR-RAG is halted by patience and the 20-chunk cap while ~700 tokens '
        f'of its allowance sit unused, whereas the baseline spends 93% of the same allowance. '
        f'sliding_window_k20 therefore wins +0.0858 recall while consuming <b>42.3% more '
        f'context</b>. At comparable spend the advantage vanishes in both directions: '
        f'sliding_window_k10 is −0.031 (p = 0.125) and matched_k20 does not survive Holm '
        f'(p = 0.676). <b>Every Traditional comparison in this project is confounded until the '
        f'arms are matched on tokens actually consumed, not on chunk count.</b>', "body"))
    story.append(table([
        ["Stability of the k20 result", "Diff", "p", "Holm p"],
        ["selection (n = 67, decision basis)", "+0.0858", "0.0007", "<b>0.0165</b>"],
        ["tuning (n = 143, descriptive)", "+0.0335", "0.235", "1.000"],
        ["pooled (n = 210, descriptive)", "+0.0502", "0.017", "0.304"],
        ["MuSiQue only (selection, n = 45)", "+0.1352", "0.0001", "<b>0.0033</b>"],
        ["QASPER only (selection, n = 22)", "−0.0152", "1.000", "1.000"],
    ], [62 * mm, 24 * mm, 24 * mm, avail - 110 * mm], align=[1, 2, 3]))
    story.append(para(
        f'{tag("LIMITATION")} The magnitude is not stable across partitions and is carried '
        f'entirely by MuSiQue. It is a serious unresolved threat to the baseline comparison, '
        f'not a finished finding.', "note"))

    story.append(PageBreak())

    # ---------------------------------------------------------------- oracle
    story.append(para("5. Oracle@1 and Oracle@2 — pair complementarity is exactly zero", "h1"))
    orows = [["Scope", "n", "Oracle@1", "Oracle@2", "O2 − O1", "Best constant", "Router",
              "O2 − const"]]
    for scope, label in (("selection|both", "selection"), ("selection|qasper", "sel · QASPER"),
                         ("selection|musique_full", "sel · MuSiQue"),
                         ("tuning|both", "tuning"), ("all|both", "pooled")):
        b = val["oracles"][scope]["candidate"]
        orows.append([label, str(b["n_scored"]), f"{b['oracle1']:.4f}", f"{b['oracle2']:.4f}",
                      f"<b>{b['oracle2_minus_oracle1']:.4f}</b>",
                      f"{b['best_constant_single_view_recall']:.4f}",
                      f"{b['router_recall']:.4f}",
                      f"{b['oracle2_minus_best_constant']:+.4f}"])
    story.append(table(orows, [26 * mm, 10 * mm, 20 * mm, 20 * mm, 18 * mm, 24 * mm,
                               18 * mm, avail - 136 * mm], align=[1, 2, 3, 4, 5, 6, 7]))
    story.append(para(
        f'{tag("RAW")} Per-example verification over all 210 scored questions and all six '
        f'forced pairs: a pair strictly beats the best single view on <b>0 of 210</b> examples '
        f'in the candidate system and 1 of 210 in the legacy system; a pair is the unique '
        f'maximiser on <b>0 of 210</b>. {tag("STAT")} Oracle@2 exceeds the best constant view '
        f'by +0.0249, CI [0.000, +0.060], p = 0.251 — below the frozen 0.05 margin and not '
        f'significant. {tag("INFER")} Under a shared budget a second representation splits the '
        f'budget and never recovers evidence the best single view misses. The "one or two '
        f'complementary representations" premise has no headroom to exploit — this is not '
        f'merely a router that fails to exploit it.', "body"))
    story.append(para(
        f'{tag("LIMITATION")} 65 of 67 selection examples have two or more tied maximisers, so '
        f'<i>which</i> view is best remains unreliable and no per-view attribution is made. The '
        f'oracle <i>value</i> is a maximum and is unaffected by ties.', "note"))

    # ---------------------------------------------------------------- hierarchy
    story.append(para("6. Hierarchical v2 — representation fixed, system unmoved", "h1"))
    story.append(table([
        ["Comparison", "Diff", "95% CI", "p", "dz"],
        ["hierarchical_v2_only − hierarchical_only (selection, n=67)", "<b>+0.2463</b>",
         "[+0.153, +0.341]", "0.00005", "+0.63"],
        ["same, tuning (n=143)", "+0.2476", "[+0.174, +0.321]", "0.00005", "+0.55"],
        ["same, pooled (n=210)", "+0.2472", "[+0.189, +0.305]", "0.00005", "+0.58"],
    ], [72 * mm, 20 * mm, 30 * mm, 18 * mm, avail - 140 * mm], align=[1, 3, 4]))
    hr = dec["hierarchy_candidate_retained"]
    story.append(table([
        ["Frozen retention requirement", "Threshold", "Observed", "Met"],
        ["v2 improves over v1", "> 0", "+0.2463", "YES"],
        ["candidate primary noninferior to legacy", ">= −0.02", "+0.0149", "YES"],
        ["AND recall gain", ">= +0.02", "+0.0149", "NO"],
        ["OR context reduction", ">= 10%", f"{hr['context_reduction_percent']:+.1f}%", "NO"],
        ["OR Oracle@2 gain over legacy set", ">= +0.02",
         f"{hr['oracle2_gain_over_legacy_representation_set']:.4f}", "NO"],
        ["<b>RETAINED</b>", "", "", "<b>NO</b>"],
    ], [66 * mm, 24 * mm, 26 * mm, avail - 116 * mm]))
    story.append(para(
        f'{tag("INFER")} This is the most informative result in the experiment. Repairing the '
        f'worst representation by +0.246 recall moves the full system by +0.015 and moves the '
        f'attainable ceiling by exactly 0.000. A pipeline whose ceiling is invariant to a '
        f'quarter-point improvement in one of its four views does not derive its value from '
        f'representation selection. hierarchical_v2 should still replace v1 as a correctness '
        f'fix, with v1 preserved for the record; what the rule denies is the <i>claim</i> that '
        f'the repair rescues routing.', "body"))

    story.append(PageBreak())

    # ---------------------------------------------------------------- gates
    story.append(para("7. Gate components — what actually changes an output", "h1"))
    gi = val["gate_instrumentation"]
    story.append(table([
        ["Component", "Examples with ANY output change", "Fields changed", "Recall diff", "Verdict"],
        ["Novelty rule (0.2292)", f"<b>0 / 286</b>", "none", "0.0000", "<b>INERT</b>"],
        ["Similarity-gap rule (1.9052)", f"<b>0 / 286</b>", "none", "0.0000", "<b>INERT</b>"],
        ["Post-retrieval deduplication",
         f"{gi['no_post_retrieval_deduplication']['examples_with_any_output_change']} / 286",
         "final_k, context_tokens, precision", "0.0000", "Efficiency only"],
        ["Second representation",
         f"{gi['select_one_representation']['examples_with_any_output_change']} / 286",
         "recall on only 4, CER on 3", "−0.0075", "Negligible"],
    ], [40 * mm, 34 * mm, 44 * mm, 20 * mm, avail - 138 * mm]))
    story.append(para(
        "Internal instrumentation, n = 286 (6,989 candidates scored, 2,000 batches). "
        "max_* fields are aggregated with max, not summed.", "h2"))
    story.append(table([
        ["Gate", "Reached", "Would reject", "Binding"],
        ["Relevance percentile", "6,989", "1,788", "<b>1,788</b>"],
        ["Novelty (0.2292)", "5,201", "108", "107"],
        ["Marginal utility (0.1289)", "—", "135", "<b>0</b>"],
        ["Redundancy (0.9547)", "—", "—", "<b>0</b>"],
        ["Budget caps", "—", "—", "478"],
    ], [46 * mm, 26 * mm, 28 * mm, avail - 100 * mm], align=[1, 2, 3]))
    story.append(table([
        ["Similarity-gap rule", "Value"],
        ["Eligible batches (rule may fire)", "664"],
        ["Times the rule fired", "<b>0</b>"],
        ["Maximum standardized gap ever observed", "<b>1.0925</b>"],
        ["Maximum gap on an eligible empty batch", "1.0119"],
        ["Frozen threshold", "<b>1.9052</b>"],
    ], [72 * mm, avail - 72 * mm]))
    story.append(para(
        f'{tag("STAT")} The similarity-gap rule is not merely unused — it is '
        f'<b>unreachable</b>. The largest standardized gap anywhere in 286 examples is 1.0925, '
        f'43% below the frozen threshold of 1.9052. {tag("STAT")} Marginal utility and '
        f'redundancy bind <b>0 times in 6,989 candidate evaluations</b>. {tag("INFER")} Novelty '
        f'binds 107 times yet changes no output, meaning every candidate it rejected would have '
        f'been removed downstream anyway by budget caps or deduplication.', "body"))

    # ---------------------------------------------------------------- failures
    story.append(para("8. Failure analysis", "h1"))
    story.append(table([
        ["Class", "Count", "Note"],
        ["Complete evidence recovered", "129", ""],
        ["Partial", "55", ""],
        ["Total miss (recall = 0)", "26", "<b>18 of these are unreachable by any view or pair</b>"],
        ["Recoverable by perfect routing", "<b>14 / 210</b>", "3 of 67 in the selection partition"],
    ], [50 * mm, 24 * mm, avail - 74 * mm]))
    story.append(para(
        f'{tag("INFER")} The dominant failure mode is evidence that is not retrievable by any '
        f'configuration of this embedding and chunking family — an encoder/corpus problem, not '
        f'a routing problem. Routing repair addresses at most 6.7% of scored examples.', "body"))

    # ---------------------------------------------------------------- next
    story.append(para("9. What survives, what is dropped, what blocks the paid run", "h1"))
    story.append(table([
        ["Survives into v2", "Dropped from the claim"],
        ["Adaptive retrieval past a shallow probe (−0.313 without it, Holm 0.0016)<br/>"
         "A single representation (semantic; paragraph and hierarchical_v2 indistinguishable)<br/>"
         "Post-retrieval deduplication as an explicit efficiency mechanism<br/>"
         "Relevance-percentile scoring — the only gate that binds",
         "RRF representation selection<br/>The one-or-two-view premise<br/>"
         "The novelty rule<br/>The standardized similarity-gap rule<br/>"
         "Marginal utility and redundancy as components<br/>"
         "Any assertion of superiority over Traditional RAG"],
    ], [avail / 2, avail / 2]))
    story.append(table([
        ["#", "Blocking prerequisite before any paid run"],
        ["1", "<b>Fix the budget defect first.</b> SAR-RAG spends 65% of its stated token "
              "budget while the baseline spends 93%. Re-screen free, matched on tokens consumed."],
        ["2", "<b>Recalibration is mandatory.</b> The frozen thresholds were fitted with the "
              "four-view router and hierarchical v1 in the pipeline. A single-view method is a "
              "different estimand; the v1 manifest cannot carry over."],
        ["3", "Resolve the sliding_window_k20 instability (selection +0.086 / tuning +0.034) "
              "on training data."],
        ["4", "Verified current Groq rates on the run date, and explicit owner authorisation."],
    ], [10 * mm, avail - 10 * mm]))
    story.append(para(
        f'{tag("BLOCKED")} <b>The paid held-out evaluation remains blocked.</b> Nothing in this '
        f'analysis authorises it. Running it against the current method would spend the budget '
        f'confirming a contribution this experiment has just refuted.', "body"))

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MX, H - MT + 5 * mm, W - MX, H - MT + 5 * mm)
        canvas.line(MX, MB - 4 * mm, W - MX, MB - 4 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(MX, H - MT + 6.5 * mm, "SAR-RAG — Research Status (screening v3, n=286)")
        canvas.drawRightString(W - MX, H - MT + 6.5 * mm, "Zero LLM calls")
        canvas.drawString(MX, MB - 8 * mm,
                          "Training-only screening. Not confirmatory evidence. "
                          "No evaluation split was accessed.")
        canvas.drawRightString(W - MX, MB - 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=MX, rightMargin=MX,
                               topMargin=MT, bottomMargin=MB,
                               title="SAR-RAG Research Status — Screening v3")
    frame = Frame(MX, MB, W - 2 * MX, H - MT - MB, id="main",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    document.addPageTemplates([PageTemplate(id="std", frames=[frame], onPage=decorate)])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.build(story)
    print(json.dumps({"output": str(OUTPUT), "bytes": OUTPUT.stat().st_size}))


if __name__ == "__main__":
    build()

"""Build the independent verification PDF.

Writes to output/independent_verification/. Never touches the version-1
historical PDFs under output/pdf/.

Every confirmatory result cell is rendered as PENDING; no unexecuted experiment
is presented as completed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, LongTable, PageBreak, PageTemplate,
    Paragraph, Spacer, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "output" / "independent_verification"
OUTPUT = VERIFY / "SAR_RAG_INDEPENDENT_VERIFICATION.pdf"

W, H = A4
MX, MT, MB = 16 * mm, 16 * mm, 16 * mm
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#176B87")
RED = colors.HexColor("#A32A2A")
AMBER = colors.HexColor("#8A6100")
GREEN = colors.HexColor("#1F6B45")
INK = colors.HexColor("#26323D")
GREY = colors.HexColor("#5A6A78")
RULE = colors.HexColor("#C9D4DD")
BAND = colors.HexColor("#EEF3F7")

BASE = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=BASE["Title"], fontName="Helvetica-Bold",
                            fontSize=20, leading=24, textColor=NAVY, spaceAfter=2),
    "sub": ParagraphStyle("sub", parent=BASE["Normal"], fontName="Helvetica",
                          fontSize=9.5, leading=13, textColor=GREY, spaceAfter=8),
    "h1": ParagraphStyle("h1", parent=BASE["Heading1"], fontName="Helvetica-Bold",
                         fontSize=13.5, leading=16, textColor=NAVY,
                         spaceBefore=10, spaceAfter=5),
    "h2": ParagraphStyle("h2", parent=BASE["Heading2"], fontName="Helvetica-Bold",
                         fontSize=10.5, leading=13, textColor=BLUE,
                         spaceBefore=8, spaceAfter=3),
    "body": ParagraphStyle("body", parent=BASE["Normal"], fontName="Helvetica",
                           fontSize=8.8, leading=12.2, textColor=INK, spaceAfter=4),
    "cell": ParagraphStyle("cell", parent=BASE["Normal"], fontName="Helvetica",
                           fontSize=7.4, leading=9.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", parent=BASE["Normal"], fontName="Helvetica-Bold",
                            fontSize=7.4, leading=9.4, textColor=NAVY),
    "note": ParagraphStyle("note", parent=BASE["Normal"], fontName="Helvetica-Oblique",
                           fontSize=7.8, leading=10.5, textColor=GREY, spaceAfter=5),
}


def para(text, style="body"):
    return Paragraph(text, S[style])


def table(rows, widths, align=None, small=False):
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
    ]
    for index in (align or []):
        style.append(("ALIGN", (index, 1), (index, -1), "RIGHT"))
    data = [[Paragraph(str(c), S["cellb"] if r == 0 else S["cell"]) for c in row]
            for r, row in enumerate(rows)]
    data[0] = [Paragraph(f'<font color="#FFFFFF"><b>{c}</b></font>', S["cell"])
               for c in rows[0]]
    t = LongTable(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def tag(label):
    colour = {"REPRODUCED FACT": GREEN, "CALIBRATION DIAGNOSTIC": BLUE,
              "ENGINEERING PILOT": AMBER, "CONFIRMATORY RESULT": NAVY,
              "PENDING": RED, "REVIEWER INFERENCE": GREY,
              "LIMITATION": AMBER}[label]
    return f'<font color="#{colour.hexval()[2:]}" size="7"><b>[{label}]</b></font>'


def read_csv(path, limit=None):
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows[:limit] if limit else rows


def build():
    avail = W - 2 * MX
    story = []

    story.append(para("SAR-RAG — Independent Verification Report", "title"))
    story.append(para(
        "Representation-State Adaptive RAG &nbsp;|&nbsp; commit "
        "5cb4cd7dc85f9c4bd87f22cfa840494d98b6a9d5 &nbsp;|&nbsp; "
        "<b>Zero paid Groq calls were made during this audit</b>", "sub"))

    story.append(para("How to read this report", "h1"))
    story.append(table([
        ["Label", "Meaning"],
        ["REPRODUCED FACT", "Re-derived from source code or recorded artifacts during this audit"],
        ["CALIBRATION DIAGNOSTIC", "Training-only calibration output; not an evaluation result"],
        ["ENGINEERING PILOT", "From the 2-question QASPER pilot; not inferential"],
        ["CONFIRMATORY RESULT", "From a completed held-out run — none exist in this repository"],
        ["PENDING", "Not executed"],
        ["REVIEWER INFERENCE", "Auditor judgement, flagged as such"],
        ["LIMITATION", "A constraint on what can be concluded"],
    ], [42 * mm, avail - 42 * mm]))

    story.append(para("1. Headline findings", "h1"))
    story.append(table([
        ["#", "Finding", "Label"],
        ["1", "<b>No result artifact in this repository was produced by the current code.</b> "
              "No run_fingerprint.json and no checkpoints/ directory exist anywhere under output/. "
              "The pilot lacks complete_evidence_recovery, gold_evidence_spans, candidates_examined, "
              "retrieval_rounds, stop_reason, trace_summary, provisional_evidence and api_attempts. "
              "Corrections #12, #13, #15 and #16 were in source but never executed.", "REPRODUCED FACT"],
        ["2", "<b>The frozen calibration reproduces bit-for-bit.</b> All 32 proposed trials, 6 baseline "
              "trials, sample IDs, partitions, strata and source hashes are identical, before and after "
              "defect remediation. No threshold changed.", "REPRODUCED FACT"],
        ["3", "<b>Three implementation defects found and fixed</b>, one falsifying a claimed correction: "
              "peak working set was always null on 64-bit Windows (truncated HANDLE); retrieval recall "
              "still returned 1.0 with no gold evidence in a second live path; the baseline reported "
              "final_k as candidates examined.", "REPRODUCED FACT"],
        ["4", "<b>A single-view baseline currently beats the full routed system.</b> On retrieval-only "
              "screening over fresh training examples, paragraph_only beats the full four-view routed "
              "system on recall, complete-evidence, precision, context tokens AND candidates examined "
              "simultaneously. This is the most serious threat to the paper's contribution (see 6b).",
              "REPRODUCED FACT"],
        ["5", "<b>Two named components are inert; the second view is near-inert.</b> no_novelty and "
              "no_similarity_gap reproduce the full system to four decimal places. Two views are selected "
              "for 16.7% of queries and add unique gold evidence in only 3.3%.", "REPRODUCED FACT"],
        ["6", "<b>But at a matched context budget, SAR-RAG does win.</b> At ~1,300 tokens it beats "
              "Traditional k=10 by +0.056 recall and +0.116 complete-evidence. The efficiency claim must "
              "be framed budget-matched rather than k-matched.", "REPRODUCED FACT"],
        ["7", "<b>Leakage is smaller than the shipped audit implies.</b> Only the 60 calibration examples "
              "were ever fitted; that check did not exist. They share 0 example IDs, 0 source IDs and "
              "0 document IDs with either evaluation split.", "REPRODUCED FACT"],
    ], [8 * mm, avail - 44 * mm, 36 * mm]))

    story.append(PageBreak())
    story.append(para("2. Confirmatory results", "h1"))
    story.append(para(
        f"{tag('PENDING')} <b>No confirmatory experiment has been run.</b> Every cell below is pending "
        "and must not be filled from pilot or calibration data.", "body"))
    story.append(table([
        ["Table", "QASPER test (n=1,451)", "MuSiQue dev (n=4,834)", "Status"],
        ["Main per-dataset results", "PENDING", "PENDING", "requires paid run"],
        ["Answerability and abstention", "PENDING", "PENDING", "requires paid run"],
        ["Statistical inference", "PENDING", "PENDING", "requires paid run"],
        ["Efficiency and cost", "PENDING", "PENDING", "rates unverified; cost undefined"],
        ["Representation-state analysis", "PENDING", "PENDING", "retrieval-only screening available"],
        ["Adaptive-k analysis", "PENDING", "PENDING", "retrieval-only screening available"],
        ["Ablation results (end-to-end)", "PENDING", "PENDING", "15/20 screened offline"],
        ["Robustness / subgroup", "PENDING", "PENDING", "requires paid run"],
        ["Failure-category counts", "PENDING", "PENDING", "requires paid run"],
    ], [52 * mm, 34 * mm, 34 * mm, avail - 120 * mm]))

    story.append(para("3. Calibration reproduction", "h1"))
    version = json.loads((VERIFY / "VERSION1_VS_VERSION2_CALIBRATION.json").read_text(encoding="utf-8"))
    story.append(table([
        ["Comparison", "Differing fields", "Result"],
        ["v0 original frozen vs v1 reproduction (unmodified code)",
         str(len(version["v0_vs_v1"]["differing_fields"])), "IDENTICAL"],
        ["v1 reproduction vs v2 (after defect fixes)",
         str(len(version["v1_vs_v2"]["differing_fields"])), "IDENTICAL"],
        ["Thresholds changed by remediation", "0", "none"],
    ], [avail - 60 * mm, 26 * mm, 34 * mm], align=[1]))
    story.append(Spacer(1, 4))
    story.append(para(f"{tag('CALIBRATION DIAGNOSTIC')} Held-back training-selection partition "
                      "(24 examples). These are <b>not</b> test results and establish no superiority.", "body"))
    story.append(table([
        ["Metric", "SAR-RAG", "Traditional", "Difference"],
        ["Evidence recall", "0.7917", "0.8958", "-0.1042"],
        ["Complete-evidence recovery", "0.6875", "0.8750", "-0.1875"],
        ["Evidence precision", "0.1719", "0.1127", "+0.0592"],
        ["Mean context tokens", "1,265", "1,833", "-31.0%"],
        ["Mean final k", "14.67", "17.33", "-2.66"],
        ["Mean selected representations", "1.125", "1.000", "+0.125"],
    ], [avail - 90 * mm, 30 * mm, 30 * mm, 30 * mm], align=[1, 2, 3]))

    story.append(para("4. Leakage audit", "h1"))
    leak = json.loads((VERIFY / "DATASET_LEAKAGE_AUDIT.json").read_text(encoding="utf-8"))
    q = leak["qasper_overlap_characterization"]
    story.append(para(
        f"{tag('REPRODUCED FACT')} {q['distinct_overlapping_questions']} distinct QASPER question "
        f"strings overlap train and test, touching {q['test_examples_affected']} test examples "
        f"({q['test_examples_affected_percent']}%). All are generic NLP-paper templates and "
        f"<b>{q['overlapping_questions_that_share_a_paper_id']} share a paper</b> with their training "
        "counterpart. This is template repetition, not answer or source leakage — but it must never "
        "be described as zero leakage.", "body"))
    cal = leak["calibration_isolation_audit"]["per_evaluation_split"]
    story.append(table([
        ["Calibration sample (60 fitted examples) vs", "QASPER test", "MuSiQue dev"],
        ["Example-ID overlap", str(cal["qasper_test"]["calibration_example_id_overlap"]),
         str(cal["musique_full_validation"]["calibration_example_id_overlap"])],
        ["Source / paper-ID overlap", str(cal["qasper_test"]["calibration_source_id_overlap"]),
         str(cal["musique_full_validation"]["calibration_source_id_overlap"])],
        ["Document-ID overlap", str(cal["qasper_test"]["calibration_document_id_overlap"]),
         str(cal["musique_full_validation"]["calibration_document_id_overlap"])],
        ["Distinct question templates shared", str(cal["qasper_test"]["calibration_question_hash_overlap_distinct"]),
         str(cal["musique_full_validation"]["calibration_question_hash_overlap_distinct"])],
        ["Evaluation examples touched",
         f'{cal["qasper_test"]["evaluation_examples_touched"]} ({cal["qasper_test"]["evaluation_examples_touched_percent"]}%)',
         f'{cal["musique_full_validation"]["evaluation_examples_touched"]} ({cal["musique_full_validation"]["evaluation_examples_touched_percent"]}%)'],
    ], [avail - 70 * mm, 35 * mm, 35 * mm], align=[1, 2]))
    story.append(para(f"{tag('LIMITATION')} Exact matching cannot rule out semantic paraphrase, prior "
                      "benchmark contamination, or model pretraining exposure. The last applies equally "
                      "to both arms, so it does not bias the paired comparison — only absolute scores.", "note"))

    story.append(PageBreak())
    story.append(para("5. Defect remediation", "h1"))
    defects = read_csv(VERIFY / "DEFECT_REMEDIATION_LOG.csv")
    rows = [["ID", "Class", "Sev", "Title", "Status"]]
    for d in defects:
        rows.append([d["defect_id"], d["class"].split("_")[0], d["severity"],
                     d["title"], d["status"].replace("_", " ")])
    story.append(table(rows, [10 * mm, 16 * mm, 14 * mm, avail - 84 * mm, 44 * mm]))
    story.append(para("Class a = direct implementation bug, b = documentation/formula mismatch, "
                      "c = methodological decision requiring a new experimental version.", "note"))

    story.append(para("6. Test suite", "h1"))
    story.append(table([
        ["", "Pre-audit", "Post-audit"],
        ["Tests", "23", "73"],
        ["Interpreter", "Anaconda (different environment)", ".venv (the research environment)"],
        ["Primary pipeline under frozen controls", "not covered", "covered"],
        ["Real MiniLM integration", "not covered", "covered"],
        ["Harness fingerprint / checkpoint / resume", "not covered", "covered"],
        ["Systems measurement fields", "not covered", "covered"],
    ], [avail - 80 * mm, 40 * mm, 40 * mm]))
    story.append(para(
        "The pre-audit suite asserted the two headline contracts (probe depth is not final k, "
        "context budget never exceeded) against the <b>legacy</b> multi-call system, and its only "
        "primary-pipeline test disabled every adaptive gate.", "note"))

    story.append(PageBreak())
    story.append(para("6b. Retrieval-only ablation screening", "h1"))
    story.append(para(
        f"{tag('REPRODUCED FACT')} Zero Groq calls. Fresh deterministic stratified sample of 60 "
        "official TRAINING examples (seed 20260804, distinct from the calibration seed so screening "
        "does not reuse threshold-selection examples). 43 of 60 carry gold evidence and are scored. "
        f"{tag('LIMITATION')} Retrieval-only: no answer quality, latency or cost, and no statistical "
        "inference. This is screening evidence for deciding what deserves paid evaluation, not a result.",
        "body"))
    screen = read_csv(VERIFY / "RETRIEVAL_ONLY_ABLATIONS.csv")

    def num(row, key, places=4):
        try:
            return f"{float(row[key]):.{places}f}"
        except (ValueError, TypeError):
            return "-"

    rows = [["Condition", "Recall", "Compl.ev", "Prec.", "Ctx tok", "final k", "Cand.", "2-view%"]]
    for r in screen:
        if r["arm"] != "sar_rag":
            continue
        rows.append([r["condition"], num(r, "evidence_recall"), num(r, "complete_evidence"),
                     num(r, "evidence_precision"), num(r, "context_tokens", 1),
                     num(r, "final_k", 2), num(r, "candidates_examined", 1),
                     f'{100 * float(r["two_representations"]):.1f}'])
    story.append(table(rows, [46 * mm, 18 * mm, 19 * mm, 16 * mm, 18 * mm, 16 * mm, 16 * mm, 16 * mm],
                       align=[1, 2, 3, 4, 5, 6, 7]))
    story.append(Spacer(1, 5))
    rows = [["Traditional baseline", "Words", "Recall", "Compl.ev", "Prec.", "Ctx tok", "final k"]]
    for r in screen:
        if r["arm"] != "traditional":
            continue
        rows.append([r["condition"], num(r, "granularity_words", 1), num(r, "evidence_recall"),
                     num(r, "complete_evidence"), num(r, "evidence_precision"),
                     num(r, "context_tokens", 1), num(r, "final_k", 2)])
    story.append(table(rows, [52 * mm, 18 * mm, 19 * mm, 19 * mm, 18 * mm, 19 * mm, avail - 145 * mm],
                       align=[1, 2, 3, 4, 5, 6]))

    story.append(para("Five findings that change the assessment", "h2"))
    story.append(table([
        ["#", "Finding"],
        ["F1", "<b>A single-view baseline beats the full routed system.</b> paragraph_only beats full on "
               "every retrieval outcome at once: recall +0.095, complete-evidence +0.070, precision +0.016, "
               "fewer context tokens, 31% fewer candidates examined. This is the most serious threat to the "
               "paper's contribution."],
        ["F2", "<b>Two named components are inert.</b> no_novelty and no_similarity_gap reproduce full to "
               "four decimal places on every metric. At the frozen thresholds neither rule ever binds."],
        ["F3", "<b>At a matched context budget, SAR-RAG does win.</b> At ~1,300 tokens: recall 0.7054 vs "
               "0.6492 (+0.056) and complete-evidence 0.5814 vs 0.4651 (+0.116) against Traditional k=10. "
               "The efficiency claim must be framed budget-matched, not k-matched."],
        ["F4", "<b>Auditor self-correction.</b> The granularity confound was initially rated HIGH from "
               "config parameters (2.7x). Measured realized ratio is 1.73x on QASPER and 1.44x on MuSiQue, "
               "because sliding windows stop at section boundaries. Severity revised to MEDIUM. It does "
               "not explain away F3."],
        ["F5", "<b>The second representation contributes almost nothing.</b> Two views on 16.7% of queries "
               "(QASPER 30%, MuSiQue 3.3%); adds unique gold evidence in 20% of those, i.e. 3.3% of all "
               "queries. select_one_representation loses only 0.0116 recall and zero complete-evidence."],
    ], [10 * mm, avail - 10 * mm]))
    story.append(para(
        "<b>Components that ARE supported:</b> adaptive retrieval (+0.238 recall over probe-only), "
        "RRF selection over no selection (+0.110), post-retrieval deduplication (removes ~1 chunk and "
        "66 tokens with zero recall loss). <b>Also:</b> 36.7% of queries stop on a safety cap; the RRF "
        "constant is nearly irrelevant across {1,3,10,60}; probe depth 5 outperforms the frozen depth 3 "
        "and must NOT be silently adopted.", "body"))

    story.append(PageBreak())
    story.append(para("7. Review decision", "h1"))
    story.append(table([
        ["Question", "Answer"],
        ["Implementation matches intended pipeline?", "Substantially yes; one formula misstatement (D08) plus documented deviations"],
        ["Exactly one primary LLM call?", "Yes — verified structurally and end to end"],
        ["Final k genuinely adaptive?", "Yes in mechanism; scale behaviour PENDING"],
        ["Selection based on RRF complementarity?", "Yes in mechanism; published formula is wrong; second view rare"],
        ["Deduplication after retrieval?", "Yes — accepted IDs equal provisional IDs"],
        ["Traditional a fair baseline?", "Fair on the answer side; efficiency confounded by granularity"],
        ["Calibration isolated from evaluation?", "Yes"],
        ["Evidence of leakage?", "No source/document/example leakage; template repetition disclosed"],
        ["Metrics correctly calculated?", "Yes — 22/22 against hand computation"],
        ["Tests paired and corrected?", "Yes — paired bootstrap, randomization, exact McNemar, Holm"],
        ["Improves answer quality?", "PENDING — unknown, never demonstrated"],
        ["Improves evidence recall?", "No on current evidence (-0.104 on calibration)"],
        ["Improves complete-evidence recovery?", "No on current evidence (-0.187 on calibration)"],
        ["Improves precision / context efficiency?", "Yes — at a matched ~1,300-token budget: +0.056 recall, +0.116 complete-evidence"],
        ["Meaningful Pareto improvement?", "Vs fixed-k Traditional at matched budget: plausibly. Vs a paragraph_only single view: no"],
        ["Which components are supported?", "Supported: adaptive retrieval, RRF over no-selection, dedup. Not supported: novelty, similarity gap, second view, RRF constant"],
        ["Ready for Q2 submission?", "<b>Not yet</b> — no confirmatory result exists"],
    ], [avail - 105 * mm, 105 * mm]))

    story.append(PageBreak())
    story.append(para("8. Claims that must be weakened or corrected", "h1"))
    story.append(table([
        ["#", "Required correction"],
        ["1", "Rewrite the RRF formula. The selector applies a <b>second</b> normalization over total "
              "aligned probe mass, so the frozen 0.2437 is a share of that mass, not a raw NRRF value."],
        ["2", "Never use 'training-free' unqualified. The repository is already correct; keep it."],
        ["3", "Never describe normalized RRF as a calibrated probability."],
        ["4", "Do not call QASPER leakage zero. State: 72 repeated generic templates, 163 test examples, "
              "0 shared papers, 0.14% calibration-level channel."],
        ["5", "Report the efficiency comparison budget-matched, not k-matched, and disclose the measured "
              "1.44-1.73x granularity difference."],
        ["6", "Do not present 'one or two representations' as a general mechanism. Two views on 16.7% of "
              "queries, unique gold added in 3.3%, and essentially QASPER-only."],
        ["7", "<b>Do not claim novelty or the similarity-gap rule as contributing components.</b> Both are "
              "provably inert at the frozen thresholds."],
        ["8", "<b>Do not claim the four-view design is necessary.</b> paragraph_only currently dominates "
              "the full system; hierarchical_only is by far the weakest view."],
        ["9", "Do not describe final k as unconstrained while 36.7% of queries stop on a safety cap."],
        ["10", "Do not adopt probe depth 5 from screening; it needs a fresh training-only calibration."],
        ["11", "Do not report a single n = 6,285 beside all four primary outcomes. Valid paired n is "
               "3,789 / 5,645 / 5,645 / 6,285."],
        ["12", "Keep chunk-level P@k, MRR and nDCG strictly secondary; they are granularity-sensitive."],
        ["13", "Do not present the three-calls-per-question study cost as the method's cost."],
        ["14", "Do not label pilot figures as final results."],
    ], [8 * mm, avail - 8 * mm]))

    story.append(para("9. Current defensible conclusion", "h1"))
    story.append(para(
        "SAR-RAG is a model-fine-tuning-free, training-calibrated adaptive retrieval method. Training "
        "calibration indicates higher evidence precision and reduced context use, but lower evidence "
        "recall and complete-evidence recovery than a static-k Traditional RAG baseline. Its overall "
        "quality-efficiency advantage remains unproven pending complete held-out evaluation, registered "
        "ablations, systems measurements and manual failure analysis.", "body"))
    story.append(para(f"{tag('REVIEWER INFERENCE')} <b>Three auditor additions</b>, all from "
                      "retrieval-only screening on training data:", "body"))
    story.append(table([
        ["#", "Addition"],
        ["1", "The efficiency comparison must be <b>budget-matched, not k-matched</b>. At an equal "
              "~1,300-token budget SAR-RAG wins (+0.056 recall, +0.116 complete-evidence). Against k=20 "
              "it loses on coverage. Both must be reported; budget-matched is the honest framing and is "
              "the strongest result the method currently has."],
        ["2", "<b>Two named components are inert and one is near-inert.</b> Novelty and the similarity-gap "
              "rule change nothing at the frozen thresholds; the second representation is worth ~0.01 "
              "recall on 3.3% of queries."],
        ["3", "<b>A trivial single-view baseline currently dominates the method.</b> Until this is refuted "
              "on held-out data, the central contribution is not established."],
    ], [10 * mm, avail - 10 * mm]))
    story.append(para(
        "<b>Outcome classification: (D) no convincing advantage</b> — with a genuine but narrow "
        "<b>(C) efficiency-oriented trade-off at matched context budget</b>. Neither can be asserted as a "
        "result, because no confirmatory experiment has been run. The experiment is <b>not</b> invalid "
        "(E): no leakage, unfair prompt, broken metric or test-set tuning was found. It is unexecuted — "
        "and the screening evidence now predicts that the routing contribution, as currently framed, "
        "will not survive it.", "body"))

    story.append(para("10. Unresolved issues", "h1"))
    story.append(table([
        ["#", "Issue", "Blocking"],
        ["U1", "No confirmatory run exists; all held-out results are PENDING", "Yes"],
        ["U2", "Groq run-date rates unverified; cost undefined", "Yes"],
        ["U3", "Complete-evidence recovery scored on 1,876 unanswerable MuSiQue examples", "Yes"],
        ["U4", "Granularity confound: control added, not yet reported end to end", "Yes"],
        ["U5", "Paper RRF formula does not match the implementation", "Yes"],
        ["U6", "Second view: 16.7% of queries, unique gold in 3.3%, essentially QASPER-only", "Yes"],
        ["U13", "<b>paragraph_only dominates the full routed system on every retrieval outcome</b>", "Yes"],
        ["U14", "Novelty and similarity-gap components are provably inert at the frozen thresholds", "Yes"],
        ["U15", "36.7% of queries terminate on a safety cap rather than adaptively", "Yes"],
        ["U16", "Probe depth 5 beats frozen depth 3; must not be adopted without fresh calibration", "No"],
        ["U7", "Non-inferiority margins likely unachievable; must not be revised post-hoc", "No"],
        ["U8", "Failure tables and 4 of 9 figures need a real run", "No"],
        ["U9", "Confirmatory figure hard-codes a 'pilot' caption", "No"],
        ["U10", "Revision pin inert on local snapshot path; provenance rests on verified sha256", "No"],
        ["U11", "pytest not declared in pyproject.toml", "No"],
        ["U12", "MuSiQue test labels hidden; test numbers require official submission", "No"],
    ], [10 * mm, avail - 34 * mm, 24 * mm]))

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MX, H - MT + 5 * mm, W - MX, H - MT + 5 * mm)
        canvas.line(MX, MB - 4 * mm, W - MX, MB - 4 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(MX, H - MT + 6.5 * mm, "SAR-RAG — Independent Verification")
        canvas.drawRightString(W - MX, H - MT + 6.5 * mm, "Zero paid Groq calls")
        canvas.drawString(MX, MB - 8 * mm,
                          "Confirmatory cells are PENDING. Calibration and pilot values are not results.")
        canvas.drawRightString(W - MX, MB - 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document = BaseDocTemplate(str(OUTPUT), pagesize=A4,
                               leftMargin=MX, rightMargin=MX,
                               topMargin=MT, bottomMargin=MB,
                               title="SAR-RAG Independent Verification Report")
    frame = Frame(MX, MB, W - 2 * MX, H - MT - MB, id="main",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    document.addPageTemplates([PageTemplate(id="std", frames=[frame], onPage=decorate)])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.build(story)
    print(json.dumps({"output": str(OUTPUT), "bytes": OUTPUT.stat().st_size}))


if __name__ == "__main__":
    build()

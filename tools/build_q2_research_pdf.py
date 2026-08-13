from __future__ import annotations

import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, LongTable, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "output" / "q2_analysis_pilot" / "q2_analysis.json"
LEAKAGE = ROOT / "output" / "q2_analysis_pilot" / "leakage_audit.json"
CALIBRATION = ROOT / "output" / "calibration" / "frozen_manifest.json"
OUTPUT = ROOT / "output" / "pdf" / "SAR_RAG_Q2_RESEARCH_EVALUATION.pdf"
FIGURES = ROOT / "output" / "q2_analysis_pilot" / "figures"

W, H = A4
MX, MT, MB = 17 * mm, 18 * mm, 17 * mm
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#176B87")
TEAL = colors.HexColor("#168C8C")
INK = colors.HexColor("#26323D")
GRAY = colors.HexColor("#5B6573")
PALE = colors.HexColor("#F3F7F9")
WARN = colors.HexColor("#FFF3D8")
LINE = colors.HexColor("#CEDAE1")


def register_fonts():
    regular, bold = Path("C:/Windows/Fonts/aptos.ttf"), Path("C:/Windows/Fonts/aptos-bold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("Body", str(regular)))
        pdfmetrics.registerFont(TTFont("Bold", str(bold)))
        return "Body", "Bold"
    return "Helvetica", "Helvetica-Bold"


BODY, BOLD = register_fonts()
base = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", fontName=BOLD, fontSize=24, leading=29, textColor=colors.white, alignment=TA_CENTER)
SUBTITLE = ParagraphStyle("Subtitle", fontName=BODY, fontSize=12, leading=16, textColor=colors.HexColor("#DDEBF2"), alignment=TA_CENTER)
H1 = ParagraphStyle("H1", fontName=BOLD, fontSize=15, leading=19, textColor=NAVY, spaceBefore=4.5*mm, spaceAfter=2*mm, keepWithNext=True)
H2 = ParagraphStyle("H2", fontName=BOLD, fontSize=11.5, leading=14.5, textColor=BLUE, spaceBefore=3*mm, spaceAfter=1.5*mm, keepWithNext=True)
P = ParagraphStyle("P", fontName=BODY, fontSize=9.1, leading=13.1, textColor=INK, spaceAfter=2.1*mm)
BULLET = ParagraphStyle("Bullet", parent=P, leftIndent=5*mm, firstLineIndent=-3.5*mm, spaceAfter=1.2*mm)
SMALL = ParagraphStyle("Small", fontName=BODY, fontSize=7.5, leading=9.6, textColor=GRAY)
CELL = ParagraphStyle("Cell", fontName=BODY, fontSize=7.3, leading=9.2, textColor=INK)
CELL_HEAD = ParagraphStyle("CellHead", fontName=BOLD, fontSize=7.2, leading=9, textColor=colors.white)
CALLOUT = ParagraphStyle("Callout", fontName=BODY, fontSize=10, leading=14.5, textColor=NAVY, backColor=WARN, borderColor=colors.HexColor("#E5C36B"), borderWidth=.6, borderPadding=8, spaceAfter=3*mm)
CODE = ParagraphStyle("Code", fontName="Courier", fontSize=7.4, leading=10, textColor=INK, backColor=PALE, borderPadding=6, spaceAfter=2*mm)


def esc(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(value, digits=4):
    if value is None:
        return "Pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(LINE); canvas.line(MX, H-13*mm, W-MX, H-13*mm)
        canvas.setFillColor(GRAY); canvas.setFont(BODY, 7.3)
        canvas.drawString(MX, H-9.6*mm, "SAR-RAG | Q2 research evaluation package")
        canvas.drawRightString(W-MX, 9*mm, f"Page {doc.page}")
    canvas.restoreState()


class Doc(BaseDocTemplate):
    def __init__(self):
        super().__init__(str(OUTPUT), pagesize=A4, leftMargin=MX, rightMargin=MX,
                         topMargin=MT, bottomMargin=MB,
                         title="SAR-RAG Q2 Research Evaluation Package", author="Research Project")
        frame = Frame(MX, MB, W-2*MX, H-MT-MB, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=page))


def table(headers, rows, widths=None, font=7.3):
    if widths is None:
        widths = [(W-2*MX)/len(headers)] * len(headers)
    header_style = CELL_HEAD
    cell_style = ParagraphStyle("CellDynamic", parent=CELL, fontSize=font, leading=font+1.9)
    data = [[Paragraph(esc(value), header_style) for value in headers]]
    data += [[Paragraph(esc(value), cell_style) for value in row] for row in rows]
    result = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("GRID", (0,0), (-1,-1), .35, LINE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PALE]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 3), ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    return result


def bullet(text):
    return Paragraph("- " + esc(text), BULLET)


def figure(path, caption):
    if not path.exists():
        return [Paragraph("Figure pending full evaluation: " + esc(caption), CALLOUT)]
    image = Image(str(path))
    scale = min((W-2*MX)/(image.imageWidth or 1), 82*mm/(image.imageHeight or 1))
    image.drawWidth *= scale; image.drawHeight *= scale
    return [image, Paragraph(esc(caption), ParagraphStyle("Caption", parent=SMALL, alignment=TA_CENTER, spaceAfter=3*mm))]


def cover(analysis):
    status = analysis["status"]
    box = Table([[Paragraph("Representation-State Adaptive RAG", TITLE)], [Paragraph(
        "Q2 journal evaluation implementation, calculations, reproducibility controls, and current evidence status", SUBTITLE
    )]], colWidths=[W-2*MX])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY), ("TOPPADDING", (0,0), (-1,0), 26*mm),
        ("BOTTOMPADDING", (0,0), (-1,0), 4*mm), ("BOTTOMPADDING", (0,1), (-1,1), 20*mm),
        ("LEFTPADDING", (0,0), (-1,-1), 10*mm), ("RIGHTPADDING", (0,0), (-1,-1), 10*mm),
    ]))
    warning = Paragraph(
        "Current status: the evaluation code is implemented and verified, but the available answer-generation result contains only two engineering-pilot questions. It is not confirmatory evidence. Empty final-result cells are deliberately marked Pending; no values are invented.",
        CALLOUT,
    )
    meta = table(["Item", "Recorded value"], [
        ["Analysis status", status], ["Embedding", "all-MiniLM-L6-v2"],
        ["Answer model", "Groq openai/gpt-oss-120b"], ["Frozen calibration", "60 stratified training examples"],
        ["Final evaluation", "QASPER test + MuSiQue development; pending execution"],
        ["Report date", "13 August 2026"],
    ], [46*mm, 117*mm], 8)
    return [Spacer(1,8*mm), box, Spacer(1,14*mm), warning, Spacer(1,8*mm), meta, PageBreak()]


def build_story(analysis, leakage, calibration):
    story = cover(analysis)
    story += [Paragraph("1. What has been implemented", H1), Paragraph(
        "The package now converts every completed question into paper tables and figures from raw recorded results. It calculates answer quality, evidence retrieval, abstention, paired uncertainty, practical effect decisions, efficiency, representation states, adaptive k, subgroups, ablations, and failure-review samples. Confirmatory runs are fingerprinted, checkpointed, resumable, and reject changes to frozen controls.", P)]
    implemented = [
        "Answer exact match plus token precision, recall, and maximum-gold F1 with documented normalization.",
        "Unique gold-span precision/recall, complete-evidence recovery, P@1/3/5/10/20, R@1/3/5/10/20, MRR, and nDCG.",
        "Answerability confusion matrix, false-answer rate, correct abstention, and incorrect abstention.",
        "Paired bootstrap confidence intervals, paired randomization, paired effect size dz, exact McNemar, and Holm correction.",
        "Mean, median, SD, quartiles, p95, min/max for latency, tokens, cost, final k, candidates, and rounds.",
        "Representation combination analysis, second-view marginal gain, routing-risk flags, adaptive-k correlations, subgroups, and deterministic failure samples.",
        "Twenty registered conditions covering four single views, RRF, stopping, novelty, gap, deduplication, one-view routing, probe/RRF sensitivity, fixed-k baselines, and legacy multi-call cost.",
    ]
    story += [bullet(value) for value in implemented]

    story += [Paragraph("2. Fixed research design", H1), table(
        ["Component", "Frozen decision", "Reason"], [
            ["Primary method", "Four views; top-3 probe; select 1-2 by normalized RRF", "Tests representation-state routing"],
            ["RRF constant", "k0 = 3", "Probe contains ranks 1-3"],
            ["Final retrieval k", "Emergent; never calibrated", "Question complexity differs"],
            ["Safety limits", "20 chunks, 2,000 tokens, 60 candidates, 8 rounds", "Bounded execution, not target k"],
            ["Baseline", "Sliding-window dense RAG; static k = 20", "Chosen on training evidence only"],
            ["Answerer", "Identical Groq prompt/schema/model for both methods", "Removes generator confounding"],
            ["Final QASPER", "Test, 1,451 questions", "Development was used for engineering"],
            ["Local MuSiQue", "Development, 4,834 questions", "Test labels are hidden"],
        ], [38*mm, 69*mm, 56*mm])]

    story += [Paragraph("3. Dataset statistics", H1)]
    ds_rows = [[row["dataset"], row["split"], row["questions"], row["answerable"], row["insufficient"],
                fmt(row["documents_mean"],2), fmt(row["document_words_mean"],1), fmt(row["gold_evidence_spans_mean"],2)]
               for row in analysis.get("dataset_statistics", [])]
    story += [table(["Dataset","Split","N","Answerable","Insufficient","Docs/q","Words/q","Gold spans/q"], ds_rows,
                    [25*mm,18*mm,17*mm,21*mm,21*mm,17*mm,23*mm,21*mm], 6.8)]

    story += [Paragraph("4. Calculation definitions", H1)]
    formulas = [
        ("Exact match", "EM = mean_i max_a 1[normalize(prediction_i) = normalize(a)]"),
        ("Token F1", "P = common/predicted; R = common/gold; F1 = 2PR/(P+R); select the gold answer with maximum F1."),
        ("Evidence recall", "Unique gold spans covered by final evidence / all annotated gold spans."),
        ("Complete evidence", "CER = mean over answerable questions of 1[every required gold span is recovered]."),
        ("MRR", "Mean reciprocal rank of the first relevant retrieved chunk."),
        ("nDCG", "DCG@k / ideal DCG@k using binary chunk relevance; granularity-sensitive and therefore secondary."),
        ("Reduction", "100 x (1 - SAR / Traditional) for context tokens, model tokens, cost, and latency."),
        ("Effect size", "Paired dz = mean(SAR - Traditional) / SD(SAR - Traditional)."),
    ]
    story += [table(["Measure", "Implemented calculation"], formulas, [42*mm,121*mm], 7.5)]
    story += [Paragraph("Normalization", H2), Paragraph(
        "Evaluation removes deterministic evidence markers such as [E1], applies Unicode-aware case folding, removes punctuation, removes the English articles a/an/the, collapses whitespace, and compares token multisets. Multiple accepted gold answers use the maximum token F1.", P)]

    story += [Paragraph("5. Primary outcomes and decision rules", H1), table(
        ["Outcome", "Role", "Practical margin", "Multiplicity"], [
            ["Answer token F1", "Primary superiority", "+0.02 absolute", "Holm corrected"],
            ["Gold-evidence recall", "Primary non-inferiority", "Lower difference bound above -0.02", "Holm corrected"],
            ["Complete-evidence recovery", "Primary non-inferiority", "Lower difference bound above -0.02", "Holm corrected"],
            ["Context tokens", "Primary efficiency", "At least 10% reduction", "Holm corrected"],
            ["Cost", "Secondary efficiency", "At least 10% reduction", "Explicit frozen prices"],
        ], [47*mm,40*mm,47*mm,29*mm])]
    story += [Paragraph(
        "A superiority claim requires confirmatory status, paired confidence intervals, corrected inference, practical margins, and failure analysis to agree. A p-value below 0.05 alone is insufficient.", CALLOUT)]

    story += [Paragraph("6. Main held-out result table", H1), Paragraph(
        "The table below is the exact publication schema. Full values remain Pending because the 6,285-question frozen run has not been executed. The engineering pilot is shown later and must not fill this table.", P)]
    pending_rows = []
    for dataset in ("QASPER test (n=1,451)", "MuSiQue dev (n=4,834)"):
        for metric in ("Exact match","Token F1","Evidence recall","Complete evidence","Evidence precision","Faithfulness","Answer relevancy","Answerability F1"):
            pending_rows.append([dataset, metric, "Pending", "Pending", "Pending", "Pending", "TBD", "TBD"])
    story += [table(["Dataset","Metric","Traditional","SAR-RAG","Difference","95% CI","p","Holm p"], pending_rows,
                    [30*mm,30*mm,20*mm,20*mm,20*mm,23*mm,10*mm,10*mm], 6.4)]

    story += [PageBreak(), Paragraph("7. Verified engineering pilot - not paper evidence", H1)]
    pilot_rows = []
    for row in analysis.get("main_results", []):
        if row["metric"] in ("answer_token_f1","evidence_set_recall","evidence_set_precision","faithfulness","answer_relevancy","context_tokens","total_tokens","latency_seconds"):
            pilot_rows.append([row["metric"], fmt(row["traditional"]), fmt(row["sar_rag"]), fmt(row["difference"]),
                               f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]", fmt(row["randomization_p"]), row["paired_n"]])
    story += [table(["Metric","Traditional","SAR-RAG","Difference","Pilot CI","p","n"], pilot_rows,
                    [42*mm,22*mm,22*mm,22*mm,35*mm,14*mm,8*mm], 7)]
    story += [Paragraph(
        "Interpretation: n = 2 only verified the end-to-end harness and identical answer generators. It cannot estimate population effects, significance, robustness, or superiority. Its confidence intervals are not used for claims.", CALLOUT)]
    for name, caption in (
        ("01_main_quality.png", "Pilot-only primary quality comparison"),
        ("02_recall_token_pareto.png", "Pilot-only evidence recall versus context tokens"),
        ("03_final_k.png", "Pilot-only emergent final-k distribution"),
    ):
        story += figure(FIGURES/name, caption)

    story += [Paragraph("8. Answerability and abstention", H1), Paragraph(
        "For each dataset and method, the final report will provide TP, TN, FP, FN; accuracy; precision; recall; F1; false-answer rate on insufficient questions; correct-abstention rate; and incorrect-abstention rate on answerable questions. Risk-coverage is intentionally omitted until a defensible confidence score is introduced.", P)]
    story += [table(["Gold / prediction", "Predicted answerable", "Predicted insufficient"],
                    [["Gold answerable","TP","FN"],["Gold insufficient","FP","TN"]], [55*mm,54*mm,54*mm])]

    story += [Paragraph("9. Statistical inference", H1)]
    for value in (
        "Continuous/per-question outcomes: paired mean difference, 10,000-sample paired bootstrap 95% CI, paired randomization p-value, and paired dz.",
        "Binary outcomes: exact McNemar test with discordant-pair counts.",
        "Four primary outcomes: Holm family-wise multiplicity correction.",
        "Every row reports valid paired n; undefined retrieval outcomes on no-gold-evidence questions are excluded rather than scored as perfect.",
        "Statistical significance and practical importance are reported separately.",
    ): story.append(bullet(value))

    story += [Paragraph("10. Efficiency and systems measurements", H1), table(
        ["Quantity","Mean","Median","SD","p95","Accounting rule"], [
            ["Online latency","Pending","Pending","Pending","Pending","Excludes one-time indexing"],
            ["Offline indexing","Pending","Pending","Pending","Pending","Four-view and one-view reported separately"],
            ["Context tokens","Pending","Pending","Pending","Pending","MiniLM tokenizer count"],
            ["Prompt/completion/total tokens","Pending","Pending","Pending","Pending","Groq returned usage"],
            ["Cost/question","Pending","Pending","Pending","Pending","Run-date rates frozen in fingerprint"],
            ["Peak memory and index size","Pending","Pending","Pending","Pending","OS process peak plus exact vector-array bytes"],
            ["Candidates and rounds","Pending","Pending","Pending","Pending","Trace-derived"],
        ], [36*mm,19*mm,19*mm,19*mm,19*mm,51*mm], 6.9)]

    story += [Paragraph("11. Representation-state analysis", H1)]
    rep_rows = [[row["dataset"], row["state"], row["questions"], fmt(row["percentage"],1), fmt(row["evidence_recall"]),
                 fmt(row["complete_evidence"]), fmt(row["answer_f1"]), fmt(row["context_tokens"],1)]
                for row in analysis.get("representation_states", [])]
    story += [table(["Dataset","State","N","%","Evidence recall","Complete","Answer F1","Tokens"], rep_rows,
                    [24*mm,36*mm,12*mm,13*mm,23*mm,20*mm,20*mm,21*mm], 6.6)]
    story += [Paragraph(
        "The full run additionally reports one-versus-two selection, each individual view, representation combinations by dataset/length/hop/answerability, second-view marginal RRF gain, final k per state, unique gold evidence added by the second view, and routing-risk cases where an unselected top probe contains gold evidence absent from selected probes.", P)]
    story += figure(FIGURES/"04_representation_frequency.png", "Pilot-only representation frequency")

    story += [Paragraph("12. Adaptive-k analysis", H1)]
    adaptive = analysis.get("adaptive_k", {})
    k = adaptive.get("final_k", {})
    story += [table(["Measure","Current recorded value","Full-paper interpretation"], [
        ["Mean / median final k", f"{fmt(k.get('mean'))} / {fmt(k.get('median'))}", "Report separately by dataset"],
        ["SD / quartiles / p95", f"{fmt(k.get('sd'))} / {fmt(k.get('p25'))}-{fmt(k.get('p75'))} / {fmt(k.get('p95'))}", "Show distribution"],
        ["At chunk cap", fmt(adaptive.get("percentage_at_chunk_cap")), "High rate would weaken adaptivity claim"],
        ["At token cap", fmt(adaptive.get("percentage_at_token_cap")), "Report as safety-cap termination"],
        ["Correlation with evidence count", fmt(adaptive.get("correlation_final_k_gold_evidence_count",{}).get("spearman_rho")), "Expected positive if k adapts to evidence need"],
        ["Correlation with hop count", fmt(adaptive.get("correlation_final_k_hop_count",{}).get("spearman_rho")), "MuSiQue only"],
    ], [47*mm,48*mm,68*mm])]

    story += [PageBreak(), Paragraph("13. Registered ablations", H1), Paragraph(
        "Every ablation is a named condition included in the immutable run fingerprint. This prevents unreported threshold combinations and makes resumed runs reject mismatches.", P)]
    ablation_rows = [[item["condition"], "Primary" if item["primary"] else "Ablation", item["online_policy"], "Pending"]
                     for item in analysis.get("ablation_registry", [])]
    story += [table(["Condition","Role","Online policy","Held-out result"], ablation_rows,
                    [72*mm,26*mm,37*mm,28*mm], 6.8)]
    story += [Paragraph(
        "Ablation results must use the same dataset rows, answerer, schema, model, decoding, budgets, metrics, and run environment. The legacy controller is reported only as a multi-call cost ablation and cannot be presented as obeying the one-call claim.", CALLOUT)]

    story += [Paragraph("14. Robustness and subgroup analysis", H1)]
    for value in (
        "Dataset and answerability.", "QASPER short/medium/long documents.",
        "MuSiQue 2/3/4 hop questions.", "Short/long gold answers and number of gold evidence spans.",
        "One versus multiple documents.", "Stop reason and one versus two selected views.",
        "Every important subgroup includes n and uncertainty; small groups receive no strong claim.",
    ): story.append(bullet(value))

    story += [Paragraph("15. Failure analysis", H1), Paragraph(
        "The implementation automatically flags candidate failures and selects a deterministic bottom-hash sample for manual review. Automated flags are not treated as final causal labels; reviewers must inspect source, retrieved evidence, trace, deduplication provenance, and answer.", P)]
    failure_rows = [[row["category"], row["count"]] for row in analysis.get("failure_categories", [])]
    if failure_rows:
        story.append(table(["Automatic flag","Pilot count"], failure_rows, [120*mm,43*mm]))
    else:
        story.append(Paragraph("No automatic failure table was produced from the two-question pilot. Full deterministic review sample: pending.", CALLOUT))
    for value in (
        "Wrong representation or necessary second representation rejected.",
        "Similarity, gap, novelty, marginal-utility, or redundancy rule stopped useful evidence.",
        "Gold evidence lost at deduplication or chunk boundary.",
        "Evidence retrieved but ignored by answerer; unsupported answer; correct or incorrect abstention.",
        "Gold annotation incomplete/ambiguous or source parsing failure.",
    ): story.append(bullet(value))

    story += [Paragraph("16. Leakage and fairness audit", H1)]
    q = leakage["within_dataset"]["qasper_train_vs_test"]
    m = leakage["within_dataset"]["musique_train_vs_dev"]
    story += [table(["Comparison","Exact question overlap","Source-ID overlap","Document-ID overlap"], [
        ["QASPER train vs test", q["exact_normalized_question_overlap"], q["source_id_overlap"], q["document_id_overlap"]],
        ["MuSiQue train vs dev", m["exact_normalized_question_overlap"], m["source_id_overlap"], m["document_id_overlap"]],
        ["Cross-dataset calibration/evaluation", "0 in both checked directions", "0", "0"],
    ], [60*mm,37*mm,33*mm,33*mm])]
    story += [Paragraph(
        "QASPER contains repeated generic question wordings across different papers: 72 normalized question hashes overlap, while source and document IDs do not. This is disclosed rather than called zero leakage. Calibration uses only aggregate retrieval evidence, not answer text. Exact matching cannot rule out paraphrase overlap, semantic contamination, or model pretraining exposure.", CALLOUT)]
    fairness = [
        "Calibration source hashes and selected sample IDs are frozen in the manifest.",
        "No QASPER test or MuSiQue development label may change controls.",
        "Traditional and SAR use identical MiniLM, context budget, answer prompt/schema, Groq model, decoding, and completion limit.",
        "Four-view offline indexing cost is measured separately from one-view baseline indexing.",
        "API usage, retries, prices, run fingerprint, and atomic per-question checkpoints are retained.",
    ]
    story += [bullet(value) for value in fairness]

    story += [Paragraph("17. Reproduction commands", H1), Paragraph(
        "The following commands execute the frozen primary conditions. Groq prices must be verified on the run date and supplied explicitly. The full calls are substantial and have not been silently executed by this report builder.", P)]
    story += [Paragraph(esc(
        ".\\.venv\\Scripts\\python.exe -m sar_rag.experiments --input .\\datasets\\normalized\\qasper_test.jsonl --output-dir .\\output\\confirmatory\\qasper --calibration-manifest .\\output\\calibration\\frozen_manifest.json"
    ), CODE), Paragraph(esc(
        ".\\.venv\\Scripts\\python.exe -m sar_rag.experiments --input .\\datasets\\normalized\\musique_full_validation.jsonl --output-dir .\\output\\confirmatory\\musique --calibration-manifest .\\output\\calibration\\frozen_manifest.json"
    ), CODE), Paragraph(esc(
        ".\\.venv\\Scripts\\python.exe -m sar_rag.q2_analysis --results <qasper-results.json> --results <musique-results.json> --dataset-input .\\datasets\\normalized\\qasper_test.jsonl --dataset-input .\\datasets\\normalized\\musique_full_validation.jsonl --output-dir .\\output\\q2_confirmatory"
    ), CODE)]

    story += [Paragraph("18. Current scientific conclusion", H1), Paragraph(
        "The requested Q2 evaluation framework is implemented end to end, and its calculations have been exercised on a real two-question engineering run. The complete held-out answer and ablation experiments have not yet been executed, so a Q2-quality superiority claim cannot currently be made.", P), Paragraph(
        "Training-only diagnostics indicate a real trade-off: SAR-RAG used less context and achieved higher evidence precision, while static-k Traditional RAG recovered more complete evidence. The current defensible statement is that SAR-RAG is a model-fine-tuning-free, training-calibrated adaptive retrieval method whose quality-efficiency advantage remains unproven.", P), Paragraph(
        "The paper may claim superiority only after the frozen QASPER test and MuSiQue development runs, corrected paired inference, registered ablations, systems measurements, and manual failure review support a meaningful improvement or a clearly defined Pareto improvement.", CALLOUT)]
    return story


def main():
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    leakage = json.loads(LEAKAGE.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Doc().build(build_story(analysis, leakage, calibration))
    print(OUTPUT)


if __name__ == "__main__":
    main()

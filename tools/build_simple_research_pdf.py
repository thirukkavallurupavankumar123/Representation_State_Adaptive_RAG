from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "RESEARCH_STATUS_SIMPLE.md"
OUTPUT = ROOT / "output" / "pdf" / "SAR_RAG_Complete_Research_Simple_Words.pdf"

PAGE_W, PAGE_H = A4
MARGIN_X = 19 * mm
MARGIN_TOP = 20 * mm
MARGIN_BOTTOM = 18 * mm

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#236B8E")
TEAL = colors.HexColor("#168C8C")
LIGHT_BLUE = colors.HexColor("#EAF3F7")
PALE = colors.HexColor("#F6F8FA")
GRAY = colors.HexColor("#5B6573")
LINE = colors.HexColor("#D5DEE5")
WHITE = colors.white


def register_fonts():
    candidates = [
        ("Body", Path("C:/Windows/Fonts/aptos.ttf")),
        ("BodyBold", Path("C:/Windows/Fonts/aptos-bold.ttf")),
        ("Body", Path("C:/Windows/Fonts/calibri.ttf")),
        ("BodyBold", Path("C:/Windows/Fonts/calibrib.ttf")),
    ]
    found = {}
    for name, path in candidates:
        if name not in found and path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
            found[name] = True
    return ("Body" if "Body" in found else "Helvetica",
            "BodyBold" if "BodyBold" in found else "Helvetica-Bold")


BODY_FONT, BOLD_FONT = register_fonts()


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="DocTitle", fontName=BOLD_FONT, fontSize=27, leading=32,
    textColor=WHITE, alignment=TA_LEFT, spaceAfter=9 * mm,
))
styles.add(ParagraphStyle(
    name="DocSubtitle", fontName=BODY_FONT, fontSize=14.5, leading=20,
    textColor=colors.HexColor("#DDEBF2"), alignment=TA_LEFT,
))
styles.add(ParagraphStyle(
    name="Section", fontName=BOLD_FONT, fontSize=15.5, leading=20,
    textColor=NAVY, spaceBefore=5 * mm, spaceAfter=2.5 * mm,
    keepWithNext=False,
))
styles.add(ParagraphStyle(
    name="Subsection", fontName=BOLD_FONT, fontSize=12.2, leading=16,
    textColor=BLUE, spaceBefore=3.5 * mm, spaceAfter=1.5 * mm,
    keepWithNext=False,
))
styles.add(ParagraphStyle(
    name="BodyText2", fontName=BODY_FONT, fontSize=9.5, leading=14.2,
    textColor=colors.HexColor("#26323D"), spaceAfter=2.3 * mm,
))
styles.add(ParagraphStyle(
    name="Bullet2", parent=styles["BodyText2"], leftIndent=5 * mm,
    firstLineIndent=-3.5 * mm, bulletIndent=1.5 * mm, spaceAfter=1.3 * mm,
))
styles.add(ParagraphStyle(
    name="Number2", parent=styles["BodyText2"], leftIndent=7 * mm,
    firstLineIndent=-5 * mm, bulletIndent=1.5 * mm, spaceAfter=1.3 * mm,
))
styles.add(ParagraphStyle(
    name="Callout", fontName=BODY_FONT, fontSize=10.2, leading=15,
    textColor=NAVY, backColor=LIGHT_BLUE, borderColor=colors.HexColor("#B9D6E3"),
    borderWidth=0.7, borderPadding=8, spaceBefore=2 * mm, spaceAfter=3 * mm,
))
styles.add(ParagraphStyle(
    name="Small", fontName=BODY_FONT, fontSize=8, leading=10.5,
    textColor=GRAY,
))


def esc(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def inline(text):
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return text


def header_footer(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN_X, PAGE_H - 14 * mm, PAGE_W - MARGIN_X, PAGE_H - 14 * mm)
        canvas.setFont(BODY_FONT, 7.6)
        canvas.setFillColor(GRAY)
        canvas.drawString(MARGIN_X, PAGE_H - 10.5 * mm, "SAR-RAG | Updated Research and Evidence Status")
        canvas.drawRightString(PAGE_W - MARGIN_X, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


class ResearchDocTemplate(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=A4,
            rightMargin=MARGIN_X,
            leftMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title="SAR-RAG Updated Research in Simple Words",
            author="Research Project",
            subject="Representation-State Adaptive RAG",
        )
        frame = Frame(
            MARGIN_X, MARGIN_BOTTOM, PAGE_W - 2 * MARGIN_X,
            PAGE_H - MARGIN_TOP - MARGIN_BOTTOM,
            id="normal", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="research", frames=[frame], onPage=header_footer))


def cover_story():
    title_box = Table([
        [Paragraph("Source-Aligned<br/>Representation-State<br/>Adaptive RAG", styles["DocTitle"])],
        [Paragraph("Complete updated research plan in simple words", styles["DocSubtitle"])],
    ], colWidths=[PAGE_W - 2 * MARGIN_X])
    title_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 15 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 15 * mm),
        ("TOPPADDING", (0, 0), (-1, 0), 27 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4 * mm),
        ("TOPPADDING", (0, 1), (-1, 1), 1 * mm),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 20 * mm),
    ]))
    summary = Table([
        [Paragraph("THE CORE IDEA", ParagraphStyle(
            "Kicker", fontName=BOLD_FONT, fontSize=9, textColor=TEAL, leading=12))],
        [Paragraph(
            "Probe four document representations at depth three, select one or two with "
            "normalized RRF, retrieve without a fixed final k, deduplicate, and make one "
            "evidence-grounded answer call.", styles["Callout"])],
    ], colWidths=[PAGE_W - 2 * MARGIN_X])
    summary.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    meta = Paragraph(
        "<b>Research specification:</b> Frozen calibration revision<br/>"
        "<b>Date:</b> 13 August 2026<br/>"
        "<b>Status:</b> Implemented and pilot-verified; full confirmatory evaluation pending",
        styles["BodyText2"],
    )
    return [Spacer(1, 7 * mm), title_box, Spacer(1, 17 * mm), summary,
            Spacer(1, 13 * mm), meta, PageBreak()]


def overview_table():
    data = [
        ["Design choice", "Final decision"],
        ["Primary views", "Sentence window, paragraph, semantic segment, hierarchy"],
        ["Probe", "Top 3 per view; this is not final k"],
        ["Selection", "Normalized RRF, k0 = 3; choose one or at most two views"],
        ["Deep retrieval", "Batches of 3; final k emerges from stopping and deduplication"],
        ["Final context", "Maximum 20 chunks and 2,000 tokenizer tokens"],
        ["LLM use", "One final Groq call per method per question"],
        ["Calibration", "60 training examples; frozen baseline k = 20"],
        ["Current conclusion", "Quality-efficiency trade-off is not yet proven superior"],
    ]
    rows = []
    for ri, row in enumerate(data):
        style = ParagraphStyle("Cell", parent=styles["BodyText2"], fontSize=8.7, leading=12,
                               textColor=WHITE if ri == 0 else colors.HexColor("#26323D"),
                               fontName=BOLD_FONT if ri == 0 else BODY_FONT, spaceAfter=0)
        rows.append([Paragraph(inline(x), style) for x in row])
    table = Table(rows, colWidths=[48 * mm, 124 * mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("BACKGROUND", (0, 1), (-1, -1), PALE),
    ]))
    return [Paragraph("Final decisions at a glance", styles["Section"]), table, Spacer(1, 4 * mm)]


def pipeline_visual():
    labels = [
        "Question", "Probe 4 views", "RRF select 1-2", "Adaptive retrieval",
        "Deduplicate", "Token compact", "One LLM call"
    ]
    cells = []
    for i, label in enumerate(labels):
        cells.append(Paragraph(label, ParagraphStyle(
            f"Pipe{i}", fontName=BOLD_FONT, fontSize=7.7, leading=9.5,
            alignment=TA_CENTER, textColor=WHITE)))
        if i < len(labels) - 1:
            cells.append(Paragraph("&gt;", ParagraphStyle(
                f"Arrow{i}", fontName=BOLD_FONT, fontSize=11, alignment=TA_CENTER,
                textColor=TEAL)))
    widths = []
    for i in range(len(cells)):
        widths.append(20 * mm if i % 2 == 0 else 5 * mm)
    table = Table([cells], colWidths=widths, hAlign="LEFT")
    commands = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm)]
    for i in range(0, len(cells), 2):
        commands.extend([
            ("BACKGROUND", (i, 0), (i, 0), NAVY if i < 8 else BLUE),
            ("BOX", (i, 0), (i, 0), 0.5, NAVY),
        ])
    table.setStyle(TableStyle(commands))
    return [Paragraph("Online pipeline", styles["Section"]), table, Spacer(1, 4 * mm)]


def parse_markdown():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = cover_story()
    story.extend(overview_table())
    story.extend(pipeline_visual())
    paragraph = []
    numbered = 0

    def flush():
        nonlocal paragraph
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), styles["BodyText2"]))
            paragraph = []

    skip_first_title = True
    for line in lines:
        stripped = line.strip()
        if skip_first_title and stripped.startswith("# "):
            skip_first_title = False
            continue
        if stripped.startswith("## "):
            flush()
            heading = stripped[3:]
            if heading.startswith("Complete Research Plan"):
                continue
            story.append(Paragraph(inline(heading), styles["Section"]))
            numbered = 0
        elif stripped.startswith("### "):
            flush()
            story.append(Paragraph(inline(stripped[4:]), styles["Subsection"]))
            numbered = 0
        elif re.match(r"^- ", stripped):
            flush()
            story.append(Paragraph(inline(stripped[2:]), styles["Bullet2"], bulletText="•"))
        elif re.match(r"^\d+\. ", stripped):
            flush()
            m = re.match(r"^(\d+)\. (.*)$", stripped)
            story.append(Paragraph(inline(m.group(2)), styles["Number2"], bulletText=m.group(1) + "."))
            numbered += 1
        elif not stripped:
            flush()
        elif stripped == "Version 1.0 - 2 August 2026":
            continue
        else:
            paragraph.append(stripped)
    flush()
    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ResearchDocTemplate(str(OUTPUT))
    doc.build(parse_markdown())
    print(OUTPUT)


if __name__ == "__main__":
    main()

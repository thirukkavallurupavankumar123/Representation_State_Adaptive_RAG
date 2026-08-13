from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "RESEARCH_STATUS_SIMPLE.md"
OUTPUT = ROOT / "output" / "pdf" / "SAR_RAG_Complete_Research_Simple_Words.pdf"
PAGE_W, PAGE_H = A4
MX, MT, MB = 18 * mm, 19 * mm, 17 * mm
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#236B8E")
TEAL = colors.HexColor("#168C8C")
INK = colors.HexColor("#26323D")
GRAY = colors.HexColor("#5B6573")
PALE = colors.HexColor("#F3F7F9")
LINE = colors.HexColor("#D5DEE5")


def fonts():
    regular = Path("C:/Windows/Fonts/aptos.ttf")
    bold = Path("C:/Windows/Fonts/aptos-bold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("DocBody", str(regular)))
        pdfmetrics.registerFont(TTFont("DocBold", str(bold)))
        return "DocBody", "DocBold"
    return "Helvetica", "Helvetica-Bold"


BODY, BOLD = fonts()
base = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "Title", fontName=BOLD, fontSize=25, leading=30, textColor=colors.white,
    alignment=TA_CENTER, spaceAfter=5 * mm,
)
SUBTITLE = ParagraphStyle(
    "Subtitle", fontName=BODY, fontSize=12.5, leading=17, textColor=colors.HexColor("#DDEBF2"),
    alignment=TA_CENTER,
)
H2 = ParagraphStyle(
    "H2", fontName=BOLD, fontSize=15, leading=19, textColor=NAVY,
    spaceBefore=4.5 * mm, spaceAfter=2.2 * mm, keepWithNext=True,
)
H3 = ParagraphStyle(
    "H3", fontName=BOLD, fontSize=11.5, leading=15, textColor=BLUE,
    spaceBefore=3 * mm, spaceAfter=1.4 * mm, keepWithNext=True,
)
BODY_STYLE = ParagraphStyle(
    "Body", fontName=BODY, fontSize=9.25, leading=13.5, textColor=INK,
    spaceAfter=2.1 * mm,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY_STYLE, leftIndent=5 * mm, firstLineIndent=-3.5 * mm,
    spaceAfter=1.2 * mm,
)
SMALL = ParagraphStyle("Small", fontName=BODY, fontSize=7.8, leading=10, textColor=GRAY)
CALLOUT = ParagraphStyle(
    "Callout", fontName=BODY, fontSize=10.2, leading=15, textColor=NAVY,
    backColor=colors.HexColor("#EAF3F7"), borderColor=colors.HexColor("#B9D6E3"),
    borderWidth=0.7, borderPadding=9,
)


def esc(value):
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(value):
    value = esc(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", value)
    return value


def page(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.line(MX, PAGE_H - 13 * mm, PAGE_W - MX, PAGE_H - 13 * mm)
        canvas.setFont(BODY, 7.5)
        canvas.setFillColor(GRAY)
        canvas.drawString(MX, PAGE_H - 9.8 * mm, "SAR-RAG | Updated research and evidence status")
        canvas.drawRightString(PAGE_W - MX, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


class Document(BaseDocTemplate):
    def __init__(self):
        super().__init__(
            str(OUTPUT), pagesize=A4, leftMargin=MX, rightMargin=MX,
            topMargin=MT, bottomMargin=MB,
            title="SAR-RAG Complete Updated Research in Simple Words",
            author="Research Project",
        )
        frame = Frame(MX, MB, PAGE_W - 2 * MX, PAGE_H - MT - MB, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=page))


def cover():
    box = Table([[Paragraph("Representation-State<br/>Adaptive RAG", TITLE)], [Paragraph(
        "Complete updated research, algorithm, calibration, evidence, and honest conclusion",
        SUBTITLE,
    )]], colWidths=[PAGE_W - 2 * MX])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING", (0, 0), (-1, 0), 27 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 3 * mm),
        ("LEFTPADDING", (0, 0), (-1, -1), 12 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12 * mm),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 22 * mm),
    ]))
    decision = Paragraph(
        "The pipeline is implemented and training-calibrated. It uses a top-3 probe, "
        "normalized RRF to select one or two representations, adaptive retrieval with "
        "an emergent final k, post-retrieval deduplication, and one final LLM call. "
        "Overall superiority is not yet demonstrated.", CALLOUT,
    )
    meta = Table([
        ["Embedding", "all-MiniLM-L6-v2"],
        ["Answer model", "Groq openai/gpt-oss-120b"],
        ["Frozen calibration", "60 official training examples"],
        ["Report date", "13 August 2026"],
        ["Evidence status", "Full confirmatory evaluation pending"],
    ], colWidths=[43 * mm, 118 * mm])
    meta.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), BOLD), ("FONTNAME", (1, 0), (1, -1), BODY),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("BACKGROUND", (0, 0), (0, -1), PALE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
    ]))
    return [Spacer(1, 8 * mm), box, Spacer(1, 14 * mm), decision, Spacer(1, 12 * mm), meta, PageBreak()]


def pipeline():
    labels = ["Top-3 probe", "RRF select 1-2", "Adaptive batches", "Deduplicate", "One LLM call"]
    cells = []
    widths = []
    for index, label in enumerate(labels):
        cells.append(Paragraph(label, ParagraphStyle(
            f"p{index}", fontName=BOLD, fontSize=7.5, leading=9, textColor=colors.white,
            alignment=TA_CENTER,
        )))
        widths.append(27 * mm)
        if index < len(labels) - 1:
            cells.append(Paragraph("&gt;", ParagraphStyle(
                f"a{index}", fontName=BOLD, fontSize=11, textColor=TEAL, alignment=TA_CENTER,
            )))
            widths.append(6 * mm)
    table = Table([cells], colWidths=widths)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
    ]
    for column in range(0, len(cells), 2):
        commands.append(("BACKGROUND", (column, 0), (column, 0), NAVY))
    table.setStyle(TableStyle(commands))
    return [Paragraph("Pipeline at a glance", H2), table, Spacer(1, 3 * mm)]


def content():
    story = cover() + pipeline()
    paragraph = []

    def flush():
        nonlocal paragraph
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), BODY_STYLE))
            paragraph = []

    first_title = True
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if first_title and line.startswith("# "):
            first_title = False
            continue
        if line.startswith("## "):
            flush()
            story.append(Paragraph(inline(line[3:]), H2))
        elif line.startswith("### "):
            flush()
            story.append(Paragraph(inline(line[4:]), H3))
        elif line.startswith("- "):
            flush()
            story.append(Paragraph(inline("- " + line[2:]), BULLET))
        elif not line:
            flush()
        else:
            paragraph.append(line)
    flush()
    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    Document().build(content())
    print(OUTPUT)


if __name__ == "__main__":
    main()

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, KeepTogether, LongTable, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "experiments" / "pilot_combined" / "EVALUATION_REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "SAR_RAG_REPRODUCTION_EVALUATION.pdf"


def inline(value: str) -> str:
    value = html.escape(value.strip())
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<a href="\2" color="#1B5E7A">\1</a>', value)
    value = re.sub(r"&lt;(https?://[^&]+)&gt;", r'<a href="\1" color="#1B5E7A">\1</a>', value)
    return value


def page_decor(canvas, document) -> None:
    canvas.saveState()
    width, height = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#D6E2E8"))
    canvas.line(16 * mm, 13 * mm, width - 16 * mm, 13 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526873"))
    canvas.drawString(16 * mm, 8 * mm, "SAR-RAG reproduction and critical evaluation - pilot evidence only")
    canvas.drawRightString(width - 16 * mm, 8 * mm, f"Page {document.page}")
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleX", parent=base["Title"], fontName="Helvetica-Bold",
                                fontSize=22, leading=27, textColor=colors.HexColor("#123B4A"),
                                alignment=TA_CENTER, spaceAfter=12),
        "h1": ParagraphStyle("H1X", parent=base["Heading1"], fontName="Helvetica-Bold",
                             fontSize=15, leading=19, textColor=colors.HexColor("#0A6674"),
                             spaceBefore=10, spaceAfter=6, keepWithNext=True),
        "h2": ParagraphStyle("H2X", parent=base["Heading2"], fontName="Helvetica-Bold",
                             fontSize=12, leading=15, textColor=colors.HexColor("#174D5B"),
                             spaceBefore=8, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("BodyX", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=9.3, leading=13, textColor=colors.HexColor("#1F2C31"),
                               spaceAfter=5),
        "bullet": ParagraphStyle("BulletX", parent=base["BodyText"], fontName="Helvetica",
                                 fontSize=9.1, leading=12.5, leftIndent=14, firstLineIndent=-8,
                                 bulletIndent=4, spaceAfter=3),
        "cell": ParagraphStyle("CellX", parent=base["BodyText"], fontName="Helvetica",
                               fontSize=7.4, leading=9.2, textColor=colors.HexColor("#233239")),
        "header": ParagraphStyle("HeaderX", parent=base["BodyText"], fontName="Helvetica-Bold",
                                 fontSize=7.5, leading=9.4, textColor=colors.white),
        "caption": ParagraphStyle("CaptionX", parent=base["BodyText"], fontName="Helvetica-Oblique",
                                  fontSize=8, leading=10, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#526873"), spaceAfter=8),
    }


def table_flow(rows: list[list[str]], st) -> LongTable:
    count = max(len(row) for row in rows)
    usable = landscape(A4)[0] - 32 * mm
    if count == 5:
        weights = [1.7, 1.0, 1.0, .7, 2.4]
    elif count == 4:
        weights = [1.4, 1.0, 1.0, 2.2]
    elif count == 3:
        weights = [1.8, 1.0, 2.6]
    else:
        weights = [1] * count
    total = sum(weights)
    widths = [usable * weight / total for weight in weights]
    data = []
    for row_index, row in enumerate(rows):
        style = st["header"] if row_index == 0 else st["cell"]
        padded = row + [""] * (count - len(row))
        data.append([Paragraph(inline(cell), style) for cell in padded])
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0A6674")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#B5C8CF")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7F8")]),
    ]))
    return table


def build() -> None:
    st = styles()
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            story.append(Paragraph(inline(" ".join(paragraph_buffer)), st["body"]))
            paragraph_buffer.clear()

    index = 0
    first_title = True
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            flush_paragraph()
            index += 1
            continue
        if line.startswith("| "):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].rstrip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = [[cell.strip() for cell in value.strip("|").split("|")] for value in table_lines]
            if len(rows) > 1 and all(set(cell) <= {"-", ":"} for cell in rows[1]):
                rows.pop(1)
            story.extend([table_flow(rows, st), Spacer(1, 7)])
            continue
        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", line)
        if image_match:
            flush_paragraph()
            image_path = SOURCE.parent / image_match.group(2)
            picture = Image(str(image_path))
            max_width, max_height = 250 * mm, 78 * mm
            scale = min(max_width / picture.imageWidth, max_height / picture.imageHeight)
            picture.drawWidth = picture.imageWidth * scale
            picture.drawHeight = picture.imageHeight * scale
            story.extend([picture, Paragraph(inline(image_match.group(1)), st["caption"])])
            index += 1
            continue
        if line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(inline(line[2:]), st["title"] if first_title else st["h1"]))
            first_title = False
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline(line[3:]), st["h1"]))
        elif line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline(line[4:]), st["h2"]))
        elif re.match(r"^[-*] ", line):
            flush_paragraph()
            story.append(Paragraph(inline(line[2:]), st["bullet"], bulletText="-"))
        elif re.match(r"^\d+\. ", line):
            flush_paragraph()
            number, value = line.split(". ", 1)
            story.append(Paragraph(inline(value), st["bullet"], bulletText=number + "."))
        else:
            paragraph_buffer.append(line.replace("  ", " "))
        index += 1
    flush_paragraph()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=landscape(A4), rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=23 * mm, title="SAR-RAG Reproduction Evaluation",
        author="Research reproduction generated from recorded experiment artifacts",
    )
    document.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT)


if __name__ == "__main__":
    build()

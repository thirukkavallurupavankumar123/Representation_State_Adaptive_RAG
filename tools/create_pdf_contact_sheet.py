from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


root = Path(__file__).resolve().parents[1]
render_dir = root / "tmp" / "pdfs" / "sar_rag_render"
paths = sorted(render_dir.glob("page-*.png"))
thumb_w = 300
label_h = 25
gap = 12
cols = 3

thumbs = []
for index, path in enumerate(paths, 1):
    image = Image.open(path).convert("RGB")
    thumb_h = round(image.height * thumb_w / image.width)
    image.thumbnail((thumb_w, thumb_h))
    tile = Image.new("RGB", (thumb_w, thumb_h + label_h), "white")
    tile.paste(image, (0, label_h))
    ImageDraw.Draw(tile).text((8, 5), f"Page {index}", fill="#17324D")
    thumbs.append(tile)

rows = (len(thumbs) + cols - 1) // cols
tile_h = max(t.height for t in thumbs)
sheet = Image.new(
    "RGB",
    (cols * thumb_w + (cols + 1) * gap, rows * tile_h + (rows + 1) * gap),
    "#DDE5EA",
)
for index, tile in enumerate(thumbs):
    x = gap + (index % cols) * (thumb_w + gap)
    y = gap + (index // cols) * (tile_h + gap)
    sheet.paste(tile, (x, y))

out = render_dir / "contact-sheet.png"
sheet.save(out)
print(out)

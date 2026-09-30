"""Render every page of a PDF to PNG files for visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--zoom", type=float, default=1.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(args.pdf)
    matrix = fitz.Matrix(args.zoom, args.zoom)
    for index, page in enumerate(document):
        target = args.output_dir / f"page_{index + 1:02d}.png"
        page.get_pixmap(matrix=matrix, alpha=False).save(target)
    print({"pages": len(document), "output_dir": str(args.output_dir.resolve())})


if __name__ == "__main__":
    main()

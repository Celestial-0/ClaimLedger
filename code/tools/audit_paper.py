"""Unified paper audit quality gate for ClaimLedger.

Consolidates:
- Manuscript PDF validation (page limits, references page, post-reference caption check)
- Printed figure geometry audit (aspect ratio, IEEE column/text width, height bounds)
- Per-page vertical density and float-heavy page analysis

Usage:
    uv run --project code --with pypdf python code/tools/audit_paper.py [paper_pdf_path] [--checks {all,pdf,geometry,page-usage}]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pypdf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = REPOSITORY_ROOT / "paper" / "build" / "claimledger" / "claimledger.pdf"
PAPER_SRC = REPOSITORY_ROOT / "paper" / "src"

COLUMN_W_IN = 3.50  # IEEE single column
TEXT_W_IN = 7.16    # IEEE double column (full width)

# Figure file -> placement
PLACEMENTS = {
    "figure_conflict_density_scatter.pdf": "column",
    "figure_performance_storage.pdf": "text",
    "figure_benchmark_composition.pdf": "column",
    "figure_method_heatmap.pdf": "column",
    "figure_error_bars.pdf": "column",
    "figure_active_claim_counts.pdf": "column",
    "figure_radar_profile.pdf": "column",
    "figure_conversation_accuracy.pdf": "column",
    "figure_paired_accuracy_differences.pdf": "column",
    "figure_extraction_error_degradation.pdf": "column",
    "figure_multillm_reader_quality.pdf": "column",
    "figure_multillm_abstention.pdf": "column",
}

BOUNDS = {"column": (1.0, 2.6), "text": (1.4, 3.6)}
CAPTION_RE = re.compile(r"^\s*(?:Fig\.|Figure)\s+\d+\.")


def validate_pdf(pdf_path: Path) -> dict[str, Any]:
    reader = pypdf.PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    reference_page = next(
        (index + 1 for index, text in enumerate(pages) if re.search(r"^\s*References\s*$", text, re.MULTILINE)),
        None,
    )
    captions = [
        {"page": index + 1, "text": line.strip()}
        for index, text in enumerate(pages)
        for line in text.splitlines()
        if CAPTION_RE.match(line)
    ]
    after_references = [item for item in captions if reference_page is not None and item["page"] > reference_page]
    result = {
        "pdf": str(pdf_path),
        "pages": len(pages),
        "references_page": reference_page,
        "caption_count": len(captions),
        "captions_after_references": after_references,
        "passed": reference_page is not None and not after_references and len(pages) <= 14,
    }
    return result


def audit_geometry(fig_dir: Path | None = None) -> list[dict[str, Any]]:
    directory = fig_dir or (PAPER_SRC / "figures")
    results = []
    print(f"\n{'figure':<44} {'place':<7} {'export pt':>12} {'aspect':>7} {'printed in':>16}  flag")
    print("-" * 92)
    for name, placement in PLACEMENTS.items():
        path = directory / name
        if not path.exists():
            print(f"{name:<44} {placement:<7} {'MISSING':>12}")
            results.append({"figure": name, "placement": placement, "status": "MISSING"})
            continue
        box = pypdf.PdfReader(str(path)).pages[0].mediabox
        w = float(box.width)
        h = float(box.height)
        target_in = COLUMN_W_IN if placement == "column" else TEXT_W_IN
        printed_h = target_in * (h / w)
        aspect = w / h
        lo, hi = BOUNDS[placement]
        flag = ""
        if printed_h > hi:
            flag = f"<-- TOO TALL (>{hi}in): bad aspect export"
        elif printed_h < lo:
            flag = f"<-- very short (<{lo}in)"
        print(
            f"{name:<44} {placement:<7} {w:>6.0f}x{h:<5.0f} {aspect:>7.2f} "
            f"{target_in:>6.2f}x{printed_h:<6.2f}  {flag}"
        )
        results.append({
            "figure": name,
            "placement": placement,
            "width_pt": w,
            "height_pt": h,
            "aspect": aspect,
            "printed_h_in": printed_h,
            "flag": flag,
        })
    return results


def analyze_page_usage(pdf_path: Path) -> None:
    reader = pypdf.PdfReader(str(pdf_path))
    print(f"\n{'pg':>3} {'words':>6} {'imgs':>5} {'img_area%':>9}  note")
    print("-" * 38)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        words = len([w for w in text.split() if any(c.isalnum() for c in w)])
        page_area = float(page.mediabox.width) * float(page.mediabox.height)
        img_area = 0.0
        n_imgs = 0
        try:
            for _ in page.images:
                n_imgs += 1
        except Exception:
            pass
        try:
            resources = page.get("/Resources") or {}
            xobj = resources.get("/XObject")
            if xobj:
                for obj in xobj.get_object().values():
                    o = obj.get_object()
                    if o.get("/Subtype") == "/Form":
                        bb = o.get("/BBox")
                        if bb:
                            img_area += abs((float(bb[2]) - float(bb[0])) * (float(bb[3]) - float(bb[1])))
        except Exception:
            pass
        pct = (img_area / page_area * 100) if page_area else 0.0
        note = "<-- float-heavy / sparse" if words < 200 else ""
        print(f"{i+1:>3} {words:>6} {n_imgs:>5} {pct:>8.1f}  {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF, help="Path to compiled paper PDF")
    parser.add_argument(
        "--checks",
        choices=["all", "pdf", "geometry", "page-usage"],
        default="all",
        help="Checks to run (default: all)",
    )
    args = parser.parse_args()

    passed = True
    if args.checks in ("all", "pdf"):
        if not args.pdf.exists():
            print(f"ERROR: PDF not found at {args.pdf}", file=sys.stderr)
            return 1
        res = validate_pdf(args.pdf)
        print("\n--- PDF Validation ---")
        print(json.dumps(res, indent=2))
        if not res["passed"]:
            passed = False

    if args.checks in ("all", "geometry"):
        audit_geometry()

    if args.checks in ("all", "page-usage"):
        if args.pdf.exists():
            analyze_page_usage(args.pdf)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_PBRT_MODEL_EVAL = Path(
    "output/pbrt_real_threshold_amp_depth_finetune_replay_e30_p8_keepamp_rw030/eval_pbrt_val97_endpoint"
)
DEFAULT_PBRT_PROPAINTER_EVAL = Path("output/pbrt_propainter_seed123/evaluation")
DEFAULT_PBRT_DEPTHCAD_EVAL = Path("output/depthcad_holeaware_kinect_n1000_ft_pilot/eval_seed123")
DEFAULT_PBRT_BAYESTOF_EVAL = Path("output/approx_bayestof_cache_n100_nt5000")

DEFAULT_REAL_RAW9_EVAL = Path("output/real_raw9_masked_self_test_ratio10_thr1m_iq6")
DEFAULT_REAL_DEPTH_EVAL = Path("output/real_depth_masked_self_test_ratio10_thr1m")
DEFAULT_REAL_LFRD2_EVAL = Path("output/lfrd2_raw9_masked_self_test_anchor_fliplr")
DEFAULT_REAL_HOLE_DIAG = Path("output/pbrt_real_new_selection/oneclick_compare/realhole_method_comparison")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a unified PBRT + real method comparison report.")
    parser.add_argument("--output_dir", type=Path, default=Path("output/unified_method_comparison"))
    parser.add_argument("--pbrt_model_eval", type=Path, default=DEFAULT_PBRT_MODEL_EVAL)
    parser.add_argument("--pbrt_propainter_eval", type=Path, default=DEFAULT_PBRT_PROPAINTER_EVAL)
    parser.add_argument("--pbrt_depthcad_eval", type=Path, default=DEFAULT_PBRT_DEPTHCAD_EVAL)
    parser.add_argument("--pbrt_bayestof_eval", type=Path, default=DEFAULT_PBRT_BAYESTOF_EVAL)
    parser.add_argument("--real_raw9_eval", type=Path, default=DEFAULT_REAL_RAW9_EVAL)
    parser.add_argument("--real_depth_eval", type=Path, default=DEFAULT_REAL_DEPTH_EVAL)
    parser.add_argument("--real_lfrd2_eval", type=Path, default=DEFAULT_REAL_LFRD2_EVAL)
    parser.add_argument("--real_hole_diag", type=Path, default=DEFAULT_REAL_HOLE_DIAG)
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def maybe_load_summary(path):
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    if not path.exists():
        return None, path
    return load_json(path), path


def fmt_float(value, digits=6):
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pct(value):
    if value is None:
        return ""
    return f"{float(value) * 100.0:.1f}%"


def standard_row(method, summary, prefix, note="", source=None):
    agg = summary["aggregate"]
    return {
        "benchmark": "pbrt",
        "method": method,
        "global_mae": agg.get(f"{prefix}_global_mae"),
        "hole_mae": agg.get(f"{prefix}_hole_mae"),
        "valid_mae": agg.get(f"{prefix}_valid_mae"),
        "anchor_mae": agg.get("anchor_hole_mae"),
        "improve_vs_anchor": None,
        "source": str(source) if source else "",
        "note": note,
    }


def selftest_row(method, summary, prefix="model", note="", source=None):
    agg = summary["aggregate"]
    return {
        "benchmark": "real_masked_self_test",
        "method": method,
        "global_mae": agg.get(f"{prefix}_global_mae"),
        "hole_mae": agg.get(f"{prefix}_mask_mae"),
        "valid_mae": agg.get(f"{prefix}_unmasked_mae"),
        "anchor_mae": agg.get("anchor_mask_mae"),
        "improve_vs_anchor": agg.get("mask_improve_vs_anchor"),
        "source": str(source) if source else "",
        "note": note,
    }


def lfrd2_row(method, summary, note="", source=None):
    agg = summary["aggregate"]
    return {
        "benchmark": "real_masked_self_test",
        "method": method,
        "global_mae": agg.get("lfrd2_full_global_mae_mean"),
        "hole_mae": agg.get("lfrd2_mask_mae_mean"),
        "valid_mae": agg.get("lfrd2_unmasked_mae_mean"),
        "anchor_mae": agg.get("anchor_mask_mae_mean"),
        "improve_vs_anchor": agg.get("mask_improvement_vs_anchor"),
        "source": str(source) if source else "",
        "note": note,
    }


def bayestof_row(method, summary, note="", source=None):
    agg = summary["aggregate"]
    return {
        "benchmark": "pbrt",
        "method": method,
        "global_mae": agg.get("bayestof_global_mae"),
        "hole_mae": agg.get("bayestof_hole_mae"),
        "valid_mae": agg.get("bayestof_valid_mae"),
        "anchor_mae": None,
        "improve_vs_anchor": None,
        "source": str(source) if source else "",
        "note": note,
    }


def hole_diag_rows(summary):
    rows = []
    if not summary:
        return rows
    for row in summary.get("rows", []):
        hole_source = Path(row["hole_source"])
        if hole_source.exists():
            mask = np.load(hole_source).astype(bool)
            hole_pixels = int(mask.sum())
            hole_ratio = hole_pixels / float(mask.size)
        else:
            hole_pixels = None
            hole_ratio = row.get("hole_ratio")
        rows.append(
            {
                "sample": row["sample"],
                "hole_pixels": hole_pixels,
                "hole_ratio": hole_ratio,
                "source": row["hole_source"],
            }
        )
    return rows


def write_markdown_table(f, headers, rows):
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for row in rows:
        f.write(
            "| "
            + " | ".join(
                [
                    str(row.get(h.lower().replace(" ", "_"), row.get(h, "")))
                    if h not in {"Global MAE", "Hole MAE", "Valid MAE", "Anchor MAE", "Improve vs Anchor", "Hole Ratio"}
                    else ""
                    for h in headers
                ]
            )
            + " |\n"
        )


def write_section_table(f, headers, rows, formatters):
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for row in rows:
        cells = []
        for key in formatters:
            value = row.get(key)
            cells.append(formatters[key](value))
        f.write("| " + " | ".join(cells) + " |\n")


def write_report(path, sections):
    with open(path, "w") as f:
        f.write("# Unified Method Comparison\n\n")
        for title, subtitle, rows, headers, formatters in sections:
            f.write(f"## {title}\n")
            if subtitle:
                f.write(f"{subtitle}\n\n")
            if rows:
                write_section_table(f, headers, rows, formatters)
            else:
                f.write("No data.\n")
            f.write("\n")


def flatten_rows(rows):
    out = []
    for row in rows:
        out.append({k: ("" if v is None else v) for k, v in row.items()})
    return out


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = [
        "benchmark",
        "method",
        "global_mae",
        "hole_mae",
        "valid_mae",
        "anchor_mae",
        "improve_vs_anchor",
        "source",
        "note",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_sections(args):
    all_rows = []
    sections = []

    pbrt_model_summary, pbrt_model_path = maybe_load_summary(args.pbrt_model_eval)
    pbrt_propainter_summary, pbrt_propainter_path = maybe_load_summary(args.pbrt_propainter_eval)
    pbrt_depthcad_summary, pbrt_depthcad_path = maybe_load_summary(args.pbrt_depthcad_eval)
    pbrt_bayestof_summary, pbrt_bayestof_path = maybe_load_summary(args.pbrt_bayestof_eval)

    pbrt_rows = []
    if pbrt_model_summary:
        pbrt_rows.extend(
            [
                standard_row("Noisy input", pbrt_model_summary, "noisy", source=pbrt_model_path),
                standard_row("Stage-1 base", pbrt_model_summary, "base", source=pbrt_model_path),
                standard_row("Ours (Depth flow)", pbrt_model_summary, "model", source=pbrt_model_path),
            ]
        )
        all_rows.extend(pbrt_rows)
    if pbrt_propainter_summary:
        pbrt_rows.extend(
            [
                standard_row("NS anchor r15", pbrt_propainter_summary, "opencv_ns_r15", source=pbrt_propainter_path),
                standard_row("ProPainter", pbrt_propainter_summary, "propainter", source=pbrt_propainter_path),
            ]
        )
        all_rows.extend(pbrt_rows[-2:])
    if pbrt_depthcad_summary:
        pbrt_rows.append(
            standard_row("DepthCAD-HoleAware", pbrt_depthcad_summary, "model", source=pbrt_depthcad_path)
        )
        all_rows.append(pbrt_rows[-1])
    if pbrt_bayestof_summary:
        pbrt_rows.append(
            bayestof_row(
                "BayesToF (approx)",
                pbrt_bayestof_summary,
                source=pbrt_bayestof_path,
                note="Approximate IQ wavelet baseline derived from BayesToF; not full tap-count reproduction.",
            )
        )
        all_rows.append(pbrt_rows[-1])
    sections.append(
        (
            "PBRT Holdout",
            "Ground-truth benchmark. Lower is better.",
            pbrt_rows,
            ["Method", "Global MAE", "Hole MAE", "Valid MAE", "Source", "Note"],
            {
                "method": lambda v: str(v),
                "global_mae": fmt_float,
                "hole_mae": fmt_float,
                "valid_mae": fmt_float,
                "source": lambda v: str(v),
                "note": lambda v: str(v),
            },
        )
    )

    real_raw9_summary, real_raw9_path = maybe_load_summary(args.real_raw9_eval)
    real_depth_summary, real_depth_path = maybe_load_summary(args.real_depth_eval)
    real_lfrd2_summary, real_lfrd2_path = maybe_load_summary(args.real_lfrd2_eval)

    real_rows = []
    if real_raw9_summary:
        real_rows.append(
            selftest_row(
                "Raw9 flow (ours)",
                real_raw9_summary,
                source=real_raw9_path,
                note="masked self-test on paired real raw9/depth",
            )
        )
        all_rows.append(real_rows[-1])
    if real_depth_summary:
        real_rows.append(
            selftest_row(
                "Depth-only self-test",
                real_depth_summary,
                source=real_depth_path,
                note="paired depth only",
            )
        )
        all_rows.append(real_rows[-1])
    if real_lfrd2_summary:
        real_rows.append(
            lfrd2_row(
                "LFRD2 proxy",
                real_lfrd2_summary,
                source=real_lfrd2_path,
                note="cross-sensor proxy baseline",
            )
        )
        all_rows.append(real_rows[-1])
    sections.append(
        (
            "Real Masked Self-Test",
            "Artificial masks on paired real raw9/depth. Lower is better.",
            real_rows,
            ["Method", "Global MAE", "Mask MAE", "Unmasked MAE", "Anchor MAE", "Improve vs Anchor", "Note", "Source"],
            {
                "method": lambda v: str(v),
                "global_mae": fmt_float,
                "hole_mae": fmt_float,
                "valid_mae": fmt_float,
                "anchor_mae": fmt_float,
                "improve_vs_anchor": fmt_pct,
                "note": lambda v: str(v),
                "source": lambda v: str(v),
            },
        )
    )

    hole_summary, hole_summary_path = maybe_load_summary(args.real_hole_diag)
    hole_rows = hole_diag_rows(hole_summary)
    mean_ratio = float(np.mean([r["hole_ratio"] for r in hole_rows])) if hole_rows else None
    sections.append(
        (
            "Real Observed-Hole Diagnostic",
            (
                "Observed holes are sparse here; use this section for qualitative inspection only. "
                f"Mean hole ratio: {fmt_pct(mean_ratio)}."
                if mean_ratio is not None
                else "Observed holes are sparse here; use this section for qualitative inspection only."
            ),
            hole_rows,
            ["Sample", "Hole Pixels", "Hole Ratio", "Source"],
            {
                "sample": lambda v: str(v),
                "hole_pixels": lambda v: "" if v is None else str(v),
                "hole_ratio": fmt_pct,
                "source": lambda v: str(v),
            },
        )
    )

    summary = {
        "sections": [
            {
                "title": title,
                "subtitle": subtitle,
                "rows": rows,
            }
            for title, subtitle, rows, _headers, _formatters in sections
        ],
        "source_paths": {
            "pbrt_model_eval": str(pbrt_model_path),
            "pbrt_propainter_eval": str(pbrt_propainter_path),
            "pbrt_depthcad_eval": str(pbrt_depthcad_path),
            "pbrt_bayestof_eval": str(pbrt_bayestof_path),
            "real_raw9_eval": str(real_raw9_path),
            "real_depth_eval": str(real_depth_path),
            "real_lfrd2_eval": str(real_lfrd2_path),
            "real_hole_diag": str(hole_summary_path),
        },
        "rows": all_rows,
    }
    return sections, summary


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    sections, summary = build_sections(args)

    report_path = args.output_dir / "summary.md"
    csv_path = args.output_dir / "summary.csv"
    json_path = args.output_dir / "summary.json"

    write_report(report_path, sections)
    write_csv(csv_path, summary["rows"])
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(report_path)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()

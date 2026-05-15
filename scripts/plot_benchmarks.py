#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY_ORDER: List[str] = [
    "Naive",
    "Brent-Kung",
    "Kinoshita-Li",
]

def _variant_chains(prefix: str = "ntt") -> Dict[str, List[Tuple[str, str]]]:
    return {
        "Naive": [
            (f"{prefix}/naive_horner", "Horner"),
        ],
        "Brent-Kung": [
            (f"{prefix}/brent_kung_basic", "basic"),
            (f"{prefix}/brent_kung_opt", "opt"),
            (f"{prefix}/brent_kung_streaming", "streaming"),
            (f"{prefix}/brent_kung_tuned_m", "tuned_m"),
        ],
        "Kinoshita-Li": [
            (f"{prefix}/kl_dual_basic", "basic"),
            (f"{prefix}/kl_dual_inplace", "inplace"),
            (f"{prefix}/kl_dual_truncated_mul", "trunc_mul"),
            (f"{prefix}/kl_dual_threshold", "threshold"),
        ],
    }

VARIANT_CHAINS = _variant_chains("ntt")
MEM_VARIANT_CHAINS = _variant_chains("mem")

def parse_json(path: str) -> Dict[str, Dict[int, float]]:
    import json
    results: Dict[str, Dict[int, float]] = {}
    with open(path) as f:
        doc = json.load(f)
    for bm in doc.get("benchmarks", []):
        full_name = bm.get("name", "")
        pieces = full_name.rsplit("/", 1)
        if len(pieces) != 2:
            continue
        bench_name, n_str = pieces
        try:
            n = int(n_str)
        except ValueError:
            continue
        t = bm.get("real_time", bm.get("cpu_time", 0.0))
        unit = bm.get("time_unit", "ms")
        if unit == "ns":
            t /= 1e6
        elif unit == "us":
            t /= 1e3
        if t < 1e-9:
            continue
        results.setdefault(bench_name, {})[n] = t
    return results


def parse_csv_text(path: str) -> Dict[str, Dict[int, float]]:
    results: Dict[str, Dict[int, float]] = {}
    with open(path) as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("name,"):
            header_idx = i
            break
    if header_idx is None:
        return results

    cols = lines[header_idx].strip().split(",")
    name_col = cols.index("name")
    time_col = None
    for cand in ("real_time", "cpu_time"):
        if cand in cols:
            time_col = cols.index(cand)
            break
    if time_col is None:
        return results
    unit_col = cols.index("time_unit") if "time_unit" in cols else None

    for line in lines[header_idx + 1:]:
        parts = line.strip().split(",")
        if len(parts) <= max(name_col, time_col):
            continue
        full_name = parts[name_col].strip('"')
        pieces = full_name.rsplit("/", 1)
        if len(pieces) != 2:
            continue
        bench_name, n_str = pieces
        try:
            n = int(n_str)
            t = float(parts[time_col])
        except ValueError:
            continue

        unit = parts[unit_col].strip('"') if unit_col and unit_col < len(parts) else "ms"
        if unit == "ns":
            t /= 1e6
        elif unit == "us":
            t /= 1e3

        if t < 1e-9:
            continue
        results.setdefault(bench_name, {})[n] = t

    return results


def parse_benchmark_file(path: str) -> Dict[str, Dict[int, float]]:
    with open(path) as f:
        first = f.read(1).strip()
    if first == "{":
        raw = parse_json(path)
    else:
        raw = parse_csv_text(path)
    return _filter_skipped(raw)


def _filter_skipped(data: Dict[str, Dict[int, float]]) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for bench, nd in data.items():
        ns_sorted = sorted(nd.keys())
        if not ns_sorted:
            continue
        filtered: Dict[int, float] = {}
        peak_so_far = 0.0
        for n in ns_sorted:
            t = nd[n]
            if peak_so_far > 0.1 and t < peak_so_far * 0.01:
                continue
            peak_so_far = max(peak_so_far, t)
            filtered[n] = t
        if filtered:
            out[bench] = filtered
    return out


def plot_time_heatmap(data: Dict[str, Dict[int, float]], out_dir: str) -> None:
    all_variants: List[Tuple[str, str, str]] = []
    for fam in FAMILY_ORDER:
        for bench, label in VARIANT_CHAINS[fam]:
            if bench in data:
                all_variants.append((fam, label, bench))

    if not all_variants:
        print("  [skip] time heatmap: no data")
        return

    all_ns = sorted(set(n for bench in data for n in data[bench]))
    all_ns = [n for n in all_ns if any(
        n in data.get(b, {}) for _, _, b in all_variants
    )]

    matrix = np.full((len(all_variants), len(all_ns)), np.nan)
    row_labels = []
    for i, (fam, label, bench) in enumerate(all_variants):
        row_labels.append(f"{fam} / {label}")
        for j, n in enumerate(all_ns):
            if n in data.get(bench, {}):
                t = data[bench][n]
                if t > 1e-9:
                    matrix[i, j] = t

    log_matrix = np.where(np.isnan(matrix), np.nan, np.log10(matrix))

    fig, ax = plt.subplots(figsize=(max(6, len(all_ns) * 1.5),
                                     max(5, len(all_variants) * 0.5)))

    valid_mask = ~np.isnan(log_matrix)
    vmin = np.nanmin(log_matrix) if valid_mask.any() else -3
    vmax = np.nanmax(log_matrix) if valid_mask.any() else 3

    masked = np.ma.masked_invalid(log_matrix)
    im = ax.imshow(masked, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)

    for i in range(len(all_variants)):
        for j in range(len(all_ns)):
            if not np.isnan(matrix[i, j]):
                t = matrix[i, j]
                text = f"{t:.1f}" if t >= 1 else f"{t:.2f}" if t >= 0.01 else f"{t:.3f}"
                color = "white" if log_matrix[i, j] > (vmin + vmax) / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
            else:
                ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="#999999")

    ax.set_xticks(range(len(all_ns)))
    ax.set_xticklabels([str(n) for n in all_ns], fontsize=9)
    ax.set_yticks(range(len(all_variants)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("n", fontsize=11)
    ax.set_title("Time (ms) \u2014 all variants \u00d7 sizes", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log\u2081\u2080(ms)", fontsize=10)

    prev_fam = None
    for i, (fam, _, _) in enumerate(all_variants):
        if prev_fam is not None and fam != prev_fam:
            ax.axhline(y=i - 0.5, color="black", linewidth=1.0, alpha=0.5)
        prev_fam = fam

    fig.tight_layout()
    out_path = os.path.join(out_dir, "heatmap_time.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def _parse_memory_json(path: str) -> Dict[str, Dict[int, float]]:
    import json
    results: Dict[str, Dict[int, float]] = {}
    with open(path) as f:
        doc = json.load(f)
    for bm in doc.get("benchmarks", []):
        full_name = bm.get("name", "")
        pieces = full_name.rsplit("/", 1)
        if len(pieces) != 2:
            continue
        bench_name, n_str = pieces
        try:
            n = int(n_str)
        except ValueError:
            continue
        kb = bm.get("peak_rss_kB", 0.0)
        mb = kb / 1024.0
        if mb <= 0:
            continue
        results.setdefault(bench_name, {})[n] = mb
    return results


def _parse_memory_csv_text(path: str) -> Dict[str, Dict[int, float]]:
    results: Dict[str, Dict[int, float]] = {}
    with open(path) as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("name,"):
            header_idx = i
            break
    if header_idx is None:
        return results

    cols = lines[header_idx].strip().split(",")
    name_col = cols.index("name")
    rss_col = None
    for cand in ("peak_rss_kB",):
        if cand in cols:
            rss_col = cols.index(cand)
            break
    if rss_col is None:
        return results

    for line in lines[header_idx + 1:]:
        parts = line.strip().split(",")
        if len(parts) <= max(name_col, rss_col):
            continue
        full_name = parts[name_col].strip('"')
        pieces = full_name.rsplit("/", 1)
        if len(pieces) != 2:
            continue
        bench_name, n_str = pieces
        try:
            n = int(n_str)
            kb = float(parts[rss_col])
        except ValueError:
            continue
        mb = kb / 1024.0
        if mb <= 0:
            continue
        results.setdefault(bench_name, {})[n] = mb

    return results


def parse_memory_csv(path: str) -> Dict[str, Dict[int, float]]:
    with open(path) as f:
        first = f.read(1).strip()
    if first == "{":
        return _parse_memory_json(path)
    return _parse_memory_csv_text(path)


def plot_memory_heatmap(data: Dict[str, Dict[int, float]], out_dir: str) -> None:
    chains = MEM_VARIANT_CHAINS
    all_variants: List[Tuple[str, str, str]] = []
    for fam in FAMILY_ORDER:
        for bench, label in chains[fam]:
            if bench in data:
                all_variants.append((fam, label, bench))

    if not all_variants:
        print("  [skip] memory heatmap: no data")
        return

    all_ns = sorted(set(n for bench in data for n in data[bench]))
    all_ns = [n for n in all_ns if any(
        n in data.get(b, {}) for _, _, b in all_variants
    )]

    matrix = np.full((len(all_variants), len(all_ns)), np.nan)
    row_labels = []
    for i, (fam, label, bench) in enumerate(all_variants):
        row_labels.append(f"{fam} / {label}")
        for j, n in enumerate(all_ns):
            if n in data.get(bench, {}):
                matrix[i, j] = data[bench][n]

    log_matrix = np.where(
        np.isnan(matrix) | (matrix <= 0), np.nan, np.log10(matrix)
    )

    fig, ax = plt.subplots(figsize=(max(6, len(all_ns) * 1.5),
                                     max(5, len(all_variants) * 0.5)))

    valid_mask = ~np.isnan(log_matrix)
    vmin = np.nanmin(log_matrix) if valid_mask.any() else -2
    vmax = np.nanmax(log_matrix) if valid_mask.any() else 4

    masked = np.ma.masked_invalid(log_matrix)
    im = ax.imshow(masked, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)

    for i in range(len(all_variants)):
        for j in range(len(all_ns)):
            if not np.isnan(matrix[i, j]):
                mb = matrix[i, j]
                if mb >= 1024:
                    text = f"{mb/1024:.1f} GB"
                elif mb >= 1:
                    text = f"{mb:.1f} MB"
                else:
                    text = f"{mb*1024:.0f} kB"
                lv = log_matrix[i, j] if not np.isnan(log_matrix[i, j]) else vmin
                color = "white" if lv > (vmin + vmax) / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)
            else:
                ax.text(j, i, "--", ha="center", va="center", fontsize=6, color="#999999")

    ax.set_xticks(range(len(all_ns)))
    ax.set_xticklabels([str(n) for n in all_ns], fontsize=9)
    ax.set_yticks(range(len(all_variants)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("n", fontsize=11)
    ax.set_title("Peak RSS (MB) \u2014 all variants \u00d7 sizes", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log\u2081\u2080(MB)", fontsize=10)

    prev_fam = None
    for i, (fam, _, _) in enumerate(all_variants):
        if prev_fam is not None and fam != prev_fam:
            ax.axhline(y=i - 0.5, color="black", linewidth=1.0, alpha=0.5)
        prev_fam = fam

    fig.tight_layout()
    out_path = os.path.join(out_dir, "heatmap_memory.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def parse_precision_csv(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "n": int(row["n"]),
                "algo_a": row["algo_a"],
                "algo_b": row["algo_b"],
                "rel_max": float(row["rel_max"]),
            })
    return rows


def plot_precision_heatmap(rows: List[dict], out_dir: str) -> None:
    ns = sorted(set(r["n"] for r in rows))
    algos = sorted(set(r["algo_a"] for r in rows) | set(r["algo_b"] for r in rows))

    if not ns or not algos:
        print("  [skip] precision heatmap: no data")
        return

    for target_n in ns:
        n_rows = [r for r in rows if r["n"] == target_n]
        n_algos = sorted(set(r["algo_a"] for r in n_rows) | set(r["algo_b"] for r in n_rows))
        idx = {a: i for i, a in enumerate(n_algos)}
        sz = len(n_algos)
        matrix = np.full((sz, sz), np.nan)

        for r in n_rows:
            i, j = idx[r["algo_a"]], idx[r["algo_b"]]
            val = r["rel_max"]
            matrix[i, j] = val
            matrix[j, i] = val
        np.fill_diagonal(matrix, 0.0)

        safe = np.where((matrix > 0) & ~np.isnan(matrix), matrix, np.nan)
        log_matrix = np.where(np.isnan(safe), np.nan, np.log10(safe))
        log_matrix = np.where(matrix == 0, np.nan, log_matrix)

        fig, ax = plt.subplots(figsize=(max(7, sz * 0.7), max(6, sz * 0.6)))

        valid = log_matrix[~np.isnan(log_matrix)]
        if valid.size > 0:
            vmin, vmax = valid.min(), valid.max()
        else:
            vmin, vmax = -20, 0

        display = np.where(matrix == 0, vmin - 1, log_matrix)
        display = np.where(np.isnan(display), vmin - 1, display)
        im = ax.imshow(display, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)

        for i in range(sz):
            for j in range(sz):
                if i == j:
                    ax.text(j, i, "=", ha="center", va="center", fontsize=6,
                            color="#999999")
                elif not np.isnan(matrix[i, j]):
                    v = matrix[i, j]
                    if v == 0:
                        text = "0"
                    else:
                        exp = int(math.floor(math.log10(v)))
                        text = f"{exp}"
                    lv = log_matrix[i, j] if not np.isnan(log_matrix[i, j]) else vmin
                    color = "white" if lv > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=6,
                            color=color)
                else:
                    ax.text(j, i, "--", ha="center", va="center", fontsize=6,
                            color="#999999")

        ax.set_xticks(range(sz))
        ax.set_xticklabels(n_algos, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(sz))
        ax.set_yticklabels(n_algos, fontsize=7)
        ax.set_title(f"Pairwise rel_max error (log\u2081\u2080), n={target_n}", fontsize=11)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("log\u2081\u2080(rel_max)", fontsize=9)

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"heatmap_precision_n{target_n}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark heatmaps")
    parser.add_argument("mode", choices=["time", "memory", "precision"],
                        help="'time' / 'memory' for Google Benchmark CSV; "
                             "'precision' for capture_precision.py CSV")
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: results)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "time":
        data = parse_benchmark_file(args.csv)
        print(f"Parsed {sum(len(v) for v in data.values())} data points "
              f"from {len(data)} benchmarks")
        print("\nGenerating figure:")
        plot_time_heatmap(data, args.output_dir)
    elif args.mode == "memory":
        data = parse_memory_csv(args.csv)
        print(f"Parsed {sum(len(v) for v in data.values())} data points "
              f"from {len(data)} benchmarks")
        print("\nGenerating figure:")
        plot_memory_heatmap(data, args.output_dir)
    else:
        rows = parse_precision_csv(args.csv)
        print(f"Parsed {len(rows)} pairwise comparisons")
        print("\nGenerating figures:")
        plot_precision_heatmap(rows, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

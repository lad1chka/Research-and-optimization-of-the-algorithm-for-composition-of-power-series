#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys


LINE_RE = re.compile(
    r"\[FFT n=(\d+)\]\s+(\S+)\s+vs\s+(\S+):\s+.*"
    r"abs\{.*max=([\d.e+\-]+).*\}\s+"
    r"rel\{max=([\d.e+\-]+)\s+mean=([\d.e+\-]+)\s+q90=([\d.e+\-]+)\}"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-binary",
                        default="build/tests/test_fft_cross_algorithm")
    parser.add_argument("--output", default="results/precision.csv")
    parser.add_argument("--filter", default="FFTPrecision.Double*",
                        help="gtest filter (default: double only)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Running {args.test_binary} --gtest_filter='{args.filter}' ...")
    proc = subprocess.run(
        [args.test_binary, f"--gtest_filter={args.filter}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"Test exited with code {proc.returncode}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)

    rows = []
    for line in proc.stdout.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            rows.append({
                "n": int(m.group(1)),
                "algo_a": m.group(2),
                "algo_b": m.group(3),
                "abs_max": float(m.group(4)),
                "rel_max": float(m.group(5)),
                "rel_mean": float(m.group(6)),
                "rel_q90": float(m.group(7)),
            })

    if not rows:
        print("No precision data parsed!", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

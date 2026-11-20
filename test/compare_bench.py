#!/usr/bin/env python3
"""Compare bench_moptim outputs: best time per label and checksum agreement.

Usage: compare_bench.py DIR [--ref LABEL] [--filter SUBSTR] [--tol 1e-12]

DIR holds outputs of bench_moptim (or of anything printing the same rows) named
`<label>.<round>`, e.g. gcc.1 gcc.2 t1.1 t1.2. For every (row name, elements) the table gives the
minimum time per label over the rounds, the ratio to the reference label and the largest relative
difference of the `key=value` numbers of the notes column against the reference. A label missing a
row prints "-". Differences above --tol are flagged with `!`.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# "<name> <elements> <time> us <ns/elem>  <notes>"; the name may contain spaces and digits, so it is
# matched lazily up to the first "<int> <float> us <float>" that follows it.
ROW = re.compile(r"^(?P<name>.+?)\s+(?P<n>\d+)\s+(?P<us>\d+(?:\.\d+)?) us\s+\d+(?:\.\d+)?(?:\s+(?P<notes>.*))?$")
KEY_VALUE = re.compile(r"(\w+)=(\S+)")


def parse_notes(text):
    values = {}
    for key, raw in KEY_VALUE.findall(text or ""):
        try:
            values[key] = float(raw)
        except ValueError:
            pass  # not a number (e.g. a status name)
    return values


def relative_difference(a, b):
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def load(directory):
    """Returns {label: {(name, elements): [(us, notes), ...]}} over all rounds."""
    data = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or "." not in path.name:
            continue
        label = path.name.rsplit(".", 1)[0]
        for line in path.read_text(errors="replace").splitlines():
            match = ROW.match(line.rstrip())
            if match:
                key = (match["name"], int(match["n"]))
                data[label][key].append((float(match["us"]), parse_notes(match["notes"])))
    return data


def checksum_difference(runs, reference_runs):
    """Largest relative difference of any note between any run and the first reference run."""
    expected = reference_runs[0][1]
    worst = 0.0
    for _, notes in runs:
        for key, value in notes.items():
            if key in expected:
                worst = max(worst, relative_difference(value, expected[key]))
            else:
                return float("inf")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--ref", help="reference label (default: gcc if present, else the first label)")
    parser.add_argument("--filter", default="", help="only rows whose name contains this substring")
    parser.add_argument("--tol", type=float, default=1e-12, help="checksum tolerance (default 1e-12)")
    args = parser.parse_args()

    data = load(args.directory)
    if not data:
        sys.exit(f"no benchmark rows found in {args.directory}")
    labels = sorted(data)
    ref = args.ref or ("gcc" if "gcc" in data else labels[0])
    if ref not in data:
        sys.exit(f"reference label '{ref}' not found; labels: {', '.join(labels)}")
    labels.remove(ref)
    labels.insert(0, ref)

    rows = sorted({key for runs in data.values() for key in runs if args.filter in key[0]},
                  key=lambda k: (k[0], k[1]))
    failures = 0
    for name, n in rows:
        print(f"\n{name}  elements={n}")
        print(f"  {'label':<14} {'rounds':>6} {'min us':>12} {'ratio':>8}  checksum max rel diff")
        reference_runs = data[ref].get((name, n))
        reference_us = min(us for us, _ in reference_runs) if reference_runs else None
        for label in labels:
            runs = data[label].get((name, n))
            if not runs:
                print(f"  {label:<14} {'-':>6}")
                continue
            best = min(us for us, _ in runs)
            ratio = f"{best / reference_us:8.3f}" if reference_us else f"{'-':>8}"
            if label == ref or not reference_runs:
                check = "(reference)" if label == ref else "-"
                if label == ref and any(notes for _, notes in runs):
                    # Rounds of the reference itself should agree with each other.
                    spread = checksum_difference(runs, reference_runs)
                    check = f"{spread:.2e}" + (" !" if spread > args.tol else " (reference)")
                    failures += spread > args.tol
            else:
                diff = checksum_difference(runs, reference_runs)
                check = f"{diff:.2e}" + (" !" if diff > args.tol else "")
                failures += diff > args.tol
            print(f"  {label:<14} {len(runs):>6} {best:12.2f} {ratio}  {check}")
    if failures:
        print(f"\n{failures} checksum difference(s) above tolerance {args.tol:g}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

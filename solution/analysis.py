"""Paired image/session bootstrap and class-size statistics from real runs."""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, _prf, parse_output


def image_counts(entries, predictions):
    if set(predictions) != {entry["image_id"] for entry in entries}:
        raise ValueError("Predictions must cover the exact evaluation split")
    rows = []
    for entry in entries:
        per, _ = _match(_gt_boxes(entry), parse_output(predictions[entry["image_id"]]) or [])
        rows.append(tuple(sum(v[k] for v in per.values()) for k in range(3)))
    return rows


def paired_ci(entries, first, second, n_boot=2000, seed=0, by_session=False):
    """CI for F1(second)-F1(first); the same sampled units serve both models."""
    groups = defaultdict(list)
    for i, entry in enumerate(entries):
        groups[entry["session"] if by_session else i].append(i)
    units = list(groups.values())
    rng = random.Random(seed)
    differences = []
    for _ in range(n_boot):
        selected = [index for _ in units for index in rng.choice(units)]
        totals = [tuple(sum(rows[i][k] for i in selected) for k in range(3))
                  for rows in (first, second)]
        differences.append(_prf(*totals[1])["f1"]-_prf(*totals[0])["f1"])
    differences.sort()
    return [differences[int(0.025*n_boot)], differences[int(0.975*n_boot)-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["e0", "e1", "e2", "e3", "e4_data_0.1"])
    args = ap.parse_args()
    entries = read_jsonl("data/splits/test.jsonl")
    counts = {}
    for name in args.names:
        predictions = {r["image_id"]: r["raw"] for r in read_jsonl(f"work/{name}_test.jsonl")}
        counts[name] = image_counts(entries, predictions)
    paired = {}
    for i, first in enumerate(args.names):
        for second in args.names[i+1:]:
            paired[f"{second}-{first}"] = dict(
                delta_f1=_prf(*(sum(r[k] for r in counts[second]) for k in range(3)))["f1"]
                         - _prf(*(sum(r[k] for r in counts[first]) for k in range(3)))["f1"],
                image_ci95=paired_ci(entries, counts[first], counts[second]),
                session_ci95=paired_ci(entries, counts[first], counts[second], by_session=True))
    area = {c: [] for c in CLASSES}
    for entry in entries:
        for obj in entry["objects"]:
            area[obj["label"]].append(obj["yolo"][2]*obj["yolo"][3]*100)
    summary = dict(n_test_sessions=len({e["session"] for e in entries}), n_boot=2000, seed=0,
        paired_f1_differences=paired, median_box_area_percent={c: statistics.median(v) for c, v in area.items()})
    Path("report/statistical_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

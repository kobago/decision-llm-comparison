"""Aggregate results/*.jsonl into results/summary.json and a markdown table.

Per question: prediction = argmax of the option probabilities.
  accuracy  - prediction == gold
  brier     - sum over options of (p - onehot(gold))^2, averaged (0 is best, 2 is worst)
  ece       - expected calibration error of the top-1 probability, 10 equal-width bins
Missing/failed answers count as wrong, with a uniform distribution for Brier/ECE.
Latency is per record (all questions of a record in one call), skipping the first 3 as warm-up.
"""

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / os.environ.get("BENCH_RESULTS", "results")
WARMUP = 3


def gold_key(q, gold):
    if q["type"] == "noul":
        return "true" if gold else "false"
    return str(gold)


def options(q):
    if q["type"] == "noul":
        return ["true", "false"]
    if q["type"] == "choice":
        return list(q["criteria"])
    return [str(i) for i in range(len(q["criteria"]))]


def ece(pairs, bins=10):
    buckets = defaultdict(list)
    for conf, correct in pairs:
        buckets[min(int(conf * bins), bins - 1)].append((conf, correct))
    n = len(pairs)
    return sum(len(b) / n * abs(statistics.mean(c for c, _ in b) - statistics.mean(k for _, k in b))
               for b in buckets.values())


def score_model(name, records):
    rows = [json.loads(line) for line in (RESULTS / f"{name}.jsonl").open()]
    meta = json.loads((RESULTS / f"{name}.meta.json").read_text())
    by_id = {r["id"]: r for r in rows}
    per_task = defaultdict(list)  # task -> [(correct, brier, conf)]
    failures = 0
    for rec in records:
        row = by_id.get(rec["id"])
        for qid, q in rec["questions"].items():
            opts = options(q)
            probs = (row or {}).get("probs", {}).get(qid) or {}
            if not probs or sum(probs.values()) == 0:
                failures += 1
                probs = {o: 1 / len(opts) for o in opts}
                pred = None
            else:
                pred = max(probs, key=lambda o: probs[o])
            g = gold_key(q, rec["gold"][qid])
            brier = sum((probs.get(o, 0.0) - (o == g)) ** 2 for o in opts)
            conf = probs.get(pred, 0.0) if pred else 1 / len(opts)
            per_task[rec["task"]].append((pred == g, brier, conf))
    all_q = [x for v in per_task.values() for x in v]
    ms = [r["ms"] for r in rows[WARMUP:]] or [r["ms"] for r in rows]
    return {
        "model": name,
        "accuracy": statistics.mean(c for c, _, _ in all_q),
        "brier": statistics.mean(b for _, b, _ in all_q),
        "ece": ece([(cf, float(c)) for c, _, cf in all_q]) if meta["has_probabilities"] else None,
        "task_accuracy": {t: statistics.mean(c for c, _, _ in v) for t, v in sorted(per_task.items())},
        "failures": failures,
        "n_questions": len(all_q),
        "p50_ms": statistics.median(ms),
        "p95_ms": statistics.quantiles(ms, n=20)[18] if len(ms) >= 20 else max(ms),
        **{k: meta[k] for k in ("repo", "device", "load_s", "peak_rss_gib", "peak_mps_gib", "has_probabilities")},
    }


def main():
    records = [json.loads(line) for line in (ROOT / "data" / "eval.jsonl").open()]
    names = sys.argv[1:] or sorted(p.name.removesuffix(".meta.json") for p in RESULTS.glob("*.meta.json"))
    summary = sorted((score_model(n, records) for n in names), key=lambda s: -s["accuracy"])
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    tasks = list(summary[0]["task_accuracy"])
    head = ["model", "acc", "brier", "ece", *tasks, "fail", "p50 ms", "p95 ms", "mem GiB", "device"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for s in summary:
        mem = max(s["peak_rss_gib"], s["peak_mps_gib"])
        cells = [s["model"], f"{s['accuracy']:.3f}", f"{s['brier']:.3f}",
                 "-" if s["ece"] is None else f"{s['ece']:.3f}",
                 *(f"{s['task_accuracy'][t]:.2f}" for t in tasks), str(s["failures"]),
                 f"{s['p50_ms']:.0f}", f"{s['p95_ms']:.0f}", f"{mem:.1f}", s["device"]]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

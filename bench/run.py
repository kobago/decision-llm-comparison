"""Run one model over data/eval.jsonl and write results/<model>.jsonl.

usage: uv run python bench/run.py <model> [--limit N]
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import MODELS, ROOT  # noqa: E402


def mps_mem_gib() -> float:
    try:
        import torch

        return torch.mps.driver_allocated_memory() / 2**30
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cls, repo, kwargs = MODELS[args.model]
    records = [json.loads(line) for line in (ROOT / "data" / "eval.jsonl").open()]
    if args.limit:
        records = records[: args.limit]

    t = time.perf_counter()
    model = cls(repo, **kwargs)
    load_s = time.perf_counter() - t
    print(f"[{args.model}] loaded in {load_s:.1f}s on {cls.device}", flush=True)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    peak_mps = mps_mem_gib()
    with (out_dir / f"{args.model}.jsonl").open("w") as f:
        for i, r in enumerate(records):
            t = time.perf_counter()
            try:
                probs, error = model(r["state"], r["questions"]), None
            except Exception as e:
                probs, error = {qid: {} for qid in r["questions"]}, f"{type(e).__name__}: {e}"
                print(f"  {r['id']} error: {error}", file=sys.stderr, flush=True)
            ms = (time.perf_counter() - t) * 1000
            peak_mps = max(peak_mps, mps_mem_gib())
            f.write(json.dumps({"id": r["id"], "ms": ms, "probs": probs, "error": error}, ensure_ascii=False) + "\n")
            if i % 20 == 0:
                print(f"  {i + 1}/{len(records)} {r['id']} {ms:.0f}ms", flush=True)

    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
    meta = {"model": args.model, "repo": repo, "device": cls.device, "load_s": load_s,
            "peak_rss_gib": rss_gib, "peak_mps_gib": peak_mps,
            "has_probabilities": getattr(cls, "has_probabilities", True), "n": len(records)}
    (out_dir / f"{args.model}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

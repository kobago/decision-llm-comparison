"""One adapter per runtime. Each takes (state, questions) in the dataset's format and
returns {question_id: {option: probability}}.

Option keys are normalised: choice -> option name, noul -> "true"/"false",
score -> level index as a string ("0", "1", ...).
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / os.environ.get("BENCH_RESULTS", "results")


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def normalise_answers(answers: dict, questions: dict) -> dict[str, dict[str, float]]:
    """SystemOne-style answers -> {qid: {option: prob}}."""
    out = {}
    for qid, q in questions.items():
        a = answers[qid]
        if q["type"] == "noul":
            p = float(a["noul"])
            out[qid] = {"true": p, "false": 1.0 - p}
        else:
            out[qid] = {str(k): float(v) for k, v in a["probabilities"].items()}
    return out


def sync(torch, device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class SystemOneHF:
    """d1 / d3: transformers AutoModel with remote code exposing system_one()."""

    device = os.environ.get("BENCH_DEVICE", "mps")

    def __init__(self, repo: str, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=getattr(torch, dtype))
        self.model = self.model.to(self.device).eval()

    def __call__(self, state, questions):
        with self.torch.inference_mode():
            out = self.model.system_one(state=state, questions=questions)
        sync(self.torch, self.device)
        return normalise_answers(out["answers"], questions)


class Strands:
    device = "mps"

    def __init__(self, repo: str):
        import torch
        from strands_decider.infer import load_engine
        from strands_decider.schema import SystemOneRequest

        self.torch = torch
        self.request = SystemOneRequest
        self.engine = load_engine(repo, device=self.device)

    def __call__(self, state, questions):
        out = self.engine.evaluate(self.request(state=state, questions=questions)).model_dump()
        self.torch.mps.synchronize()
        return normalise_answers(out["answers"], questions)


class Laya:
    device = "cpu"

    def __init__(self, repo: str):
        from laya import Router

        self.router = Router()

    def __call__(self, state, questions):
        out = self.router.predict(state, questions)
        return normalise_answers(out["answers"], questions)


class Needle:
    """Needle generates structured output without probabilities: one-hot on the pick."""

    device = "cpu"
    has_probabilities = False

    def __init__(self, repo: str):
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")
        import needle

        self.needle = needle

    def _schema(self, questions):
        from pydantic import Field, create_model

        fields = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                fields[qid] = (bool, Field(description=q["instructions"]))
            elif q["type"] == "choice":
                desc = q["instructions"] + " / " + "; ".join(f"{k}: {v}" for k, v in q["criteria"].items())
                fields[qid] = (Literal[tuple(q["criteria"])], Field(description=desc))
            else:
                fields[qid] = (Literal[tuple(q["criteria"])], Field(description=q["instructions"]))
        return create_model("Decision", **fields)

    def __call__(self, state, questions):
        try:
            out = self.needle.extract(state_text(state), self._schema(questions))
        except Exception as e:  # validation failures count as wrong answers
            print(f"  needle error: {type(e).__name__}: {e}", file=sys.stderr)
            return {qid: {} for qid in questions}
        res = {}
        for qid, q in questions.items():
            v = getattr(out, qid)
            if q["type"] == "noul":
                res[qid] = {"true": float(bool(v)), "false": float(not v)}
            elif q["type"] == "choice":
                res[qid] = {k: float(k == v) for k in q["criteria"]}
            else:
                res[qid] = {str(i): float(lvl == v) for i, lvl in enumerate(q["criteria"])}
        return res


class Nimble:
    """Bespoke-Nimble-9B via its reference runner, patched from CUDA to MPS."""

    device = "mps"

    def __init__(self, repo: str):
        import torch
        from huggingface_hub import snapshot_download

        self.torch = torch
        src = Path(snapshot_download(repo))
        dst = ROOT / "vendor" / "nimble-model"
        if not dst.exists():
            shutil.copytree(src, dst)
            code = (dst / "inference.py").read_text()
            code = code.replace(
                "if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():",
                "if not torch.backends.mps.is_available():",
            )
            code = code.replace(".to('cuda')", ".to('mps')")
            code = code.replace("torch.autocast('cuda',dtype=torch.bfloat16)", "contextlib.nullcontext()")
            code = "import contextlib\n" + code
            (dst / "inference.py").write_text(code)
        sys.path.insert(0, str(dst))
        from inference import NimbleModel

        self.model = NimbleModel(str(dst))

    def __call__(self, state, questions):
        schema, score_fields = {}, []
        for qid, q in questions.items():
            if q["type"] == "noul":
                schema[qid] = {"type": "boolean", "description": q["instructions"]}
            elif q["type"] == "choice":
                schema[qid] = {"type": "enum", "description": q["instructions"],
                               "choices": list(q["criteria"]), "choice_descriptions": dict(q["criteria"])}
            else:
                levels = [str(i) for i in range(len(q["criteria"]))]
                schema[qid] = {"type": "enum", "description": q["instructions"], "choices": levels,
                               "choice_descriptions": dict(zip(levels, q["criteria"]))}
                score_fields.append(qid)
        out = self.model.score(context=state_text(state), schema=schema, score_fields=score_fields)
        self.torch.mps.synchronize()
        res = {}
        for qid, q in questions.items():
            probs = out["fields"][qid]["probabilities"]
            res[qid] = {str(k).lower() if q["type"] == "noul" else str(k): float(v) for k, v in probs.items()}
        return res


MODELS = {
    "strands-2b": (Strands, "StrandsAgents/strands-decider-2B-hobson-v19", {}),
    "laya": (Laya, "convaiinnovations/laya", {}),
    "needle3": (Needle, "Cactus-Compute/needle3", {}),
    "d1-2b": (SystemOneHF, "LiquidAI/d1-2B", {}),
    "d1-3b": (SystemOneHF, "LiquidAI/d1-3B", {}),
    "d1-omni-600m": (SystemOneHF, "LiquidAI/d1-omni-600M", {"dtype": "float16"}),
    "d3-edge": (SystemOneHF, "vllm-sr/d3-edge", {}),
    "d3-lite": (SystemOneHF, "vllm-sr/d3-lite", {}),
    "d3-nano": (SystemOneHF, "vllm-sr/d3-nano", {}),
    "d3-mini": (SystemOneHF, "vllm-sr/d3-mini", {}),
    "d3-flash": (SystemOneHF, "vllm-sr/d3-flash", {}),
    "nimble-9b": (Nimble, "bespokelabs/Bespoke-Nimble-9B", {}),
}

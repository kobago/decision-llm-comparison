import json, sys, time
import torch
from transformers import AutoModel

repo, dtype_name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "bfloat16")
device = sys.argv[3] if len(sys.argv) > 3 else "mps"
dtype = getattr(torch, dtype_name)

questions = {
    "department": {"type": "choice", "instructions": "どの部署に回すべきか", "criteria": {"billing": "請求・支払い", "technical_support": "技術的な不具合", "sales": "購入前の相談", "shipping": "配送"}},
    "refund": {"type": "noul", "instructions": "顧客は返金を求めていますか"},
    "urgency": {"type": "score", "instructions": "緊急度は", "criteria": ["低", "中", "高"]},
}
state = "先月の請求が二重に引き落とされています。1件返金してください。"

t = time.perf_counter()
model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=dtype).to(device).eval()
print("load", round(time.perf_counter() - t, 1), "s")
for i in range(3):
    t = time.perf_counter()
    with torch.inference_mode():
        out = model.system_one(state=state, questions=questions)
    if device == "mps":
        torch.mps.synchronize()
    print("call", i, round((time.perf_counter() - t) * 1000), "ms")
if device == "mps":
    print("mps driver mem", round(torch.mps.driver_allocated_memory() / 2**30, 2), "GiB")
print(json.dumps(out, ensure_ascii=False)[:1500])

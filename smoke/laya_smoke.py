import json, time
from laya import Router

state = "先月の請求が二重に引き落とされています。1件返金してください。"
questions = {
    "department": {"type": "choice", "instructions": "どの部署に回すべきか", "criteria": ["billing", "technical_support", "sales", "shipping"]},
    "refund": {"type": "noul", "instructions": "顧客は返金を求めていますか"},
    "urgency": {"type": "score", "instructions": "緊急度は", "criteria": ["低", "中", "高"]},
}
t = time.perf_counter(); r = Router(); print("load", round(time.perf_counter() - t, 1), "s")
for _ in range(2):
    t = time.perf_counter(); out = r.predict(state, questions); print("predict", round((time.perf_counter() - t) * 1000), "ms")
print(json.dumps(out, ensure_ascii=False, indent=1)[:2000])

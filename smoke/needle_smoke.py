import time
from typing import Literal
from pydantic import BaseModel, Field
import needle

class Ticket(BaseModel):
    department: Literal["billing", "technical_support", "sales", "shipping"] = Field(description="どの部署に回すべきか")
    refund: bool = Field(description="顧客は返金を求めていますか")
    urgency: Literal["低", "中", "高"] = Field(description="緊急度")

for text in ["先月の請求が二重に引き落とされています。1件返金してください。",
             "I was charged twice last month, please refund one of them."]:
    for _ in range(2):
        t = time.perf_counter(); out = needle.extract(text, Ticket); ms = (time.perf_counter() - t) * 1000
    print(round(ms), "ms", out)

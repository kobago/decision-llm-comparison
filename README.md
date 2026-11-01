# decision-llm-comparison

TypeSafe Jev 以降に公開されたオープンウェイトの Decision モデル (state + 型付き質問 → 選択肢ごとの確率) を、
Mac mini (Apple M2 Pro / 32GB) 上で実務寄りのタスクで比較する。

## 対象モデル

| key | repo | 実行方法 |
|---|---|---|
| d3-mini / d3-flash / d3-nano / d3-lite / d3-edge | `vllm-sr/d3-*` | transformers `system_one()` (MPS, bf16) |
| d1-2b / d1-3b / d1-omni-600m | `LiquidAI/d1-2B`, `LiquidAI/d1-3B`, `LiquidAI/d1-omni-600M` | transformers `system_one()` (MPS, bf16 / fp16) |
| strands-2b | `StrandsAgents/strands-decider-2B-hobson-v19` | `strands-decider` engine (MPS) |
| nimble-9b | `bespokelabs/Bespoke-Nimble-9B` | 同梱 `inference.py` を MPS 用に書き換えて実行 (`vendor/`) |
| laya | `convaiinnovations/laya` | `laya.Router` (CPU、日本語は multilingual に自動ルーティング) |
| needle3 | `Cactus-Compute/needle3` | `needle.extract()` (CPU)。確率を返さないため one-hot 扱い |

27B 以上 (Clef, Clef-Omni, JEV-27B-VL, d3, d3-turbo) は BF16 でメモリに載らないため対象外。

## 評価データ (`data/eval.jsonl`)

`uv run python data/build_dataset.py` で生成。120 レコード / 260 問 (日本語主体、英語を一部含む)。

| task | 件数 | 質問 |
|---|---|---|
| support | 40 (ja 30 / en 10) | 部署振り分け (choice 6), 返金要求 (noul), 緊急度 (score 3), 解約意向 (noul) |
| invoice | 20 (JSON state) | 経理規程に基づく承認/差戻し/要確認 (choice 3), 金額一致 (noul) |
| helpdesk | 20 | 呼び出すツール (choice 6) |
| rag | 20 | 回答が文書だけで裏付けられるか (noul) |
| injection | 20 | プロンプトインジェクションか (noul) |

## 実行

```bash
uv sync
./bench/run_all.sh d3-mini d1-3b laya   # results/<model>.jsonl と .meta.json を出力
uv run python bench/report.py          # results/summary.json と表を出力
```

d1 / d3 は `BENCH_DEVICE` で実行デバイス (既定 `mps`)、`BENCH_RESULTS` で出力先 (既定 `results`) を変えられる。

```bash
BENCH_DEVICE=cuda BENCH_RESULTS=results-rtx2060s ./bench/run_all.sh d1-2b d1-3b
BENCH_RESULTS=results-rtx2060s uv run python bench/report.py
```

## 指標

- acc: argmax の正解率 / brier: 多クラス Brier (0 が最良) / ece: top-1 確信度の ECE (10 bin)
- p50/p95 ms: 1 レコード (1〜4 問をまとめて 1 回で問う) あたりの応答時間。先頭 3 件はウォームアップとして除外
- mem: プロセスの最大 RSS と MPS ドライバ確保量の大きい方

## 結果 (2026-10-26, M2 Pro 32GB, torch 2.14.1, transformers 5.20.2)

| model | acc | brier | ece | helpdesk | injection | invoice | rag | support | fail | p50 ms | p95 ms | mem GiB | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d3-mini | 0.958 | 0.060 | 0.031 | 1.00 | 0.85 | 0.97 | 1.00 | 0.96 | 0 | 3639 | 9148 | 9.9 | mps |
| d3-flash | 0.954 | 0.064 | 0.018 | 1.00 | 0.80 | 0.95 | 1.00 | 0.96 | 0 | 1934 | 7273 | 16.9 | mps |
| d1-3b | 0.942 | 0.090 | 0.033 | 1.00 | 0.85 | 0.90 | 1.00 | 0.95 | 0 | 348 | 718 | 6.8 | mps |
| d3-nano | 0.942 | 0.094 | 0.021 | 1.00 | 0.95 | 0.90 | 0.95 | 0.94 | 0 | 528 | 1991 | 4.6 | mps |
| nimble-9b | 0.931 | 0.096 | 0.024 | 1.00 | 0.90 | 0.82 | 1.00 | 0.94 | 0 | 3080 | 19753 | 19.0 | mps |
| strands-2b | 0.908 | 0.180 | 0.099 | 1.00 | 0.75 | 0.80 | 0.95 | 0.94 | 0 | 473 | 1088 | 4.2 | mps |
| d3-lite | 0.873 | 0.173 | 0.042 | 1.00 | 0.95 | 0.70 | 0.90 | 0.89 | 0 | 393 | 1397 | 2.3 | mps |
| d3-edge | 0.862 | 0.203 | 0.040 | 0.95 | 0.80 | 0.82 | 0.90 | 0.86 | 0 | 220 | 589 | 1.4 | mps |
| laya | 0.704 | 0.444 | 0.151 | 0.95 | 0.75 | 0.57 | 0.80 | 0.69 | 0 | 46 | 190 | 3.3 | cpu |
| d1-omni-600m | 0.627 | 0.477 | 0.057 | 1.00 | 0.40 | 0.57 | 0.75 | 0.61 | 0 | 85 | 161 | 3.6 | mps |
| needle3 | 0.177 | 0.824 | - | 0.30 | 0.25 | 0.55 | 0.50 | 0.02 | 155 | 186 | 543 | 0.3 | cpu |

## 追加計測: d1-2B vs d1-3B (2026-11-01, RTX 2060 SUPER 8GB, CUDA bf16, torch 2.14.1+cu130, transformers 5.20.2)

d1-3B の直後に公開された d1-2B (LFM3-VL-2B ベース, 1.92B) を、手元の Linux 機で d1-3B と同条件で計測した (`results-rtx2060s/`)。

| model | acc | brier | ece | helpdesk | injection | invoice | rag | support | fail | p50 ms | p95 ms | mem GiB | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1-2b | 0.950 | 0.087 | 0.023 | 1.00 | 0.85 | 0.90 | 1.00 | 0.96 | 0 | 144 | 290 | 5.1 | cuda |
| d1-3b | 0.946 | 0.089 | 0.034 | 1.00 | 0.85 | 0.90 | 1.00 | 0.96 | 0 | 231 | 461 | 6.7 | cuda |

- 正解数は 247 / 246 (260 問中) で実質同等。差が出たのは injection の 2 問ずつの入れ替わりと urgency の 1 問のみ。
  モデルカードが示すほどの差 (Decision Index 51.1 vs 49.0 など) はこのタスクでは出なかった。
- d1-2B は p50 で約 1.6 倍速く、メモリも約 1.6 GiB 少ない。ece もやや良い。
- d1-3B の CUDA 結果は M2 Pro (MPS) の結果と 260 問中 259 問で argmax が一致したため、精度の列は上の表と比較してよい
  (d1-2B は d3-mini / d3-flash に次ぐ 3 位相当)。応答時間・メモリはハードウェアが違うため比較できない。

## 注意点

- d3 系 / strands / nimble は CUDA 専用カーネル (`flash-linear-attention`, `causal_conv1d`) が使えず PyTorch 参照実装で動くため、
  応答時間はモデルカード (GPU) の値より大幅に遅い。
- nimble-9b は 32GB 機ではスワップが発生する (重み約 18GB)。
- 緊急度 (support.urgency) は主観が入るラベルで、上位モデルでも 0.80〜0.85 に留まる。ラベル自体の見直し余地がある。
- 請求書の登録番号の形式チェック (「T」+13 桁) は全モデルが苦手で、桁数違いを見逃して approve しがち。
  形式チェックのような決定的なルールはコード側で行うべき。
- needle3 は判断モデルではなくツール呼び出し/抽出モデル。日本語の判断タスクではツールを呼ばないことが多く、失敗は不正解として扱った。

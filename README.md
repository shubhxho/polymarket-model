# Cross-market fusion

Dual-stream model for 15-minute Polymarket crypto binaries. Fast venue is Binance futures. Slow venue is the Polymarket CLOB. Same thesis as [LACUNA](https://github.com/humanplane/cross-market-state-fusion) — information shows up on Binance first — but the implementation is not a 5-tick concat MLP.

Trained on Apple Silicon with **MLX 0.32**. Feature extraction is **C++20** (OFI, Kyle λ, VPIN, Hawkes intensity, Hayashi–Yoshida lead-lag) via nanobind.

## Why this is not LACUNA

| | LACUNA | This |
|---|---|---|
| Fusion | Flatten last 5 states, concat, tanh MLP | Bidirectional cross-attention over 64-tick streams |
| Features | 18 hand-scaled floats in Python | C++20 microstructure + lead-lag |
| Decision | 3-way actor softmax | Calibrated `P(resolve UP)` versus the live bid/ask |
| Train | Online PPO on live ticks only | Supervised pretrain on lag/expiry + PPO on a lag-aware simulator |
| Stack | tanh / LayerNorm / Adam | RMSNorm, RoPE, SwiGLU, `mx.fast` SDPA, AdamW + cosine |

The shipped policy is not `argmax(actor)`. It is:

```
buy UP    if  P(up)  − ask  > 4.5¢
buy DOWN  if  1−P(up) − (1−bid) > 4.5¢
else HOLD
```

That is the lag trade: a probability the CLOB has not printed yet.

## Held-out simulator eval (80 episodes)

Same environment for every policy. Share-based PnL **after paying the bid/ask**. Not live Polymarket PnL.

| policy | mean PnL | win % | trades/ep | Sharpe | P(resolve UP) |
|---|---:|---:|---:|---:|---:|
| **fusion** | **+0.04** | **82.5%** | 2.41 | **0.58** | **90.5%** |
| LACUNA clone | −0.47 | 33.8% | 3.81 | −7.04 | — |
| lag oracle | +0.51 | 43.8% | 1.99 | 3.64 | — |
| random | −6.25 | 0.0% | 60.0 | −26.0 | — |

Fusion is more conservative than the oracle (higher win rate, smaller average PnL). The LACUNA clone, trained on the same simulator, overtrades and loses.

## Is this SOTA?

No. It is a stronger *implementation* than LACUNA on a **controlled lag simulator**. That is not state of the art for prediction-market making, binary-option pricing, or live Polymarket trading.

What it is not:
- Not compared to published LOB transformers, temporal fusion transformers, or calibrated market-making baselines on real fills
- Not live out-of-sample; paper trading here still assumes you can hit the displayed bid/ask
- Not a claim that 90% expiry accuracy transfers to production 15-minute markets
- The lag oracle still beats the trained policy on PnL (+0.51 vs +0.04)

What it *is*:
- Better than a faithful LACUNA clone **on this simulator** (fusion +0.04 / Sharpe 0.58 vs clone −0.47 / Sharpe −7)
- A modern stack (C++20 features, MLX 0.32 dual-stream attention, `P(up)` vs CLOB)

If you need a one-line label: **SOTA vs LACUNA in-sim, not SOTA in the field.**

## Install, train, test (uv only)

```bash
uv sync
uv run pytest
uv run cmf train --pretrain-steps 400 --ppo-updates 40 --eval-episodes 64
uv run cmf live --load checkpoints/fusion.safetensors --size 50
```

`uv.lock` pins the tree. `uv sync` builds the C++ extension and the MLX env. Do not use pip.

## Notation book

Every symbol used in the C++ engine, simulator, network, and policy is written out as a short monograph (KaTeX, theorem boxes, index):

```bash
uv run python -m http.server --directory docs 4173
# or
uv run cmf docs
```

Then open [http://127.0.0.1:4173](http://127.0.0.1:4173). Hosted copy: [shubhxho.github.io/polymarket-model](https://shubhxho.github.io/polymarket-model/).

## Layout

```
csrc/           C++20 fusion engine + nanobind
cmf/model.py    Dual-stream transformer (MLX)
cmf/policy.py   P(up) vs CLOB decision
cmf/simulator.py  Lag-aware 15-minute binary
cmf/baseline.py LACUNA Phase-5 clone
cmf/train.py    Pretrain + PPO + eval
```

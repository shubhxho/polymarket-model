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

## Install and train (Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cmf train --pretrain-steps 400 --ppo-updates 40 --eval-episodes 64
```

Weights land in `checkpoints/fusion.safetensors`. Paper-trade live 15-minute markets:

```bash
cmf live --load checkpoints/fusion.safetensors --size 50
```

## Layout

```
csrc/           C++20 fusion engine + nanobind
cmf/model.py    Dual-stream transformer (MLX)
cmf/policy.py   P(up) vs CLOB decision
cmf/simulator.py  Lag-aware 15-minute binary
cmf/baseline.py LACUNA Phase-5 clone
cmf/train.py    Pretrain + PPO + eval
```

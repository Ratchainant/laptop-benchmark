# Benchmark Results

**Date:** 2026-04-24  
**Devices tested:** 3  
**Scoring:** Normalized 0–100 (best in group = 100)

---

## Devices

| Device | CPU | RAM | OS | Recorded |
|--------|-----|-----|----|----------|
| MacBook Air M3 | Apple M3 | 17.2 GB | Darwin arm64 | 2026-04-24 |
| Mac Mini M4 | Apple M4 | 17.2 GB | Darwin arm64 | 2026-04-24 |
| Dell Pro (Ryzen AI) | AMD64 Family 26 Model 96 Stepping 0, AuthenticAMD | 33.5 GB | Windows AMD64 | 2026-04-24 |

---

## Final Scores

| Rank | Device | Score | Bar | Tests run |
|------|--------|------:|-----|-----------|
| 🥇 | Mac Mini M4 | **93.2** | `███████████████████░` | 6/6 |
| 🥈 | MacBook Air M3 | **93.0** | `███████████████████░` | 6/6 |
| 🥉 | Dell Pro (Ryzen AI) | **40.5** | `████████░░░░░░░░░░░░` | 6/6 |

---

## Per-Test Results

### LLM Inference — 30% weight

_↑ higher is better_

| Device | Result | Score | Bar |
|--------|-------:|------:|-----|
| MacBook Air M3 | 39.01 tokens/sec | 90.1/100 | `██████████████░░` |
| Mac Mini M4 ★ | 43.30 tokens/sec | 100.0/100 | `████████████████` |
| Dell Pro (Ryzen AI) | 15.87 tokens/sec | 36.7/100 | `██████░░░░░░░░░░` |

### Image Classification (ONNX ResNet-50) — 20% weight

_↑ higher is better_

| Device | Result | Score | Bar |
|--------|-------:|------:|-----|
| MacBook Air M3 | 692.51 images/sec | 94.9/100 | `███████████████░` |
| Mac Mini M4 ★ | 729.98 images/sec | 100.0/100 | `████████████████` |
| Dell Pro (Ryzen AI) | 161.13 images/sec | 22.1/100 | `████░░░░░░░░░░░░` |

### Pandas groupby + join (5M rows) — 15% weight

_↓ lower is better_

| Device | Result | Score | Bar |
|--------|-------:|------:|-----|
| MacBook Air M3 | 0.27 seconds | 97.8/100 | `████████████████` |
| Mac Mini M4 ★ | 0.26 seconds | 100.0/100 | `████████████████` |
| Dell Pro (Ryzen AI) | 0.53 seconds | 49.7/100 | `████████░░░░░░░░` |

### DuckDB analytical query (50M rows) — 15% weight

_↓ lower is better_

| Device | Result | Score | Bar |
|--------|-------:|------:|-----|
| MacBook Air M3 ★ | 0.16 seconds | 100.0/100 | `████████████████` |
| Mac Mini M4 | 0.20 seconds | 80.3/100 | `█████████████░░░` |
| Dell Pro (Ryzen AI) | 0.23 seconds | 71.8/100 | `███████████░░░░░` |

### Sequential I/O Read — 10% weight

_↑ higher is better_

| Device | Result | Score | Bar |
|--------|-------:|------:|-----|
| MacBook Air M3 ★ | 4.66 GB/s | 100.0/100 | `████████████████` |
| Mac Mini M4 | 2.86 GB/s | 61.5/100 | `██████████░░░░░░` |
| Dell Pro (Ryzen AI) | 1.65 GB/s | 35.3/100 | `██████░░░░░░░░░░` |

### NumPy matrix multiply — 10% weight

_↑ higher is better_

| Device | Result | Score | Bar |
|--------|-------:|------:|-----|
| MacBook Air M3 | 1337.87 GFLOPS | 73.8/100 | `████████████░░░░` |
| Mac Mini M4 ★ | 1812.87 GFLOPS | 100.0/100 | `████████████████` |
| Dell Pro (Ryzen AI) | 603.69 GFLOPS | 33.3/100 | `█████░░░░░░░░░░░` |

---

## Category Breakdown

| Category | Device | Score | Bar |
|----------|--------|------:|-----|
| AI Inference | Mac Mini M4 | 100.0/100 | `████████████████` |
|  | MacBook Air M3 | 92.0/100 | `███████████████░` |
|  | Dell Pro (Ryzen AI) | 30.8/100 | `█████░░░░░░░░░░░` |
| Data / Analytics | MacBook Air M3 | 98.9/100 | `████████████████` |
|  | Mac Mini M4 | 90.1/100 | `██████████████░░` |
|  | Dell Pro (Ryzen AI) | 60.8/100 | `██████████░░░░░░` |
| I/O + Memory | MacBook Air M3 | 86.9/100 | `██████████████░░` |
|  | Mac Mini M4 | 80.7/100 | `█████████████░░░` |
|  | Dell Pro (Ryzen AI) | 34.3/100 | `█████░░░░░░░░░░░` |

---

## Scoring Formula

```
score_i = (result_i / best) × 100   # higher-is-better
score_i = (best / result_i) × 100   # lower-is-better
final   = Σ(score_i × weight_i) / Σ(weight_i)
```

_Each device is scored relative to the best result in the group (= 100)._
_Each device uses its own hardware accelerator (Metal / CoreML / DirectML)._

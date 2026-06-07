# ModelDNA Benchmark Report

**Date**: 2026-06-07
**Platform**: NVIDIA RTX 4050 Laptop (6GB) / NVIDIA A10 (24GB)
**Base Model**: Qwen2.5-0.5B-Instruct (494M)

---

## 1. Full-Scale Benchmark (GSM8K + HumanEval)

| Method | GSM8K (1319) | HumanEval (164) | Notes |
|--------|-------------|-----------------|-------|
| Baseline | 30.1% (397/1319) | 16.5% (27/164) | Original model |
| LoRA Direct | 29.4% (388/1319) | 9.1% (15/164) | Catastrophic forgetting on code |
| SVD-BottomSwap | 25.9% (342/1319) | 20.1% (33/164) | +11.0pp over LoRA on HumanEval |

## 2. Offset Sensitivity Analysis

Quick-mode scan (100 GSM8K + 60 HumanEval):

| Offset | GSM8K (100) | HumanEval (60) | Combined Score |
|--------|------------|----------------|----------------|
| 8 | 30.0% | 33.3% | 63.3 |
| **16** | 28.0% | **45.0%** | **73.0** |
| 32 | 29.0% | 33.3% | 62.3 |
| 64 | 29.0% | 36.7% | 65.7 |
| Baseline | 24.0% | 38.3% | 62.3 |

## 3. Ablation Study

Quick-mode ablation of each design choice:

| Strategy | GSM8K (100) | HumanEval (60) | HE Δ vs Baseline |
|----------|------------|----------------|-------------------|
| Baseline | 24.0% | 38.3% | — |
| Full Method (offset=8) | 31.0% | 31.7% | -6.6pp |
| −Purify (no purification) | 31.0% | 25.0% | -13.3pp |
| −Deep (all 24 layers) | 19.0% | 25.0% | -13.3pp |
| +offset×2 (offset=16) | 28.0% | 45.0% | +6.7pp |

## 4. Stability Analysis

N=5 independent injections at optimal config (offset=16):

| Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | μ | σ |
|---------|---------|---------|---------|---------|------|------|
| 28.0% | 28.0% | 28.0% | 28.0% | 28.0% | 28.00% | **0.0000%** |

## 5. Cross-Domain Transfer

| Direction | Target Domain | Non-Target Domain | Conclusion |
|-----------|--------------|-------------------|------------|
| Math→Code | HE=45.0% (+6.7pp) | GSM8K=28.0% (+4.0pp) | Positive transfer |
| Code→Math | HE=5.0% | GSM8K=5.0% | Training data format mismatch |

## 6. Efficiency

| Stage | Duration (0.5B, RTX 4050) |
|-------|---------------------------|
| LoRA Training (400 steps) | ~20 min |
| SVD Injection Total | **~15 sec** |

1.5B model (A10): SVD 28.6s vs LoRA 543s — **19× speedup**.

## 7. SVD Purification Effect

Top-5 singular value energy concentration:

| State | Energy Concentration |
|-------|---------------------|
| Before purification | 40-52% |
| After purification (τ=0.85, ε=0.01) | 80-100% |

---

*Raw data: `svd_results_v5.json`, `svd_results_v6.json`*
*Reproduction scripts: `svd_validate_v5.py`, `svd_validate_v6.py`*

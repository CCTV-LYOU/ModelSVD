# ModelDNA: Zero-Training Knowledge Transfer via SVD Injection

**ModelDNA** is a novel zero-training knowledge extraction and injection method based on Singular Value Decomposition (SVD). It extracts "knowledge fingerprints" from trained model weight deltas and injects them into target models without any gradient computation or backpropagation.

## Key Results

| Metric | LoRA Fine-tuning | ModelDNA (ours) |
|--------|-----------------|-----------------|
| HumanEval pass@1 | 9.1% | **20.1%** (+11.0pp) |
| Catastrophic Forgetting | -7.4pp on HE | **0.0pp** (eliminated) |
| Injection Stability (σ) | 24.18% (N=10) | **0.0000%** (N=5) |
| Injection Speed | ~20 min (training) | **~15 sec** (80× faster) |
| Cross-domain Transfer | -7.4pp (negative) | **+6.7pp** (positive) |

## Repository Contents

- `paper/` — Academic papers (Chinese & English)
- `results/` — Benchmark reports with experimental data
- `pseudocode/` — Algorithm sketches (academic reference only, not executable)

## Contact

For collaboration, investment, or licensing inquiries:
- **Email**: 1417981857@qq.com
- **GitHub**: [@CCTV-LYOU](https://github.com/CCTV-LYOU)

## License

This repository contains academic reference materials only. For commercial use or technology licensing, please contact the author.

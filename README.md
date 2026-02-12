# Agent Robustness Via Theory of Mind

> ⚠️ **Work in Progress**: This repository is under active development. Code, methods, and results are subject to change.

Research on **Theory of Mind (ToM)** capabilities in language models using steering vectors and benchmark evaluations. Explores how models understand that others may hold different beliefs and knowledge—critical for robust AI agents.

## Overview

This repository implements:

- **Steering Vector Extraction**: Extract ToM-specific activation patterns from language models
- **Multi-Benchmark Evaluation**: Test across ToMi, FANToM, SimpleToM, and ToMBench
- **Contrastive Analysis**: Compare ToM-required vs. non-ToM scenarios

## Repository Structure

```
├── getting_steering_vecs.ipynb          # Extract ToM steering vectors
├── tom_benchmarks/
│   ├── tom_benchmark_suite.ipynb        # Unified benchmark evaluation
│   ├── tomi/                            # ToMi benchmark & pair extraction
│   ├── fantom/                          # FANToM benchmark
│   └── eval_outputs/                    # Results
└── steering_vectors/                    # Saved steering vectors
```

## Benchmarks

| Benchmark           | Source            | Focus                                      |
| ------------------- | ----------------- | ------------------------------------------ |
| **ToMi**      | Facebook Research | First/second-order false belief in stories |
| **FANToM**    | Allen AI          | Information asymmetry in conversations     |
| **SimpleToM** | Allen AI          | Mental state & behavior prediction         |
| **ToMBench**  | Tsinghua          | 8 ToM tasks, 31 social reasoning abilities |

## Quick Start

```bash
git clone https://github.com/AndresCotton/Agent-Robustness-Via-ToM.git
cd Agent-Robustness-Via-ToM

# For FANToM evaluation
cd tom_benchmarks/fantom
conda env create -f environment.yml
conda activate fantom
python eval_fantom.py --model gpt-4-0613
```

**Main Notebooks:**

- `getting_steering_vecs.ipynb` - Extract steering vectors from paired ToM/non-ToM examples
- `tom_benchmarks/tom_benchmark_suite.ipynb` - Evaluate models across all benchmarks

## Methodology

**Steering Vector Extraction:**

1. Load paired examples (ToM-required vs. non-ToM)
2. Filter for correctly answered examples
3. Identify important attention heads
4. Extract: `steering_vec = mean(ToM_activations) - mean(non_ToM_activations)`
5. Apply to model and evaluate on held-out set

**Example Contrastive Pair:**

```
ToM Required: "Emma put the ball in the basket. Emma left. Jack moved it to the box.
               Where does Emma think the ball is?" → basket (false belief)

No ToM: "Emma put the ball in the basket. Jack moved it to the box.
         Where does Emma think the ball is?" → box (Emma observed)
```

## Results

See preliminary results in `tom_benchmarks/eval_outputs/`.

## Citation

```bibtex
@inproceedings{le-etal-2019-revisiting,
    title = "Revisiting the Evaluation of Theory of Mind through Question Answering",
    author = "Le, Matthew and Boureau, Y-Lan and Nickel, Maximilian",
    booktitle = "Proceedings of EMNLP-IJCNLP",
    year = "2019"
}

@inproceedings{kim2023fantom,
    title = {FANToM: A Benchmark for Stress-testing Machine Theory of Mind in Interactions},
    author = {Kim, Hyunwoo and Sclar, Melanie and Zhou, Xuhui and Le Bras, Ronan and Kim, Gunhee and Choi, Yejin and Sap, Maarten},
    booktitle = {EMNLP},
    year = {2023}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

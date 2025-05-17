# WaKA: Data Attribution using K-Nearest Neighbors and Membership Privacy Principles

[![arXiv](https://img.shields.io/badge/arXiv-2411.01357-b31b1b.svg)](https://arxiv.org/abs/2411.01357)

This repository contains the official implementation of the WaKA (Wasserstein K-nearest-neighbors Attribution) algorithm as described in the paper ["WaKA: Data Attribution using K-Nearest Neighbors and Membership Privacy Principles"](https://arxiv.org/abs/2411.01357).

## Overview

WaKA is a novel attribution method that leverages principles from the LiRA (Likelihood Ratio Attack) framework and k-nearest neighbors classifiers (k-NN). It efficiently measures the contribution of individual data points to the model's loss distribution, analyzing every possible k-NN that can be constructed using the training set, without requiring subset sampling.

WaKA can be used for:
- Membership inference attack (MIA) to assess privacy risks
- Privacy influence measurement
- Data valuation

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -U "ray[default]"
   ```

2. Optional: For Ray parallelization on macOS:
   ```bash
   pip install -e ../rayPlus/ --config-settings editable_mode=strict
   ```

## Usage

### Utility Experiments

Run utility-driven data minimization experiments:

```bash
python utility_driven_data_minimization.py
```

These experiments demonstrate how WaKA can be used for data valuation and data minimization tasks.

### Privacy Experiments

Run privacy evaluation and auditing experiments:

```bash
python privacy_evaluation_and_audit.py
```

These experiments demonstrate how WaKA can be used for membership inference attacks and privacy risk assessment.

## Core Components

- `waka.py`: Core implementation of the WaKA algorithm
- `knn_shapley_valuation.py`: Implementation of Shapley value computation for k-NN models
- `utils.py`: Utility functions for data handling and analysis
- `valuation_analysis.py`: Tools for analyzing data valuation methods
- `utility_driven_data_minimization.py`: Experiments for data minimization based on utility metrics
- `privacy_evaluation_and_audit.py`: Experiments for privacy evaluation and auditing
- `lira_attack.py`: Implementation of LiRA (Likelihood Ratio Attack)

## Paper Abstract

In this paper, we introduce WaKA (Wasserstein K-nearest-neighbors Attribution), a novel attribution method that leverages principles from the LiRA (Likelihood Ratio Attack) framework and k-nearest neighbors classifiers (k-NN). WaKA efficiently measures the contribution of individual data points to the model's loss distribution, analyzing every possible k-NN that can be constructed using the training set, without requiring to sample subsets of the training set. WaKA is versatile and can be used a posteriori as a membership inference attack (MIA) to assess privacy risks or a priori for privacy influence measurement and data valuation. Thus, WaKA can be seen as bridging the gap between data attribution and membership inference attack (MIA) by providing a unified framework to distinguish between a data point's value and its privacy risk. For instance, we have shown that self-attribution values are more strongly correlated with the attack success rate than the contribution of a point to the model generalization. WaKA's different usage were also evaluated across diverse real-world datasets, demonstrating performance very close to LiRA when used as an MIA on k-NN classifiers, but with greater computational efficiency. Additionally, WaKA shows greater robustness than Shapley Values for data minimization tasks (removal or addition) on imbalanced datasets.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mesana2024waka,
      title={WaKA: Data Attribution using K-Nearest Neighbors and Membership Privacy Principles}, 
      author={Patrick Mesana and Clément Bénesse and Hadrien Lautraite and Gilles Caporossi and Sébastien Gambs},
      year={2024},
      eprint={2411.01357},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


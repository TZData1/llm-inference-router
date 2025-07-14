# Contextual Multi-Armed Bandits for LLM Routing

This repository contains the implementation of contextual multi-armed bandit algorithms for dynamic LLM routing, optimizing the accuracy-efficiency trade-off in large language model inference.

## Overview

This project explores how contextual bandits can intelligently route queries to different LLMs based on:
- Query complexity and task type
- Semantic features
- Desired accuracy-energy trade-offs

Key algorithms implemented:
- Epsilon-Greedy (with contextual variants)
- LinUCB (Linear Upper Confidence Bound)
- Thompson Sampling

## Repository Structure

```
experiments/         # Main experiment implementations
├── a2_warmup/      # Algorithm warm-up analysis
├── a3_feature_ablation/  # Feature importance study
├── a4_hyperparameter_tuning/  # Parameter sensitivity
├── a5_algorithm_bakeoff/  # Algorithm comparison
├── a6_lambda_sweep/  # Accuracy-efficiency trade-off
└── a8_adaptability/  # Model pool adaptation

src/                # Core implementations
├── bandit/         # MAB algorithm implementations
├── feature_extractor/  # Context feature extraction
└── services/       # Routing and evaluation services
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker with docker-compose
- CUDA-capable GPU (recommended)

### Installation

Quick setup using Make:
```bash
make setup
```

Or manually:
1. Clone the repository
2. Set up the environment: 
   ```bash
   python -m venv venv && source venv/bin/activate
   ```
3. Install dependencies: 
   ```bash
   pip install -r requirements.txt
   ```
4. Initialize PostgreSQL database:
   ```bash
   docker-compose -f db/docker-compose-db.yaml up -d
   ```
5. Load the database backup:
   ```bash
   gunzip -c db/llm_db_backup.sql.gz | docker exec -i db-llm_db-1 psql -U tz -d llm_db
   ```

## Running Experiments

Using Make:
```bash
make a2  # Warmup analysis
make a3  # Feature ablation
make a4  # Hyperparameter tuning
make a5  # Algorithm comparison
make a6  # Lambda parameter sweep
make a8  # Adaptability test

# Or run all experiments in screen sessions
make run-all
```

Or run directly:
```bash
# Run algorithm comparison
python -m experiments.a5_algorithm_bakeoff.run_experiment

# Run feature ablation
python -m experiments.a3_feature_ablation.run_experiment

# Run lambda parameter sweep
python -m experiments.a6_lambda_sweep.run_experiment
```

Results are saved in `experiments/<experiment_name>/results/` and plots in `experiments/<experiment_name>/plots/`.

## Configuration

Experiment parameters are configured in `experiments/config/experiments.yaml`. Key parameters:
- `n_runs`: Number of runs per experiment (default: 20)
- `samples_per_dataset`: Queries per dataset (default: 500)
- `lambda_weight`: Accuracy-efficiency trade-off parameter

## Key Results

The experiments demonstrate that:
- Contextual bandits significantly outperform non-contextual baselines
- LinUCB achieves the best balance of exploration and exploitation
- Task type and complexity features are most informative for routing

## Limitations

- Dataset size limited to 2500 queries due to computational constraints
- Hyperparameter tuning was exploratory rather than exhaustive

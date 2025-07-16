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

## Quick Start

```bash
# Setup everything
make setup

# Run an experiment
make a5-full

# Generate plots from results
## If not 'make setup' before: make setup-venv install-deps
make a5-plot

# View all available commands
make [TAB][TAB]
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

### Full Experiments

Using Make (recommended):
```bash
# Run individual experiments
make a2-full  # Warmup analysis
make a3-full  # Feature ablation
make a4-full  # Hyperparameter tuning
make a5-full  # Algorithm comparison
make a6-full  # Lambda parameter sweep
make a8-full  # Adaptability test

# Legacy shortcuts (same as -full)
make a2, make a3, make a4, make a5, make a6, make a8

# Run all experiments in screen sessions
make run-all-full
```

Or run directly:
```bash
python -m experiments.a5_algorithm_bakeoff.run_experiment
```

### Generating Plots from Existing Results

You can regenerate plots without re-running experiments. Each experiment has a standalone plotting module that loads results from the `results/` directory:

```bash
# Generate plots using latest results
make a2-plot  # Generates: cumulative regret, selection heatmap
make a3-plot  # Generates: regret boxplot
make a4-plot  # Generates: hyperparameter performance heatmaps
make a5-plot  # Generates: Pareto plots, bar plots, regret curves, model timeline
make a6-plot  # Generates: Pareto subplots, accuracy/energy boxplots
make a8-plot  # Generates: model selection frequency plot

# Generate plots from specific timestamp
make a5-plot TS=20250714_130345
make a6-plot TS=20250714_132438

# Generate all plots
make run-all-plot
```

You can also run plotting modules directly:
```bash
python -m experiments.a5_algorithm_bakeoff.plotting [timestamp]
```

### Managing Plots

```bash
# Clean plots for individual experiments
make clean-a2-plots
make clean-a3-plots
# ... etc

# Clean all plots
make clean-all-plots
```

### Results Storage

- Results are saved in `experiments/<experiment_name>/results/` with timestamps
- Plots are saved in `experiments/<experiment_name>/plots/`
- Each experiment run creates files with format: `<experiment>_<type>_<YYYYMMDD>_<HHMMSS>.csv`

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

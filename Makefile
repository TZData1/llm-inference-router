# Simple Makefile for Contextual LLM Router

.PHONY: help setup clean-docker setup-venv install-deps start-db load-db stop-db clean \
        a2 a3 a4 a5 a6 a8 \
        a2-full a3-full a4-full a5-full a6-full a8-full \
        a2-plot a3-plot a4-plot a5-plot a6-plot a8-plot \
        clean-a2-plots clean-a3-plots clean-a4-plots clean-a5-plots clean-a6-plots clean-a8-plots \
        clean-all-plots \
        run-all run-all-full run-all-plot check-status

# Setup targets
setup: clean-docker setup-venv install-deps start-db load-db
	@echo "Setup complete!"

setup-venv:
	@echo "Creating virtual environment..."
	python3 -m venv venv

install-deps:
	@echo "Installing dependencies..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

start-db:
	@echo "Starting database containers..."
	docker-compose -f db/docker-compose-db.yaml up -d
	@echo "Waiting for database to be ready..."
	@sleep 25

load-db:
	@echo "Loading database backup after sleep..."
	gunzip -c db/llm_db_backup.sql.gz | docker exec -i db-llm_db-1 psql -U tz -d llm_db

stop-db:
	@echo "Stopping database containers..."
	docker-compose -f db/docker-compose-db.yaml down

clean-docker:
	@echo "Cleaning up Docker containers..."
	-docker-compose -f db/docker-compose-db.yaml down

clean: stop-db
	@echo "Cleaning up..."
	rm -rf venv
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Experiment targets (full runs)
a2-full:
	@echo "Running A2 Warmup experiment..."
	./venv/bin/python -m experiments.a2_warmup.run_experiment

a3-full:
	@echo "Running A3 Feature Ablation experiment..."
	./venv/bin/python -m experiments.a3_feature_ablation.run_experiment

a4-full:
	@echo "Running A4 Hyperparameter Tuning experiment..."
	./venv/bin/python -m experiments.a4_hyperparameter_tuning.run_experiment

a5-full:
	@echo "Running A5 Algorithm Bakeoff experiment..."
	./venv/bin/python -m experiments.a5_algorithm_bakeoff.run_experiment

a6-full:
	@echo "Running A6 Lambda Sweep experiment..."
	./venv/bin/python -m experiments.a6_lambda_sweep.run_experiment

a8-full:
	@echo "Running A8 Adaptability experiment..."
	./venv/bin/python -m experiments.a8_adaptability.run_experiment

# Plot-only targets (regenerate plots from existing data)
# Usage: make a2-plot [TS=20250714_123456]
a2-plot:
	@echo "Generating A2 Warmup plots..."
	./venv/bin/python -m experiments.a2_warmup.plotting $(TS)

a3-plot:
	@echo "Generating A3 Feature Ablation plots..."
	./venv/bin/python -m experiments.a3_feature_ablation.plotting $(TS)

a4-plot:
	@echo "Generating A4 Hyperparameter Tuning plots..."
	./venv/bin/python -m experiments.a4_hyperparameter_tuning.plotting $(TS)

a5-plot:
	@echo "Generating A5 Algorithm Bakeoff plots..."
	./venv/bin/python -m experiments.a5_algorithm_bakeoff.plotting $(TS)

a6-plot:
	@echo "Generating A6 Lambda Sweep plots..."
	./venv/bin/python -m experiments.a6_lambda_sweep.plotting $(TS)

a8-plot:
	@echo "Generating A8 Adaptability plots..."
	./venv/bin/python -m experiments.a8_adaptability.plotting $(TS)

# Legacy shortcuts 
a2: a2-full
a3: a3-full
a4: a4-full
a5: a5-full
a6: a6-full
a8: a8-full

# Clean plot directories
clean-a2-plots:
	@echo "Cleaning A2 plots..."
	@rm -f experiments/a2_warmup/plots/*.png experiments/a2_warmup/plots/*.jpg

clean-a3-plots:
	@echo "Cleaning A3 plots..."
	@rm -f experiments/a3_feature_ablation/plots/*.png experiments/a3_feature_ablation/plots/*.jpg

clean-a4-plots:
	@echo "Cleaning A4 plots..."
	@rm -f experiments/a4_hyperparameter_tuning/plots/*.png experiments/a4_hyperparameter_tuning/plots/*.jpg

clean-a5-plots:
	@echo "Cleaning A5 plots..."
	@rm -f experiments/a5_algorithm_bakeoff/plots/*.png experiments/a5_algorithm_bakeoff/plots/*.jpg

clean-a6-plots:
	@echo "Cleaning A6 plots..."
	@rm -f experiments/a6_lambda_sweep/plots/*.png experiments/a6_lambda_sweep/plots/*.jpg

clean-a8-plots:
	@echo "Cleaning A8 plots..."
	@rm -f experiments/a8_adaptability/plots/*.png experiments/a8_adaptability/plots/*.jpg

# Clean all plots
clean-all-plots: clean-a2-plots clean-a3-plots clean-a4-plots clean-a5-plots clean-a6-plots clean-a8-plots
	@echo "All plots cleaned!"

# Run all experiments in screen sessions (super slow)
run-all-full:
	@echo "Starting all experiments in screen sessions with logging..."
	@mkdir -p logs
	@screen -L -Logfile logs/exp_a2.log -dmS exp_a2 ./venv/bin/python -m experiments.a2_warmup.run_experiment
	@echo "Started A2 in screen session 'exp_a2' (log: logs/exp_a2.log)"
	@screen -L -Logfile logs/exp_a3.log -dmS exp_a3 ./venv/bin/python -m experiments.a3_feature_ablation.run_experiment
	@echo "Started A3 in screen session 'exp_a3' (log: logs/exp_a3.log)"
	@screen -L -Logfile logs/exp_a4.log -dmS exp_a4 ./venv/bin/python -m experiments.a4_hyperparameter_tuning.run_experiment
	@echo "Started A4 in screen session 'exp_a4' (log: logs/exp_a4.log)"
	@screen -L -Logfile logs/exp_a5.log -dmS exp_a5 ./venv/bin/python -m experiments.a5_algorithm_bakeoff.run_experiment
	@echo "Started A5 in screen session 'exp_a5' (log: logs/exp_a5.log)"
	@screen -L -Logfile logs/exp_a6.log -dmS exp_a6 ./venv/bin/python -m experiments.a6_lambda_sweep.run_experiment
	@echo "Started A6 in screen session 'exp_a6' (log: logs/exp_a6.log)"
	@screen -L -Logfile logs/exp_a8.log -dmS exp_a8 ./venv/bin/python -m experiments.a8_adaptability.run_experiment
	@echo "Started A8 in screen session 'exp_a8' (log: logs/exp_a8.log)"
	@echo "All experiments started! Logs are in logs/ directory."
	@echo "Use 'screen -ls' to see sessions and 'screen -r exp_XX' to attach."

# Check status of running experiments
check-status:
	@echo "=== Running Screen Sessions ==="
	@screen -ls || echo "No screen sessions running"
	@echo ""
	@echo "Check logs in logs/ directory to see experiment progress."

# Run all plot generations
run-all-plot:
	@echo "Generating plots for all experiments..."
	@echo "=== A2 Warmup ==="
	@./venv/bin/python -m experiments.a2_warmup.plotting || echo "A2 plot generation failed"
	@echo ""
	@echo "=== A3 Feature Ablation ==="
	@./venv/bin/python -m experiments.a3_feature_ablation.plotting || echo "A3 plot generation failed"
	@echo ""
	@echo "=== A4 Hyperparameter Tuning ==="
	@./venv/bin/python -m experiments.a4_hyperparameter_tuning.plotting || echo "A4 plot generation failed"
	@echo ""
	@echo "=== A5 Algorithm Bakeoff ==="
	@./venv/bin/python -m experiments.a5_algorithm_bakeoff.plotting || echo "A5 plot generation failed"
	@echo ""
	@echo "=== A6 Lambda Sweep ==="
	@./venv/bin/python -m experiments.a6_lambda_sweep.plotting || echo "A6 plot generation failed"
	@echo ""
	@echo "=== A8 Adaptability ==="
	@./venv/bin/python -m experiments.a8_adaptability.plotting || echo "A8 plot generation failed"
	@echo ""
	@echo "All plot generation complete!"

# Legacy run-all (points to run-all-full)
run-all: run-all-full
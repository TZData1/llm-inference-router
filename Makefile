# Simple Makefile for Contextual LLM Router

.PHONY: setup clean-docker setup-venv install-deps start-db load-db stop-db clean a2 a3 a4 a5 a6 a8 run-all

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

# Experiment targets
a2:
	@echo "Running A2 Warmup experiment..."
	./venv/bin/python -m experiments.a2_warmup.run_experiment

a3:
	@echo "Running A3 Feature Ablation experiment..."
	./venv/bin/python -m experiments.a3_feature_ablation.run_experiment

a4:
	@echo "Running A4 Hyperparameter Tuning experiment..."
	./venv/bin/python -m experiments.a4_hyperparameter_tuning.run_experiment

a5:
	@echo "Running A5 Algorithm Bakeoff experiment..."
	./venv/bin/python -m experiments.a5_algorithm_bakeoff.run_experiment

a6:
	@echo "Running A6 Lambda Sweep experiment..."
	./venv/bin/python -m experiments.a6_lambda_sweep.run_experiment

a8:
	@echo "Running A8 Adaptability experiment..."
	./venv/bin/python -m experiments.a8_adaptability.run_experiment

# Run all experiments in screen sessions
run-all:
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
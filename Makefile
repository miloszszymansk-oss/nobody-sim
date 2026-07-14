# nbody-sim control center (SPEC §10). Run `make` for the target list.
.DEFAULT_GOAL := help
PY ?= python3
IMAGE ?= nbody-sim

.PHONY: help install test lint profile simulate figures serve docker-build docker-run

help: ## list available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install: ## editable install with dev dependencies
	$(PY) -m pip install -e ".[dev]"

test: ## physics test suite (pytest)
	$(PY) -m pytest -q

lint: ## static checks (ruff)
	ruff check src tests experiments benchmarks

profile: ## cProfile digest + A/B grouping benchmark -> benchmarks/out/
	$(PY) benchmarks/profile_bh.py

simulate: ## two-Plummer collision -> JSON for the player + montage figure
	$(PY) experiments/exp_cluster.py
	cp experiments/out/cluster_merger.json visualization/data/

figures: ## regenerate every committed figure from scratch
	$(PY) experiments/exp_energy_drift.py --e 0.9 --orbits 300 --steps-per-orbit 3000
	$(PY) experiments/exp_energy_drift.py --e 0.6 --orbits 2500 --steps-per-orbit 200
	$(PY) experiments/fig_energy_drift.py
	$(PY) benchmarks/bench_scaling.py
	$(PY) experiments/exp_cluster.py

serve: ## player at http://localhost:8000 (bypasses file:// CORS)
	$(PY) -m http.server -d visualization 8000

docker-build: ## build the runtime image
	docker build -t $(IMAGE) .

docker-run: ## default cluster experiment in the container; results land in ./
	docker run --rm \
		-v "$(PWD)/experiments/out:/app/experiments/out" \
		-v "$(PWD)/figures:/app/figures" \
		$(IMAGE)

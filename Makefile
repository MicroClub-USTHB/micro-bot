# Micro Bot: one-line commands. Run `make` or `make help` to list them.
.DEFAULT_GOAL := help
SHELL := /bin/bash

API_URL ?= http://localhost:8000
DOCS    ?= data/docs
VIS     ?= public
Q       ?= What is Micro Club?
FAST_TESTS  := not ocr and not tokenizer and not embedder and not reranker
MODEL_TESTS := ocr or tokenizer or embedder or reranker

.PHONY: help dev env install up down restart logs ps health ingest ask docs models \
        test test-models lint format check eval clean reset

help: ## List the commands
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk -F':.*## ' '{printf "  \033[36mmake %-12s\033[0m %s\n", $$1, $$2}'

## --- Run --------------------------------------------------------------------

dev: env up health ## Create .env if missing, build and start the stack, wait for health

env: ## Create .env with fresh secrets (never overwrites)
	@scripts/init_env.sh

up: ## Build and start the API + Qdrant containers
	docker compose up -d --build

down: ## Stop the containers (data kept)
	docker compose down

restart: ## Restart the containers
	docker compose restart

logs: ## Follow the API logs
	docker compose logs -f app

ps: ## Show container status
	docker compose ps

health: ## Wait for the API and print its health
	@scripts/wait_healthy.sh $(API_URL)

## --- Use --------------------------------------------------------------------

ingest: ## Upload a folder: make ingest DOCS=data/docs VIS=public
	@API_URL=$(API_URL) scripts/ingest.sh $(DOCS) $(VIS)

ask: ## Ask a question: make ask Q="Quand le club a-t-il été fondé ?"
	@curl -sS $(API_URL)/api/v1/chat -H 'content-type: application/json' \
		-d "$$(python3 -c 'import json,sys; print(json.dumps({"message": sys.argv[1]}))' "$(Q)")" | python3 -m json.tool

docs: ## List indexed documents
	@curl -sS $(API_URL)/api/v1/documents -H "X-Admin-Key: $$(grep '^ADMIN_API_KEY=' .env | cut -d= -f2-)" | python3 -m json.tool

## --- Develop ----------------------------------------------------------------

install: ## Install Python deps locally (uv)
	uv sync --locked

models: install ## Download the model files into ./models (for model tests and eval)
	MODELS_DIR=$(CURDIR)/models uv run python scripts/download_models.py

test: install ## Fast tests (no model files)
	uv run pytest -m "$(FAST_TESTS)"

test-models: models ## Tests that need the model files
	MODELS_DIR=$(CURDIR)/models TESSDATA_PREFIX=$(CURDIR)/models/tessdata uv run pytest -m "$(MODEL_TESTS)"

lint: ## Ruff lint + format check
	uv run ruff check . && uv run ruff format --check .

format: ## Auto-fix lint and format
	uv run ruff check --fix . && uv run ruff format .

check: ## What CI runs: lock check, lint, fast tests
	uv lock --check && $(MAKE) --no-print-directory lint test

eval: models ## Retrieval eval with real models (in-memory Qdrant)
	MODELS_DIR=$(CURDIR)/models TESSDATA_PREFIX=$(CURDIR)/models/tessdata uv run python -m eval.run_eval --label local

## --- Clean ------------------------------------------------------------------

clean: ## Remove caches and build leftovers
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -not -path './.venv/*' -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info

reset: clean ## Stop the stack and DELETE its data (index, docs), models and .venv
	@read -p "Delete the Qdrant index, stored documents, ./models and .venv? [y/N] " a && [[ $$a == y ]]
	docker compose down -v --rmi local
	rm -rf models .venv

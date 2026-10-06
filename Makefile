.DEFAULT_GOAL := help
.PHONY: help install serve dev test lint format eval-offline eval-real agent-demo set free-models docker-build docker-up clean

CONFIG ?= configs/eval.yaml

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies incl. dev tools and the `classifier` extra (CPU torch + transformers)
	uv sync --all-extras

serve:  ## Gateway + dashboard on http://localhost:8000 (fake gullible upstream unless configured)
	uv run bulwark serve --port 8000

dev:  ## Same as serve, with auto-reload
	uv run bulwark serve --port 8000 --reload

test:  ## Test suite (no API keys; classifier tests run only if the model is cached)
	uv run pytest

lint:  ## Ruff lint + format check + mypy (strict)
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:  ## Auto-format and fix lint issues
	uv run ruff format src tests
	uv run ruff check --fix src tests

set:  ## Rebuild data/handwritten/handwritten.jsonl from the hand-written YAML source
	uv run bulwark eval build-set

agent-demo:  ## Email agent under attack, without vs with Bulwark (fake gullible model, offline)
	uv run bulwark agent-demo

eval-offline:  ## Detection by layer and dataset, classifier choice, agent with the fake model: no API calls
	uv run bulwark eval classifiers -c $(CONFIG)
	uv run bulwark eval detection -c $(CONFIG)
	uv run bulwark eval agent -c $(CONFIG) --fake

eval-real:  ## LLM-judge layer and the agent with a free OpenRouter model (needs OPENROUTER_API_KEY)
	uv run bulwark eval detection -c $(CONFIG) --judge
	uv run bulwark eval agent -c $(CONFIG) --real

free-models:  ## List free OpenRouter models and smoke-test tool calling on three of them
	uv run bulwark models free --smoke 3 --tools

docker-build:  ## Build the gateway image (classifier runtime included, model downloaded on first start)
	docker compose build

docker-up:  ## Gateway in Docker on http://localhost:8000
	docker compose up --build

clean:  ## Remove caches (keeps results/)
	rm -rf .cache .pytest_cache .mypy_cache .ruff_cache .hypothesis

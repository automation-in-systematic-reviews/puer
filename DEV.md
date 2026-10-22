# Development guide

This guide is the entry point for returning contributors working on the inference API.
For native local setup and troubleshooting, see [Local development](docs/local-development.md).

## Development workflows

Use the existing Docker Compose workflow when developing in the containerized environment described in `README.md`.
Use the native workflow when running the API directly from the local `inference-api` Conda environment.
Native setup steps are single-sourced in [Local development](docs/local-development.md).
Application keys, environment loading, and Docker Compose forwarding are documented in [Environment variables](docs/environment-variables.md).
LLM model and reasoning scopes, provider compatibility, and provenance alignment are documented in [LLM configuration](docs/llm-configuration.md).

## Dependency ownership

Dependencies are layered as Docker > Conda > uv.
Docker creates the named `inference-api` Conda environment for the container image.
Conda owns Python 3.12, PyTorch, Git LFS, `uv`, and `just`.
uv owns all remaining Python runtime and development dependencies declared in `inference-api/pyproject.toml` and locked in `inference-api/uv.lock`.

## Just recipes

Run recipes from the PUER repository root (`puer/`).
Activate the named `inference-api` Conda environment for runtime and quality recipes; health checks do not require activation.
The root `justfile` runs recipe commands inside `inference-api/`, where the Python project and application paths live.
Invoking `just` from `inference-api/` also discovers the root justfile.

- `just` runs the default recipe, which lists the available recipes in unsorted order.
- `just --list` lists the available recipes directly.
- `just checkhealth` runs all member health checks without installing dependencies, downloading assets, or starting the API.
- `just check-conda` checks that the named `inference-api` Conda environment exists.
- `just check-datasets` checks the required threshold dataset and fails if its contents are missing.
- `just check-bert-models` checks the two BERT model directories and only warns when either is missing.
- `just run-local` starts Uvicorn with reload on port 12306 for native local development.
- `just run-local-no-bert` uses the same local server settings without loading local BERT models or threshold data; see [Local development](docs/local-development.md) for route availability.
- `just test` runs the pytest suite.
- `just fmt` applies Ruff lint fixes, including import sorting, and formats `app` and `tests`.
- `just lint` checks Ruff formatting, runs Ruff linting, and runs `ty` on `app` and `tests` without modifying source files.

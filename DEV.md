# Development guide

This guide is the entry point for returning contributors working on the inference API.
For native local setup and troubleshooting, see [Local development](docs/local-development.md).

## Development workflows

Use the existing Docker Compose workflow when developing in the containerized environment described in `README.md`.
Use the native workflow when running the API directly from the local `inference-api` Conda environment.
Native setup steps are single-sourced in [Local development](docs/local-development.md).

## Dependency ownership

Dependencies are layered as Docker > Conda > uv.
Docker creates the named `inference-api` Conda environment for the container image.
Conda owns Python 3.12, PyTorch, Git LFS, `uv`, and `just`.
uv owns all remaining Python runtime and development dependencies declared in `inference-api/pyproject.toml` and locked in `inference-api/uv.lock`.

## Just recipes

Run recipes from `inference-api` with the named `inference-api` Conda environment active.

- `just` runs the default recipe, which lists the available recipes in unsorted order.
- `just --list` lists the available recipes directly.
- `just init` synchronizes the locked uv dependencies into the active Conda environment.
- `just run-local` starts Uvicorn with reload on port 12306 for native local development.
- `just test` runs the pytest suite.
- `just fmt` applies Ruff lint fixes, including import sorting, and formats `app` and `tests`.
- `just lint` checks Ruff formatting, runs Ruff linting, and runs `ty` on `app` and `tests` without modifying source files.

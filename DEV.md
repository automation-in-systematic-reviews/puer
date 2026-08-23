# Development guide

This guide is the entry point for returning contributors working on the inference API.
For native local setup and troubleshooting, see [Local development](docs/local-development.md).

## Development workflows

Use the existing Docker Compose workflow when developing in the containerized environment described in `README.md`.
Use the native workflow when running the API directly from the local `inference-api` Conda environment.
Native setup steps are single-sourced in [Local development](docs/local-development.md).

## Make targets

Run these targets from `inference-api` after activating the `inference-api` environment.

- `make test` runs the pytest suite.
- `make run-local` starts Uvicorn with reload on port 12306 for native local development.
- `make fmt` formats `app` and `tests` with Black and isort.
- `make lint` checks `app` and `tests` with flake8.

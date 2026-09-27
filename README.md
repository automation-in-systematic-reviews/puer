# PU'ER

**P**U'ER **U**tilities for **E**nhancing systematic **R**eviews

PU'ER is a FastAPI inference service for utilities that enhance systematic reviews.
The service provides health checks, a debug text classifier, embedding-based study screening, prompt-based title and abstract extraction and screening, and PDF risk-of-bias assessment endpoints.

Key locations:
- `docs/api-endpoints.md`: docs for the API endpoints
- `docs/environment-variables.md`: canonical API-key, environment-loading, and Docker Compose reference
- `docs/llm-configuration.md`: canonical model, reasoning, provider, and provenance reference
- `docs/local-development.md`: native development without Docker Compose
- `inference-api/app`: API code

## Usage

Start the API from the repository root with Docker Compose:

```sh
docker-compose up
```

The service is available at `http://localhost:12306` by default.
Open `http://localhost:12306/docs` for the FastAPI interactive API docs.

See [Environment variables](docs/environment-variables.md) for API keys, application environment loading, and the Docker Compose port input.
See [LLM configuration](docs/llm-configuration.md) for model and reasoning settings used by the prompt and risk-of-bias endpoints.

The Docker container uses `/inference-api` as its working directory.

Endpoint details are documented in [docs/api-endpoints.md](docs/api-endpoints.md).

## Setup

Install Docker and Git LFS before building the service.
On macOS, one way to install them is:

```sh
brew install --cask docker
brew install git-lfs
git lfs install
```

Configure API keys and environment loading as described in [Environment variables](docs/environment-variables.md), then use [LLM configuration](docs/llm-configuration.md) for nonsecret model and reasoning overrides.

The service expects the following model and data paths inside `inference-api`:

- `models/albert-base-v2-imdb`
- `models/cup_multi_gpu_24_05_30`
- `data/summary_26_01_05.csv`

The ALBERT debug model can be fetched from Hugging Face:

```sh
git clone https://huggingface.co/textattack/albert-base-v2-imdb \
  inference-api/models/albert-base-v2-imdb
```

The code uses `data/summary_26_01_05.csv` for study-screening thresholds.
The repository also includes the earlier `data/summary_24_08_08.csv` threshold file.

Build the Docker image from the repository root:

```sh
docker-compose build
```

## Development

The Docker Compose service mounts `./inference-api` into the container.
The API is started with Uvicorn reload enabled, so changes under `inference-api/app` restart the service.
Model loading can take time after each restart.
The image builds the project's dependencies into the named `inference-api` Conda environment.

To open a shell in the running container:

```sh
docker-compose exec inference-api micromamba run -n inference-api bash
```

This starts the shell with the named Conda environment active.
Run development commands from inside the container:

```sh
just --list
just test
just fmt
just lint
```

`just test` runs pytest.
`just fmt` applies Ruff lint fixes, including import sorting, and formats `app` and `tests`.
`just lint` checks Ruff formatting, runs Ruff linting, and runs `ty` on `app` and `tests`.
For native development, follow [Local development](docs/local-development.md).

## Verify Docker changes

Run this Docker verification after changes to the image, dependencies, or Compose configuration.
Build and start the service from the repository root:

```sh
docker-compose build
docker-compose up -d
```

Verify the available recipes and quality checks inside the running container:

```sh
docker-compose exec inference-api micromamba run -n inference-api bash
just --list
just lint
just test
```

Then verify `http://localhost:12306/check` and open `http://localhost:12306/docs`.
These checks require the configured authentication value and model and threshold-data assets, and `/check` returns `true` only when all configured asset paths are available.
See [Environment variables](docs/environment-variables.md) for configuration details.

## Deployment

After completing setup and building the image, run the service in the background:

```sh
docker-compose up -d
```

## Configuration

See [Environment variables](docs/environment-variables.md) for API keys, supported application environment behavior, native shell setup, and Docker Compose forwarding.
See [LLM configuration](docs/llm-configuration.md) for model and reasoning defaults, endpoint scope, provider compatibility, and provenance alignment.

## References

- FastAPI documentation: https://fastapi.tiangolo.com/
- ALBERT debug model: https://huggingface.co/textattack/albert-base-v2-imdb

## Citation

> TBD

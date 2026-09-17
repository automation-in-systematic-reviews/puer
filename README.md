# PU'ER

**P**U'ER **U**tilities for **E**nhancing systematic **R**eviews

PU'ER is a FastAPI inference service for utilities that enhance systematic reviews.
The service provides health checks, a debug text classifier, embedding-based study screening, prompt-based title and abstract extraction and screening, and PDF risk-of-bias assessment endpoints.

Key locations:
- `docs/api-endpoints.md`: docs for the API endpoints
- `inference-api/app`: API code

## Usage

Start the API from the repository root with Docker Compose:

```sh
docker-compose up
```

The service is available at `http://localhost:12306` by default.
Open `http://localhost:12306/docs` for the FastAPI interactive API docs.

The default port can be changed with `INFERENCE_API_PORT` in the project-level `.env` file.

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

Create a `.env` file at the repository root and set the API key used by the protected study-screening endpoint:

```sh
WCRF_API_KEY=<api-key>
```

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
docker-compose exec -it inference-api bash
```

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
docker-compose exec -it inference-api bash
just --list
just lint
just test
```

Then verify `http://localhost:12306/check` and open `http://localhost:12306/docs`.
These checks require `WCRF_API_KEY` and the configured model and threshold-data assets, and `/check` returns `true` only when all configured asset paths are available.

## Deployment

After completing setup and building the image, run the service in the background:

```sh
docker-compose up -d
```

## Configuration

Environment variables are loaded from the project-level `.env` file.

- `WCRF_API_KEY`: API key accepted by `/study_screening/encode`.
- `INFERENCE_API_PORT`: Optional host port for Docker Compose.
  Defaults to `12306`.

## References

- FastAPI documentation: https://fastapi.tiangolo.com/
- ALBERT debug model: https://huggingface.co/textattack/albert-base-v2-imdb

## Citation

> TBD

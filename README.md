# PU'ER

**P**U'ER **U**tilities for **E**nhancing systematic **R**eviews

PU'ER is a FastAPI inference service for utilities that enhance systematic reviews.
The current service provides health checks, a debug text classifier endpoint, and a study-screening endpoint that scores a study against a review topic.

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

The current code expects `data/summary_26_01_05.csv` for study-screening thresholds.
At the time of writing, the repository contains `data/summary_24_08_08.csv`, so the expected threshold file should be supplied or the configured path should be updated before relying on `/check` or `/study_screening/encode`.

Build the Docker image from the repository root:

```sh
docker-compose build
```

## Development

The Docker Compose service mounts `./inference-api` into the container.
The API is started with Uvicorn reload enabled, so changes under `inference-api/app` restart the service.
Model loading can take time after each restart.

To open a shell in the running container:

```sh
docker-compose exec -it inference-api bash
```

Run development commands from inside the container:

```sh
make test
make fmt
make lint
```

`make test` runs pytest.
`make fmt` runs Black and isort on `app` and `tests`.
`make lint` runs flake8 on `app` and `tests`.

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

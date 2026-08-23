# Local development

Use this guide to run the inference API natively rather than through Docker Compose.

## Prerequisites

Install a Conda-compatible environment manager.
Install Git LFS when the required model or data assets are provided through LFS.
Run `git lfs install` before obtaining LFS-managed assets.

## Create and activate the environment

Run all commands in this guide from `inference-api`.

```sh
conda env create -f environment.yml
conda activate inference-api
```

Use the equivalent commands if your Conda-compatible manager has different syntax.

## Obtain required assets

The API requires these paths relative to `inference-api`:

- `models/albert-base-v2-imdb/` for the debug ALBERT classifier.
- `models/cup_multi_gpu_24_05_30/` for the study-screening model.
- `data/summary_26_01_05.csv` for study-screening thresholds.

Obtain the public ALBERT model from Hugging Face:

```sh
git clone https://huggingface.co/textattack/albert-base-v2-imdb models/albert-base-v2-imdb
```

Contact the project maintainer for the private screening model at `models/cup_multi_gpu_24_05_30/` and the threshold data at `data/summary_26_01_05.csv`.

## Configure local environment variables

Create `inference-api/.env` with a non-secret pseudo key for protected local endpoints:

```sh
WCRF_API_KEY=local-development-key
```

`OPENAI_API_KEY` is optional for most local work.
Set it in `inference-api/.env` only when exercising the risk-of-bias endpoint.

## Validate and run the API

With the `inference-api` environment active, first validate the local checkout:

```sh
make test
```

Use `make test` as the first troubleshooting step after changing the environment, dependencies, or local assets.

Start the API after validation succeeds:

```sh
make run-local
```

Verify that configured model and data paths are available at http://localhost:12306/check.
Open http://localhost:12306/docs to inspect the running API.

## Troubleshooting

If startup reports a missing environment variable, confirm that `inference-api/.env` exists and defines `WCRF_API_KEY`.
Add `OPENAI_API_KEY` only when using the risk-of-bias endpoint.

If `/check` returns `false` or model loading fails, confirm that all three required asset paths exist exactly as listed above.
Request the private screening model or threshold data from the project maintainer when either private asset is unavailable.

If Make cannot find its targets or the API cannot resolve relative paths, confirm that the current directory is `inference-api`.
If Python uses unexpected dependencies, reactivate the `inference-api` Conda environment and rerun `make test`.

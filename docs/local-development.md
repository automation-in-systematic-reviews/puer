# Local development

Use this guide to run the inference API natively rather than through Docker Compose.

## Prerequisites

Install a Conda-compatible environment manager.
Install Git LFS when the required model or data assets are provided through LFS.
Run `git lfs install` before obtaining LFS-managed assets.

## Create and activate the environment

Run all commands in this guide from `inference-api`.
Create a new environment, activate it, and synchronize the locked uv dependencies.

```sh
conda env create -f environment.yml
conda activate inference-api
just init
```

Use the equivalent commands if your Conda-compatible manager has different syntax.
Recreate an environment that predates the uv migration so legacy Conda-owned Python packages do not remain installed.

```sh
conda env remove --name inference-api
conda env create -f environment.yml
conda activate inference-api
just init
```

After that migration, update the environment following later changes to `environment.yml`, then activate it and synchronize the locked uv dependencies again.

```sh
conda env update -f environment.yml --prune
conda activate inference-api
just init
```

The recipes reject any active Conda environment other than `inference-api`.
`just init` makes uv target the active Conda prefix, so dependencies are not installed into a nested `.venv`.

## Obtain required assets

The API requires these paths relative to `inference-api`.

- `models/albert-base-v2-imdb/` for the debug ALBERT classifier.
- `models/cup_multi_gpu_24_05_30/` for the legacy study-screening model.
- `data/summary_26_01_05.csv` for legacy study-screening thresholds.

Obtain the public ALBERT model from Hugging Face.

```sh
git clone https://huggingface.co/textattack/albert-base-v2-imdb models/albert-base-v2-imdb
```

Contact the project maintainer for the private screening model at `models/cup_multi_gpu_24_05_30/` and the threshold data at `data/summary_26_01_05.csv`.

## Configure local environment variables

Set `WCRF_API_KEY` for protected local endpoints without recording its value in project files or documentation.
Set `OPENAI_API_KEY` only when deliberately invoking an OpenAI-backed endpoint.
The prompt-based screening endpoints use `OPENAI_STUDY_SCREENING_MODEL`, which defaults to `gpt-5.6-terra`.
They use `OPENAI_STUDY_SCREENING_REASONING_EFFORT`, which defaults to `medium`.
Set either screening-specific variable only to override its default for a local deployment.
The risk-of-bias endpoint remains separate and uses `OPENAI_MODEL`, defaulting to `gpt-5.2`, and `OPENAI_REASONING_EFFORT`, defaulting to `medium`.

The OpenAI client is initialized lazily when a prompt-based extraction or prediction request reaches the provider boundary.
Starting the API, importing the screening service, and running its mocked tests do not construct the screening OpenAI client.
Missing or invalid provider configuration therefore affects a prompt-based request as a `503` configuration error rather than native startup.

## Offline verification

With the `inference-api` Conda environment active, run the focused prompt-based verification command below.
It runs the extraction route, prediction route, and screening-service tests with mocked services or OpenAI clients and does not require or permit paid provider calls.

```sh
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv run --offline --locked python -m pytest -vv tests/test_data_extraction.py tests/test_study_screening.py tests/test_title_abstract_screening.py
```

Run the full offline suite with the following command.
The full suite uses mocks for OpenAI-backed behavior and does not require or permit paid provider calls.

```sh
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv run --offline --locked python -m pytest -vv
```

The focused and full commands require an initialized local environment and use only locally installed locked dependencies.
They may require the local model and threshold assets for legacy tests that start the full application lifespan.

## Run the API

Start the API after offline verification succeeds.

```sh
just run-local
```

Verify that configured model and data paths are available at `http://localhost:12306/check`.
Open `http://localhost:12306/docs` to inspect the running API.
Invoking a prompt-based endpoint against a configured provider is an explicit operator action and is outside the offline verification workflow.

## Troubleshooting

If startup reports a missing environment variable, confirm that the local environment supplies `WCRF_API_KEY`.
Add `OPENAI_API_KEY` only when deliberately using an OpenAI-backed endpoint.
If a prompt-based request returns a configuration `503`, check `OPENAI_API_KEY` and any screening-specific overrides.
If `/check` returns `false` or model loading fails, confirm that all three required asset paths exist exactly as listed above.
Request the private screening model or threshold data from the project maintainer when either private asset is unavailable.
If just cannot find its recipes or the API cannot resolve relative paths, confirm that the current directory is `inference-api`.
If Python uses unexpected dependencies, reactivate the `inference-api` Conda environment, run `just init`, and rerun the offline verification command.

# Local development

Use this guide to run the inference API natively rather than through Docker Compose.

## Prerequisites

Install a Conda-compatible environment manager.
The commands below use Micromamba; Conda users can replace `micromamba` with `conda`.
Install Git LFS when the required model or data assets are provided through LFS.
Run `git lfs install` before obtaining LFS-managed assets.

## Create and activate the environment

Run all commands in this guide from `inference-api`.
Create the environment and activate it as a replacement for the current environment.
Use `micromamba activate inference-api`, not `micromamba activate --stack inference-api`, so tools cannot leak in from another environment.

```sh
micromamba env create -f environment.yml
micromamba activate inference-api
```

Verify that Python, uv, and just all come from the new environment before synchronizing project dependencies.

```sh
python --version
command -v python
command -v uv
command -v just
```

Python must report version 3.12, and all three command paths must be under `$CONDA_PREFIX/bin`.
Synchronize the locked uv dependencies after those checks pass.

```sh
just init
```

### Recreate an environment from before the uv migration

Recreate an existing `inference-api` environment if it uses a Python version other than 3.12 or does not contain its own `uv` and `just` executables.
Updating that legacy environment in place can retain incompatible Conda-owned packages, so remove it instead.
Deactivate `inference-api` before removing it.

```sh
micromamba deactivate
micromamba env remove --name inference-api
micromamba env create -f environment.yml
micromamba activate inference-api
python --version
command -v python
command -v uv
command -v just
just init
```

### Update an environment created after the migration

After the initial migration, update the environment following later changes to `environment.yml`, then activate it normally and synchronize the locked uv dependencies again.

```sh
micromamba env update -f environment.yml --prune
micromamba activate inference-api
just init
```

The recipes reject an active Conda environment other than `inference-api`, an unsupported Python version, or development tools inherited from another environment.
`just init` makes uv target the active Conda prefix, so dependencies are not installed into a nested `.venv`.

## Obtain required assets

The default full-model API launch requires these paths relative to `inference-api`.
Skip these asset downloads when using `just run-local-no-bert`; the Python dependencies are still required.

- `models/albert-base-v2-imdb/` for the debug ALBERT classifier.
- `models/cup_multi_gpu_24_05_30/` for the legacy study-screening model.
- `data/summary_26_01_05.csv` for legacy study-screening thresholds.

Obtain the public ALBERT model from Hugging Face.

```sh
git clone https://huggingface.co/textattack/albert-base-v2-imdb models/albert-base-v2-imdb
```

Contact the project maintainer for the private screening model at `models/cup_multi_gpu_24_05_30/` and the threshold data at `data/summary_26_01_05.csv`.

## Configure local environment variables

Use [Environment variables](environment-variables.md) as the canonical reference for API keys, environment loading, native shell setup, restart behavior, and Docker forwarding.
Use [LLM configuration](llm-configuration.md) for model and reasoning defaults, endpoint scope, provider compatibility, lazy initialization, and provenance alignment.
For native development, export values in the shell that launches `just run-local` or `just run-local-no-bert` so the task runner and API process inherit them.
Restart the API after changing an application variable.

## Offline verification

With the `inference-api` Conda environment active, run the focused prompt-based verification command below.
It runs the extraction route, prediction route, and screening-service tests with mocked services or OpenAI clients and does not require or permit paid provider calls.

```sh
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv run --offline --locked python -m pytest -vv tests/test_data_extraction.py tests/test_study_screening.py tests/test_title_abstract_screening.py
```

Run the full offline suite with the following command.
The full suite mocks local models, threshold loading, and OpenAI-backed behavior and does not require or permit paid provider calls.

```sh
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv run --offline --locked python -m pytest -vv
```

The focused and full commands require an initialized local environment and use only locally installed locked dependencies.
Neither command requires local model or threshold assets.
The test configuration disables `.env` loading and uses synthetic API keys.

## Run the API

Start the default full-model API after offline verification succeeds.

```sh
just run-local
```

Verify that configured model and data paths are available at `http://localhost:12306/check`.

For prompt-based development without local BERT models or threshold data, use:

```sh
just run-local-no-bert
```

This recipe retains the local port and reload behavior and skips both local model loaders and the threshold CSV.
The startup switch is documented under `PUER_LOAD_BERT_MODELS` in [Environment variables](environment-variables.md).
In this mode, `/debug/encode` and `/study_screening/encode` return `503`, while prompt extraction, prompt prediction, and risk-of-bias routes remain available with their normal authentication and provider configuration.
`WCRF_API_KEY` is still required at startup; `OPENAI_API_KEY` is required only for provider-backed requests.
The boolean `/check` response is `true` when local assets are disabled, even if their paths are absent; it does not verify provider readiness.
Use the authenticated `/check/extraction` endpoint for local extraction-provider configuration diagnostics without a paid provider request.

Both launch recipes serve `http://localhost:12306/docs` for inspecting the running API.
Invoking a prompt-based endpoint against a configured provider is an explicit operator action and is outside the offline verification workflow.

## Troubleshooting

If startup reports a missing environment variable, check [Environment variables](environment-variables.md) and the shell that launched the API.
If an LLM request returns a configuration `503`, check [LLM configuration](llm-configuration.md) and the shell that launched the API.
In full-model mode, if `/check` returns `false` or model loading fails, confirm that all three required asset paths exist exactly as listed above.
Request the private screening model or threshold data from the project maintainer when either private asset is unavailable, or use `just run-local-no-bert` for prompt-based development.
If a legacy encode route reports that local BERT models are disabled, restart with local model loading enabled and the required assets available.
If `just` reports that no justfile or recipes are available, confirm that the current directory is `inference-api`.
If the shell reports `just: command not found`, activate `inference-api` normally and check `$CONDA_PREFIX/bin/just`.
Recreate the environment using the migration procedure above only if that executable remains absent.
If a recipe reports that Python, uv, or just came from an unexpected path, activate `inference-api` without `--stack` and confirm that each command resolves under `$CONDA_PREFIX/bin`.
If uv reports that the project environment is incompatible and cannot be recreated because it is not a virtual environment, recreate the Conda environment rather than asking uv to replace its prefix.
If Python uses unexpected dependencies after the command-path checks pass, run `just init` and rerun the offline verification command.

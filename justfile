# Keep API commands rooted in the Python project.
set working-directory := "inference-api"

# List available recipes.
default:
    @just --list --unsorted

# ==== Health ====

# Check the Conda environment and local assets without changing them.
[group('health')]
checkhealth:
    #!/usr/bin/env bash
    status=0
    for recipe in check-conda check-datasets check-bert-models; do
        just --no-dotenv --justfile "{{ justfile() }}" "$recipe" || status=1
    done
    exit "$status"

# Check that the named Conda environment exists; activation is not required.
[group('health')]
check-conda:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ "${CONDA_DEFAULT_ENV:-}" == inference-api && \
          -d "${CONDA_PREFIX:-}/conda-meta" ]]; then
        printf '%s\n' 'OK: inference-api Conda environment exists.'
        exit 0
    fi

    for manager in micromamba conda; do
        if command -v "$manager" >/dev/null 2>&1 && \
           "$manager" run -n inference-api sh -c \
               'test -d "$CONDA_PREFIX/conda-meta"' >/dev/null 2>&1; then
            printf '%s\n' 'OK: inference-api Conda environment exists.'
            exit 0
        fi
    done

    printf '%s\n' \
        'ERROR: Cannot find the inference-api Conda environment.' \
        'Create it from inference-api/environment.yml using Conda or Micromamba.' \
        >&2
    exit 1

# Check that the required threshold dataset is present and nonempty.
[group('health')]
check-datasets:
    #!/usr/bin/env bash
    set -euo pipefail
    dataset=data/summary_26_01_05.csv
    if [[ ! -s "$dataset" ]] || \
       grep -q '^version https://git-lfs.github.com/spec/v1$' "$dataset"; then
        printf 'ERROR: Missing dataset contents: inference-api/%s\n' \
            "$dataset" >&2
        exit 1
    fi
    printf 'OK: Dataset is present: inference-api/%s\n' "$dataset"

# Warn about missing BERT model directories without failing the check.
[group('health')]
check-bert-models:
    #!/usr/bin/env bash
    set -euo pipefail
    for model in models/albert-base-v2-imdb models/cup_multi_gpu_24_05_30; do
        if [[ -d "$model" ]]; then
            printf 'OK: Model directory is present: inference-api/%s\n' "$model"
        else
            printf 'WARNING: Missing BERT model: inference-api/%s\n' \
                "$model" >&2
        fi
    done

# ==== Application ====

# Start the API server for the container runtime on port 80.
[group('application')]
run:
    UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX:?Activate inference-api first.}" \
        uv run --locked uvicorn app.main:app --reload \
        --reload-dir /inference-api/app --host 0.0.0.0 --port 80

# Start the API server locally on port 12306.
[group('application')]
run-local:
    UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX:?Activate inference-api first.}" \
        uv run --locked uvicorn app.main:app --reload \
        --reload-dir app --host 0.0.0.0 --port 12306

# Start the API locally without BERT models or threshold data.
[group('application')]
run-local-no-bert:
    PUER_LOAD_BERT_MODELS=false \
        UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX:?Activate inference-api first.}" \
        uv run --locked uvicorn app.main:app --reload \
        --reload-dir app --host 0.0.0.0 --port 12306

# ==== Quality ====

# Run the pytest suite.
[group('quality')]
test:
    UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX:?Activate inference-api first.}" \
        uv run --locked python -m pytest -vv

# Apply Ruff lint fixes, import sorting, and formatting.
[group('quality')]
fmt:
    UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX:?Activate inference-api first.}" \
        uv run --locked ruff check --fix app tests
    UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX:?Activate inference-api first.}" \
        uv run --locked ruff format app tests

# Check formatting, linting, and types without modifying source files.
[group('quality')]
lint:
    UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX:?Activate inference-api first.}" \
        uv run --locked ruff format --check app tests
    UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX:?Activate inference-api first.}" \
        uv run --locked ruff check app tests
    UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX:?Activate inference-api first.}" \
        uv run --locked ty check app tests

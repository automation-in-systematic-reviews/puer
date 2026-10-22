"""Exercise task-runner health checks with isolated local assets and tools."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

MODELS = ("albert-base-v2-imdb", "cup_multi_gpu_24_05_30")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Copy the real justfile into a synthetic, offline project."""
    just = shutil.which("just")
    if just is None:
        pytest.skip("just is required for task-runner checks")

    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "justfile", tmp_path / "justfile"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "just").symlink_to(just)
    for manager in ("micromamba", "conda"):
        executable = bin_dir / manager
        executable.write_text(
            "#!/bin/sh\n"
            '[ "$1" = run ] && [ "$2" = -n ] && '
            '[ "$3" = inference-api ] || exit 99\n'
            'exit "${HEALTH_TEST_CONDA_STATUS:-0}"\n'
        )
        executable.chmod(0o755)

    api = tmp_path / "inference-api"
    (api / "data").mkdir(parents=True)
    (api / "data" / "summary_26_01_05.csv").write_text(
        "topic,strategy,threshold\ntest,default,0.5\n"
    )
    for model in MODELS:
        (api / "models" / model).mkdir(parents=True)
    return tmp_path


def run_just(
    workspace: Path,
    *arguments: str,
    from_api: bool = False,
    conda_status: str = "0",
) -> subprocess.CompletedProcess[str]:
    """Run real Just with only synthetic Conda managers on the search path."""
    res = subprocess.run(
        [str(workspace / "bin" / "just"), "--no-dotenv", *arguments],
        cwd=workspace / "inference-api" if from_api else workspace,
        env={
            "PATH": f"{workspace / 'bin'}{os.pathsep}{os.defpath}",
            "HEALTH_TEST_CONDA_STATUS": conda_status,
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    return res


@pytest.mark.parametrize("from_api", [False, True])
def test_checkhealth_succeeds_without_activation(
    workspace: Path, from_api: bool
) -> None:
    """Discover the relocated justfile and check every component."""
    res = run_just(workspace, "checkhealth", from_api=from_api)
    assert res.returncode == 0, res.stderr
    assert "OK: inference-api Conda environment exists." in res.stdout
    assert "OK: Dataset is present:" in res.stdout
    assert res.stdout.count("OK: Model directory is present:") == 2


@pytest.mark.parametrize("recipe", ["checkhealth", "check-bert-models"])
def test_missing_models_only_warn(workspace: Path, recipe: str) -> None:
    """Missing local BERT models must not make health checks fail."""
    shutil.rmtree(workspace / "inference-api" / "models")
    res = run_just(workspace, recipe)
    assert res.returncode == 0, res.stderr
    assert res.stderr.count("WARNING: Missing BERT model:") == 2


@pytest.mark.parametrize(
    "contents",
    [None, "", "version https://git-lfs.github.com/spec/v1\noid sha256:test\n"],
)
def test_missing_dataset_contents_fail(workspace: Path, contents: str | None) -> None:
    """Missing, empty, and LFS-pointer datasets are not usable assets."""
    dataset = workspace / "inference-api" / "data" / "summary_26_01_05.csv"
    if contents is None:
        dataset.unlink()
    else:
        dataset.write_text(contents)
    res = run_just(workspace, "checkhealth")
    assert res.returncode != 0
    assert "ERROR: Missing dataset contents:" in res.stderr
    assert res.stdout.count("OK: Model directory is present:") == 2


def test_missing_conda_environment_fails(workspace: Path) -> None:
    """Report Conda failure without skipping the independent asset checks."""
    res = run_just(workspace, "checkhealth", conda_status="1")
    assert res.returncode != 0
    assert "ERROR: Cannot find the inference-api Conda environment." in res.stderr
    assert "OK: Dataset is present:" in res.stdout
    assert res.stdout.count("OK: Model directory is present:") == 2


def test_removed_recipes_are_unavailable(workspace: Path) -> None:
    """The health workflow replaces init and ensure-conda."""
    res = run_just(workspace, "--summary")
    assert res.returncode == 0, res.stderr
    assert "init" not in res.stdout.split()
    assert "ensure-conda" not in res.stdout.split()


def test_runtime_requires_an_active_prefix(workspace: Path) -> None:
    """Do not let uv create a nested environment when Conda is inactive."""
    res = run_just(workspace, "test")
    assert res.returncode != 0
    assert "Activate inference-api first." in res.stderr
    assert not (workspace / "inference-api" / ".venv").exists()

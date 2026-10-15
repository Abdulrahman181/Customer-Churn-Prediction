import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "Customer Churn prediction.ipynb"
sys.path.insert(0, str(REPO_ROOT))

import validate_project  # noqa: E402


def test_repository_notebook_passes_static_validation():
    assert validate_project.validate_notebook(NOTEBOOK) == []


def test_validator_rejects_stale_execution_state_and_ipython_commands(tmp_path):
    notebook = {
        "nbformat": 4,
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 7,
                "outputs": [{"output_type": "stream", "text": "old"}],
                "source": ["%matplotlib inline\n"],
            }
        ],
    }
    path = tmp_path / "bad.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")

    errors = validate_project.validate_notebook(path)

    assert any("stale outputs" in error for error in errors)
    assert any("stale execution_count" in error for error in errors)
    assert any("magic/shell" in error for error in errors)

"""Static validation for the customer churn project.

This intentionally does not execute model training because the source dataset is
not committed and heavy ML dependencies may not be installed in every environment.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOK = REPO_ROOT / "Customer Churn prediction.ipynb"
REQUIRED_ENV_MARKER = "CUSTOMER_CHURN_DATA"
NOTEBOOK_MAGIC_RE = re.compile(r"^\s*(?:%|!|\?|get_ipython\s*\()", re.MULTILINE)


def validate_notebook(notebook_path: Path) -> list[str]:
    """Return actionable validation errors for a notebook without executing it."""
    errors: list[str] = []
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read valid notebook JSON: {exc}"]

    if notebook.get("nbformat", 0) < 4:
        errors.append("notebook must use nbformat 4 or newer")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return errors + ["notebook cells must be a list"]

    code = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {index} is not an object")
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            errors.append(f"cell {index} has invalid cell_type: {cell_type!r}")
        source = cell.get("source", "")
        if not isinstance(source, (str, list)):
            errors.append(f"cell {index} source must be a string or list")
            source = ""
        source_text = source if isinstance(source, str) else "".join(source)
        if cell_type == "code":
            code.append(source_text)
            if cell.get("outputs"):
                errors.append(f"code cell {index} contains stale outputs")
            if cell.get("execution_count") is not None:
                errors.append(f"code cell {index} has a stale execution_count")
            try:
                ast.parse(source_text)
            except SyntaxError as exc:
                errors.append(f"code cell {index} has syntax error: {exc}")
            if NOTEBOOK_MAGIC_RE.search(source_text):
                errors.append(f"code cell {index} contains a notebook-only magic/shell command")

    combined = "\n".join(code)
    if REQUIRED_ENV_MARKER not in combined:
        errors.append(f"notebook does not expose the {REQUIRED_ENV_MARKER} data-path setting")
    if "train_test_split" not in combined or "stratify=y" not in combined:
        errors.append("notebook is missing a stratified train/test split")
    if "Pipeline" not in combined or "ColumnTransformer" not in combined:
        errors.append("notebook is missing a leakage-safe preprocessing pipeline")
    if "cross_validate" not in combined or "StratifiedKFold" not in combined:
        errors.append("notebook is missing training-only cross-validated model selection")
    for forbidden in ("D:\\", "C:\\", "/content/", "Users/"):
        if forbidden in combined:
            errors.append(f"notebook contains a non-portable path fragment: {forbidden}")
    return errors


def main() -> int:
    if not NOTEBOOK.is_file():
        print(f"ERROR: missing notebook: {NOTEBOOK}")
        return 1

    errors = validate_notebook(NOTEBOOK)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    print(f"OK: {NOTEBOOK.name}: {len(code_cells)} code cells, valid AST, clean execution state, portable data configuration")
    return 0


if __name__ == "__main__":
    sys.exit(main())

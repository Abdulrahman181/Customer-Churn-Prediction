"""Static validation for the customer churn project.

This intentionally does not execute model training because the source dataset is
not committed and heavy ML dependencies may not be installed in every environment.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOK = REPO_ROOT / "Customer Churn prediction.ipynb"
REQUIRED_ENV_MARKER = "CUSTOMER_CHURN_DATA"


def main() -> int:
    if not NOTEBOOK.is_file():
        print(f"ERROR: missing notebook: {NOTEBOOK}")
        return 1

    try:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid notebook JSON: {exc}")
        return 1

    if notebook.get("nbformat", 0) < 4:
        print("ERROR: notebook must use nbformat 4 or newer")
        return 1

    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    errors: list[str] = []
    code = []
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        code.append(source)
        if cell.get("outputs"):
            errors.append(f"code cell {index} contains stale outputs")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            errors.append(f"code cell {index} has syntax error: {exc}")
        if "%matplotlib" in source or source.lstrip().startswith("!"):
            errors.append(f"code cell {index} contains a notebook-only magic/shell command")

    combined = "\n".join(code)
    if REQUIRED_ENV_MARKER not in combined:
        errors.append(f"notebook does not expose the {REQUIRED_ENV_MARKER} data-path setting")
    if "train_test_split" not in combined or "stratify=y" not in combined:
        errors.append("notebook is missing a stratified train/test split")
    if "Pipeline" not in combined or "ColumnTransformer" not in combined:
        errors.append("notebook is missing a leakage-safe preprocessing pipeline")
    for forbidden in ("D:\\", "C:\\", "/content/", "Users/"):
        if forbidden in combined:
            errors.append(f"notebook contains a non-portable path fragment: {forbidden}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"OK: {NOTEBOOK.name}: {len(code_cells)} code cells, valid AST, no outputs, portable data configuration")
    return 0


if __name__ == "__main__":
    sys.exit(main())

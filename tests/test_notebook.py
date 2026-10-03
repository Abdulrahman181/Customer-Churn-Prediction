from __future__ import annotations

import ast
import json
from pathlib import Path


def test_notebook_code_parses_and_contains_no_saved_outputs():
    notebook_path = Path(__file__).resolve().parents[1] / "Customer Churn prediction.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        assert not cell.get("outputs"), f"Notebook cell {index} contains saved output."
        assert cell.get("execution_count") is None, f"Notebook cell {index} has an execution count."
        source = "".join(cell.get("source", []))
        # Jupyter line magics are not Python syntax; all remaining code should parse.
        python_source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
        )
        if python_source.strip():
            ast.parse(python_source, filename=f"notebook-cell-{index}")

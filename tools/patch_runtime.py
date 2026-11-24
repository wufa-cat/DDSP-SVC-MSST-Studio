# -*- coding: utf-8 -*-
"""Idempotent fixes for third-party packages shipped in runtime\\.

Run:  runtime\\python.exe tools\\patch_runtime.py
Called automatically by start.bat (only once, marker: runtime\\.patched)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / "runtime" / "Lib" / "site-packages"
MARK = ROOT / "runtime" / ".patched"

OLD = """        if "additionalProperties" in schema:
            des += [
                f"str, {_json_schema_to_python_type(schema['additionalProperties'], defs)}"
            ]
"""

NEW = """        if "additionalProperties" in schema:
            _ap = schema["additionalProperties"]
            if isinstance(_ap, dict):
                des += [
                    f"str, {_json_schema_to_python_type(_ap, defs)}"
                ]
"""


def patch_gradio_client():
    """gradio_client chokes on JSON-Schema 'additionalProperties: true' (bool).
    crashes api_info() -> webui dies on launch with 'localhost is not accessible'."""
    f = SP / "gradio_client" / "utils.py"
    if not f.exists():
        return "skip (gradio_client not installed)"
    s = f.read_text(encoding="utf-8")
    if NEW in s:
        return "already patched"
    if OLD not in s:
        return "skip (pattern not found - different gradio version)"
    f.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
    return "patched"


def main():
    report = {"gradio_client": patch_gradio_client()}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if all(v.startswith("skip (gradio_client not") for v in report.values()):
        return 0
    try:
        MARK.write_text(json.dumps(report), encoding="utf-8")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

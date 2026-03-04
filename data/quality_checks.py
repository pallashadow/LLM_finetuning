"""
Data quality checks for RAG training/eval JSONL.

- Malformed JSONL rows (invalid JSON or missing required fields)
- refs must be a non-empty list of strings
- answer must be non-empty
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import REQUIRED_KEYS


class QualityError:
    def __init__(
        self,
        line_no: int | None,
        error_type: str,
        message: str,
        sample_id: str | None = None,
    ):
        self.line_no = line_no
        self.error_type = error_type
        self.message = message
        self.sample_id = sample_id

    def __str__(self) -> str:
        loc = f"line {self.line_no}" if self.line_no is not None else "in-memory"
        if self.sample_id:
            return f"[{loc}] {self.error_type}: {self.message} (id={self.sample_id})"
        return f"[{loc}] {self.error_type}: {self.message}"


def check_malformed_jsonl(path: Path) -> list[QualityError]:
    """Check each line is valid JSON and has required keys; refs is list of strings."""
    errors: list[QualityError] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(QualityError(line_no, "malformed_json", str(e)))
                continue
            if not isinstance(obj, dict):
                errors.append(
                    QualityError(line_no, "not_object", "Each line must be a JSON object.")
                )
                continue
            for key in REQUIRED_KEYS:
                if key not in obj:
                    errors.append(
                        QualityError(
                            line_no,
                            "missing_key",
                            f"Missing required key: {key}",
                            sample_id=obj.get("sample_id"),
                        )
                    )
            refs = obj.get("refs")
            if refs is not None:
                if not isinstance(refs, list):
                    errors.append(
                        QualityError(
                            line_no,
                            "invalid_refs",
                            "`refs` must be a list of strings.",
                            sample_id=obj.get("sample_id"),
                        )
                    )
                else:
                    for i, r in enumerate(refs):
                        if not isinstance(r, str):
                            errors.append(
                                QualityError(
                                    line_no,
                                    "invalid_ref_item",
                                    f"refs[{i}] must be a string.",
                                    sample_id=obj.get("sample_id"),
                                )
                            )
            if "refs" in obj and isinstance(obj["refs"], list) and len(obj["refs"]) == 0:
                errors.append(
                    QualityError(
                        line_no,
                        "empty_refs",
                        "`refs` must not be empty.",
                        sample_id=obj.get("sample_id"),
                    )
                )
    return errors


def check_empty_answers(rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Return list of (index, sample_id) for rows with empty or whitespace-only answer."""
    out: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        answer = (row.get("answer") or "").strip()
        if not answer:
            out.append((i, row.get("sample_id", "?") or "?"))
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_quality_checks(path: Path) -> dict[str, Any]:
    """
    Run all checks on a JSONL file. Returns a report dict with keys:
    - malformed_errors: list of QualityError (as strings)
    - empty_answer_indices: list of (index, sample_id)
    - total_lines: int
    - passed: bool
    """
    malformed = check_malformed_jsonl(path)
    if malformed:
        return {
            "malformed_errors": [str(e) for e in malformed],
            "empty_answer_indices": [],
            "total_lines": 0,
            "passed": False,
        }
    rows = read_jsonl(path)
    empty = check_empty_answers(rows)
    return {
        "malformed_errors": [],
        "empty_answer_indices": [(i, sid) for i, sid in empty],
        "total_lines": len(rows),
        "passed": len(empty) == 0,
    }


def run_quality_checks_and_print(path: Path) -> bool:
    """Run checks and print a short report. Returns True if passed."""
    report = run_quality_checks(path)
    print(f"Quality check: {path}")
    print(f"  Total lines: {report['total_lines']}")
    if report["malformed_errors"]:
        print(f"  Malformed JSON / schema: {len(report['malformed_errors'])} error(s)")
        for e in report["malformed_errors"][:10]:
            print(f"    - {e}")
        if len(report["malformed_errors"]) > 10:
            print(f"    ... and {len(report['malformed_errors']) - 10} more")
    if report["empty_answer_indices"]:
        print(
            f"  Empty answers: {len(report['empty_answer_indices'])} (e.g. {report['empty_answer_indices'][:3]})"
        )
    if report["passed"]:
        print("  Result: PASSED")
    else:
        print("  Result: FAILED")
    return report["passed"]

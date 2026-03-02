import json
import tempfile
import unittest
from pathlib import Path

from data.loaders import _resolve_source_spec, load_split_rows


class TestLoaders(unittest.TestCase):
    def _write_jsonl(self, rows: list[object]) -> str:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        with tmp as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return tmp.name

    def test_resolve_source_spec_legacy_file_key(self) -> None:
        cfg = {"train_file": "data/synthetic_rag_train.jsonl"}
        spec = _resolve_source_spec(cfg, "train")
        self.assertEqual(spec["type"], "jsonl_file")
        self.assertEqual(spec["config"]["path"], "data/synthetic_rag_train.jsonl")

    def test_load_split_rows_with_pipeline(self) -> None:
        path = self._write_jsonl(
            [
                {"question": " q1 ", "answer": " a1 "},
                {"question": " q2 ", "answer": "   "},
            ]
        )
        cfg = {
            "train_source": {"type": "jsonl_file", "config": {"path": path}},
            "pipeline": [{"name": "strip_fields", "config": {"fields": ["question", "answer"]}}],
            "train_pipeline": [{"name": "drop_empty_answer"}],
        }
        rows = load_split_rows(cfg, "train")
        self.assertEqual(rows, [{"question": "q1", "answer": "a1"}])

    def test_load_split_rows_unknown_source(self) -> None:
        cfg = {"train_source": {"type": "not_exists", "config": {}}}
        with self.assertRaises(KeyError):
            load_split_rows(cfg, "train")

    def test_load_split_rows_unknown_pipeline_step(self) -> None:
        path = self._write_jsonl([{"answer": "ok"}])
        cfg = {
            "train_source": {"type": "jsonl_file", "config": {"path": path}},
            "pipeline": [{"name": "not_exists"}],
        }
        with self.assertRaises(KeyError):
            load_split_rows(cfg, "train")

    def test_load_split_rows_jsonl_line_must_be_object(self) -> None:
        path = self._write_jsonl([{"answer": "ok"}, ["not", "object"]])
        cfg = {"train_source": {"type": "jsonl_file", "config": {"path": path}}}
        with self.assertRaises(ValueError):
            load_split_rows(cfg, "train")


if __name__ == "__main__":
    unittest.main()

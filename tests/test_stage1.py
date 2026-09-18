"""Offline preparation contract checks using a pinned miniature source fixture."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("prepare_stage1", Path(__file__).parents[1] / "scripts/prepare_stage1.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

class Stage1Tests(unittest.TestCase):
    def rows(self):
        return [{"id": split, "state_id": split, "family_id": "test", "split": split,
                 "state": "Unique state " + split,
                 "questions": {"q": {"type": "boolean", "instructions": "Allowed?"}},
                 "gold_probs": {"q": {"false": 1, "true": 0}},
                 "gold_probs_kind": "deterministic_truth",
                 "metadata": {"license": "CC0-1.0", "source": "self_authored_programmatic", "source_group_id": split},
                 "teacher": {"native_probs": {"q": {"false": 0.1, "true": 0.9}}}}
                for split in prep.SPLITS]

    def run_fixture(self, rows, check=None):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.jsonl", Path(tmp) / "out"
            payload = "\n".join(json.dumps(r) for r in rows).encode()
            source.write_bytes(payload)
            old = prep.SHA256
            prep.SHA256 = hashlib.sha256(payload).hexdigest()
            try:
                manifest = prep.prepare(source, output)
                if check:
                    check(manifest, output)
            finally:
                prep.SHA256 = old

    def test_whitelist_preserves_gold_and_never_teacher(self):
        def check(manifest, output):
            row = json.loads((output / "train.jsonl").read_text())
            self.assertEqual(row["target"], [1.0, 0.0])
            self.assertNotIn("teacher", row)
            self.assertEqual(manifest["removed_root_field_counts"]["teacher"], 5)
        self.run_fixture(self.rows(), check)

    def test_reject_cross_split_group(self):
        rows = self.rows()
        rows[1]["metadata"]["source_group_id"] = "train"
        with self.assertRaisesRegex(AssertionError, "cross-split"):
            self.run_fixture(rows)

    def test_accept_exact_soft_probability(self):
        rows = self.rows()
        rows[0]["gold_probs_kind"] = "programmatic_conditional_distribution"
        rows[0]["gold_probs"]["q"] = {"false": 0.7, "true": 0.3}
        self.run_fixture(rows, lambda manifest, _: self.assertEqual(manifest["target_kind_counts"]["programmatic_conditional_distribution"], 1))

    def test_conflicting_duplicate_rejected(self):
        rows = self.rows()
        rows[1]["state"] = rows[0]["state"]
        rows[1]["gold_probs"]["q"] = {"false": 0, "true": 1}
        with self.assertRaisesRegex(AssertionError, "conflicting targets"):
            self.run_fixture(rows)

if __name__ == "__main__":
    unittest.main()

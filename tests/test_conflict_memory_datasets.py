import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _load_module(name: str, path: str, package: str = None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if package:
        module.__package__ = package
    spec.loader.exec_module(module)
    return module


if "memory_chain" not in sys.modules:
    pkg = types.ModuleType("memory_chain")
    pkg.__path__ = ["memory_chain"]
    sys.modules["memory_chain"] = pkg

CONFLICT_MODULE = _load_module("memory_chain.conflict_memory", "memory_chain/conflict_memory.py", package="memory_chain")
ConflictAwareMetadataProcessor = CONFLICT_MODULE.ConflictAwareMetadataProcessor


class TestConflictMemoryDatasets(unittest.TestCase):
    def _run_case(self, processor, case):
        profiles = {}
        decisions = []
        for item in case["messages"]:
            metadata = processor.build_metadata_envelope(
                item["raw_message"],
                item["understanding"],
                item["timestamp"],
                item["user_speak"],
            )
            subject = metadata["who"]["subject"]
            if subject not in profiles:
                profiles[subject] = {"event": {}, "fact": {}, "attr": {}}
            event_profile = profiles[subject]["event"]
            fact_profile = profiles[subject]["fact"]
            attr_profile = profiles[subject]["attr"]
            history = processor.get_versions_by_scope(metadata["which"]["scope"], subject)
            result = processor.detect_conflicts(metadata, event_profile, fact_profile, attr_profile, version_history=history)
            processor.maybe_record_conflict(metadata, result)
            processor.register_version(metadata, result, item["understanding"]["index"])
            decisions.append(result["decision"])

            event_profile[metadata["which"]["scope"]] = metadata["what"]["polarity"]
            fact_profile[item["understanding"]["fact"]] = "present"
            for attr in item["understanding"].get("attribue", []):
                attr_profile[attr] = "present"

        return decisions

    def test_conflict_dataset_json_covers_5w_behaviors(self):
        dataset_file = Path("conflict_dataset.json")
        payload = json.loads(dataset_file.read_text(encoding="utf-8"))
        datasets = payload["datasets"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = ConflictAwareMetadataProcessor(
                str(Path(tmp_dir) / "conflicts.json"),
                str(Path(tmp_dir) / "versions.json"),
            )

            observed = {}
            for case in datasets:
                decisions = self._run_case(processor, case)
                observed[case["name"]] = decisions
                expected = case.get("expected", {})
                if "contains_conflict" in expected:
                    if expected["contains_conflict"]:
                        self.assertIn("CONFLICT_EVENT", decisions, msg=case["name"])
                    else:
                        self.assertNotIn("CONFLICT_EVENT", decisions, msg=case["name"])
                if "first_decision" in expected:
                    self.assertGreater(len(decisions), 0, msg=case["name"])
                    self.assertEqual(decisions[0], expected["first_decision"], msg=case["name"])

            hints = processor.get_conflict_hints_for_query("Do I still like running now?")
            self.assertTrue(any("running" in hint for hint in hints))
            self.assertIn("what_preference_flip", observed)
            self.assertIn("who_subject_switch_user_agent", observed)
            self.assertIn("when_temporal_conflict", observed)
            self.assertIn("where_location_conflict", observed)
            self.assertIn("which_scope_missing", observed)


if __name__ == "__main__":
    unittest.main()

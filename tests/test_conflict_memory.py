import importlib.util
import json
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


CONFLICT_MODULE = _load_module("memory_chain.conflict_memory", "memory_chain/conflict_memory.py", package="memory_chain")

# Build lightweight package context so relative import '.utils' in working_memory.py resolves.
import sys
if "memory_chain" not in sys.modules:
    memory_chain_pkg = types.ModuleType("memory_chain")
    memory_chain_pkg.__path__ = ["memory_chain"]
    sys.modules["memory_chain"] = memory_chain_pkg
utils_mod = _load_module("memory_chain.utils", "memory_chain/utils.py", package="memory_chain")
sys.modules["memory_chain.utils"] = utils_mod
WORKING_MODULE = _load_module("memory_chain.working_memory", "memory_chain/working_memory.py", package="memory_chain")

ConflictAwareMetadataProcessor = CONFLICT_MODULE.ConflictAwareMetadataProcessor
Working_Memory = WORKING_MODULE.Working_Memory


class TestConflictAwareMetadataProcessor(unittest.TestCase):
    def test_build_metadata_envelope_extracts_5w(self):
        processor = ConflictAwareMetadataProcessor()
        understanding = {
            "topics": "running",
            "emotions": "Positive",
            "fact": "I run at home every day",
            "attribue": ["athletic"],
            "index": 12,
        }
        envelope = processor.build_metadata_envelope(
            raw_message="I run at home every day",
            message_understanding=understanding,
            timestamp="2026-02-01T10:00:00Z",
            user_speak=True,
        )

        self.assertEqual(envelope["who"]["subject"], "user")
        self.assertEqual(envelope["which"]["scope"], "running")
        self.assertEqual(envelope["what"]["polarity"], "positive")
        self.assertNotEqual(envelope["where"]["location_value"], "unknown")
        self.assertIn(envelope["when"]["time_type"], {"point", "relative"})


    def test_custom_signal_rules_override_polarity(self):
        processor = ConflictAwareMetadataProcessor(
            signal_rules={
                "polarity": {
                    "positive": [r"\bupbeat\b"],
                    "negative": [r"\bdownbeat\b"],
                }
            }
        )
        self.assertEqual(processor._normalize_polarity("I feel upbeat"), "positive")
        self.assertEqual(processor._normalize_polarity("I feel downbeat"), "negative")

    def test_infer_where_avoids_generic_in_pattern_false_positive(self):
        processor = ConflictAwareMetadataProcessor()
        where = processor._infer_where("I am in coding mode and thinking about design")
        self.assertEqual(where["location_value"], "unknown")



    def test_build_metadata_envelope_prefers_llm_fields_when_present(self):
        processor = ConflictAwareMetadataProcessor()
        understanding = {
            "topics": "running",
            "emotions": "Positive",
            "fact": "I run daily",
            "attribue": [],
            "index": 2,
        }
        llm_fields = {
            "who": {"subject": "user", "entity_id": "alice", "confidence": 0.99},
            "what": {"claim_type": "preference", "proposition": "running", "polarity": "negative", "confidence": 0.91},
            "which": {"scope": "jogging", "evidence_span": ["I run daily"], "source_turn_ids": [2]},
            "where": {"location_type": "named", "location_value": "home", "confidence": 0.88},
            "when": {"time_type": "absolute", "time_value": "2026-01-01", "recency_score": 0.95},
        }
        envelope = processor.build_metadata_envelope(
            raw_message="I run daily",
            message_understanding=understanding,
            timestamp="2026-01-01T10:00:00Z",
            user_speak=True,
            llm_fields=llm_fields,
        )

        self.assertEqual(envelope["which"]["scope"], "jogging")
        self.assertEqual(envelope["what"]["polarity"], "negative")
        self.assertEqual(envelope["who"]["entity_id"], "alice")
    def test_detect_conflict_event_for_opposite_attitude(self):
        processor = ConflictAwareMetadataProcessor()
        envelope = {
            "who": {"subject": "user", "entity_id": "self", "confidence": 0.95},
            "what": {"claim_type": "preference", "proposition": "basketball", "polarity": "negative", "confidence": 0.8},
            "which": {"scope": "basketball", "evidence_span": ["I hate basketball"], "source_turn_ids": [3]},
            "where": {"location_type": "unknown", "location_value": "unknown", "confidence": 0.2},
            "when": {"time_type": "point", "time_value": "2026-01-01", "recency_score": 1.0},
        }
        event_profile = {"basketball": "The 1 round attitude: Positive; "}
        result = processor.detect_conflicts(envelope, event_profile, {}, {})

        self.assertEqual(result["decision"], "CONFLICT_EVENT")
        self.assertGreaterEqual(result["dimensions"]["what"], 0.85)



    def test_register_version_marks_previous_active_on_conflict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            conflict_path = Path(tmp_dir) / "conflicts.json"
            version_path = Path(tmp_dir) / "versions.json"
            processor = ConflictAwareMetadataProcessor(str(conflict_path), str(version_path))

            metadata1 = {
                "who": {"subject": "user", "entity_id": "self", "confidence": 0.95},
                "what": {"claim_type": "preference", "proposition": "basketball", "polarity": "positive", "confidence": 0.8},
                "which": {"scope": "basketball", "evidence_span": ["I like basketball"], "source_turn_ids": [1]},
                "where": {"location_type": "contextual", "location_value": "home", "confidence": 0.7},
                "when": {"time_type": "point", "time_value": "2026-01-01", "recency_score": 1.0},
            }
            processor.register_version(metadata1, {"decision": "ADD"}, 1)

            metadata2 = {
                "who": {"subject": "user", "entity_id": "self", "confidence": 0.95},
                "what": {"claim_type": "preference", "proposition": "basketball", "polarity": "negative", "confidence": 0.8},
                "which": {"scope": "basketball", "evidence_span": ["I hate basketball"], "source_turn_ids": [2]},
                "where": {"location_type": "contextual", "location_value": "office", "confidence": 0.7},
                "when": {"time_type": "point", "time_value": "2026-02-01", "recency_score": 1.0},
            }
            processor.register_version(metadata2, {"decision": "CONFLICT_EVENT"}, 2)

            versions = json.loads(version_path.read_text(encoding="utf-8"))
            self.assertEqual(len(versions), 2)
            self.assertEqual(versions[0]["status"], "superseded_by_conflict")
            self.assertEqual(versions[1]["status"], "conflicted")

    def test_conflict_hints_contains_time_conflict_message(self):
        processor = ConflictAwareMetadataProcessor()
        processor.conflict_ledger = [
            {
                "metadata": {
                    "which": {"scope": "running"},
                    "what": {"proposition": "run every day"}
                },
                "conflict": {
                    "reason": "该偏好存在时间冲突",
                    "dimensions": {"when": 0.8}
                }
            }
        ]
        hints = processor.get_conflict_hints_for_query("Do I still like running now?")
        self.assertTrue(any("该偏好存在时间冲突" in hint for hint in hints))

    def test_records_conflict_to_ledger_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ledger_path = Path(tmp_dir) / "conflict.json"
            processor = ConflictAwareMetadataProcessor(str(ledger_path))
            envelope = {
                "who": {"subject": "user", "entity_id": "self", "confidence": 0.95},
                "what": {"claim_type": "preference", "proposition": "coffee", "polarity": "negative", "confidence": 0.8},
                "which": {"scope": "coffee", "evidence_span": ["I hate coffee"], "source_turn_ids": [1]},
                "where": {"location_type": "unknown", "location_value": "unknown", "confidence": 0.2},
                "when": {"time_type": "unknown", "time_value": "unknown", "recency_score": 0.2},
            }
            conflict = {
                "decision": "CONFLICT_EVENT",
                "score": 0.7,
                "dimensions": {"what": 0.8},
                "boundaries": {},
            }

            processor.maybe_record_conflict(envelope, conflict)
            self.assertTrue(ledger_path.exists())
            data = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["conflict"]["decision"], "CONFLICT_EVENT")


class TestWorkingMemoryMetadata(unittest.TestCase):
    def test_working_memory_persists_metadata_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wm_path = Path(tmp_dir) / "wm.json"
            wm = Working_Memory(str(wm_path), "user-1", "model", 2, None, 1)
            wm.add_message_to_working_memory(
                raw_message="message",
                message="[1] summary",
                topics="topic",
                emotions="positive",
                reason="reason",
                index=1,
                timestamp="2026-01-01",
                fact="fact",
                attribute=["attr"],
                metadata={"who": {"subject": "user"}},
                conflict={"decision": "ADD"},
            )

            payload = json.loads(wm_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["metadata"]["who"]["subject"], "user")
            self.assertEqual(payload[0]["conflict"]["decision"], "ADD")


if __name__ == "__main__":
    unittest.main()

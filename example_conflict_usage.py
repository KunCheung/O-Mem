#!/usr/bin/env python
# coding=utf-8

"""
Offline demo for conflict-aware memory ingestion with diverse memory datasets.
This script does not require OpenAI calls and focuses on phase-2/3 logic in
ConflictAwareMetadataProcessor.
"""

import json
import tempfile
from pathlib import Path

import importlib.util


def _load_conflict_processor():
    spec = importlib.util.spec_from_file_location("memory_chain.conflict_memory", "memory_chain/conflict_memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConflictAwareMetadataProcessor


ConflictAwareMetadataProcessor = _load_conflict_processor()


DATASET_FILE = Path("conflict_dataset.json")


def load_datasets():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("datasets", [])


def run_dataset(processor: ConflictAwareMetadataProcessor, dataset: dict):
    print(f"\n=== Dataset: {dataset['name']} ===")
    profiles = {}

    for item in dataset["messages"]:
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

        scope_history = processor.get_versions_by_scope(metadata["which"]["scope"], subject)
        conflict = processor.detect_conflicts(
            metadata,
            event_profile,
            fact_profile,
            attr_profile,
            version_history=scope_history,
        )
        processor.maybe_record_conflict(metadata, conflict)
        processor.register_version(metadata, conflict, item["understanding"]["index"])

        # naive profile update for demo progression
        event_profile[metadata["which"]["scope"]] = f"{metadata['what']['polarity']}"
        fact_profile[item["understanding"]["fact"]] = "present"
        for attr in item["understanding"].get("attribue", []):
            attr_profile[attr] = "present"

        print(
            f"turn={item['understanding']['index']} scope={metadata['which']['scope']} "
            f"decision={conflict['decision']} score={conflict['score']} reason={conflict['reason']}"
        )


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        conflict_path = Path(tmp_dir) / "conflicts.json"
        version_path = Path(tmp_dir) / "versions.json"
        processor = ConflictAwareMetadataProcessor(str(conflict_path), str(version_path))

        datasets = load_datasets()
        for dataset in datasets:
            run_dataset(processor, dataset)

        print("\n=== Retrieval conflict hints ===")
        query = "Do I still like running now?"
        hints = processor.get_conflict_hints_for_query(query, top_k=5)
        print(json.dumps({"query": query, "hints": hints}, ensure_ascii=False, indent=2))

        print("\n=== Ledger summary ===")
        print(f"conflict_events={len(processor.conflict_ledger)}")
        print(f"versions={len(processor.version_ledger)}")


if __name__ == "__main__":
    main()

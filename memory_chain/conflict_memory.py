#!/usr/bin/env python
# coding=utf-8

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


DEFAULT_SIGNAL_RULES = {
    "polarity": {
        "positive": [r"\bpositive\b", r"\blike\b", r"\blove\b", r"\benjoy\b", r"\bsupport\b", r"喜欢", r"热爱"],
        "negative": [r"\bnegative\b", r"\bdislike\b", r"\bhate\b", r"\bavoid\b", r"\breject\b", r"不喜欢", r"讨厌"],
    },
    "time": {
        "relative": [
            r"\btoday\b", r"\byesterday\b", r"\btomorrow\b", r"\blast\s+(day|week|month|year)\b",
            r"\bthis\s+(day|week|month|year)\b", r"\bnext\s+(day|week|month|year)\b", r"\brecently\b",
            r"今天", r"昨天", r"明天", r"上周", r"上个月", r"下周", r"最近",
        ],
        "absolute": [
            r"\b\d{4}-\d{1,2}-\d{1,2}\b", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", r"\b\d{4}年\d{1,2}月\d{1,2}日\b",
        ],
    },
    "location": {
        "named": [
            r"\bat\s+home\b", r"\bin\s+the\s+office\b", r"\bat\s+school\b", r"\bin\s+hospital\b", r"\bonline\b",
            r"在家", r"在公司", r"在学校", r"在医院", r"线上",
        ],
        "contextual": [
            r"\b(in|at)\s+([A-Za-z\u4e00-\u9fff][\w\-\u4e00-\u9fff]{1,30}(?:\s+[A-Za-z\u4e00-\u9fff][\w\-\u4e00-\u9fff]{1,30}){0,3})\b",
            r"在([\u4e00-\u9fffA-Za-z0-9]{2,20})",
        ],
    },
}

_LOCATION_INVALID_TAILS = {
    "general", "coding", "working", "learning", "thinking", "doing", "it", "that", "this"
}


class ConflictAwareMetadataProcessor:
    """Extract 5W metadata, detect conflicts, persist conflict events and version history."""

    def __init__(
        self,
        conflict_ledger_path: Optional[str] = None,
        version_ledger_path: Optional[str] = None,
        signal_rules: Optional[Dict[str, Any]] = None,
    ):
        self.conflict_ledger_path = conflict_ledger_path
        self.version_ledger_path = version_ledger_path
        self.signal_rules = self._merge_signal_rules(signal_rules)
        self._compiled_patterns = self._compile_signal_patterns(self.signal_rules)
        self.conflict_ledger: List[Dict[str, Any]] = []
        self.version_ledger: List[Dict[str, Any]] = []

        if conflict_ledger_path:
            self.conflict_ledger = self._load_json_list(conflict_ledger_path)
        if version_ledger_path:
            self.version_ledger = self._load_json_list(version_ledger_path)

    def build_metadata_envelope(
        self,
        raw_message: str,
        message_understanding: Dict[str, Any],
        timestamp: Any,
        user_speak: bool,
        llm_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        topic = str(message_understanding.get("topics", "")).strip()
        emotion = str(message_understanding.get("emotions", "")).strip()
        fact = str(message_understanding.get("fact", "")).strip()

        when = self._infer_when(raw_message, timestamp)
        where = self._infer_where(raw_message)
        claim_type = self._infer_claim_type(fact, message_understanding.get("attribue", []), emotion)
        proposition = fact if fact else topic

        heuristic = {
            "who": {
                "subject": "user" if user_speak else "agent",
                "entity_id": "self",
                "confidence": 0.95,
            },
            "what": {
                "claim_type": claim_type,
                "proposition": proposition,
                "polarity": self._normalize_polarity(emotion),
                "confidence": 0.8 if proposition else 0.2,
            },
            "which": {
                "scope": topic,
                "evidence_span": [raw_message],
                "source_turn_ids": [message_understanding.get("index")],
            },
            "where": where,
            "when": when,
        }

        return self._merge_llm_envelope(heuristic, llm_fields)


    def _merge_llm_envelope(self, fallback: Dict[str, Any], llm_fields: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not llm_fields or not isinstance(llm_fields, dict):
            return fallback

        merged = json.loads(json.dumps(fallback))
        for section in ["who", "what", "which", "where", "when"]:
            source = llm_fields.get(section)
            if isinstance(source, dict):
                merged.setdefault(section, {}).update({k: v for k, v in source.items() if v not in [None, ""]})
        return merged

    def detect_conflicts(
        self,
        metadata_envelope: Dict[str, Any],
        event_profile: Dict[str, str],
        fact_profile: Dict[str, str],
        attr_profile: Dict[str, str],
        version_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        scope = metadata_envelope["which"].get("scope", "")
        polarity = metadata_envelope["what"].get("polarity", "neutral")
        proposition = metadata_envelope["what"].get("proposition", "")

        dimensions = {
            "who": 0.0,
            "what": self._score_what_conflict(scope, polarity, proposition, event_profile, fact_profile),
            "which": 0.0 if scope else 0.7,
            "where": self._score_where_conflict(metadata_envelope, version_history or []),
            "when": self._score_when_conflict(metadata_envelope, version_history or []),
        }

        if proposition and self._fact_conflicts_with_attributes(proposition, attr_profile):
            dimensions["what"] = max(dimensions["what"], 0.8)

        weighted_score = (
            0.10 * dimensions["who"] +
            0.45 * dimensions["what"] +
            0.10 * dimensions["which"] +
            0.15 * dimensions["where"] +
            0.20 * dimensions["when"]
        )

        boundaries = {
            "who_boundary": metadata_envelope["who"]["confidence"] >= 0.6,
            "what_boundary": bool(metadata_envelope["what"]["proposition"]),
            "which_boundary": bool(scope),
            "where_when_boundary": bool(metadata_envelope["where"].get("location_value")) or metadata_envelope["when"]["time_type"] != "unknown",
            "conflict_boundary": weighted_score < 0.6,
        }

        if not boundaries["who_boundary"] or not boundaries["what_boundary"]:
            decision = "IGNORE"
        elif weighted_score >= 0.6 or dimensions["what"] >= 0.85 or dimensions["when"] >= 0.75:
            decision = "CONFLICT_EVENT"
        elif weighted_score >= 0.3:
            decision = "UPDATE_CANDIDATE"
        else:
            decision = "ADD"

        return {
            "score": round(weighted_score, 4),
            "dimensions": dimensions,
            "decision": decision,
            "boundaries": boundaries,
            "reason": self._build_conflict_reason(dimensions),
        }

    def maybe_record_conflict(self, metadata_envelope: Dict[str, Any], conflict_result: Dict[str, Any]) -> None:
        if conflict_result.get("decision") != "CONFLICT_EVENT":
            return
        conflict_event = {
            "metadata": metadata_envelope,
            "conflict": conflict_result,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        self.conflict_ledger.append(conflict_event)
        self._persist_conflict_ledger()

    def register_version(self, metadata_envelope: Dict[str, Any], conflict_result: Dict[str, Any], message_index: Any) -> Dict[str, Any]:
        scope = metadata_envelope.get("which", {}).get("scope", "")
        proposition = metadata_envelope.get("what", {}).get("proposition", "")
        status = "active"
        if conflict_result.get("decision") == "CONFLICT_EVENT":
            status = "conflicted"

        existing_active_ids = []
        for version in self.version_ledger:
            same_scope = version.get("scope") == scope and scope
            same_subject = version.get("subject") == metadata_envelope.get("who", {}).get("subject")
            if same_scope and same_subject and version.get("status") == "active" and status == "conflicted":
                version["status"] = "superseded_by_conflict"
                existing_active_ids.append(version.get("version_id"))

        version_id = len(self.version_ledger) + 1
        version_entry = {
            "version_id": version_id,
            "subject": metadata_envelope.get("who", {}).get("subject"),
            "scope": scope,
            "proposition": proposition,
            "polarity": metadata_envelope.get("what", {}).get("polarity"),
            "where": metadata_envelope.get("where", {}).get("location_value"),
            "when": metadata_envelope.get("when", {}).get("time_value"),
            "status": status,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message_index": message_index,
            "decision": conflict_result.get("decision"),
            "supersedes": existing_active_ids,
        }
        self.version_ledger.append(version_entry)
        self._persist_version_ledger()
        return version_entry

    def get_versions_by_scope(self, scope: str, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        matched = [v for v in self.version_ledger if v.get("scope") == scope]
        if subject:
            matched = [v for v in matched if v.get("subject") == subject]
        return matched

    def get_conflict_hints_for_query(self, query: str, top_k: int = 3) -> List[str]:
        query_norm = self._normalize_text(query)
        hints: List[str] = []
        for item in reversed(self.conflict_ledger):
            scope = self._normalize_text(item.get("metadata", {}).get("which", {}).get("scope", ""))
            prop = self._normalize_text(item.get("metadata", {}).get("what", {}).get("proposition", ""))
            if not scope and not prop:
                continue
            scope_hit = bool(scope and scope in query_norm)
            prop_hit = bool(prop and any(token and token in query_norm for token in prop.split()[:4]))
            if scope_hit or prop_hit:
                reason = item.get("conflict", {}).get("reason", "存在冲突")
                when_dim = item.get("conflict", {}).get("dimensions", {}).get("when", 0)
                if when_dim >= 0.5:
                    reason = "该偏好存在时间冲突"
                hints.append(f"[冲突提示] {item.get('metadata', {}).get('which', {}).get('scope', '未知主题')}: {reason}")
            if len(hints) >= top_k:
                break
        return hints

    def _persist_conflict_ledger(self) -> None:
        if self.conflict_ledger_path:
            with open(self.conflict_ledger_path, "w", encoding="utf-8") as f:
                json.dump(self.conflict_ledger, f, ensure_ascii=False, indent=2)

    def _persist_version_ledger(self) -> None:
        if self.version_ledger_path:
            with open(self.version_ledger_path, "w", encoding="utf-8") as f:
                json.dump(self.version_ledger, f, ensure_ascii=False, indent=2)

    def _load_json_list(self, path: str) -> List[Dict[str, Any]]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def _infer_when(self, raw_message: str, timestamp: Any) -> Dict[str, Any]:
        relative_match = self._first_pattern_hit(raw_message, self._compiled_patterns["time"]["relative"])
        if relative_match:
            return {"time_type": "relative", "time_value": relative_match.group(0), "recency_score": 0.7}

        absolute_match = self._first_pattern_hit(raw_message, self._compiled_patterns["time"]["absolute"])
        if absolute_match:
            return {"time_type": "absolute", "time_value": absolute_match.group(0), "recency_score": 0.9}

        if timestamp:
            return {"time_type": "point", "time_value": str(timestamp), "recency_score": 1.0}

        return {"time_type": "unknown", "time_value": "unknown", "recency_score": 0.2}

    def _infer_where(self, raw_message: str) -> Dict[str, Any]:
        named_match = self._first_pattern_hit(raw_message, self._compiled_patterns["location"]["named"])
        if named_match:
            return {"location_type": "named", "location_value": named_match.group(0).strip(), "confidence": 0.75}

        for pattern in self._compiled_patterns["location"]["contextual"]:
            match = pattern.search(raw_message)
            if not match:
                continue
            if match.lastindex and match.lastindex >= 2:
                location_value = f"{match.group(1)} {match.group(2)}".strip()
                tail = match.group(2).strip().lower()
            elif match.lastindex and match.lastindex >= 1:
                location_value = match.group(0).strip()
                tail = match.group(1).strip().lower()
            else:
                location_value = match.group(0).strip()
                tail = location_value.lower()
            first_token = tail.split()[0] if tail else ""
            if tail in _LOCATION_INVALID_TAILS or first_token in _LOCATION_INVALID_TAILS:
                continue
            return {"location_type": "contextual", "location_value": location_value, "confidence": 0.6}

        return {"location_type": "unknown", "location_value": "unknown", "confidence": 0.2}

    def _infer_claim_type(self, fact: str, attributes: Any, emotion: str) -> str:
        if fact and any(k in fact.lower() for k in ["will", "plan", "going to", "tomorrow", "next"]):
            return "plan"
        if fact:
            return "fact"
        if attributes:
            return "identity"
        if emotion:
            return "preference"
        return "status"

    def _normalize_polarity(self, emotion: str) -> str:
        if self._first_pattern_hit(emotion, self._compiled_patterns["polarity"]["positive"]):
            return "positive"
        if self._first_pattern_hit(emotion, self._compiled_patterns["polarity"]["negative"]):
            return "negative"
        return "neutral"

    def _score_what_conflict(
        self,
        scope: str,
        polarity: str,
        proposition: str,
        event_profile: Dict[str, str],
        fact_profile: Dict[str, str],
    ) -> float:
        score = 0.0
        if scope in event_profile:
            profile_text = event_profile[scope].lower()
            if polarity == "positive" and "negative" in profile_text:
                score = max(score, 0.9)
            if polarity == "negative" and "positive" in profile_text:
                score = max(score, 0.9)

        proposition_tokens = set(self._normalize_text(proposition).split())
        for fact_key in fact_profile.keys():
            fact_tokens = set(self._normalize_text(fact_key).split())
            if not proposition_tokens or not fact_tokens:
                continue
            overlap = len(proposition_tokens & fact_tokens) / max(1, len(proposition_tokens | fact_tokens))
            if overlap > 0.65 and self._is_negation_mismatch(proposition, fact_key):
                score = max(score, 0.8)
        return score

    def _score_when_conflict(self, metadata_envelope: Dict[str, Any], version_history: List[Dict[str, Any]]) -> float:
        current_time = str(metadata_envelope.get("when", {}).get("time_value", "")).strip().lower()
        if not current_time or current_time == "unknown":
            return 0.0
        for version in reversed(version_history):
            previous_time = str(version.get("when", "")).strip().lower()
            if previous_time and previous_time != "unknown" and previous_time != current_time:
                return 0.8
        return 0.0

    def _score_where_conflict(self, metadata_envelope: Dict[str, Any], version_history: List[Dict[str, Any]]) -> float:
        current_where = str(metadata_envelope.get("where", {}).get("location_value", "")).strip().lower()
        if not current_where or current_where == "unknown":
            return 0.0
        for version in reversed(version_history):
            previous_where = str(version.get("where", "")).strip().lower()
            if previous_where and previous_where != "unknown" and previous_where != current_where:
                return 0.6
        return 0.0

    def _fact_conflicts_with_attributes(self, proposition: str, attr_profile: Dict[str, str]) -> bool:
        proposition_norm = self._normalize_text(proposition)
        for attr_key in attr_profile.keys():
            attr_norm = self._normalize_text(attr_key)
            if not attr_norm:
                continue
            if attr_norm in proposition_norm and self._is_negation_mismatch(proposition_norm, attr_norm):
                return True
        return False

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", str(text).lower())
        return re.sub(r"\s+", " ", text).strip()

    def _is_negation_mismatch(self, left: str, right: str) -> bool:
        neg_words = ["not", "never", "no", "n't", "没有", "不", "别"]
        left_neg = any(word in left.lower() for word in neg_words)
        right_neg = any(word in right.lower() for word in neg_words)
        return left_neg != right_neg

    def _build_conflict_reason(self, dimensions: Dict[str, float]) -> str:
        if dimensions.get("when", 0) >= 0.5:
            return "该偏好存在时间冲突"
        if dimensions.get("where", 0) >= 0.5:
            return "该偏好存在地点冲突"
        if dimensions.get("what", 0) >= 0.5:
            return "该偏好存在语义冲突"
        return "存在潜在冲突"

    def _merge_signal_rules(self, signal_rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = json.loads(json.dumps(DEFAULT_SIGNAL_RULES))
        if not signal_rules:
            return merged
        for level_1, value_1 in signal_rules.items():
            if level_1 not in merged or not isinstance(value_1, dict):
                merged[level_1] = value_1
                continue
            for level_2, value_2 in value_1.items():
                merged[level_1][level_2] = value_2
        return merged

    def _compile_signal_patterns(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "polarity": {
                "positive": [re.compile(p, flags=re.IGNORECASE) for p in rules["polarity"]["positive"]],
                "negative": [re.compile(p, flags=re.IGNORECASE) for p in rules["polarity"]["negative"]],
            },
            "time": {
                "relative": [re.compile(p, flags=re.IGNORECASE) for p in rules["time"]["relative"]],
                "absolute": [re.compile(p, flags=re.IGNORECASE) for p in rules["time"]["absolute"]],
            },
            "location": {
                "named": [re.compile(p, flags=re.IGNORECASE) for p in rules["location"]["named"]],
                "contextual": [re.compile(p, flags=re.IGNORECASE) for p in rules["location"]["contextual"]],
            },
        }

    def _first_pattern_hit(self, text: str, patterns: List[re.Pattern]) -> Optional[re.Match]:
        for pattern in patterns:
            match = pattern.search(str(text))
            if match:
                return match
        return None

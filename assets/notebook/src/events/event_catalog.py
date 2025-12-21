from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EventMeta:
    name: str
    category: str
    start_date: str  # ISO YYYY-MM-DD
    end_date: str    # ISO YYYY-MM-DD
    include_patterns: List[str]
    exclude_patterns: List[str]
    matched_subreddits: List[str]


def load_event_lexicon(path: str | Path) -> Dict[str, Any]:
    """
    Load the JSON produced by scripts/find_related.py into a dict.
    The JSON schema is assumed to match event_related_subreddits_lexicon.json.

    This file is the authoritative event-to-subreddit mapping.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Event lexicon JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def get_event_names(lex: Dict[str, Any]) -> List[str]:
    return sorted(list(lex.keys()))


def get_event_meta(lex: Dict[str, Any], event_name: str) -> EventMeta:
    if event_name not in lex:
        raise KeyError(f"Event '{event_name}' not found. Available: {get_event_names(lex)[:20]} ...")

    obj = lex[event_name]
    lexicon = obj.get("lexicon", {})
    return EventMeta(
        name=event_name,
        category=str(obj.get("category", "")),
        start_date=str(obj.get("start_date", "")),
        end_date=str(obj.get("end_date", "")),
        include_patterns=list(lexicon.get("include_patterns", [])),
        exclude_patterns=list(lexicon.get("exclude_patterns", [])),
        matched_subreddits=list(obj.get("matched_subreddits", [])),
    )


def find_event_by_substring(lex: Dict[str, Any], query: str) -> Optional[str]:
    """
    Convenience helper for notebooks: returns the first event whose name contains `query` (case-insensitive).
    """
    q = query.lower().strip()
    for name in get_event_names(lex):
        if q in name.lower():
            return name
    return None

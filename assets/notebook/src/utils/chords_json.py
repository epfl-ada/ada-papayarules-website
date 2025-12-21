from __future__ import annotations

import json
import re
import unicodedata
import difflib
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ---------- helpers ----------

def slugify_for_chord(title: str) -> str:
    """
    Must match JS slugify exactly.
    """
    s = (title or "").lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"^_+|_+$", "", s)
    s = re.sub(r"_+", "_", s)
    return s


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_event_objects(data: Any) -> List[Dict[str, Any]]:
    """
    Supports:
      - [ {event}, ... ]
      - { "events": [ {event}, ... ] }
      - { "id": {event}, ... }
    """
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("events"), list):
            return [e for e in data["events"] if isinstance(e, dict)]
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())

    raise ValueError("Unsupported event_cards.json structure")


def build_chord_index(chords_dir: Path) -> Dict[str, Tuple[Path, str]]:
    """
    slug -> (path, media_type)
    html preferred over png
    """
    out = {}

    for p in chords_dir.glob("chord_*.png"):
        slug = p.stem.replace("chord_", "", 1)
        out[slug] = (p, "img")

    for p in chords_dir.glob("chord_*.html"):
        slug = p.stem.replace("chord_", "", 1)
        out[slug] = (p, "iframe")

    return out


def ensure_media(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "media" not in ev or not isinstance(ev["media"], list):
        ev["media"] = []
    ev["media"] = [m for m in ev["media"] if isinstance(m, dict)]
    return ev["media"]


def upsert_media(
    ev: Dict[str, Any],
    *,
    src: str,
    media_type: str,
    caption: str,
    insert_first: bool = True,
) -> bool:
    """
    Returns True if modified.
    """
    media = ensure_media(ev)

    for m in media:
        if m.get("src") == src:
            changed = False
            if m.get("type") != media_type:
                m["type"] = media_type
                changed = True
            if caption and m.get("caption") != caption:
                m["caption"] = caption
                changed = True
            return changed

    item = {"type": media_type, "src": src, "caption": caption}
    media.insert(0, item) if insert_first else media.append(item)
    return True

def canonical_key(s: str) -> str:
    """
    Canonicalize strings so event titles and chord filenames match robustly.

    Handles:
      - diacritics: 'Niño' -> 'nino'
      - dashes: '–'/'—' -> '-'
      - '&' -> 'and'
      - parentheses removed
      - any non-alnum -> '_'
      - collapse multiple underscores
    """
    s = (s or "").strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("(", " ").replace(")", " ")

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_chord_index_canonical(chords_dir: Path) -> Dict[str, Tuple[Path, str]]:
    """
    canonical_key -> (path, media_type)
    Prefer html over png when both exist.
    """
    out: Dict[str, Tuple[Path, str]] = {}

    for p in chords_dir.glob("chord_*.png"):
        stem = p.stem[len("chord_"):]  # remove prefix
        key = canonical_key(stem)
        out[key] = (p, "img")

    for p in chords_dir.glob("chord_*.html"):
        stem = p.stem[len("chord_"):]
        key = canonical_key(stem)
        out[key] = (p, "iframe")  # overrides png if present

    return out



def best_close_match(
    query: str,
    choices: List[str],
    *,
    cutoff: float = 0.82,
    require_token_overlap: bool = True,
) -> str | None:
    """
    Return the closest match using difflib, or None if nothing passes cutoff.

    If require_token_overlap is True, we additionally require the query and match
    to share at least one token (len >= 4) to avoid absurd pairings.
    """
    if not query or not choices:
        return None

    hit = difflib.get_close_matches(query, choices, n=1, cutoff=cutoff)
    if not hit:
        return None

    cand = hit[0]
    if require_token_overlap:
        q_tokens = {t for t in query.split("_") if len(t) >= 4}
        c_tokens = {t for t in cand.split("_") if len(t) >= 4}
        if q_tokens and c_tokens and not (q_tokens & c_tokens):
            return None

    return cand


def attach_chords(
    data: Any,
    *,
    chords_dir: Path,
    base_for_src: Path,
    caption: str = "Chord diagram (cross-community linking)",
    title_key: str = "event",
    use_canonical_matching: bool = True,
    fuzzy_fallback: bool = True,
    fuzzy_cutoff: float = 0.82,
    expected_prefix: str = "assets/notebook/img/chord_diagrams/",
    run_sanity_check: bool = False,
) -> Dict[str, Any]:
    """
    Mutates data in-place by upserting ev["media"] entries for chord diagrams.

    Matching:
      - If use_canonical_matching=True (default): matches by canonical_key(event_title)
        to canonical_key(chord filename stem without 'chord_').
      - Else: strict JS-style slugify_for_chord() match (legacy behavior).

    Robustness:
      - If no exact match and fuzzy_fallback=True, picks the closest chord key
        (above fuzzy_cutoff) with an optional token-overlap safety check.

    Paths:
      - chord `src` is stored relative to base_for_src when possible.
      - chords_dir and base_for_src are resolved to absolute paths to avoid
        relative-path edge cases.

    Returns a stats dict. If run_sanity_check=True, validates that chord src paths
    start with expected_prefix and do not contain "..".
    """
    chords_dir = Path(chords_dir).resolve()
    base_for_src = Path(base_for_src).resolve()

    # Normalize any previously-written bad src paths (optional hygiene).
    for ev in iter_event_objects(data):
        media = ev.get("media")
        if not isinstance(media, list):
            continue
        for m in media:
            if not isinstance(m, dict):
                continue
            src = m.get("src")
            if not isinstance(src, str):
                continue
            if src.startswith("img/chord_diagrams/"):
                m["src"] = "assets/notebook/" + src
            elif src.startswith("notebook/img/chord_diagrams/"):
                m["src"] = "assets/" + src

    events = iter_event_objects(data)

    if use_canonical_matching:
        chord_index = build_chord_index_canonical(chords_dir)
    else:
        chord_index = build_chord_index(chords_dir)

    chord_keys = list(chord_index.keys())

    stats: Dict[str, Any] = dict(
        events_total=len(events),
        matched=0,
        matched_fuzzy=0,
        modified=0,
        missing_titles=0,
        no_chord_file=0,
        unmatched_events=[],
        fuzzy_pairs=[],
    )

    for ev in events:
        title = ev.get(title_key)
        if not isinstance(title, str) or not title.strip():
            stats["missing_titles"] += 1
            continue

        if use_canonical_matching:
            key = canonical_key(title)
        else:
            key = slugify_for_chord(title)

        hit = chord_index.get(key)

        # Fuzzy fallback when no exact hit
        if not hit and fuzzy_fallback:
            close_key = best_close_match(key, chord_keys, cutoff=fuzzy_cutoff, require_token_overlap=True)
            if close_key is not None:
                maybe_hit = chord_index.get(close_key)
                if maybe_hit:
                    hit = maybe_hit
                    stats["matched_fuzzy"] += 1
                    stats["fuzzy_pairs"].append(
                        {"event": title, "event_key": key, "chord_key": close_key}
                    )

        if not hit:
            stats["no_chord_file"] += 1
            stats["unmatched_events"].append(title)
            continue

        path, media_type = hit
        stats["matched"] += 1

        # Store src relative to base_for_src when possible
        try:
            src = Path(path).resolve().relative_to(base_for_src).as_posix()
        except Exception:
            # Fall back to a POSIX-ish string; prefer keeping forward slashes.
            src = Path(path).as_posix()

        if upsert_media(ev, src=src, media_type=media_type, caption=caption):
            stats["modified"] += 1

    if run_sanity_check:
        bad: List[str] = []
        for ev in iter_event_objects(data):
            media = ev.get("media")
            if not isinstance(media, list):
                continue
            for m in media:
                if not isinstance(m, dict):
                    continue
                src = m.get("src")
                if not isinstance(src, str):
                    continue
                if "chord_" not in src:
                    continue
                if expected_prefix and not src.startswith(expected_prefix):
                    bad.append(src)
                if ".." in src:
                    bad.append(src)

        if bad:
            raise AssertionError(
                "Invalid chord media paths detected:\n"
                + "\n".join(f"  - {p}" for p in sorted(set(bad)))
            )

    return stats


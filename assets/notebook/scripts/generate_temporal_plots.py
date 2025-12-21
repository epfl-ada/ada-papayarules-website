"""
Generate interactive HTML temporal comparison plots for all events in the lexicon,
and attach the generated plot paths into assets/data/event_cards.json.

- Reads events from JSON lexicon
- Generates plotly HTML files into assets/notebook/img/temporal_events/
- Updates assets/data/event_cards.json by adding an iframe media entry per event
  (robust matching using canonicalized event names)

Run from: assets/notebook/ (recommended)
"""

from __future__ import annotations

import json
import re
import sys
import shutil
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]  # .../assets/notebook
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))



# ---------------------------
# Robust name / path utilities
# ---------------------------

import difflib

def best_close_match(query: str, choices: list[str], cutoff: float = 0.82) -> str | None:
    hit = difflib.get_close_matches(query, choices, n=1, cutoff=cutoff)
    return hit[0] if hit else None


def build_html_index(output_dir: Path, prefix: str) -> dict[str, Path]:
    """
    Build canonical_key -> html path for files like {prefix}_{something}.html
    """
    out: dict[str, Path] = {}
    for p in output_dir.glob(f"{prefix}_*.html"):
        stem = p.stem[len(prefix) + 1 :]  # strip "prefix_"
        out[canonical_key(stem)] = p.resolve()
    return out


def attach_temporal_html_to_event_cards(
    *,
    event_cards_path: Path,
    html_index: dict[str, Path],
    repo_root: Path,
    caption: str = "Temporal activity (event-aligned)",
    title_key: str = "event",
    fuzzy_fallback: bool = True,
    fuzzy_cutoff: float = 0.82,
) -> dict[str, Any]:
    data = load_json(event_cards_path)
    events = iter_event_objects(data)

    # index cards by canonical event key
    card_by_key: dict[str, dict[str, Any]] = {}
    for ev in events:
        t = ev.get(title_key)
        if isinstance(t, str) and t.strip():
            card_by_key[canonical_key(t)] = ev

    html_keys = list(html_index.keys())

    stats: dict[str, Any] = dict(
        events_total=len(events),
        matched=0,
        matched_fuzzy=0,
        modified=0,
        missing_titles=0,
        no_plot_file=0,
        unmatched_events=[],
        fuzzy_pairs=[],
    )

    for ev in events:
        title = ev.get(title_key)
        if not isinstance(title, str) or not title.strip():
            stats["missing_titles"] += 1
            continue

        k = canonical_key(title)
        plot_path = html_index.get(k)

        if plot_path is None and fuzzy_fallback:
            close_k = best_close_match(k, html_keys, cutoff=fuzzy_cutoff)
            if close_k is not None:
                plot_path = html_index.get(close_k)
                if plot_path is not None:
                    stats["matched_fuzzy"] += 1
                    stats["fuzzy_pairs"].append({"event": title, "event_key": k, "plot_key": close_k})

        if plot_path is None:
            stats["no_plot_file"] += 1
            stats["unmatched_events"].append(title)
            continue

        # src must be site-stable (keep 'assets/..' prefix)
        src = plot_path.relative_to(repo_root).as_posix()

        changed = upsert_media(
            ev,
            src=src,
            media_type="iframe",
            caption=caption,
            insert_first=False,   # put after chord, or switch to True if you want it first
        )
        stats["matched"] += 1
        if changed:
            stats["modified"] += 1

    backup = event_cards_path.with_suffix(event_cards_path.suffix + ".bak")
    shutil.copy2(event_cards_path, backup)
    save_json(event_cards_path, data)

    return stats


def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Walk upwards until we find a directory that contains 'assets/'.
    """
    p = (start or Path.cwd()).resolve()
    while p != p.parent and not (p / "assets").exists():
        p = p.parent
    if not (p / "assets").exists():
        raise RuntimeError("Could not find repo root containing 'assets/'")
    return p

def load_hyperlink_tsvs(notebook_root: Path) -> pd.DataFrame:
    data_dir = notebook_root / "data"
    body_file = data_dir / "soc-redditHyperlinks-body.tsv"
    title_file = data_dir / "soc-redditHyperlinks-title.tsv"

    if not body_file.exists():
        raise FileNotFoundError(f"Missing body TSV: {body_file}")
    if not title_file.exists():
        raise FileNotFoundError(f"Missing title TSV: {title_file}")

    body = pd.read_csv(body_file, sep="\t")
    title = pd.read_csv(title_file, sep="\t")
    df = pd.concat([body, title], ignore_index=True)

    if "TIMESTAMP" not in df.columns:
        raise KeyError(f"TIMESTAMP column not found in TSVs. Columns: {list(df.columns)}")

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    return df



def canonical_key(s: str) -> str:
    """
    Canonicalize strings so event titles and filenames match robustly.

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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
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


def ensure_media_list(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "media" not in ev or not isinstance(ev["media"], list):
        ev["media"] = []
    ev["media"] = [m for m in ev["media"] if isinstance(m, dict)]
    return ev["media"]


def upsert_media(ev: Dict[str, Any], *, src: str, media_type: str, caption: str, insert_first: bool = False) -> bool:
    """
    Upsert a media dict by src; returns True if modified.
    """
    media = ensure_media_list(ev)
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
    if insert_first:
        media.insert(0, item)
    else:
        media.append(item)
    return True


# ---------------------------
# Temporal computation + plot
# ---------------------------

def compute_event_aligned_activity_from_json(
    df: pd.DataFrame,
    event_name: str,
    event_data: dict,
    days_before: int = 30,
    days_after: int = 30
) -> pd.DataFrame:
    """
    Align activity around a specific event date using lexicon JSON data.
    """
    start_date = event_data["start_date"]
    event_date = pd.to_datetime(start_date)
    subs = event_data.get("matched_subreddits", [])
    event_set = set(s.lower() for s in subs)

    window_start = event_date - timedelta(days=days_before)
    window_end = event_date + timedelta(days=days_after)

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["TIMESTAMP"]):
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    window_df = df[(df["TIMESTAMP"] >= window_start) & (df["TIMESTAMP"] <= window_end)].copy()
    if len(window_df) == 0:
        return pd.DataFrame()

    window_df["date"] = window_df["TIMESTAMP"].dt.date
    window_df["event_related"] = (
        window_df["SOURCE_SUBREDDIT"].str.lower().isin(event_set) |
        window_df["TARGET_SUBREDDIT"].str.lower().isin(event_set)
    )

    daily = window_df.groupby(["date", "event_related"]).agg({
        "SOURCE_SUBREDDIT": "count",
        "LINK_SENTIMENT": "mean"
    }).reset_index()
    daily.columns = ["date", "event_related", "count", "avg_sentiment"]
    daily["days_from_event"] = (pd.to_datetime(daily["date"]) - event_date).dt.days
    return daily


def plot_temporal_event_html(
    df: pd.DataFrame,
    event_name: str,
    event_data: dict,
    output_path: Path,
    days_before: int = 30,
    days_after: int = 30
) -> bool:
    """
    Create an interactive Plotly HTML plot for a single event's temporal pattern.
    """
    daily = compute_event_aligned_activity_from_json(
        df, event_name, event_data, days_before, days_after
    )

    if len(daily) == 0:
        print(f"  No data for {event_name}, skipping...")
        return False

    event_daily = daily[daily["event_related"] == True].copy()
    if len(event_daily) == 0:
        print(f"  No event-related activity for {event_name}, skipping...")
        return False

    event_daily = event_daily.sort_values("days_from_event")

    category = event_data.get("category", "Unknown")
    start_date = event_data.get("start_date", "")
    end_date = event_data.get("end_date", "")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=event_daily["days_from_event"],
        y=event_daily["count"],
        mode="lines",
        name="Daily Link Count",
        line=dict(color="steelblue", width=2),
        fill="tozeroy",
        fillcolor="rgba(70, 130, 180, 0.3)",
        hovertemplate="Day %{x}: %{y:,.0f} links<extra></extra>"
    ))

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Event Start",
        annotation_position="top"
    )

    fig.update_layout(
        title=dict(
            text=f"{event_name}<br><sup>{category} | {start_date} to {end_date}</sup>",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Days from Event Start",
        yaxis_title="Daily Link Count",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return True


# ---------------------------
# Attach plots to event_cards
# ---------------------------

def attach_temporal_plots_to_event_cards(
    *,
    event_cards_path: Path,
    generated: Dict[str, Path],
    repo_root: Path,
    caption: str = "Temporal activity (event-aligned)",
) -> Dict[str, Any]:
    """
    generated: canonical_event_key -> absolute path to HTML plot
    Writes 'assets/notebook/...' src into media.
    """
    data = load_json(event_cards_path)
    events = iter_event_objects(data)

    # Build index for event_cards by canonical key of ev["event"]
    card_by_key: Dict[str, Dict[str, Any]] = {}
    missing_event_field = 0
    for ev in events:
        name = ev.get("event")
        if not isinstance(name, str) or not name.strip():
            missing_event_field += 1
            continue
        card_by_key[canonical_key(name)] = ev

    stats: Dict[str, Any] = dict(
        event_cards_total=len(events),
        event_cards_missing_event_field=missing_event_field,
        generated_plots=len(generated),
        matched_cards=0,
        modified_cards=0,
        unmatched_plot_keys=[],
    )

    for k, plot_path in generated.items():
        ev = card_by_key.get(k)
        if ev is None:
            stats["unmatched_plot_keys"].append(k)
            continue

        try:
            src = plot_path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            src = plot_path.resolve().as_posix()

        # We want src like: assets/notebook/img/temporal_events/temporal_<...>.html
        if not src.startswith("assets/"):
            # if user ran from an odd place, still force a stable URL if possible
            # (this is conservative; the relative_to above should normally yield assets/..)
            pass

        changed = upsert_media(ev, src=src, media_type="iframe", caption=caption, insert_first=False)
        stats["matched_cards"] += 1
        if changed:
            stats["modified_cards"] += 1

    # Backup + save
    backup = event_cards_path.with_suffix(event_cards_path.suffix + ".bak")
    shutil.copy2(event_cards_path, backup)
    save_json(event_cards_path, data)
    return stats


def main():
    repo_root = find_repo_root()
    # Script lives in assets/notebook/scripts; we use repo_root paths to be robust.

    # Load event lexicon from JSON
    lexicon_path = repo_root / "assets" / "notebook" / "data" / "event_related_subreddits_lexicon.json"
    if not lexicon_path.exists():
        # fallback to old relative path if needed
        lexicon_path = Path("data/event_related_subreddits_lexicon.json")

    with open(lexicon_path, "r", encoding="utf-8") as f:
        events_lexicon = json.load(f)

    print(f"Found {len(events_lexicon)} events in JSON lexicon")

    # Load data - body + title
    print("Loading Reddit data (body + title) from assets/notebook/data ...")
    df = load_hyperlink_tsvs(NOTEBOOK_ROOT)
    print(f"Loaded {len(df):,} total links")


    # Output directory (must be under assets/notebook so the website can serve it)
    output_dir = repo_root / "assets" / "notebook" / "img" / "temporal_events"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots for each event, track what we generated (canonical_key -> path)
    generated: Dict[str, Path] = {}
    success_count = 0

    for event_name, event_data in events_lexicon.items():
        safe = canonical_key(event_name)
        output_path = output_dir / f"temporal_{safe}.html"

        print(f"Processing: {event_name}...")

        if output_path.exists():
            print(f"  [skip] already exists: {output_path.name}")
            success_count += 1
            generated[canonical_key(event_name)] = output_path.resolve()
            continue

        if plot_temporal_event_html(df, event_name, event_data, output_path):
            print(f"  Saved to {output_path}")
            success_count += 1
            generated[canonical_key(event_name)] = output_path.resolve()

    print(f"\nDone! Generated {success_count}/{len(events_lexicon)} plots in {output_dir}")

    # Build index of generated HTML plots and attach to event_cards.json
    html_index = build_html_index(output_dir, prefix="temporal")

    event_cards_path = repo_root / "assets" / "data" / "event_cards.json"
    print("\nAttaching temporal plots to assets/data/event_cards.json ...")

    attach_stats = attach_temporal_html_to_event_cards(
        event_cards_path=event_cards_path,
        html_index=html_index,
        repo_root=repo_root,
        caption="Temporal activity (event-aligned)",
        title_key="event",
        fuzzy_fallback=True,
        fuzzy_cutoff=0.82,
    )

    print("Attach stats:", attach_stats)



if __name__ == "__main__":
    main()

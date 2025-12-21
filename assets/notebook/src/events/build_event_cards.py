from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.events.event_catalog import EventMeta, get_event_meta, get_event_names, load_event_lexicon
from src.events.event_filtering import filter_links_by_event, split_event_vs_non_event
from src.events.event_metrics import (
    dominant_sentiment,
    key_subreddits,
    propagation_components,
    propagation_score_from_components,
)
from src.utils.data_utils import get_links_within_period


def _pad_window(start_iso: str, end_iso: str, pre_days: int, post_days: int) -> Tuple[str, str]:
    start = pd.to_datetime(start_iso) - pd.Timedelta(days=pre_days)
    end = pd.to_datetime(end_iso) + pd.Timedelta(days=post_days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def build_all_event_cards(
    body_df: pd.DataFrame,
    lex_json_path: str | Path,
    all_subreddits: Optional[set[str]] = None,
    pre_days: int = 7,
    post_days: int = 14,
    mode: str = "either",
    top_k_subs: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build one card per event, including:
      - propagation components + z-scored propagation score
      - dominant sentiment (label + ratios)
      - key subreddits list

    Notes:
      - We compute propagation z-scores across events (global normalization)
      - We pad the event window by (pre_days, post_days) for robustness
    """
    lex = load_event_lexicon(lex_json_path)
    event_names = get_event_names(lex)

    cards: List[Dict[str, Any]] = []
    comps_list: List[Dict[str, Any]] = []

    # First pass: compute components + sentiment + key subs (no z-score yet)
    for name in event_names:
        meta: EventMeta = get_event_meta(lex, name)
        event_subs = set(meta.matched_subreddits)

        # padded analysis window
        w_start, w_end = _pad_window(meta.start_date, meta.end_date, pre_days, post_days)
        df_window = get_links_within_period(body_df, w_start, w_end)

        df_event, _ = split_event_vs_non_event(df_window, event_subs, mode=mode)
        df_cross = filter_links_by_event(df_window, event_subs, mode="cross")

        sent = dominant_sentiment(df_event)
        keys = key_subreddits(df_event, df_cross=df_cross, top_k=top_k_subs)
        comps = propagation_components(df_window, df_event, df_cross, all_subreddits=all_subreddits)

        card = {
            "event": meta.name,
            "category": meta.category,
            "start_date": meta.start_date,
            "end_date": meta.end_date,
            "window_start": w_start,
            "window_end": w_end,
            "matched_subreddits_count": len(event_subs),
            "Dominant Sentiment": sent["label"],
            "Key Subreddits": [k["subreddit"] for k in keys],
            "debug": {
                "sentiment": sent,
                "key_subreddits": keys,
                "propagation_components": comps,
            },
        }

        cards.append(card)
        comps_list.append(comps)

    # Second pass: compute global z-score stats and add Propagation Score
    reach_vals = np.array([float(c["reach"]) for c in comps_list], dtype=float)
    boundary_vals = np.array([float(c["boundary_ratio"]) for c in comps_list], dtype=float)
    spill_vals = np.array([float(c["spillover_ratio"]) for c in comps_list], dtype=float)

    reach_mean, reach_std = float(reach_vals.mean()), float(reach_vals.std(ddof=0))
    boundary_mean, boundary_std = float(boundary_vals.mean()), float(boundary_vals.std(ddof=0))
    spill_mean, spill_std = float(spill_vals.mean()), float(spill_vals.std(ddof=0))

    for card in cards:
        comps = card["debug"]["propagation_components"]
        score = propagation_score_from_components(
            comps,
            reach_mean=reach_mean,
            reach_std=reach_std,
            boundary_mean=boundary_mean,
            boundary_std=boundary_std,
            spill_mean=spill_mean,
            spill_std=spill_std,
            use_spillover=True,
        )
        card["Propagation Score"] = score

    return cards


def save_cards(cards: List[Dict[str, Any]], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cards, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def label_from_mean_sentiment(mean_val: float, pos_thr: float = 0.05, neg_thr: float = -0.05) -> str:
    """
    Conservative binning of mean LINK_SENTIMENT into a coarse label.
    """
    if mean_val > pos_thr:
        return "Positive"
    if mean_val < neg_thr:
        return "Negative"
    return "Neutral"


def dominant_sentiment(df_event: pd.DataFrame) -> Dict[str, float | str | int]:
    """
    Compute event-level sentiment from LINK_SENTIMENT.
    LINK_SENTIMENT is assumed to be in {-1, 0, +1} or {-1, +1}.
    """
    _require_cols(df_event, ["LINK_SENTIMENT"])
    n = int(len(df_event))
    if n == 0:
        return {
            "n_links": 0,
            "mean_link_sentiment": 0.0,
            "pos_ratio": 0.0,
            "neg_ratio": 0.0,
            "label": "Neutral",
        }

    s = df_event["LINK_SENTIMENT"].astype(float)
    mean_val = float(s.mean())

    pos = int((s == 1).sum())
    neg = int((s == -1).sum())
    denom = pos + neg

    pos_ratio = _safe_div(pos, denom)
    neg_ratio = _safe_div(neg, denom)

    return {
        "n_links": n,
        "mean_link_sentiment": mean_val,
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
        "label": label_from_mean_sentiment(mean_val),
    }


def _zscore(x: float, mean: float, std: float) -> float:
    if std <= 1e-12:
        return 0.0
    return float((x - mean) / std)


def key_subreddits(
    df_event: pd.DataFrame,
    df_cross: Optional[pd.DataFrame] = None,
    top_k: int = 5,
    source_col: str = "SOURCE_SUBREDDIT",
    target_col: str = "TARGET_SUBREDDIT",
) -> List[Dict[str, float | str | int]]:
    """
    Rank key subreddits structurally, not by raw volume alone.

    score(sub) = z(in_deg) + z(out_deg) + z(boundary_deg)

    boundary_deg comes from df_cross if provided; otherwise 0.
    """
    _require_cols(df_event, [source_col, target_col])

    # Degrees inside event graph
    out_deg = df_event[source_col].value_counts()
    in_deg = df_event[target_col].value_counts()

    # Boundary involvement
    boundary_deg = None
    if df_cross is not None and len(df_cross) > 0:
        _require_cols(df_cross, [source_col, target_col])
        boundary_deg = pd.concat([df_cross[source_col], df_cross[target_col]]).value_counts()
    else:
        boundary_deg = pd.Series(dtype=int)

    subs = sorted(set(out_deg.index).union(in_deg.index).union(boundary_deg.index))

    rows = []
    for s in subs:
        rows.append(
            {
                "subreddit": s,
                "in_deg": int(in_deg.get(s, 0)),
                "out_deg": int(out_deg.get(s, 0)),
                "boundary_deg": int(boundary_deg.get(s, 0)),
            }
        )

    df_rank = pd.DataFrame(rows)
    if df_rank.empty:
        return []

    # z-score normalize within this event to avoid scale issues
    for col in ["in_deg", "out_deg", "boundary_deg"]:
        mu = float(df_rank[col].mean())
        sd = float(df_rank[col].std(ddof=0))
        df_rank[col + "_z"] = df_rank[col].apply(lambda v: _zscore(float(v), mu, sd))

    df_rank["score"] = df_rank["in_deg_z"] + df_rank["out_deg_z"] + df_rank["boundary_deg_z"]

    df_rank = df_rank.sort_values(["score", "boundary_deg", "out_deg", "in_deg"], ascending=False)

    out = []
    for _, r in df_rank.head(top_k).iterrows():
        out.append(
            {
                "subreddit": str(r["subreddit"]),
                "score": float(r["score"]),
                "in_deg": int(r["in_deg"]),
                "out_deg": int(r["out_deg"]),
                "boundary_deg": int(r["boundary_deg"]),
            }
        )
    return out


def propagation_components(
    df_window: pd.DataFrame,
    df_event: pd.DataFrame,
    df_cross: pd.DataFrame,
    all_subreddits: Optional[set[str]] = None,
    source_col: str = "SOURCE_SUBREDDIT",
    target_col: str = "TARGET_SUBREDDIT",
) -> Dict[str, float | int]:
    """
    Compute interpretable propagation components, all within an event-aligned time window:

    - reach: |V_event| / |V_total|  (total can be all_subreddits if provided; else V_window)
    - boundary_ratio: |E_cross| / |E_event|
    - spillover_ratio: |E_cross| / |E_window|

    These are diffusion proxies, not "influence".
    """
    _require_cols(df_window, [source_col, target_col])
    _require_cols(df_event, [source_col, target_col])
    _require_cols(df_cross, [source_col, target_col])

    V_event = set(df_event[source_col]).union(set(df_event[target_col]))
    V_window = set(df_window[source_col]).union(set(df_window[target_col]))

    V_total = all_subreddits if all_subreddits is not None and len(all_subreddits) > 0 else V_window

    E_event = int(len(df_event))
    E_cross = int(len(df_cross))
    E_window = int(len(df_window))

    reach = _safe_div(len(V_event), len(V_total))
    boundary_ratio = _safe_div(E_cross, E_event)
    spillover_ratio = _safe_div(E_cross, E_window)

    return {
        "V_event": int(len(V_event)),
        "V_total": int(len(V_total)),
        "E_event": E_event,
        "E_cross": E_cross,
        "E_window": E_window,
        "reach": float(reach),
        "boundary_ratio": float(boundary_ratio),
        "spillover_ratio": float(spillover_ratio),
    }


def propagation_score_from_components(
    comps: Dict[str, float | int],
    reach_mean: float,
    reach_std: float,
    boundary_mean: float,
    boundary_std: float,
    spill_mean: Optional[float] = None,
    spill_std: Optional[float] = None,
    use_spillover: bool = True,
) -> float:
    """
    Convert components to a single scalar score using z-scores across events.
    This should be computed AFTER you compute comps for all events (global normalization).
    """
    z_reach = _zscore(float(comps["reach"]), reach_mean, reach_std)
    z_boundary = _zscore(float(comps["boundary_ratio"]), boundary_mean, boundary_std)

    if use_spillover and spill_mean is not None and spill_std is not None:
        z_spill = _zscore(float(comps["spillover_ratio"]), spill_mean, spill_std)
        return float(z_reach + z_boundary + z_spill)

    return float(z_reach + z_boundary)

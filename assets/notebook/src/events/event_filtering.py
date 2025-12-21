from __future__ import annotations

from typing import Iterable, Literal, Tuple

import pandas as pd

Mode = Literal["either", "both", "cross"]


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")


def filter_links_by_event(
    df: pd.DataFrame,
    event_subs: set[str],
    mode: Mode = "either",
    source_col: str = "SOURCE_SUBREDDIT",
    target_col: str = "TARGET_SUBREDDIT",
) -> pd.DataFrame:
    """
    Filter hyperlink rows based on whether endpoints are in the event-related subreddit set.

    mode:
      - 'either': source OR target in event_subs (broadest, recommended default)
      - 'both'  : source AND target in event_subs (strict core)
      - 'cross' : exactly one endpoint in event_subs (boundary links)
    """
    _require_cols(df, [source_col, target_col])

    src_in = df[source_col].isin(event_subs)
    tgt_in = df[target_col].isin(event_subs)

    if mode == "either":
        mask = src_in | tgt_in
    elif mode == "both":
        mask = src_in & tgt_in
    elif mode == "cross":
        mask = src_in ^ tgt_in
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df.loc[mask].copy()


def split_event_vs_non_event(
    df: pd.DataFrame,
    event_subs: set[str],
    mode: Mode = "either",
    source_col: str = "SOURCE_SUBREDDIT",
    target_col: str = "TARGET_SUBREDDIT",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into (event_related, non_event_related) according to `mode`.
    Non-event is simply the complement within df.
    """
    df_event = filter_links_by_event(df, event_subs, mode=mode, source_col=source_col, target_col=target_col)
    df_non = df.drop(index=df_event.index).copy()
    return df_event, df_non

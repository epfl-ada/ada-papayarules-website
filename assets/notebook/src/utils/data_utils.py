import pandas as pd
import numpy as np
from datetime import datetime

from src.data.sentiment_analysis import categorization, sentiment_selection

def _normalize_timestamp_series(ts: pd.Series) -> pd.Series:
    """Return tz-naive UTC timestamps (datetime64[ns]) for safe comparisons/plotting.

    - If `ts` is tz-aware: convert to UTC, then drop tz (tz-naive).
    - If `ts` is tz-naive: keep as tz-naive.
    - Coerces errors to NaT.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        # tz-aware?
        if getattr(ts.dt, "tz", None) is not None:
            return ts.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        # In case `.dt` accessor fails (e.g. object dtype)
        pass
    return ts

def ensure_timestamp_utc_naive(df: pd.DataFrame, column: str = "TIMESTAMP") -> pd.DataFrame:
    """Ensure df[column] is a tz-naive UTC datetime64[ns] series (in-place copy-safe)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
    out = df.copy()
    out[column] = _normalize_timestamp_series(out[column])
    return out


def get_links_within_period(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    column: str = "TIMESTAMP",
    by_day: bool = True,
) -> pd.DataFrame:
    """
    Filter rows whose timestamp column falls within [start_date, end_date].

    Robust to tz-naive / tz-aware mismatches:
    - df[column] is normalized to tz-naive UTC.
    - start_date/end_date are parsed and normalized to tz-naive UTC.

    If by_day=True and start/end have no time parts, treat them as whole days
    by using < end_date + 1 day.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    ts = _normalize_timestamp_series(df[column])

    # Parse start/end; treat them as UTC; then drop tz for alignment with `ts`.
    start_dt = pd.to_datetime(start_date, utc=True).tz_localize(None)
    end_dt = pd.to_datetime(end_date, utc=True).tz_localize(None)

    if by_day and start_dt.time() == datetime.min.time() and end_dt.time() == datetime.min.time():
        end_dt = end_dt + pd.Timedelta(days=1)
        mask = (ts >= start_dt) & (ts < end_dt)
    else:
        mask = ts.between(start_dt, end_dt, inclusive="both")

    return df.loc[mask].copy()

def parse_props(s):
    """Parse a string representation of a list/array into a numpy array."""
    if pd.isna(s):
        return np.array([])
    return np.fromstring(str(s).strip('[]() '), sep=',')

def properties_into_dataframe_columns(df, col_name='PROPERTIES'):
    props_df = df[col_name].apply(parse_props)

    props_cols = [
    "n_chars","n_chars_no_ws","frac_alpha","frac_digits","frac_upper","frac_ws","frac_special",
    "n_words","n_unique_words","n_long_words","avg_word_len","n_unique_stopwords","frac_stopwords",
    "n_sentences","n_long_sentences","avg_chars_per_sentence","avg_words_per_sentence","automated_readability_index",
    "vader_pos","vader_neg","vader_compound",
    "LIWC_Funct","LIWC_Pronoun","LIWC_Ppron","LIWC_I","LIWC_We","LIWC_You","LIWC_SheHe","LIWC_They","LIWC_Ipron",
    "LIWC_Article","LIWC_Verbs","LIWC_AuxVb","LIWC_Past","LIWC_Present","LIWC_Future","LIWC_Adverbs","LIWC_Prep",
    "LIWC_Conj","LIWC_Negate","LIWC_Quant","LIWC_Numbers","LIWC_Swear","LIWC_Social","LIWC_Family","LIWC_Friends",
    "LIWC_Humans","LIWC_Affect","LIWC_Posemo","LIWC_Negemo","LIWC_Anx","LIWC_Anger","LIWC_Sad","LIWC_CogMech",
    "LIWC_Insight","LIWC_Cause","LIWC_Discrep","LIWC_Tentat","LIWC_Certain","LIWC_Inhib","LIWC_Incl","LIWC_Excl",
    "LIWC_Percept","LIWC_See","LIWC_Hear","LIWC_Feel","LIWC_Bio","LIWC_Body","LIWC_Health","LIWC_Sexual","LIWC_Ingest",
    "LIWC_Relativ","LIWC_Motion","LIWC_Space","LIWC_Time","LIWC_Work","LIWC_Achiev","LIWC_Leisure","LIWC_Home",
    "LIWC_Money","LIWC_Relig","LIWC_Death","LIWC_Assent","LIWC_Dissent","LIWC_Nonflu","LIWC_Filler"
    ]

    valid_props_df = props_df.apply(lambda a: a.size == len(props_cols))

    props_mat_df = np.vstack(props_df[valid_props_df].values) if valid_props_df.any() else np.empty((0, len(props_cols)))

    props_df_combined = pd.DataFrame(props_mat_df, columns=props_cols, index=df[valid_props_df].index)

    df_result = df.drop(columns=[col_name]).join(props_df_combined)

    return df_result


LIWC_COLS = [
"LIWC_Funct","LIWC_Pronoun","LIWC_Ppron","LIWC_I","LIWC_We","LIWC_You","LIWC_SheHe","LIWC_They","LIWC_Ipron",
"LIWC_Article","LIWC_Verbs","LIWC_AuxVb","LIWC_Past","LIWC_Present","LIWC_Future","LIWC_Adverbs","LIWC_Prep",
"LIWC_Conj","LIWC_Negate","LIWC_Quant","LIWC_Numbers","LIWC_Swear","LIWC_Social","LIWC_Family","LIWC_Friends",
"LIWC_Humans","LIWC_Affect","LIWC_Posemo","LIWC_Negemo","LIWC_Anx","LIWC_Anger","LIWC_Sad","LIWC_CogMech",
"LIWC_Insight","LIWC_Cause","LIWC_Discrep","LIWC_Tentat","LIWC_Certain","LIWC_Inhib","LIWC_Incl","LIWC_Excl",
"LIWC_Percept","LIWC_See","LIWC_Hear","LIWC_Feel","LIWC_Bio","LIWC_Body","LIWC_Health","LIWC_Sexual","LIWC_Ingest",
"LIWC_Relativ","LIWC_Motion","LIWC_Space","LIWC_Time","LIWC_Work","LIWC_Achiev","LIWC_Leisure","LIWC_Home",
"LIWC_Money","LIWC_Relig","LIWC_Death","LIWC_Assent","LIWC_Dissent","LIWC_Nonflu","LIWC_Filler"
]

def df_liwc_ranking(df, sentiment_value=0, ascending=False):
    df_LIWC = sentiment_selection(df, sentiment_value)[LIWC_COLS]
    df_liwc_ranking = df_LIWC.mean(numeric_only=True).sort_values(ascending=ascending)
    return df_liwc_ranking

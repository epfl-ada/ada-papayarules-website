
import pandas as pd
import numpy as np

def compute_core_vs_expanded_metrics(body_df: pd.DataFrame, expanded_event_periods: dict) -> pd.DataFrame:
    """
    Computes volume and sentiment metrics for core (seed) vs. expanded (added) subreddits across events.
    
    Args:
        body_df (pd.DataFrame): The dataframe containing subreddit interaction data (must have TIMESTAMP, SOURCE_SUBREDDIT, TARGET_SUBREDDIT, LINK_SENTIMENT).
        expanded_event_periods (dict): Dictionary mapping event names to tuples: (start_date, end_date, subreddit_dict).
                                     subreddit_dict should have keys 'seed' and 'added' (and optionally 'all').
    
    Returns:
        pd.DataFrame: DataFrame containing metrics for each event.
    """
    results = []
    
    for event_name, (start, end, sub_info) in expanded_event_periods.items():
        # Ensure timestamp column is datetime if not already (assuming processed externally or checked)
        # Filter to event period
        mask = (body_df["TIMESTAMP"] >= start) & (body_df["TIMESTAMP"] <= end)
        event_df = body_df[mask]
        
        if len(event_df) == 0:
            continue
            
        seed_subs = set(s.lower() for s in sub_info.get('seed', []))
        added_subs = set(s.lower() for s in sub_info.get('added', []))
        
        # Helper to identify links involving specific subreddits
        def get_links_involving(df, subs_set):
            return df[
                df["SOURCE_SUBREDDIT"].str.lower().isin(subs_set) | 
                df["TARGET_SUBREDDIT"].str.lower().isin(subs_set)
            ]

        seed_interactions = get_links_involving(event_df, seed_subs)
        added_interactions = get_links_involving(event_df, added_subs)
        
        seed_vol = len(seed_interactions)
        added_vol = len(added_interactions)
        
        seed_sent = seed_interactions["LINK_SENTIMENT"].mean() if seed_vol > 0 else np.nan
        added_sent = added_interactions["LINK_SENTIMENT"].mean() if added_vol > 0 else np.nan
        
        results.append({
            "event": event_name,
            "seed_volume": seed_vol,
            "added_volume": added_vol,
            "seed_sentiment": seed_sent,
            "added_sentiment": added_sent,
            "volume_growth_pct": ((added_vol - seed_vol) / seed_vol * 100) if seed_vol > 0 else np.nan,
            "sentiment_diff": added_sent - seed_sent if (not np.isnan(seed_sent) and not np.isnan(added_sent)) else np.nan
        })
        
    return pd.DataFrame(results)

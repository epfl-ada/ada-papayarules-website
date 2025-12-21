"""
Event-Based Analysis Functions for the Reddit Political Network Project.

This module implements analysis functions that directly address the research questions:
1. Cross-community propagation
2. Community alignment and discourse
3. Temporal dynamics
4. Stance and sentiment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

from src.data.event_subreddits import (
    EVENT_PERIODS, EVENT_CATEGORIES,
    ALL_POLITICAL, ALL_HEALTH, ALL_DISASTERS, ALL_DIPLOMACY, ALL_CONFLICT, ALL_CLIMATE
)


# =============================================================================
# RQ1: CROSS-COMMUNITY PROPAGATION
# =============================================================================

def compute_cross_domain_links(body: pd.DataFrame, event_subs: List[str]) -> pd.DataFrame:
    """
    Compute links that cross from event-related subreddits to non-event subreddits.
    
    Answers: "To what extent do discussions about major global events spread beyond 
    their original communities?"
    """
    event_set = set(s.lower() for s in event_subs)
    
    body = body.copy()
    body["src_event"] = body["SOURCE_SUBREDDIT"].str.lower().isin(event_set)
    body["tgt_event"] = body["TARGET_SUBREDDIT"].str.lower().isin(event_set)
    
    # Classify propagation type
    def classify_propagation(row):
        if row["src_event"] and row["tgt_event"]:
            return "Within Event Community"
        elif row["src_event"] and not row["tgt_event"]:
            return "Event → External (Spreading)"
        elif not row["src_event"] and row["tgt_event"]:
            return "External → Event (Engaging)"
        else:
            return "Unrelated"
    
    body["propagation_type"] = body.apply(classify_propagation, axis=1)
    
    return body.groupby("propagation_type").agg({
        "SOURCE_SUBREDDIT": "count",
        "LINK_SENTIMENT": "mean"
    }).rename(columns={"SOURCE_SUBREDDIT": "count", "LINK_SENTIMENT": "avg_sentiment"})


def compute_event_propagation_metrics(body: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, compute propagation metrics.
    """
    results = []
    
    for event_name, (start, end, subs) in EVENT_PERIODS.items():
        # Filter to event period
        mask = (body["TIMESTAMP"] >= start) & (body["TIMESTAMP"] <= end)
        event_df = body[mask]
        
        if len(event_df) == 0:
            continue
            
        event_set = set(s.lower() for s in subs)
        
        # Count links involving event subreddits
        event_involved = event_df[
            event_df["SOURCE_SUBREDDIT"].str.lower().isin(event_set) |
            event_df["TARGET_SUBREDDIT"].str.lower().isin(event_set)
        ]
        
        # Cross-domain links (event → non-event)
        spreading = event_df[
            event_df["SOURCE_SUBREDDIT"].str.lower().isin(event_set) &
            ~event_df["TARGET_SUBREDDIT"].str.lower().isin(event_set)
        ]
        
        # Non-event → event (engagement)
        engaging = event_df[
            ~event_df["SOURCE_SUBREDDIT"].str.lower().isin(event_set) &
            event_df["TARGET_SUBREDDIT"].str.lower().isin(event_set)
        ]
        
        results.append({
            "event": event_name,
            "start": start,
            "end": end,
            "total_links": len(event_df),
            "event_involved": len(event_involved),
            "event_involvement_pct": len(event_involved) / len(event_df) * 100 if len(event_df) > 0 else 0,
            "spreading_links": len(spreading),
            "engaging_links": len(engaging),
            "cross_domain_ratio": (len(spreading) + len(engaging)) / len(event_involved) * 100 if len(event_involved) > 0 else 0,
            "avg_sentiment_event": event_involved["LINK_SENTIMENT"].mean() if len(event_involved) > 0 else 0,
            "avg_sentiment_spreading": spreading["LINK_SENTIMENT"].mean() if len(spreading) > 0 else 0,
        })
    
    return pd.DataFrame(results)


def plot_event_propagation(propagation_df: pd.DataFrame, save_path: str = None):
    """
    Visualize cross-community propagation for each event.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort by event involvement
    df = propagation_df.sort_values("event_involved", ascending=True)
    
    # 1. Event involvement
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    axes[0, 0].barh(range(len(df)), df["event_involved"], color=colors)
    axes[0, 0].set_yticks(range(len(df)))
    axes[0, 0].set_yticklabels(df["event"], fontsize=9)
    axes[0, 0].set_xlabel("Number of Links Involving Event Subreddits")
    axes[0, 0].set_title("RQ1: Event Community Activity", fontweight='bold')
    
    # 2. Cross-domain ratio
    axes[0, 1].barh(range(len(df)), df["cross_domain_ratio"], color="coral")
    axes[0, 1].set_yticks(range(len(df)))
    axes[0, 1].set_yticklabels(df["event"], fontsize=9)
    axes[0, 1].set_xlabel("Cross-Domain Link Ratio (%)")
    axes[0, 1].set_title("RQ1: Propagation Beyond Original Communities", fontweight='bold')
    
    # 3. Spreading vs Engaging
    x = np.arange(len(df))
    width = 0.35
    axes[1, 0].barh(x - width/2, df["spreading_links"], width, label="Event → External", color="steelblue")
    axes[1, 0].barh(x + width/2, df["engaging_links"], width, label="External → Event", color="orange")
    axes[1, 0].set_yticks(x)
    axes[1, 0].set_yticklabels(df["event"], fontsize=9)
    axes[1, 0].set_xlabel("Number of Links")
    axes[1, 0].set_title("RQ1: Direction of Cross-Community Flow", fontweight='bold')
    axes[1, 0].legend()
    
    # 4. Sentiment comparison
    colors_sent = ["green" if s > 0.85 else "orange" if s > 0.75 else "red" for s in df["avg_sentiment_event"]]
    axes[1, 1].barh(range(len(df)), df["avg_sentiment_event"], color=colors_sent, alpha=0.7)
    axes[1, 1].set_yticks(range(len(df)))
    axes[1, 1].set_yticklabels(df["event"], fontsize=9)
    axes[1, 1].set_xlabel("Average Sentiment (1 = Positive)")
    axes[1, 1].set_title("RQ4: Event Discussion Sentiment", fontweight='bold')
    axes[1, 1].axvline(0.85, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return df


# =============================================================================
# RQ2: COMMUNITY ALIGNMENT AND DISCOURSE
# =============================================================================

def compute_category_interactions(body: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze interactions between different event category communities.
    
    Answers: "Which communities amplify or oppose narratives about particular global issues?"
    """
    categories = {
        "Political": ALL_POLITICAL,
        "Health": ALL_HEALTH,
        "Disasters": ALL_DISASTERS,
        "Diplomacy": ALL_DIPLOMACY,
        "Conflict": ALL_CONFLICT,
        "Climate": ALL_CLIMATE,
    }
    
    # Create sets for each category
    cat_sets = {cat: set(s.lower() for s in subs) for cat, subs in categories.items()}
    
    def get_category(subreddit):
        sub_lower = subreddit.lower()
        for cat, subs in cat_sets.items():
            if sub_lower in subs:
                return cat
        return "Other"
    
    body = body.copy()
    body["src_category"] = body["SOURCE_SUBREDDIT"].apply(get_category)
    body["tgt_category"] = body["TARGET_SUBREDDIT"].apply(get_category)
    
    # Compute cross-category interactions
    cross_cat = body[body["src_category"] != body["tgt_category"]]
    
    interaction_matrix = body.groupby(["src_category", "tgt_category"]).agg({
        "SOURCE_SUBREDDIT": "count",
        "LINK_SENTIMENT": "mean"
    }).reset_index()
    interaction_matrix.columns = ["Source Category", "Target Category", "Count", "Avg Sentiment"]
    
    return interaction_matrix


def compute_amplifier_resistor_scores(body: pd.DataFrame, event_subs: List[str], 
                                       event_name: str) -> pd.DataFrame:
    """
    Score subreddits as amplifiers or resistors for a specific event.
    
    Amplifier: Many positive links about the event
    Resistor: Many negative links about the event
    """
    event_set = set(s.lower() for s in event_subs)
    
    # Links involving event subreddits
    body = body.copy()
    body["event_related"] = (
        body["SOURCE_SUBREDDIT"].str.lower().isin(event_set) |
        body["TARGET_SUBREDDIT"].str.lower().isin(event_set)
    )
    
    event_links = body[body["event_related"]]
    
    # Score each subreddit
    subreddit_scores = []
    
    for sub in set(event_links["SOURCE_SUBREDDIT"]):
        sub_links = event_links[event_links["SOURCE_SUBREDDIT"] == sub]
        if len(sub_links) >= 5:  # Minimum threshold
            pos = (sub_links["LINK_SENTIMENT"] == 1).sum()
            neg = (sub_links["LINK_SENTIMENT"] == -1).sum()
            total = len(sub_links)
            
            # Amplifier score: positive ratio * volume_weight
            amplifier_score = (pos / total) * np.log1p(total)
            resistor_score = (neg / total) * np.log1p(total)
            
            subreddit_scores.append({
                "subreddit": sub,
                "event": event_name,
                "total_links": total,
                "positive": pos,
                "negative": neg,
                "positive_ratio": pos / total,
                "amplifier_score": amplifier_score,
                "resistor_score": resistor_score,
                "role": "Amplifier" if amplifier_score > resistor_score * 2 else 
                        "Resistor" if resistor_score > amplifier_score * 2 else "Neutral"
            })
    
    return pd.DataFrame(subreddit_scores).sort_values("total_links", ascending=False)


def plot_category_heatmap(interaction_matrix: pd.DataFrame, save_path: str = None):
    """
    Create a heatmap of category-to-category interactions.
    """
    # Pivot for heatmap
    count_pivot = interaction_matrix.pivot_table(
        values="Count", index="Source Category", columns="Target Category", fill_value=0
    )
    sentiment_pivot = interaction_matrix.pivot_table(
        values="Avg Sentiment", index="Source Category", columns="Target Category", fill_value=0.5
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Link volume heatmap
    sns.heatmap(count_pivot, annot=True, fmt=".0f", cmap="Blues", ax=axes[0],
                cbar_kws={'label': 'Number of Links'}, norm=LogNorm())
    axes[0].set_title("RQ2: Cross-Category Link Volume", fontweight='bold')
    axes[0].set_xlabel("Target Category")
    axes[0].set_ylabel("Source Category")
    
    # Sentiment heatmap
    sns.heatmap(sentiment_pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[1],
                vmin=0.6, vmax=1.0, center=0.8, cbar_kws={'label': 'Avg Sentiment'})
    axes[1].set_title("RQ2: Cross-Category Sentiment", fontweight='bold')
    axes[1].set_xlabel("Target Category")
    axes[1].set_ylabel("Source Category")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return count_pivot, sentiment_pivot


# =============================================================================
# RQ3: TEMPORAL DYNAMICS
# =============================================================================

def compute_event_aligned_activity(body: pd.DataFrame, event_name: str,
                                    days_before: int = 30, days_after: int = 30) -> pd.DataFrame:
    """
    Align activity around a specific event date.
    
    Answers: "How do patterns of discussion and hyperlinking vary across different events?"
    """
    if event_name not in EVENT_PERIODS:
        raise ValueError(f"Unknown event: {event_name}")
    
    start_date, end_date, subs = EVENT_PERIODS[event_name]
    event_date = pd.to_datetime(start_date)
    event_set = set(s.lower() for s in subs)
    
    # Get window
    window_start = event_date - timedelta(days=days_before)
    window_end = event_date + timedelta(days=days_after)
    
    body = body.copy()
    if not pd.api.types.is_datetime64_any_dtype(body["TIMESTAMP"]):
        body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"])
    
    window_df = body[(body["TIMESTAMP"] >= window_start) & (body["TIMESTAMP"] <= window_end)]
    
    # Compute daily activity
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


def plot_temporal_event_comparison(body: pd.DataFrame, events: List[str] = None,
                                    save_path: str = None):
    """
    Compare temporal patterns across multiple events.
    """
    if events is None:
        events = ["Brexit Referendum", "US Election 2016", "Crimea Annexation", "ISIS Caliphate"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, event_name in enumerate(events[:4]):
        if event_name not in EVENT_PERIODS:
            continue
            
        daily = compute_event_aligned_activity(body, event_name)
        
        # Plot event-related activity
        event_daily = daily[daily["event_related"] == True]
        
        if len(event_daily) > 0:
            # Rolling mean for smoothing
            event_daily = event_daily.sort_values("days_from_event")
            
            axes[idx].plot(event_daily["days_from_event"], event_daily["count"], 
                          color="steelblue", linewidth=2, alpha=0.7)
            axes[idx].fill_between(event_daily["days_from_event"], 0, event_daily["count"],
                                   color="steelblue", alpha=0.3)
            axes[idx].axvline(0, color="red", linestyle="--", linewidth=2, label="Event Start")
            axes[idx].set_xlabel("Days from Event")
            axes[idx].set_ylabel("Daily Link Count")
            axes[idx].set_title(f"RQ3: {event_name}", fontweight='bold')
            axes[idx].yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compute_event_lift(body: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the "lift" in activity during each event compared to baseline.
    
    Answers: "Do spikes in interaction correlate with real-world developments?"
    """
    results = []
    
    for event_name, (start, end, subs) in EVENT_PERIODS.items():
        event_set = set(s.lower() for s in subs)
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        # Baseline: 60 days before event
        baseline_start = start_dt - timedelta(days=60)
        baseline_end = start_dt - timedelta(days=1)
        
        body = body.copy()
        if not pd.api.types.is_datetime64_any_dtype(body["TIMESTAMP"]):
            body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"])
        
        # Event period activity
        event_mask = (body["TIMESTAMP"] >= start_dt) & (body["TIMESTAMP"] <= end_dt)
        event_df = body[event_mask]
        event_related = event_df[
            event_df["SOURCE_SUBREDDIT"].str.lower().isin(event_set) |
            event_df["TARGET_SUBREDDIT"].str.lower().isin(event_set)
        ]
        
        # Baseline period activity
        baseline_mask = (body["TIMESTAMP"] >= baseline_start) & (body["TIMESTAMP"] <= baseline_end)
        baseline_df = body[baseline_mask]
        baseline_related = baseline_df[
            baseline_df["SOURCE_SUBREDDIT"].str.lower().isin(event_set) |
            baseline_df["TARGET_SUBREDDIT"].str.lower().isin(event_set)
        ]
        
        # Compute daily rates
        event_days = (end_dt - start_dt).days + 1
        baseline_days = (baseline_end - baseline_start).days + 1
        
        event_daily_rate = len(event_related) / event_days if event_days > 0 else 0
        baseline_daily_rate = len(baseline_related) / baseline_days if baseline_days > 0 else 0
        
        lift = (event_daily_rate / baseline_daily_rate - 1) * 100 if baseline_daily_rate > 0 else 0
        
        results.append({
            "event": event_name,
            "event_links": len(event_related),
            "baseline_links": len(baseline_related),
            "event_daily_rate": event_daily_rate,
            "baseline_daily_rate": baseline_daily_rate,
            "lift_pct": lift
        })
    
    return pd.DataFrame(results).sort_values("lift_pct", ascending=False)


# =============================================================================
# RQ4: STANCE AND SENTIMENT
# =============================================================================

def compute_event_sentiment_profile(body: pd.DataFrame, event_name: str) -> pd.DataFrame:
    """
    Compute detailed sentiment profile for an event.
    
    Answers: "When a subreddit references another in the context of a global event, 
    does its tone appear supportive, critical, or neutral?"
    """
    if event_name not in EVENT_PERIODS:
        raise ValueError(f"Unknown event: {event_name}")
    
    start, end, subs = EVENT_PERIODS[event_name]
    event_set = set(s.lower() for s in subs)
    
    body = body.copy()
    if not pd.api.types.is_datetime64_any_dtype(body["TIMESTAMP"]):
        body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"])
    
    # Filter to event period
    event_df = body[(body["TIMESTAMP"] >= start) & (body["TIMESTAMP"] <= end)]
    
    # Get event-related links
    event_links = event_df[
        event_df["SOURCE_SUBREDDIT"].str.lower().isin(event_set) |
        event_df["TARGET_SUBREDDIT"].str.lower().isin(event_set)
    ]
    
    # Compute sentiment by source subreddit
    sentiment_profile = event_links.groupby("SOURCE_SUBREDDIT").agg({
        "LINK_SENTIMENT": ["count", "mean", "sum"],
        "TARGET_SUBREDDIT": "nunique"
    }).reset_index()
    sentiment_profile.columns = ["subreddit", "link_count", "avg_sentiment", "sentiment_sum", "targets"]
    
    # Classify stance
    def classify_stance(row):
        if row["avg_sentiment"] >= 0.9:
            return "Supportive"
        elif row["avg_sentiment"] <= 0.7:
            return "Critical"
        else:
            return "Neutral"
    
    sentiment_profile["stance"] = sentiment_profile.apply(classify_stance, axis=1)
    
    return sentiment_profile.sort_values("link_count", ascending=False)


def plot_event_sentiment_comparison(body: pd.DataFrame, save_path: str = None):
    """
    Compare sentiment profiles across all events.
    """
    results = []
    
    for event_name, (start, end, subs) in EVENT_PERIODS.items():
        event_set = set(s.lower() for s in subs)
        
        body_copy = body.copy()
        if not pd.api.types.is_datetime64_any_dtype(body_copy["TIMESTAMP"]):
            body_copy["TIMESTAMP"] = pd.to_datetime(body_copy["TIMESTAMP"])
        
        event_df = body_copy[(body_copy["TIMESTAMP"] >= start) & (body_copy["TIMESTAMP"] <= end)]
        event_links = event_df[
            event_df["SOURCE_SUBREDDIT"].str.lower().isin(event_set) |
            event_df["TARGET_SUBREDDIT"].str.lower().isin(event_set)
        ]
        
        if len(event_links) > 0:
            results.append({
                "event": event_name,
                "avg_sentiment": event_links["LINK_SENTIMENT"].mean(),
                "positive_pct": (event_links["LINK_SENTIMENT"] == 1).sum() / len(event_links) * 100,
                "negative_pct": (event_links["LINK_SENTIMENT"] == -1).sum() / len(event_links) * 100,
                "link_count": len(event_links)
            })
    
    df = pd.DataFrame(results).sort_values("avg_sentiment", ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sentiment ranking
    colors = ["green" if s > 0.85 else "orange" if s > 0.75 else "red" for s in df["avg_sentiment"]]
    axes[0].barh(range(len(df)), df["avg_sentiment"], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(df)))
    axes[0].set_yticklabels(df["event"], fontsize=9)
    axes[0].set_xlabel("Average Sentiment (1 = Positive)")
    axes[0].set_title("RQ4: Event Sentiment Ranking", fontweight='bold')
    axes[0].axvline(0.85, color='gray', linestyle='--', alpha=0.5, label="Baseline")
    axes[0].legend()
    
    # Positive vs Negative breakdown
    df_sorted = df.sort_values("negative_pct", ascending=True)
    x = np.arange(len(df_sorted))
    width = 0.35
    axes[1].barh(x - width/2, df_sorted["positive_pct"], width, label="Positive %", color="green", alpha=0.7)
    axes[1].barh(x + width/2, df_sorted["negative_pct"], width, label="Negative %", color="red", alpha=0.7)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(df_sorted["event"], fontsize=9)
    axes[1].set_xlabel("Percentage of Links")
    axes[1].set_title("RQ4: Sentiment Distribution by Event", fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return df

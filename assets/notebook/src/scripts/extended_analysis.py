"""
Extended analysis functions for the Reddit Political Network project.
This module provides comprehensive analysis for the data story.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter
import networkx as nx


def _coerce_item_set(x) -> set:
    """Accept DataFrame/Series/list/set and return a lowercase set of subreddit names."""
    if x is None:
        return set()
    if isinstance(x, set):
        return {str(s).lower() for s in x}
    if isinstance(x, (list, tuple)):
        return {str(s).lower() for s in x}
    # pandas
    if hasattr(x, "columns") and "item" in getattr(x, "columns", []):
        return set(x["item"].astype(str).str.lower())
    if hasattr(x, "astype") and not hasattr(x, "columns"):
        # Series-like
        return set(pd.Series(x).astype(str).str.lower())
    return {str(x).lower()}

def compute_dataset_stats(body: pd.DataFrame, title: pd.DataFrame, pol_sb: pd.DataFrame) -> dict:
    """
    Compute comprehensive dataset statistics.
    
    Returns a dict with all key metrics about the dataset.
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(body["TIMESTAMP"]):
        body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"], errors="coerce")
    # Normalize tz-aware → tz-naive UTC for safe plotting/comparisons
    try:
        if getattr(body["TIMESTAMP"].dt, "tz", None) is not None:
            body["TIMESTAMP"] = body["TIMESTAMP"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass

    
        body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"])
    if not pd.api.types.is_datetime64_any_dtype(title["TIMESTAMP"]):
        title["TIMESTAMP"] = pd.to_datetime(title["TIMESTAMP"])
    
    stats = {
        # Volume metrics
        "total_body_links": len(body),
        "total_title_links": len(title),
        "total_links": len(body) + len(title),
        
        # Subreddit metrics
        "unique_source_subs_body": body["SOURCE_SUBREDDIT"].nunique(),
        "unique_target_subs_body": body["TARGET_SUBREDDIT"].nunique(),
        "unique_source_subs_title": title["SOURCE_SUBREDDIT"].nunique(),
        "unique_target_subs_title": title["TARGET_SUBREDDIT"].nunique(),
        
        # Time metrics
        "date_start": body["TIMESTAMP"].min(),
        "date_end": body["TIMESTAMP"].max(),
        "span_days": (body["TIMESTAMP"].max() - body["TIMESTAMP"].min()).days,
        
        # Sentiment metrics
        "body_positive_count": (body["LINK_SENTIMENT"] == 1).sum(),
        "body_negative_count": (body["LINK_SENTIMENT"] == -1).sum(),
        "title_positive_count": (title["LINK_SENTIMENT"] == 1).sum(),
        "title_negative_count": (title["LINK_SENTIMENT"] == -1).sum(),
        
        # Political metrics
        "political_subreddits_count": len(pol_sb),
    }
    
    # Compute ratios
    stats["body_positive_ratio"] = stats["body_positive_count"] / stats["total_body_links"]
    stats["body_negative_ratio"] = stats["body_negative_count"] / stats["total_body_links"]
    stats["title_positive_ratio"] = stats["title_positive_count"] / stats["total_title_links"]
    stats["title_negative_ratio"] = stats["title_negative_count"] / stats["total_title_links"]
    
    return stats


def print_dataset_overview(stats: dict):
    """Print a formatted overview of dataset statistics."""
    print("=" * 70)
    print("REDDIT POLITICAL NETWORK - DATASET OVERVIEW")
    print("=" * 70)
    
    print("\n📊 VOLUME METRICS")
    print(f"   Total Body Hyperlinks:  {stats['total_body_links']:>12,}")
    print(f"   Total Title Hyperlinks: {stats['total_title_links']:>12,}")
    print(f"   Combined Total:         {stats['total_links']:>12,}")
    
    print("\n🔗 SUBREDDIT COVERAGE")
    print(f"   Unique Source Subreddits (Body):  {stats['unique_source_subs_body']:>8,}")
    print(f"   Unique Target Subreddits (Body):  {stats['unique_target_subs_body']:>8,}")
    print(f"   Political Subreddits in Filter:   {stats['political_subreddits_count']:>8,}")
    
    print("\n📅 TIME RANGE")
    print(f"   Start Date: {stats['date_start'].date()}")
    print(f"   End Date:   {stats['date_end'].date()}")
    print(f"   Span:       {stats['span_days']:,} days ({stats['span_days']//365} years, {(stats['span_days']%365)//30} months)")
    
    print("\n😊 SENTIMENT DISTRIBUTION")
    print(f"   Body - Positive: {stats['body_positive_count']:>10,} ({stats['body_positive_ratio']*100:>5.1f}%)")
    print(f"   Body - Negative: {stats['body_negative_count']:>10,} ({stats['body_negative_ratio']*100:>5.1f}%)")
    print(f"   Title - Positive: {stats['title_positive_count']:>9,} ({stats['title_positive_ratio']*100:>5.1f}%)")
    print(f"   Title - Negative: {stats['title_negative_count']:>9,} ({stats['title_negative_ratio']*100:>5.1f}%)")
    
    print("=" * 70)


def get_top_subreddits(df: pd.DataFrame, n: int = 20, by: str = "source") -> pd.DataFrame:
    """
    Get top N subreddits by link count.
    
    Args:
        df: DataFrame with SOURCE_SUBREDDIT and TARGET_SUBREDDIT columns
        n: Number of top subreddits to return
        by: "source", "target", or "both"
    
    Returns:
        DataFrame with subreddit rankings
    """
    if by == "source":
        counts = df["SOURCE_SUBREDDIT"].value_counts().head(n)
        return pd.DataFrame({"subreddit": counts.index, "outgoing_links": counts.values})
    elif by == "target":
        counts = df["TARGET_SUBREDDIT"].value_counts().head(n)
        return pd.DataFrame({"subreddit": counts.index, "incoming_links": counts.values})
    else:
        source_counts = df["SOURCE_SUBREDDIT"].value_counts()
        target_counts = df["TARGET_SUBREDDIT"].value_counts()
        all_subs = set(source_counts.index) | set(target_counts.index)
        data = []
        for sub in all_subs:
            data.append({
                "subreddit": sub,
                "outgoing_links": source_counts.get(sub, 0),
                "incoming_links": target_counts.get(sub, 0),
                "total_links": source_counts.get(sub, 0) + target_counts.get(sub, 0)
            })
        return pd.DataFrame(data).nlargest(n, "total_links")


def plot_top_subreddits(body: pd.DataFrame, n: int = 15, save_path: str = None):
    """Create visualization of top subreddits by incoming and outgoing links."""
    top_sources = body["SOURCE_SUBREDDIT"].value_counts().head(n)
    top_targets = body["TARGET_SUBREDDIT"].value_counts().head(n)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top sources (outgoing links)
    colors_out = plt.cm.Blues(np.linspace(0.4, 0.9, n))[::-1]
    axes[0].barh(range(n), top_sources.values[::-1], color=colors_out)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f"r/{s}" for s in top_sources.index[::-1]])
    axes[0].set_xlabel("Number of Outgoing Links", fontsize=12)
    axes[0].set_title("Top 15 Subreddits by Outgoing Links\n(Most Active Cross-Linkers)", fontsize=14)
    for i, v in enumerate(top_sources.values[::-1]):
        axes[0].text(v + max(top_sources.values)*0.01, i, f'{v:,}', va='center', fontsize=9)
    
    # Top targets (incoming links)
    colors_in = plt.cm.Oranges(np.linspace(0.4, 0.9, n))[::-1]
    axes[1].barh(range(n), top_targets.values[::-1], color=colors_in)
    axes[1].set_yticks(range(n))
    axes[1].set_yticklabels([f"r/{s}" for s in top_targets.index[::-1]])
    axes[1].set_xlabel("Number of Incoming Links", fontsize=12)
    axes[1].set_title("Top 15 Subreddits by Incoming Links\n(Most Referenced Communities)", fontsize=14)
    for i, v in enumerate(top_targets.values[::-1]):
        axes[1].text(v + max(top_targets.values)*0.01, i, f'{v:,}', va='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return top_sources, top_targets


def compute_monthly_activity(df: pd.DataFrame) -> pd.Series:
    """Compute monthly link counts."""
    if not pd.api.types.is_datetime64_any_dtype(df["TIMESTAMP"]):
        df = df.copy()
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        try:
            if getattr(df["TIMESTAMP"].dt, "tz", None) is not None:
                df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            pass
    
    monthly = df.groupby(df["TIMESTAMP"].dt.to_period("M")).size()
    monthly.index = monthly.index.to_timestamp()
    return monthly


def plot_temporal_activity(body: pd.DataFrame, title: pd.DataFrame = None, 
                           events: list = None, save_path: str = None):
    """
    Plot temporal activity with optional event markers.
    
    Args:
        body: Body links DataFrame
        title: Optional title links DataFrame  
        events: List of (date_str, label) tuples for event markers
    """
    monthly_body = compute_monthly_activity(body)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot body links
    ax.fill_between(monthly_body.index, monthly_body.values, alpha=0.3, color="steelblue", label="Body Links")
    ax.plot(monthly_body.index, monthly_body.values, color="steelblue", linewidth=2)
    
    # Plot title links if provided
    if title is not None:
        monthly_title = compute_monthly_activity(title)
        ax.fill_between(monthly_title.index, monthly_title.values, alpha=0.3, color="coral", label="Title Links")
        ax.plot(monthly_title.index, monthly_title.values, color="coral", linewidth=2)
    
    # Default events if none provided
    if events is None:
        events = [
            ("2014-03-18", "Crimea\nAnnexation"),
            ("2014-06-29", "ISIS\nCaliphate"),
            ("2015-06-16", "Trump\nAnnounces"),
            ("2015-12-12", "Paris\nAgreement"),
            ("2016-06-23", "Brexit\nVote"),
            ("2016-11-08", "US\nElection"),
            ("2017-01-20", "Trump\nInauguration"),
        ]
    
    # Add event markers
    ymax = ax.get_ylim()[1]
    for date_str, label in events:
        date = pd.to_datetime(date_str)
        ax.axvline(date, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.annotate(label, xy=(date, ymax * 0.95), fontsize=8, ha="center", 
                   color="darkred", rotation=0)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Number of Hyperlinks", fontsize=12)
    ax.set_title("Monthly Cross-Community Linking Activity (2014-2017)\nwith Major Global Events", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return monthly_body


def classify_link_type(row, focus_set: set, focus_label: str = "Event") -> str:
    """Classify a link based on membership in a focus subreddit set.

    By default, labels use 'Event' / 'Non-Event'. For political analysis, pass focus_label='Political'.
    """
    src_in = str(row["SOURCE_SUBREDDIT"]).lower() in focus_set
    tgt_in = str(row["TARGET_SUBREDDIT"]).lower() in focus_set

    other_label = f"Non-{focus_label}"

    if src_in and tgt_in:
        return f"{focus_label} → {focus_label}"
    if src_in and not tgt_in:
        return f"{focus_label} → {other_label}"
    if (not src_in) and tgt_in:
        return f"{other_label} → {focus_label}"
    return f"{other_label} → {other_label}"



def analyze_political_interactions(body: pd.DataFrame, pol_sb, focus_label: str = "Event") -> pd.DataFrame:
    """
    Analyze interactions between political and non-political communities.
    
    Returns DataFrame with counts and sentiment by link type.
    """
    pol_set = _coerce_item_set(pol_sb)
    
    # Classify each link
    body = body.copy()
    body["link_type"] = body.apply(lambda row: classify_link_type(row, pol_set, focus_label=focus_label), axis=1)
    
    # Compute statistics by link type
    results = []
    for lt in body["link_type"].unique():
        subset = body[body["link_type"] == lt]
        pos = (subset["LINK_SENTIMENT"] == 1).sum()
        neg = (subset["LINK_SENTIMENT"] == -1).sum()
        total = len(subset)
        
        results.append({
            "Link Type": lt,
            "Total Links": total,
            "Positive": pos,
            "Negative": neg,
            "Positive %": pos / total * 100 if total > 0 else 0,
            "Negative %": neg / total * 100 if total > 0 else 0,
            "Share %": total / len(body) * 100
        })
    
    return pd.DataFrame(results).sort_values("Total Links", ascending=False)


def plot_political_interactions(body: pd.DataFrame, pol_sb, save_path: str = None, focus_label: str = "Event"):
    """Create comprehensive visualization of political interaction patterns."""
    pol_set = _coerce_item_set(pol_sb)
    body = body.copy()
    body["link_type"] = body.apply(lambda row: classify_link_type(row, pol_set, focus_label=focus_label), axis=1)
    
    link_type_counts = body["link_type"].value_counts()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Pie chart of link types
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    axes[0, 0].pie(link_type_counts, labels=None, autopct="%1.1f%%",
                   colors=colors, startangle=90, pctdistance=0.75)
    axes[0, 0].legend(link_type_counts.index, loc="upper left", fontsize=9)
    axes[0, 0].set_title(f"Distribution of Link Types ({focus_label} vs Non-{focus_label})", fontsize=12, fontweight='bold')
    
    # 2. Bar chart of total counts
    axes[0, 1].bar(range(len(link_type_counts)), link_type_counts.values, color=colors)
    axes[0, 1].set_xticks(range(len(link_type_counts)))
    axes[0, 1].set_xticklabels([lt.replace(" → ", "\n→ ") for lt in link_type_counts.index], 
                               fontsize=9)
    axes[0, 1].set_ylabel("Number of Links")
    axes[0, 1].set_title(f"Link Volume by Type ({focus_label} focus)", fontsize=12, fontweight='bold')
    for i, v in enumerate(link_type_counts.values):
        axes[0, 1].text(i, v + max(link_type_counts.values)*0.02, f'{v:,}', 
                       ha='center', fontsize=9)
    
    # 3. Sentiment breakdown by link type
    sentiment_data = body.groupby("link_type").apply(
        lambda x: pd.Series({
            "Positive": (x["LINK_SENTIMENT"] == 1).sum(),
            "Negative": (x["LINK_SENTIMENT"] == -1).sum()
        })
    )
    
    x = np.arange(len(sentiment_data))
    width = 0.35
    axes[1, 0].bar(x - width/2, sentiment_data["Positive"], width, label="Positive", color="green", alpha=0.7)
    axes[1, 0].bar(x + width/2, sentiment_data["Negative"], width, label="Negative", color="red", alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([lt.replace(" → ", "\n→ ") for lt in sentiment_data.index], fontsize=9)
    axes[1, 0].set_ylabel("Number of Links")
    axes[1, 0].set_title("Sentiment Distribution by Link Type", fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    
    # 4. Negative sentiment ratio by link type
    neg_ratios = body.groupby("link_type").apply(
        lambda x: (x["LINK_SENTIMENT"] == -1).sum() / len(x) * 100
    ).sort_values(ascending=True)
    
    barcolors = ["red" if r > 15 else "orange" if r > 10 else "green" for r in neg_ratios.values]
    axes[1, 1].barh(range(len(neg_ratios)), neg_ratios.values, color=barcolors, alpha=0.7)
    axes[1, 1].set_yticks(range(len(neg_ratios)))
    axes[1, 1].set_yticklabels([lt.replace(" → ", " → ") for lt in neg_ratios.index], fontsize=9)
    axes[1, 1].set_xlabel("Negative Sentiment Ratio (%)")
    axes[1, 1].set_title("Negativity by Link Type", fontsize=12, fontweight='bold')
    axes[1, 1].axvline(x=neg_ratios.mean(), color='black', linestyle='--', alpha=0.5, label=f'Mean: {neg_ratios.mean():.1f}%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return sentiment_data


def analyze_event_period(body: pd.DataFrame, start_date: str, end_date: str, 
                         pol_set: set = None, event_name: str = "") -> dict:
    """
    Analyze activity during a specific event period.
    
    Args:
        body: DataFrame with links
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        pol_set: Set of political subreddit names (lowercase)
        event_name: Name of the event for display
    
    Returns:
        Dict with event statistics
    """
    from src.utils.data_utils import get_links_within_period
    
    event_df = get_links_within_period(body, start_date, end_date)
    
    if len(event_df) == 0:
        return {"event": event_name, "total": 0}
    
    total = len(event_df)
    pos = (event_df["LINK_SENTIMENT"] == 1).sum()
    neg = (event_df["LINK_SENTIMENT"] == -1).sum()
    
    result = {
        "event": event_name,
        "start": start_date,
        "end": end_date,
        "total": total,
        "positive": pos,
        "negative": neg,
        "positive_pct": pos / total * 100,
        "negative_pct": neg / total * 100,
    }
    
    if pol_set:
        pol_involved = event_df.apply(
            lambda x: x["SOURCE_SUBREDDIT"].lower() in pol_set or x["TARGET_SUBREDDIT"].lower() in pol_set,
            axis=1
        ).sum()
        result["political_involved"] = pol_involved
        result["political_pct"] = pol_involved / total * 100
    
    return result


def analyze_all_events(body: pd.DataFrame, pol_sb: pd.DataFrame) -> pd.DataFrame:
    """Analyze activity across all major events in the dataset period."""
    pol_set = set(pol_sb["item"].str.lower())
    
    events = [
        ("Crimea Annexation", "2014-02-20", "2014-03-25"),
        ("MH370 Disappearance", "2014-03-08", "2014-03-31"),
        ("ISIS Caliphate Declaration", "2014-06-25", "2014-07-15"),
        ("Ebola Crisis Peak", "2014-09-01", "2014-11-30"),
        ("Iran Nuclear Deal", "2015-07-10", "2015-07-20"),
        ("Paris Climate Agreement", "2015-11-25", "2015-12-15"),
        ("Brexit Referendum", "2016-06-15", "2016-07-05"),
        ("Trump vs Clinton Debates", "2016-09-20", "2016-10-20"),
        ("US Election 2016", "2016-11-01", "2016-11-15"),
        ("Trump Inauguration", "2017-01-15", "2017-01-25"),
    ]
    
    results = []
    for event_name, start, end in events:
        result = analyze_event_period(body, start, end, pol_set, event_name)
        results.append(result)
    
    return pd.DataFrame(results)


def plot_event_analysis(event_df: pd.DataFrame, save_path: str = None):
    """Create comprehensive event analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Total activity by event
    axes[0, 0].barh(range(len(event_df)), event_df["total"], color="steelblue", alpha=0.7)
    axes[0, 0].set_yticks(range(len(event_df)))
    axes[0, 0].set_yticklabels(event_df["event"], fontsize=9)
    axes[0, 0].set_xlabel("Number of Links")
    axes[0, 0].set_title("Total Activity During Major Events", fontsize=12, fontweight='bold')
    
    # 2. Sentiment comparison
    x = np.arange(len(event_df))
    width = 0.35
    axes[0, 1].bar(x - width/2, event_df["positive_pct"], width, label="Positive %", color="green", alpha=0.7)
    axes[0, 1].bar(x + width/2, event_df["negative_pct"], width, label="Negative %", color="red", alpha=0.7)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(event_df["event"], rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_ylabel("Percentage")
    axes[0, 1].set_title("Sentiment Distribution by Event", fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    
    # 3. Political involvement
    if "political_pct" in event_df.columns:
        colors = ["darkred" if p > 15 else "orange" if p > 10 else "darkgreen" for p in event_df["political_pct"]]
        axes[1, 0].barh(range(len(event_df)), event_df["political_pct"], color=colors, alpha=0.7)
        axes[1, 0].set_yticks(range(len(event_df)))
        axes[1, 0].set_yticklabels(event_df["event"], fontsize=9)
        axes[1, 0].set_xlabel("Political Subreddits Involved (%)")
        axes[1, 0].set_title("Political Community Involvement by Event", fontsize=12, fontweight='bold')
        axes[1, 0].axvline(event_df["political_pct"].mean(), color='black', linestyle='--', alpha=0.5)
    
    # 4. Negativity vs Political involvement scatter
    if "political_pct" in event_df.columns:
        axes[1, 1].scatter(event_df["political_pct"], event_df["negative_pct"], 
                          s=event_df["total"]/100, alpha=0.6, c="purple")
        for i, row in event_df.iterrows():
            axes[1, 1].annotate(row["event"][:15], (row["political_pct"], row["negative_pct"]), 
                               fontsize=7, alpha=0.8)
        axes[1, 1].set_xlabel("Political Involvement (%)")
        axes[1, 1].set_ylabel("Negative Sentiment (%)")
        axes[1, 1].set_title("Negativity vs Political Involvement\n(bubble size = total links)", 
                            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def build_subreddit_network(body: pd.DataFrame, min_weight: int = 5) -> nx.DiGraph:
    """
    Build a directed network of subreddit interactions.
    
    Args:
        body: DataFrame with SOURCE_SUBREDDIT and TARGET_SUBREDDIT
        min_weight: Minimum number of links to include an edge
    
    Returns:
        NetworkX DiGraph
    """
    # Count edges
    edge_counts = body.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]).size().reset_index(name="weight")
    edge_counts = edge_counts[edge_counts["weight"] >= min_weight]
    
    # Build graph
    G = nx.DiGraph()
    for _, row in edge_counts.iterrows():
        G.add_edge(row["SOURCE_SUBREDDIT"], row["TARGET_SUBREDDIT"], weight=row["weight"])
    
    return G


def compute_network_metrics(G: nx.DiGraph) -> dict:
    """Compute key network metrics."""
    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
    }
    
    # Degree statistics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    metrics["avg_in_degree"] = np.mean(list(in_degrees.values()))
    metrics["avg_out_degree"] = np.mean(list(out_degrees.values()))
    metrics["max_in_degree"] = max(in_degrees.values())
    metrics["max_out_degree"] = max(out_degrees.values())
    metrics["top_in_degree_node"] = max(in_degrees, key=in_degrees.get)
    metrics["top_out_degree_node"] = max(out_degrees, key=out_degrees.get)
    
    # Weakly connected components
    metrics["weakly_connected_components"] = nx.number_weakly_connected_components(G)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    metrics["largest_wcc_size"] = len(largest_wcc)
    metrics["largest_wcc_pct"] = len(largest_wcc) / G.number_of_nodes() * 100
    
    return metrics


def print_network_summary(metrics: dict):
    """Print formatted network summary."""
    print("=" * 60)
    print("SUBREDDIT NETWORK SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 SIZE & DENSITY")
    print(f"   Nodes (Subreddits):     {metrics['nodes']:,}")
    print(f"   Edges (Link Pairs):     {metrics['edges']:,}")
    print(f"   Network Density:        {metrics['density']:.6f}")
    
    print(f"\n🔗 DEGREE STATISTICS")
    print(f"   Average In-Degree:      {metrics['avg_in_degree']:.2f}")
    print(f"   Average Out-Degree:     {metrics['avg_out_degree']:.2f}")
    print(f"   Max In-Degree:          {metrics['max_in_degree']:,} (r/{metrics['top_in_degree_node']})")
    print(f"   Max Out-Degree:         {metrics['max_out_degree']:,} (r/{metrics['top_out_degree_node']})")
    
    print(f"\n🌐 CONNECTIVITY")
    print(f"   Weakly Connected Components: {metrics['weakly_connected_components']:,}")
    print(f"   Largest Component Size:      {metrics['largest_wcc_size']:,} ({metrics['largest_wcc_pct']:.1f}%)")
    
    print("=" * 60)


def compute_subreddit_roles(body: pd.DataFrame, pol_sb: pd.DataFrame) -> pd.DataFrame:
    """
    Classify subreddits into roles: Amplifiers, Observers, and Resistors.
    
    - Amplifiers: High outgoing, mostly positive
    - Resistors: High outgoing, mostly negative  
    - Observers: High incoming, low outgoing
    """
    pol_set = set(pol_sb["item"].str.lower())
    
    # Compute metrics per subreddit
    source_counts = body.groupby("SOURCE_SUBREDDIT").agg({
        "LINK_SENTIMENT": ["count", "mean"]
    }).reset_index()
    source_counts.columns = ["subreddit", "outgoing", "avg_sentiment"]
    
    target_counts = body["TARGET_SUBREDDIT"].value_counts().reset_index()
    target_counts.columns = ["subreddit", "incoming"]
    
    # Merge
    roles = source_counts.merge(target_counts, on="subreddit", how="outer").fillna(0)
    roles["is_political"] = roles["subreddit"].str.lower().isin(pol_set)
    
    # Classify roles
    def classify_role(row):
        if row["outgoing"] > 100:
            if row["avg_sentiment"] > 0.7:
                return "Amplifier"
            elif row["avg_sentiment"] < 0.3:
                return "Resistor"
        if row["incoming"] > row["outgoing"] * 3 and row["incoming"] > 100:
            return "Observer"
        return "Neutral"
    
    roles["role"] = roles.apply(classify_role, axis=1)
    
    return roles


def plot_subreddit_roles(roles: pd.DataFrame, top_n: int = 30, save_path: str = None):
    """Visualize subreddit roles."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Role distribution
    role_counts = roles["role"].value_counts()
    colors = {"Amplifier": "green", "Resistor": "red", "Observer": "blue", "Neutral": "gray"}
    axes[0].pie(role_counts, labels=role_counts.index, autopct="%1.1f%%",
               colors=[colors.get(r, "gray") for r in role_counts.index])
    axes[0].set_title("Distribution of Subreddit Roles", fontsize=12, fontweight='bold')
    
    # 2. Scatter plot: incoming vs outgoing
    sample = roles.nlargest(top_n * 3, "outgoing")
    
    for role in ["Amplifier", "Resistor", "Observer", "Neutral"]:
        subset = sample[sample["role"] == role]
        axes[1].scatter(subset["outgoing"], subset["incoming"], 
                       c=colors.get(role, "gray"), label=role, alpha=0.6, s=50)
    
    axes[1].set_xlabel("Outgoing Links (Activity)")
    axes[1].set_ylabel("Incoming Links (Popularity)")
    axes[1].set_title("Subreddit Activity vs Popularity by Role", fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return roles


def analyze_hourly_patterns(body: pd.DataFrame) -> pd.DataFrame:
    """Analyze activity patterns by hour of day."""
    body = body.copy()
    if not pd.api.types.is_datetime64_any_dtype(body["TIMESTAMP"]):
        body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"])
    
    body["hour"] = body["TIMESTAMP"].dt.hour
    body["day_of_week"] = body["TIMESTAMP"].dt.day_name()
    
    hourly = body.groupby("hour").agg({
        "LINK_SENTIMENT": ["count", "mean"]
    }).reset_index()
    hourly.columns = ["hour", "count", "avg_sentiment"]
    
    return hourly


def plot_hourly_patterns(body: pd.DataFrame, save_path: str = None):
    """Visualize hourly activity patterns."""
    hourly = analyze_hourly_patterns(body)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Activity by hour
    axes[0].bar(hourly["hour"], hourly["count"], color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Hour of Day (UTC)")
    axes[0].set_ylabel("Number of Links")
    axes[0].set_title("Linking Activity by Hour of Day", fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(0, 24, 3))
    
    # Sentiment by hour
    colors = ["green" if s > 0.85 else "orange" if s > 0.8 else "red" for s in hourly["avg_sentiment"]]
    axes[1].bar(hourly["hour"], hourly["avg_sentiment"], color=colors, alpha=0.7)
    axes[1].set_xlabel("Hour of Day (UTC)")
    axes[1].set_ylabel("Average Sentiment (1=Positive)")
    axes[1].set_title("Average Sentiment by Hour of Day", fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].axhline(y=hourly["avg_sentiment"].mean(), color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return hourly


def find_controversial_pairs(body: pd.DataFrame, min_links: int = 10) -> pd.DataFrame:
    """
    Find subreddit pairs with high negative sentiment.
    These represent contentious cross-community relationships.
    """
    pair_stats = body.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]).agg({
        "LINK_SENTIMENT": ["count", "mean", "sum"]
    }).reset_index()
    pair_stats.columns = ["source", "target", "count", "avg_sentiment", "sentiment_sum"]
    
    # Filter by minimum links
    pair_stats = pair_stats[pair_stats["count"] >= min_links]
    
    # Calculate negative ratio
    pair_stats["negative_ratio"] = 1 - pair_stats["avg_sentiment"]
    
    # Sort by negativity
    controversial = pair_stats.nlargest(50, "negative_ratio")
    
    return controversial


def plot_controversial_pairs(controversial: pd.DataFrame, top_n: int = 20, save_path: str = None):
    """Visualize most controversial subreddit pairs."""
    top = controversial.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = [f"r/{row['source']} → r/{row['target']}" for _, row in top.iterrows()]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))
    
    ax.barh(range(top_n), top["negative_ratio"].values[::-1] * 100, color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("Negative Sentiment Ratio (%)")
    ax.set_title(f"Top {top_n} Most Contentious Subreddit Relationships\n(minimum {controversial['count'].min()} links)", 
                fontsize=12, fontweight='bold')
    
    # Add count annotations
    for i, (_, row) in enumerate(top.iloc[::-1].iterrows()):
        ax.text(row["negative_ratio"] * 100 + 1, i, f'n={row["count"]}', va='center', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return top


def analyze_vader_sentiment(body: pd.DataFrame, sample_size: int = 50000, save_path: str = None):
    """
    Analyze VADER sentiment score distribution.
    
    Args:
        body: DataFrame with links and PROPERTIES column
        sample_size: Number of rows to sample for performance
        save_path: Optional path to save the figure
    
    Returns:
        Tuple of (positive_vader, negative_vader) Series or None if parsing fails
    """
    from src.utils.data_utils import properties_into_dataframe_columns
    
    # Sample for performance
    sample_size = min(sample_size, len(body))
    body_sample = body.sample(n=sample_size, random_state=42)
    
    try:
        body_props = properties_into_dataframe_columns(body_sample, col_name="PROPERTIES")
        
        if "vader_compound" not in body_props.columns:
            print("VADER columns not found in parsed properties.")
            return None
        
        # Split by sentiment
        pos_vader = body_props[body_props["LINK_SENTIMENT"] == 1]["vader_compound"].dropna()
        neg_vader = body_props[body_props["LINK_SENTIMENT"] == -1]["vader_compound"].dropna()
        
        return pos_vader, neg_vader
        
    except Exception as e:
        print(f"Could not parse properties: {e}")
        return None


def plot_vader_distribution(body: pd.DataFrame, sample_size: int = 50000, save_path: str = None):
    """
    Plot VADER sentiment score distribution comparing positive vs negative links.
    
    Args:
        body: DataFrame with links and PROPERTIES column
        sample_size: Number of rows to sample for performance
        save_path: Optional path to save the figure
    """
    result = analyze_vader_sentiment(body, sample_size)
    
    if result is None:
        return None
    
    pos_vader, neg_vader = result
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution by link sentiment
    axes[0].hist(pos_vader, bins=50, alpha=0.6, label="Positive Links", color="green", density=True)
    axes[0].hist(neg_vader, bins=50, alpha=0.6, label="Negative Links", color="red", density=True)
    axes[0].set_xlabel("VADER Compound Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("VADER Score Distribution by Link Sentiment", fontweight='bold')
    axes[0].legend()
    axes[0].axvline(0, color="gray", linestyle="--", alpha=0.5)
    
    # Box plot
    vader_data = [pos_vader.values, neg_vader.values]
    bp = axes[1].boxplot(vader_data, labels=["Positive Links", "Negative Links"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightgreen")
    bp["boxes"][1].set_facecolor("lightcoral")
    axes[1].set_ylabel("VADER Compound Score")
    axes[1].set_title("VADER Score Box Plot by Link Sentiment", fontweight='bold')
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print stats
    print(f"\nVADER Statistics (sample n={sample_size:,}):")
    print(f"  Positive Links - Mean: {pos_vader.mean():.3f}, Median: {pos_vader.median():.3f}")
    print(f"  Negative Links - Mean: {neg_vader.mean():.3f}, Median: {neg_vader.median():.3f}")
    
    return pos_vader, neg_vader


def compute_link_type_stats(body: pd.DataFrame, pol_sb: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive statistics for each link type.
    
    Returns DataFrame with detailed metrics per link type.
    """
    pol_set = set(pol_sb["item"].str.lower())
    body = body.copy()
    body["link_type"] = body.apply(lambda row: classify_link_type(row, pol_set, focus_label=focus_label), axis=1)
    
    stats = []
    for lt in body["link_type"].unique():
        subset = body[body["link_type"] == lt]
        pos = (subset["LINK_SENTIMENT"] == 1).sum()
        neg = (subset["LINK_SENTIMENT"] == -1).sum()
        total = len(subset)
        
        stats.append({
            "link_type": lt,
            "total": total,
            "share_pct": total / len(body) * 100,
            "positive": pos,
            "negative": neg,
            "positive_pct": pos / total * 100 if total > 0 else 0,
            "negative_pct": neg / total * 100 if total > 0 else 0,
            "unique_sources": subset["SOURCE_SUBREDDIT"].nunique(),
            "unique_targets": subset["TARGET_SUBREDDIT"].nunique(),
        })
    
    return pd.DataFrame(stats).sort_values("total", ascending=False)


def print_link_type_summary(stats_df: pd.DataFrame):
    """Print formatted summary of link type statistics."""
    print("=" * 70)
    print("POLITICAL/NON-POLITICAL INTERACTION SUMMARY")
    print("=" * 70)
    
    for _, row in stats_df.iterrows():
        print(f"\n{row['link_type']}")
        print(f"   Total Links:      {row['total']:>10,} ({row['share_pct']:.1f}% of all links)")
        print(f"   Positive:         {row['positive']:>10,} ({row['positive_pct']:.1f}%)")
        print(f"   Negative:         {row['negative']:>10,} ({row['negative_pct']:.1f}%)")
        print(f"   Unique Sources:   {row['unique_sources']:>10,}")
        print(f"   Unique Targets:   {row['unique_targets']:>10,}")
    
    print("=" * 70)


def print_event_summary(event_df: pd.DataFrame):
    """Print formatted summary of event analysis."""
    print("=" * 70)
    print("EVENT-SPECIFIC ACTIVITY SUMMARY")
    print("=" * 70)
    
    for _, row in event_df.iterrows():
        print(f"\n{row['event']}")
        print(f"   Period:           {row['start']} to {row['end']}")
        print(f"   Total Links:      {row['total']:>10,}")
        print(f"   Positive:         {row['positive']:>10,} ({row['positive_pct']:.1f}%)")
        print(f"   Negative:         {row['negative']:>10,} ({row['negative_pct']:.1f}%)")
        if 'political_involved' in row:
            print(f"   Political Links:  {row['political_involved']:>10,} ({row['political_pct']:.1f}%)")
    
    print("=" * 70)


def compute_network_assortativity(body: pd.DataFrame, pol_sb: pd.DataFrame) -> dict:
    """
    Compute network assortativity metrics (Degree and Attribute).
    
    Quantifies 'Echo Chambers':
    - Positive Attribute Assortativity: Political nodes link to Political nodes.
    - Positive Degree Assortativity: High volume nodes link to High volume nodes (Rich Club).
    """
    import networkx as nx
    
    print("Building full graph for metrics...")
    # Build full graph (not just top nodes)
    edge_counts = body.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]).size().reset_index(name="weight")
    G = nx.DiGraph()
    for _, row in edge_counts.iterrows():
        G.add_edge(row["SOURCE_SUBREDDIT"], row["TARGET_SUBREDDIT"], weight=row["weight"])
    
    # 1. Degree Assortativity
    degree_r = nx.degree_assortativity_coefficient(G)
    
    # 2. Attribute Assortativity (Political vs Non-Political)
    pol_set = set(pol_sb["item"].str.lower())
    
    # Label nodes
    for node in G.nodes():
        G.nodes[node]["type"] = "Political" if node.lower() in pol_set else "Non-Political"
        
    attribute_r = nx.attribute_assortativity_coefficient(G, "type")
    
    return {
        "degree_assortativity": degree_r,
        "attribute_assortativity": attribute_r,
        "details": "r > 0 implies modularity (echo chambers), r < 0 implies disassortativity."
    }
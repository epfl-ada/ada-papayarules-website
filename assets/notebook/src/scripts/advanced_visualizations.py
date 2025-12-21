"""
Advanced visualization functions for the Reddit Political Network project.
Extends the existing plot_generation.py with additional visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import matplotlib.patches as mpatches
from collections import Counter
from matplotlib.colors import LogNorm
import networkx as nx


def _coerce_item_set(x) -> set:
    """Accept DataFrame/Series/list/set and return a lowercase set of subreddit names."""
    if x is None:
        return set()
    if isinstance(x, set):
        return {str(s).lower() for s in x}
    if isinstance(x, (list, tuple)):
        return {str(s).lower() for s in x}
    if hasattr(x, "columns") and "item" in getattr(x, "columns", []):
        return set(x["item"].astype(str).str.lower())
    # Series-like
    try:
        return set(pd.Series(x).astype(str).str.lower())
    except Exception:
        return {str(x).lower()}

def plot_sentiment_heatmap(body: pd.DataFrame, pol_sb, save_path: str = None, focus_label: str = "Event"):
    """
    Create a heatmap showing sentiment patterns between community types.
    """
    pol_set = _coerce_item_set(pol_sb)
    
    body = body.copy()
    body["src_type"] = body["SOURCE_SUBREDDIT"].str.lower().isin(pol_set).map({True: focus_label, False: f"Non-{focus_label}"})
    body["tgt_type"] = body["TARGET_SUBREDDIT"].str.lower().isin(pol_set).map({True: focus_label, False: f"Non-{focus_label}"})
    
    # Compute average sentiment for each combination
    heatmap_data = body.groupby(["src_type", "tgt_type"])["LINK_SENTIMENT"].mean().unstack()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sentiment heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5,
               ax=axes[0], vmin=0, vmax=1, cbar_kws={'label': 'Avg Sentiment (1=Positive)'})
    axes[0].set_title(f"Average Sentiment by Community Type ({focus_label} focus)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Target Type")
    axes[0].set_ylabel("Source Type")
    
    # Count heatmap
    count_data = body.groupby(["src_type", "tgt_type"]).size().unstack()
    sns.heatmap(count_data, annot=True, fmt=",d", cmap="Blues",
               ax=axes[1], cbar_kws={'label': 'Number of Links'}, norm=LogNorm())
    axes[1].set_title(f"Link Volume by Community Type ({focus_label} focus)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Target Type")
    axes[1].set_ylabel("Source Type")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return heatmap_data


def plot_day_of_week_patterns(body: pd.DataFrame, save_path: str = None):
    """Analyze and visualize activity patterns by day of week."""
    body = body.copy()
    if not pd.api.types.is_datetime64_any_dtype(body["TIMESTAMP"]):
        body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"], errors="coerce")
    try:
        if getattr(body["TIMESTAMP"].dt, "tz", None) is not None:
            body["TIMESTAMP"] = body["TIMESTAMP"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    
    body["day_of_week"] = body["TIMESTAMP"].dt.day_name()
    
    # Order days properly
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    daily = body.groupby("day_of_week").agg({
        "LINK_SENTIMENT": ["count", "mean"]
    }).reindex(day_order)
    daily.columns = ["count", "avg_sentiment"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Activity by day
    colors = ["steelblue" if d in ["Saturday", "Sunday"] else "coral" for d in day_order]
    axes[0].bar(range(7), daily["count"], color=colors, alpha=0.7)
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels([d[:3] for d in day_order])
    axes[0].set_ylabel("Number of Links")
    axes[0].set_title("Linking Activity by Day of Week\n(Blue = Weekend)", fontsize=12, fontweight='bold')
    
    # Sentiment by day
    colors_sent = ["green" if s > daily["avg_sentiment"].mean() else "red" for s in daily["avg_sentiment"]]
    axes[1].bar(range(7), daily["avg_sentiment"], color=colors_sent, alpha=0.7)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels([d[:3] for d in day_order])
    axes[1].set_ylabel("Average Sentiment")
    axes[1].set_title("Average Sentiment by Day of Week", fontsize=12, fontweight='bold')
    axes[1].axhline(daily["avg_sentiment"].mean(), color='black', linestyle='--', alpha=0.5)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return daily


def plot_subreddit_wordcloud(body: pd.DataFrame, by: str = "source", 
                              sentiment: str = None, save_path: str = None):
    """
    Create a word cloud of subreddit names.
    
    Args:
        body: DataFrame with links
        by: "source" or "target"
        sentiment: None (all), "positive", or "negative"
    """
    df = body.copy()
    
    if sentiment == "positive":
        df = df[df["LINK_SENTIMENT"] == 1]
        title_suffix = " (Positive Links)"
    elif sentiment == "negative":
        df = df[df["LINK_SENTIMENT"] == -1]
        title_suffix = " (Negative Links)"
    else:
        title_suffix = " (All Links)"
    
    if by == "source":
        text = " ".join(df["SOURCE_SUBREDDIT"].tolist())
        title = f"Most Common Source Subreddits{title_suffix}"
    else:
        text = " ".join(df["TARGET_SUBREDDIT"].tolist())
        title = f"Most Common Target Subreddits{title_suffix}"
    
    wordcloud = WordCloud(width=800, height=400, background_color="white",
                         colormap="viridis", max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return wordcloud


def plot_sentiment_over_time(body: pd.DataFrame, window: str = "M", save_path: str = None):
    """
    Plot how sentiment evolves over time with confidence intervals.
    
    Args:
        body: DataFrame with links
        window: Time window for aggregation ("W", "M", "Q")
    """
    body = body.copy()
    if not pd.api.types.is_datetime64_any_dtype(body["TIMESTAMP"]):
        body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"], errors="coerce")
    try:
        if getattr(body["TIMESTAMP"].dt, "tz", None) is not None:
            body["TIMESTAMP"] = body["TIMESTAMP"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    
    body["period"] = body["TIMESTAMP"].dt.to_period(window)
    
    # Compute stats per period
    period_stats = body.groupby("period").agg({
        "LINK_SENTIMENT": ["mean", "std", "count"]
    })
    period_stats.columns = ["mean", "std", "count"]
    period_stats.index = period_stats.index.to_timestamp()
    
    # Compute confidence interval
    period_stats["ci"] = 1.96 * period_stats["std"] / np.sqrt(period_stats["count"])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot with confidence band
    ax.fill_between(period_stats.index, 
                    period_stats["mean"] - period_stats["ci"],
                    period_stats["mean"] + period_stats["ci"],
                    alpha=0.3, color="steelblue", label="95% CI")
    ax.plot(period_stats.index, period_stats["mean"], color="steelblue", 
            linewidth=2, marker="o", markersize=4, label="Mean Sentiment")
    
    # Add reference line
    overall_mean = body["LINK_SENTIMENT"].mean()
    ax.axhline(overall_mean, color="red", linestyle="--", alpha=0.7, 
              label=f"Overall Mean: {overall_mean:.3f}")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Sentiment (1 = Positive)")
    ax.set_title("Sentiment Evolution Over Time", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return period_stats


def plot_link_volume_distribution(body: pd.DataFrame, save_path: str = None):
    """
    Analyze the distribution of link volumes per subreddit.
    Shows the long-tail distribution typical of social networks.
    """
    source_counts = body["SOURCE_SUBREDDIT"].value_counts()
    target_counts = body["TARGET_SUBREDDIT"].value_counts()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of source counts
    axes[0, 0].hist(source_counts.values, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Number of Outgoing Links")
    axes[0, 0].set_ylabel("Number of Subreddits")
    axes[0, 0].set_title("Distribution of Outgoing Links per Subreddit", fontweight='bold')
    axes[0, 0].set_yscale("log")
    
    # Log-log plot for source
    ranks = np.arange(1, len(source_counts) + 1)
    axes[0, 1].loglog(ranks, source_counts.values, 'b.', alpha=0.5)
    axes[0, 1].set_xlabel("Rank")
    axes[0, 1].set_ylabel("Number of Outgoing Links")
    axes[0, 1].set_title("Source Subreddit Rank Distribution (Log-Log)", fontweight='bold')
    
    # Histogram of target counts
    axes[1, 0].hist(target_counts.values, bins=50, color="coral", alpha=0.7, edgecolor="black")
    axes[1, 0].set_xlabel("Number of Incoming Links")
    axes[1, 0].set_ylabel("Number of Subreddits")
    axes[1, 0].set_title("Distribution of Incoming Links per Subreddit", fontweight='bold')
    axes[1, 0].set_yscale("log")
    
    # Log-log plot for target
    ranks = np.arange(1, len(target_counts) + 1)
    axes[1, 1].loglog(ranks, target_counts.values, 'r.', alpha=0.5)
    axes[1, 1].set_xlabel("Rank")
    axes[1, 1].set_ylabel("Number of Incoming Links")
    axes[1, 1].set_title("Target Subreddit Rank Distribution (Log-Log)", fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary stats
    print("\nLink Volume Statistics:")
    print(f"  Source (Outgoing):")
    print(f"    Mean: {source_counts.mean():.1f}, Median: {source_counts.median():.0f}, Max: {source_counts.max():,}")
    print(f"    Subreddits with 1 link: {(source_counts == 1).sum():,} ({(source_counts == 1).sum()/len(source_counts)*100:.1f}%)")
    print(f"  Target (Incoming):")  
    print(f"    Mean: {target_counts.mean():.1f}, Median: {target_counts.median():.0f}, Max: {target_counts.max():,}")
    print(f"    Subreddits with 1 link: {(target_counts == 1).sum():,} ({(target_counts == 1).sum()/len(target_counts)*100:.1f}%)")
    
    return source_counts, target_counts


def plot_yearly_comparison(body: pd.DataFrame, save_path: str = None):
    """
    Compare activity and sentiment across years.
    """
    body = body.copy()
    if not pd.api.types.is_datetime64_any_dtype(body["TIMESTAMP"]):
        body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"], errors="coerce")
    try:
        if getattr(body["TIMESTAMP"].dt, "tz", None) is not None:
            body["TIMESTAMP"] = body["TIMESTAMP"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    
    body["year"] = body["TIMESTAMP"].dt.year
    
    yearly = body.groupby("year").agg({
        "LINK_SENTIMENT": ["count", "mean", lambda x: (x == -1).sum() / len(x)]
    })
    yearly.columns = ["total_links", "avg_sentiment", "negative_ratio"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Total links by year
    axes[0].bar(yearly.index, yearly["total_links"], color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Number of Links")
    axes[0].set_title("Total Links by Year", fontweight='bold')
    for i, v in enumerate(yearly["total_links"]):
        axes[0].text(yearly.index[i], v + max(yearly["total_links"])*0.02, f'{v:,.0f}', 
                    ha='center', fontsize=9)
    
    # Average sentiment by year
    colors = ["green" if s > 0.85 else "orange" if s > 0.8 else "red" for s in yearly["avg_sentiment"]]
    axes[1].bar(yearly.index, yearly["avg_sentiment"], color=colors, alpha=0.7)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Average Sentiment")
    axes[1].set_title("Average Sentiment by Year", fontweight='bold')
    axes[1].set_ylim(0.7, 1.0)
    axes[1].axhline(yearly["avg_sentiment"].mean(), color='black', linestyle='--', alpha=0.5)
    
    # Negative ratio by year
    axes[2].bar(yearly.index, yearly["negative_ratio"] * 100, color="red", alpha=0.7)
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Negative Link Ratio (%)")
    axes[2].set_title("Negative Link Ratio by Year", fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return yearly


def plot_reciprocal_links(body: pd.DataFrame, min_links: int = 5, save_path: str = None):
    """
    Analyze reciprocal linking patterns (A links to B and B links to A).
    """
    # Count links in each direction
    forward = body.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]).size().reset_index(name="forward")
    backward = body.groupby(["TARGET_SUBREDDIT", "SOURCE_SUBREDDIT"]).size().reset_index(name="backward")
    backward.columns = ["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "backward"]
    
    # Merge to find reciprocal pairs
    merged = forward.merge(backward, on=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"], how="outer").fillna(0)
    merged["total"] = merged["forward"] + merged["backward"]
    merged["reciprocity"] = np.minimum(merged["forward"], merged["backward"]) / np.maximum(merged["forward"], merged["backward"])
    
    # Filter by minimum links
    significant = merged[merged["total"] >= min_links * 2]
    
    # Find most reciprocal pairs
    reciprocal = significant[significant["reciprocity"] > 0.3].nlargest(20, "total")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribution of reciprocity
    axes[0].hist(significant["reciprocity"], bins=30, color="purple", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Reciprocity Score")
    axes[0].set_ylabel("Number of Subreddit Pairs")
    axes[0].set_title("Distribution of Link Reciprocity\n(0 = one-way, 1 = fully reciprocal)", fontweight='bold')
    axes[0].axvline(significant["reciprocity"].mean(), color='red', linestyle='--', 
                   label=f'Mean: {significant["reciprocity"].mean():.2f}')
    axes[0].legend()
    
    # Top reciprocal pairs
    if len(reciprocal) > 0:
        labels = [f"r/{row['SOURCE_SUBREDDIT'][:15]}\n↔ r/{row['TARGET_SUBREDDIT'][:15]}" 
                 for _, row in reciprocal.head(10).iterrows()]
        axes[1].barh(range(min(10, len(reciprocal))), reciprocal["total"].head(10).values[::-1], color="purple", alpha=0.7)
        axes[1].set_yticks(range(min(10, len(reciprocal))))
        axes[1].set_yticklabels(labels[::-1], fontsize=8)
        axes[1].set_xlabel("Total Bidirectional Links")
        axes[1].set_title("Top Reciprocal Subreddit Pairs", fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return reciprocal


def plot_political_activity_timeline(body: pd.DataFrame, pol_sb, save_path: str = None, focus_label: str = "Event"):
    """
    Plot political vs non-political activity over time.
    """
    pol_set = _coerce_item_set(pol_sb)
    
    body = body.copy()
    if not pd.api.types.is_datetime64_any_dtype(body["TIMESTAMP"]):
        body["TIMESTAMP"] = pd.to_datetime(body["TIMESTAMP"], errors="coerce")
    try:
        if getattr(body["TIMESTAMP"].dt, "tz", None) is not None:
            body["TIMESTAMP"] = body["TIMESTAMP"].dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    
    body["is_focus"] = body.apply(
        lambda x: str(x["SOURCE_SUBREDDIT"]).lower() in pol_set or str(x["TARGET_SUBREDDIT"]).lower() in pol_set,
        axis=1
    )
    body["month"] = body["TIMESTAMP"].dt.to_period("M")
    
    monthly = body.groupby(["month", "is_focus"]).size().unstack(fill_value=0)
    monthly.index = monthly.index.to_timestamp()
    monthly.columns = [f"Non-{focus_label}", focus_label]
    
    # Calculate political ratio
    monthly["focus_ratio"] = monthly["Political"] / (monthly["Political"] + monthly["Non-Political"]) * 100
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Stacked area chart
    axes[0].fill_between(monthly.index, 0, monthly["Non-Political"], alpha=0.5, color="steelblue", label="Non-Political")
    axes[0].fill_between(monthly.index, monthly["Non-Political"], monthly["Non-Political"] + monthly["Political"], 
                        alpha=0.5, color="coral", label="Political")
    axes[0].set_ylabel("Number of Links")
    axes[0].set_title(f"Monthly Activity: {focus_label} vs Non-{focus_label} Links", fontsize=12, fontweight='bold')
    axes[0].legend()
    
    # Political ratio over time
    axes[1].plot(monthly.index, monthly["focus_ratio"], color="coral", linewidth=2)
    axes[1].fill_between(monthly.index, monthly["focus_ratio"], alpha=0.3, color="coral")
    axes[1].set_ylabel("Political Link Share (%)")
    axes[1].set_xlabel("Date")
    axes[1].set_title(f"Share of Links Involving {focus_label} Subreddits", fontsize=12, fontweight='bold')
    axes[1].axhline(monthly["focus_ratio"].mean(), color='black', linestyle='--', 
                   label=f'Mean: {monthly["focus_ratio"].mean():.1f}%')
    axes[1].legend()
    
    # Add event markers
    events = [("2016-06-23", "Brexit"), ("2016-11-08", "Election")]
    for ax in axes:
        for date_str, label in events:
            ax.axvline(pd.to_datetime(date_str), color="red", linestyle=":", alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return monthly



def plot_network_visualization(body: pd.DataFrame, pol_sb: pd.DataFrame, top_n: int = 75, save_path: str = None):
    """
    Visualize the network structure of top interactions with community detection.
    
    Args:
        body: DataFrame with links
        pol_sb: DataFrame with political subreddits
        top_n: Number of top nodes to visualize (75-100 recommended)
        save_path: Path to save the image
    """
    from matplotlib.lines import Line2D
    import networkx.algorithms.community as nx_comm
    
    # 1. Filter for top interactions
    # Group by pair to get weights
    edge_counts = body.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]).size().reset_index(name="weight")
    
    # Filter nodes by degree (total volume) to get the most important ones
    source_vol = body["SOURCE_SUBREDDIT"].value_counts()
    target_vol = body["TARGET_SUBREDDIT"].value_counts()
    total_vol = source_vol.add(target_vol, fill_value=0)
    top_nodes = total_vol.nlargest(top_n).index
    
    # Keep edges only between top nodes
    filtered_edges = edge_counts[
        (edge_counts["SOURCE_SUBREDDIT"].isin(top_nodes)) & 
        (edge_counts["TARGET_SUBREDDIT"].isin(top_nodes))
    ]
    
    # Create graph
    G = nx.DiGraph()
    for _, row in filtered_edges.iterrows():
        G.add_edge(row["SOURCE_SUBREDDIT"], row["TARGET_SUBREDDIT"], weight=row["weight"])
    
    # Keep largest component
    largest_cc = max(nx.weakly_connected_components(G), key=len)
    G = G.subgraph(largest_cc)
    
    # 2. Community Detection (Louvain or Greedy Modularity)
    try:
        communities = nx_comm.louvain_communities(G, weight='weight', seed=42)
    except AttributeError:
        # Fallback for older networkx versions
        communities = nx_comm.greedy_modularity_communities(G, weight='weight')
        
    # Map node to community color
    community_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx
            
    # Select top k communities to color distinctively, others gray
    top_comms = sorted(communities, key=len, reverse=True)[:5]
    top_comm_ids = {community_map[list(c)[0]] for c in top_comms}
    
    # Generate colors using a colormap
    cmap = plt.cm.get_cmap('tab10')
    
    node_colors = []
    node_sizes = []
    labels = {}
    
    # Calculate degrees for sizing
    degrees = dict(G.degree(weight="weight"))
    max_degree = max(degrees.values()) if degrees else 1
    
    for node in G.nodes():
        # Color by community
        comm_id = community_map.get(node)
        if comm_id in top_comm_ids:
            # Map comm_id to a color index
            color_idx = list(top_comm_ids).index(comm_id)
            node_colors.append(cmap(color_idx))
        else:
            node_colors.append("#aaaaaa") # Gray for small communities
        
        # Size: based on degree
        size = 100 + (degrees[node] / max_degree) * 2000
        node_sizes.append(size)
        
        labels[node] = f"r/{node}"

    # 3. Layout and Plot
    pos = nx.spring_layout(G, k=0.8, seed=42, iterations=50)
    
    plt.figure(figsize=(16, 14))
    
    # Draw edges with varying width
    edges = G.edges(data=True)
    weights = [d['weight'] for u, v, d in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [0.5 + (w / max_weight) * 4 for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.2, 
                          arrowsize=10, connectionstyle='arc3,rad=0.1')
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, linewidths=1, edgecolors='white')
    
    # Labels with outline for readability
    import matplotlib.patheffects as PathEffects
    text_items = nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold")
    for t in text_items.values():
        t.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])
    
    plt.title(f"Network Structure of Top {len(G.nodes())} Subreddits (Modularity Clustering)", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved network graph to {save_path}")
    plt.show()
    
    return G


def plot_comparative_sentiment_distribution(body: pd.DataFrame, pol_sb, save_path: str = None, focus_label: str = "Event"):
    """
    Compare sentiment distributions between Political and Non-Political subreddits.
    Replacing simple histograms with informative violin plots.
    """
    pol_set = _coerce_item_set(pol_sb)
    
    df = body.copy()
    # Tag Source Type
    df["Community Type"] = df["SOURCE_SUBREDDIT"].str.lower().apply(
        lambda x: focus_label if x in pol_set else f"Non-{focus_label}"
    )
    
    plt.figure(figsize=(12, 6))
    
    # Violin plot with split distribution if we had a hue (e.g. source vs target), 
    # but here we compare categories directly.
    sns.violinplot(data=df, x="LINK_SENTIMENT", y="Community Type", 
                   palette={focus_label: "coral", f"Non-{focus_label}": "steelblue"},
                   inner="quartile", alpha=0.7)
    
    # Add vertical lines for neutral, positive, negative thresholds
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f"Sentiment Distribution: {focus_label} vs Non-{focus_label} Communities", fontsize=14, fontweight='bold')
    plt.xlabel("Sentiment Score (-1 to +1)")
    plt.ylabel("Source Community Type")
    
    # Add statistical summary annotation
    pol_mean = df[df["Community Type"]==focus_label]["LINK_SENTIMENT"].mean()
    non_mean = df[df["Community Type"]==f"Non-{focus_label}"]["LINK_SENTIMENT"].mean()
    
    plt.text(0.8, 0, f"{focus_label} Mean: {pol_mean:.3f}", color='maroon', fontweight='bold', ha='center')
    plt.text(0.8, 1, f"Non-{focus_label} Mean: {non_mean:.3f}", color='navy', fontweight='bold', ha='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
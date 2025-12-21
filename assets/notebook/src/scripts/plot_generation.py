import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from src.utils.evaluation_utils import *
from src.data.sentiment_analysis import *
from src.utils.data_utils import *
from sklearn.manifold import TSNE
import plotly.express as px

import colorsys
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from src.utils.data_utils import ensure_timestamp_utc_naive


def _coerce_item_set(x):
    """Accept DataFrame/Series/list/set and return a lowercase set of subreddit names."""
    if x is None:
        return set()
    if isinstance(x, set):
        return {str(s).lower() for s in x}
    if isinstance(x, (list, tuple)):
        return {str(s).lower() for s in x}
    if hasattr(x, "columns") and "item" in getattr(x, "columns", []):
        return set(x["item"].astype(str).str.lower())
    try:
        import pandas as pd
        return set(pd.Series(x).astype(str).str.lower())
    except Exception:
        return {str(x).lower()}

def plot_body_vs_title_sentiment(body, title, subreddits_emb, pol_sb):
    body_source_pol, body_target_pol, body_pol_nonpol = categorization(body, {"item": list(_coerce_item_set(pol_sb))})
    title_source_pol, title_target_pol, title_pol_nonpol = categorization(title, {"item": list(_coerce_item_set(pol_sb))})
    subreddits_emb_pol = categorization(subreddits_emb, {"item": pol_sb["item"]}, xor=False)

    body_positive_ratio = positive_sentiment_ratio(body)
    title_positive_ratio = positive_sentiment_ratio(title)
    body_pol_non_pol_positive_ratio = positive_sentiment_ratio(body_pol_nonpol)
    title_pol_non_pol_positive_ratio = positive_sentiment_ratio(title_pol_nonpol)
    body_pol_source_positive_ratio = positive_sentiment_ratio(body_source_pol)
    title_pol_source_positive_ratio = positive_sentiment_ratio(title_source_pol)
    body_pol_target_positive_ratio = positive_sentiment_ratio(body_target_pol)
    title_pol_target_positive_ratio = positive_sentiment_ratio(title_target_pol)
    body_negative_ratio = 1-body_positive_ratio
    title_negative_ratio = 1-title_positive_ratio
    body_pol_non_pol_negative_ratio = 1-body_pol_non_pol_positive_ratio
    title_pol_non_pol_negative_ratio = 1-title_pol_non_pol_positive_ratio
    body_pol_source_negative_ratio = 1-body_pol_source_positive_ratio
    title_pol_source_negative_ratio = 1-title_pol_source_positive_ratio
    body_pol_target_negative_ratio = 1-body_pol_target_positive_ratio
    title_pol_target_negative_ratio = 1-title_pol_target_positive_ratio

    print(f"Body positive link ratio: {body_positive_ratio}")
    print(f"Body negative link ratio: {body_negative_ratio}\n")

    print(f"Title positive link ratio: {title_positive_ratio}")
    print(f"Title negative link ratio: {title_negative_ratio}\n\n")

    print(f"Non Political Body positive link ratio: {body_pol_non_pol_positive_ratio}")
    print(f"Non Political Body negative link ratio: {body_pol_non_pol_negative_ratio}\n")

    print(f"Non Political Title positive link ratio: {title_pol_non_pol_positive_ratio}")
    print(f"Non Political Title negative link ratio: {title_pol_non_pol_negative_ratio}\n\n")

    print(f"Political Body source positive link ratio: {body_pol_source_positive_ratio}")
    print(f"Political Body source negative link ratio: {body_pol_source_negative_ratio}\n")

    print(f"Political Body target positive link ratio: {body_pol_target_positive_ratio}")
    print(f"Political Body target negative link ratio: {body_pol_target_negative_ratio}\n\n")

    print(f"Political Title source positive link ratio: {title_pol_source_positive_ratio}")
    print(f"Political Title source negative link ratio: {title_pol_source_negative_ratio}\n")

    print(f"Political Title target positive link ratio: {title_pol_target_positive_ratio}")
    print(f"Political Title target negative link ratio: {title_pol_target_negative_ratio}\n")



    df = pd.DataFrame([
        {"Section": "Body",  "Sentiment": "Positive", "Ratio": body_positive_ratio},
        {"Section": "Body",  "Sentiment": "Negative", "Ratio": body_negative_ratio},
        {"Section": "Title", "Sentiment": "Positive", "Ratio": title_positive_ratio},
        {"Section": "Title", "Sentiment": "Negative", "Ratio": title_negative_ratio},
        {"Section": "Body Political x Non-Political",  "Sentiment": "Positive", "Ratio": body_pol_non_pol_positive_ratio},
        {"Section": "Body Political x Non-Political",  "Sentiment": "Negative", "Ratio": body_pol_non_pol_negative_ratio},
        {"Section": "Title Political x Non-Political", "Sentiment": "Positive", "Ratio": title_pol_non_pol_positive_ratio},
        {"Section": "Title Political x Non-Political", "Sentiment": "Negative", "Ratio": title_pol_non_pol_negative_ratio},
    ])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="Section", y="Ratio", hue="Sentiment")

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Share of links")

    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.1%}",
                    (p.get_x() + p.get_width(), h),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                    textcoords="offset points", clip_on=False)


    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    plt.title("Positive vs. Negative Link Ratios in Political x Non-political (Body vs. Title)")

    leg = plt.legend(title="Sentiment", loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_timeline():
    def adjust_lightness(color, amount=1.0):
        c = mcolors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*c)
        l = max(0, min(1, amount * l))
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        return (r2, g2, b2)

    events = [
        # Political/Governance
        ("Annexation of Crimea", datetime(2014,2,22), datetime(2014,3,18), "Political/Governance"),
        ("Donald Trump Inauguration", datetime(2017,1,20), datetime(2017,1,20), "Political/Governance"),
        ("US Presidential Campaign", datetime(2015,6,16), datetime(2016,11,8), "Political/Governance"),
        ("Brexit Campaign & Referendum", datetime(2016,2,20), datetime(2016,6,23), "Political/Governance"),

        # Health/Public Health
        ("Ebola Outbreak", datetime(2014,3,1), datetime(2016,1,14), "Health/Public Health"),

        # Disasters
        ("MH370 Disappearance", datetime(2014,3,8), datetime(2014,3,8), "Disasters"),
        ("Cyclone Pam (Vanuatu)", datetime(2015,3,13), datetime(2015,3,14), "Disasters"),
        ("Nepal Earthquake", datetime(2015,4,25), datetime(2015,4,25), "Disasters"),
        ("Fort McMurray Wildfire", datetime(2016,5,1), datetime(2016,5,31), "Disasters"),

        # Diplomacy
        ("Iran Nuclear Deal (JCPOA) Signed", datetime(2015,7,14), datetime(2015,7,14), "Diplomacy"),
        ("Obama Visits Cuba", datetime(2016,3,20), datetime(2016,3,20), "Diplomacy"),
        ("Colombia–FARC Peace Accord", datetime(2016,9,26), datetime(2016,9,26), "Diplomacy"),

        # Conflict/Security
        ("El Chapo Arrest", datetime(2014,2,22), datetime(2014,2,22), "Conflict/Security"),
        ("Turkish Coup Attempt", datetime(2016,7,15), datetime(2016,7,15), "Conflict/Security"),
        ("Rise of ISIS / Global Terrorism", datetime(2014,6,1), datetime(2017,4,1), "Conflict/Security"),

        # Climate / Environment
        ("UN Climate Summit (NYC)", datetime(2014,9,23), datetime(2014,9,23), "Climate/Environment"),
        ("COP21 Climate Conference", datetime(2015,11,30), datetime(2015,12,12), "Climate/Environment"),
        ("COP22 Marrakech", datetime(2016,11,7), datetime(2016,11,18), "Climate/Environment"),
    ]

    #Colors
    category_base = {
        "Political/Governance": "#1f77b4",
        "Conflict/Security":    "#ff7f0e",
        "Disasters":            "#2ca02c",
        "Diplomacy":            "#d62728",
        "Climate/Environment":  "#9467bd",
        "Health/Public Health": "#8c564b",
    }

    lanes = sorted({cat for _, _, _, cat in events})
    lane_y = {cat: i for i, cat in enumerate(lanes)}

    #Color variations
    cat_counts = {}
    for _, _, _, cat in events:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    cat_indices = {cat: 0 for cat in cat_counts}

    fig, ax = plt.subplots(figsize=(14, 2.5))
    legend_handles, legend_labels = [], []

    for label, start, end, category in events:
        y = lane_y[category]
        base_col = category_base[category]
        idx = cat_indices[category]
        total = cat_counts[category]
        variation = 0.75 + (idx / max(1, total - 1)) * 0.5 if total > 1 else 0.95
        color_variant = adjust_lightness(base_col, variation)

        bar_height = 1.0
        y_center = y

        # Brexit overlaps US campaign
        if label == "Brexit Campaign & Referendum":
            bar_height = 0.5
            y_center = y - 0.25

        duration_days = (end - start).days

        if duration_days <= 32:
            marker = ax.plot(start, y_center, marker='o', color=color_variant, markersize=6, zorder=2)[0]
            legend_handles.append(marker)
            legend_labels.append(label)
        else:
            ax.broken_barh(
                [(mdates.date2num(start), mdates.date2num(end) - mdates.date2num(start))],
                (y_center - bar_height/2, bar_height),
                facecolors=color_variant,
                edgecolors='black',
                linewidth=0.5,
                zorder=2
            )
            proxy = mpatches.Rectangle((0, 0), 1, 1, facecolor=color_variant, edgecolor='black')
            legend_handles.append(proxy)
            legend_labels.append(label)

        cat_indices[category] += 1

    ax.set_yticks(list(lane_y.values()))
    ax.set_yticklabels(lanes)
    ax.set_ylim(-0.5, len(lanes) - 0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.title("Global Events Timeline (Jan 2014 – Apr 2017)")

    # Set axis scale
    all_dates = [d for _, s, e, _ in events for d in (s, e)]
    left_edge  = datetime(min(d.year for d in all_dates), 1, 1)
    right_edge = max(all_dates) + timedelta(days=1) 
    ax.set_xlim(left_edge, right_edge)
    ax.margins(x=0) 

    plt.tight_layout()

    x_min, x_max = ax.get_xlim()
    lane_color = "0.65"
    year_color = "0.7"
    lane_lw = 0.9
    year_lw = 0.9

    for y in lane_y.values():
        ax.hlines(y, x_min, x_max, linestyles=":", linewidth=lane_lw, color=lane_color, zorder=0)

    start_year = int(mdates.num2date(x_min).year)
    end_year   = int(mdates.num2date(x_max).year)
    for year in range(start_year, end_year + 1):
        x = mdates.date2num(datetime(year, 1, 1))
        if x_min <= x <= x_max:
            ax.axvline(x, linestyle="--", linewidth=year_lw, color=year_color, zorder=0)

    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.show()


def plot_top_neg_liwc(body, title, pol_sb):
    sns.set_theme(style="whitegrid")

    _, _, body_pol_nonpol = categorization(body, {"item": list(_coerce_item_set(pol_sb))})
    _, _, title_pol_nonpol = categorization(title, {"item": list(_coerce_item_set(pol_sb))})

    body_pol_non_pol_prop = properties_into_dataframe_columns(body_pol_nonpol, col_name='PROPERTIES')
    title_pol_non_pol_prop = properties_into_dataframe_columns(title_pol_nonpol, col_name='PROPERTIES')
    
    body_pol_non_pol_liwc_ranking_full = df_liwc_ranking(body_pol_non_pol_prop, sentiment_value="neg", ascending=False)
    title_pol_non_pol_liwc_ranking_full = df_liwc_ranking(title_pol_non_pol_prop, sentiment_value="neg", ascending=False)

    # Union of categories that ranked in either top-20
    cats_union = list(set(body_pol_non_pol_liwc_ranking_full.head(20).index).union(title_pol_non_pol_liwc_ranking_full.head(20).index))

    # Sort by the average (body+title)/2 so the most important overall are at the top
    avg_score = pd.Series(
        {c: (float(body_pol_non_pol_liwc_ranking_full.get(c, np.nan)) + float(title_pol_non_pol_liwc_ranking_full.get(c, np.nan))) / 2 for c in cats_union}
    )
    cats_sorted = list(avg_score.sort_values(ascending=True).index)

    # Build long-form dataframe for plotting
    plot_df = pd.DataFrame({
        "LIWC": np.repeat(cats_sorted, 2),
        "Section": ["Body", "Title"] * len(cats_sorted),
        "Mean": sum([
            [float(body_pol_non_pol_liwc_ranking_full[c]), float(title_pol_non_pol_liwc_ranking_full[c])] for c in cats_sorted
        ], [])
    })

    # Plot: horizontal grouped bars
    plt.figure(figsize=(8, max(4, 0.35 * len(cats_sorted))))
    ax = sns.barplot(
        data=plot_df, x="Mean", y="LIWC", hue="Section",
        dodge=True
    )

    # Cosmetics
    ax.set_title("Top Negative LIWC (mean among negative LINK_SENTIMENT) — Body vs Title")
    ax.set_xlabel("Mean LIWC score")
    ax.set_ylabel("LIWC Category")
    ax.legend(title="", loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_top_pos_liwc(body, title, pol_sb):
    sns.set_theme(style="whitegrid")

    _, _, body_pol_nonpol = categorization(body, {"item": list(_coerce_item_set(pol_sb))})
    _, _, title_pol_nonpol = categorization(title, {"item": list(_coerce_item_set(pol_sb))})

    body_pol_non_pol_prop = properties_into_dataframe_columns(body_pol_nonpol, col_name='PROPERTIES')
    title_pol_non_pol_prop = properties_into_dataframe_columns(title_pol_nonpol, col_name='PROPERTIES')

    body_pol_non_pol_liwc_ranking_full = df_liwc_ranking(body_pol_non_pol_prop, sentiment_value="pos", ascending=False)
    title_pol_non_pol_liwc_ranking_full = df_liwc_ranking(title_pol_non_pol_prop, sentiment_value="pos", ascending=False)

    # Union of categories that ranked in either top-20
    cats_union = list(set(body_pol_non_pol_liwc_ranking_full.head(20).index).union(title_pol_non_pol_liwc_ranking_full.head(20).index))

    # Sort by the average (body+title)/2 so the most important overall are at the top
    avg_score = pd.Series(
        {c: (float(body_pol_non_pol_liwc_ranking_full.get(c, np.nan)) + float(title_pol_non_pol_liwc_ranking_full.get(c, np.nan))) / 2 for c in cats_union}
    )
    cats_sorted = list(avg_score.sort_values(ascending=True).index)

    plot_df = pd.DataFrame({
        "LIWC": np.repeat(cats_sorted, 2),
        "Section": ["Body", "Title"] * len(cats_sorted),
        "Mean": sum([
            [float(body_pol_non_pol_liwc_ranking_full[c]), float(title_pol_non_pol_liwc_ranking_full[c])] for c in cats_sorted
        ], [])
    })

    plt.figure(figsize=(8, max(4, 0.35 * len(cats_sorted))))
    ax = sns.barplot(
        data=plot_df, x="Mean", y="LIWC", hue="Section",
        dodge=True
    )

    # Cosmetics
    ax.set_title("Top Positive LIWC (mean among positive LINK_SENTIMENT) — Body vs Title")
    ax.set_xlabel("Mean LIWC score")
    ax.set_ylabel("LIWC Category")
    ax.legend(title="", loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_pos_vs_neg_liwc(body, title, pol_sb):
    sns.set_theme(style="whitegrid")

    _, _, body_pol_nonpol = categorization(body, {"item": list(_coerce_item_set(pol_sb))})

    body_pol_non_pol_prop = properties_into_dataframe_columns(body_pol_nonpol, col_name='PROPERTIES')

    body_pol_non_pol_liwc_ranking_full_pos = df_liwc_ranking(body_pol_non_pol_prop, sentiment_value="pos", ascending=False)
    body_pol_non_pol_liwc_ranking_full_neg = df_liwc_ranking(body_pol_non_pol_prop, sentiment_value="neg", ascending=False)

    cats_union = body_pol_non_pol_liwc_ranking_full_pos.index.union(body_pol_non_pol_liwc_ranking_full_neg.index)

    both = pd.DataFrame(
        {"Body positive": body_pol_non_pol_liwc_ranking_full_pos, "Body negative": body_pol_non_pol_liwc_ranking_full_neg}
    ).reindex(cats_union)

    cats_sorted = both.mean(axis=1, skipna=True).sort_values(ascending=False).index

    plot_df = (
        both.loc[cats_sorted]
            .reset_index(names="LIWC")
            .melt(id_vars="LIWC", var_name="Section", value_name="Mean")
    )

    plt.figure(figsize=(9, max(4, 0.35 * len(cats_sorted))))
    ax = sns.barplot(
        data=plot_df, x="Mean", y="LIWC", hue="Section", dodge=True
    )

    ax.set_title("Body LIWC means — Positive vs Negative LINK_SENTIMENT")
    ax.set_xlabel("Mean LIWC score")
    ax.set_ylabel("LIWC Category")
    ax.legend(title="", loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_emb_proj_TSNE(subreddits_emb, pol_sb, inter):
    subreddits_mask = subreddits_emb[0].isin(inter)
    subreddits = subreddits_emb[0][subreddits_mask]
    X_2 = subreddits_emb[subreddits_mask].drop(columns=[0]).values
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_2_2d = tsne.fit_transform(X_2)

    subreddits_emb_mask2 = subreddits_emb[0][subreddits_mask].isin(list(_coerce_item_set(pol_sb)))
    cat = np.where(subreddits_emb_mask2, "Political", "Non-political")

    fig = px.scatter(
        x=X_2_2d[:, 0], y=X_2_2d[:, 1],
        color=cat,
        hover_name=subreddits,
        title="Subreddit Embedding Cloud (Interactive)",
        opacity=0.7
    )
    fig.show()


def plot_emb_proj_TSNE_events(subreddits_emb, events_lexicon, inter=None, notebook=False):
    """
    Plot t-SNE projection of subreddit embeddings colored by event categories.
    
    Parameters
    ----------
    subreddits_emb : pd.DataFrame
        DataFrame where column 0 contains subreddit names and remaining columns
        contain embedding values.
    events_lexicon : dict
        Dictionary loaded from event_related_subreddits_lexicon.json where keys
        are event names and values contain 'matched_subreddits' lists.
    inter : set or list, optional
        Set of subreddits to include in the plot. If None, automatically uses
        the intersection of all event-related subreddits and available embeddings.
    notebook : bool, optional
        If True, outputs a static matplotlib plot (non-interactive).
        If False (default), outputs an interactive Plotly plot.
    
    Returns
    -------
    None
        Displays the plot.
    """
    # Build a mapping from subreddit -> event name
    subreddit_to_event = {}
    all_event_subreddits = set()
    for event_name, event_data in events_lexicon.items():
        matched = event_data.get("matched_subreddits", [])
        for sr in matched:
            sr_lower = str(sr).lower()
            all_event_subreddits.add(sr_lower)
            # If subreddit is matched by multiple events, use the first one found
            if sr_lower not in subreddit_to_event:
                subreddit_to_event[sr_lower] = event_name
    
    # If inter is not provided, use event-related subreddits that have embeddings
    if inter is None:
        available_subreddits = set(subreddits_emb[0].str.lower())
        inter = all_event_subreddits & available_subreddits
    
    # Filter subreddits to those in 'inter'
    subreddits_mask = subreddits_emb[0].str.lower().isin(inter)
    subreddits = subreddits_emb[0][subreddits_mask]
    X_2 = subreddits_emb[subreddits_mask].drop(columns=[0]).values
    
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    X_2_2d = tsne.fit_transform(X_2)
    
    # Map each subreddit to its event category (or "Other")
    categories = []
    for sr in subreddits:
        sr_lower = str(sr).lower()
        categories.append(subreddit_to_event.get(sr_lower, "Other"))
    
    if notebook:
        # Static matplotlib plot
        unique_cats = list(set(categories))
        color_map = {cat: plt.cm.tab20(i / len(unique_cats)) for i, cat in enumerate(unique_cats)}
        colors = [color_map[cat] for cat in categories]
        
        plt.figure(figsize=(12, 8))
        for cat in unique_cats:
            mask = [c == cat for c in categories]
            plt.scatter(
                X_2_2d[mask, 0], X_2_2d[mask, 1],
                c=[color_map[cat]], label=cat, alpha=0.7, s=30
            )
        plt.title("Subreddit Embedding Cloud by Event")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Event", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.show()
    else:
        # Interactive Plotly plot
        fig = px.scatter(
            x=X_2_2d[:, 0], y=X_2_2d[:, 1],
            color=categories,
            hover_name=subreddits,
            title="Subreddit Embedding Cloud by Event (Interactive)",
            opacity=0.7
        )
        fig.update_layout(legend_title_text="Event")
        fig.show()

def plot_temporal_activity(body, title, freq="MS"):
    """
    Plot temporal hyperlink activity for body vs title.

    Parameters
    ----------
    body, title : pd.DataFrame
        Must have a 'TIMESTAMP' column.
    freq : str
        Pandas resample frequency. "MS" = Month Start (monthly).

    Returns
    -------
    pd.DataFrame
        Two-column dataframe with counts for body and title per period.
    """
    body = body.copy()
    title = title.copy()
    ensure_timestamp_utc_naive(body)
    ensure_timestamp_utc_naive(title)

    b = body.set_index("TIMESTAMP").resample(freq).size().rename("body")
    t = title.set_index("TIMESTAMP").resample(freq).size().rename("title")
    df = pd.concat([b, t], axis=1).fillna(0)

    ax = df.plot()
    ax.set_title(f"Hyperlink activity ({freq}) — body vs title")
    ax.set_ylabel("links")
    plt.tight_layout()
    plt.show()

    return df
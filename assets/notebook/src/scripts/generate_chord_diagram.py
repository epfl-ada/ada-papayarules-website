#!/usr/bin/env python3
"""
Interactive Chord Diagram Generator for Event-Subreddit Links

Generates interactive chord diagrams showing links between event-related 
subreddits and external subreddits during event time periods.

Usage:
    python src/scripts/generate_chord_diagram.py --event "Brexit Referendum"
    python src/scripts/generate_chord_diagram.py --all
    python src/scripts/generate_chord_diagram.py --list
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, FancyBboxPatch
from matplotlib.path import Path as MplPath
import matplotlib.patches as mpatches_path
from src.utils.chords_json import canonical_key



def load_event_lexicon(path: str = "data/event_related_subreddits_lexicon.json") -> dict:
    """Load the event-related subreddits lexicon."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hyperlinks_data() -> pd.DataFrame:
    """Load and combine both body and title hyperlinks data."""
    body_path = "data/soc-redditHyperlinks-body.tsv"
    title_path = "data/soc-redditHyperlinks-title.tsv"
    
    print("Loading body hyperlinks data...")
    body = pd.read_csv(body_path, sep="\t")
    print(f"  Loaded {len(body):,} body links")
    
    print("Loading title hyperlinks data...")
    title = pd.read_csv(title_path, sep="\t")
    print(f"  Loaded {len(title):,} title links")
    
    # Combine both datasets
    combined = pd.concat([body, title], ignore_index=True)
    combined["TIMESTAMP"] = pd.to_datetime(combined["TIMESTAMP"])
    print(f"  Combined: {len(combined):,} total links")
    
    return combined

def chord_already_exists(
    event_title: str,
    *,
    chords_dir: Path,
) -> Path | None:
    """
    Return the existing chord file path if one already exists for this event,
    otherwise None.
    """
    key = canonical_key(event_title)

    for p in chords_dir.glob("chord_*.html"):
        stem = p.stem[len("chord_"):]
        if canonical_key(stem) == key:
            return p

    return None



def get_event_external_links(
    df: pd.DataFrame,
    event_subreddits: set,
    start_date: str,
    end_date: str,
    top_n: int = 30
) -> tuple:
    """
    Get links between event subreddits (as a single group) and external subreddits.
    
    All event-related subreddits are aggregated into one category.
    
    Returns:
        Tuple of (external_link_counts dict, external_sentiments dict, total_links int)
        - external_link_counts: dict mapping external subreddit -> link count
        - external_sentiments: dict mapping external subreddit -> average sentiment (-1 to 1)
    """
    # Filter to time window
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (df["TIMESTAMP"] >= start) & (df["TIMESTAMP"] <= end)
    window_df = df[mask]
    
    if len(window_df) == 0:
        return None, None, 0
    
    # Normalize subreddit names
    event_subs_lower = {s.lower() for s in event_subreddits}
    
    # Count external links and track sentiment (one side is event-related, other is not)
    external_counts = defaultdict(int)
    external_sentiment_sums = defaultdict(float)
    total_links = 0
    
    for _, row in window_df.iterrows():
        source = row["SOURCE_SUBREDDIT"].lower()
        target = row["TARGET_SUBREDDIT"].lower()
        sentiment = row.get("LINK_SENTIMENT", 0)  # 1 for positive, -1 for negative
        
        source_is_event = source in event_subs_lower
        target_is_event = target in event_subs_lower
        
        # Only keep links where exactly one side is event-related
        if source_is_event and not target_is_event:
            external_counts[target] += 1
            external_sentiment_sums[target] += sentiment
            total_links += 1
        elif not source_is_event and target_is_event:
            external_counts[source] += 1
            external_sentiment_sums[source] += sentiment
            total_links += 1
    
    if not external_counts:
        return None, None, 0
    
    # Get top N external subreddits by link count
    sorted_external = sorted(external_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_external_counts = dict(sorted_external)
    
    # Calculate average sentiment for top external subreddits
    top_external_sentiments = {
        sub: external_sentiment_sums[sub] / external_counts[sub]
        for sub in top_external_counts.keys()
    }
    
    return top_external_counts, top_external_sentiments, total_links


def sentiment_to_color(sentiment: float, opacity: float = 0.6) -> str:
    """
    Convert sentiment value (-1 to 1) to a color.
    
    Positive sentiment -> Dark Green
    Neutral (0) -> Black/Dark Gray
    Negative sentiment -> Red
    
    Colors fade smoothly: less intense sentiment = closer to black.
    """
    # Clamp sentiment to [-1, 1]
    sentiment = max(-1, min(1, sentiment))
    
    # Neutral color (black/dark gray)
    neutral_r, neutral_g, neutral_b = 30, 30, 30
    
    if sentiment >= 0:
        # Positive: interpolate from black to dark green
        # Target: dark green (0, 100, 0)
        target_r, target_g, target_b = 0, 140, 40
        t = sentiment  # 0 = neutral, 1 = full green
        r = int(neutral_r + (target_r - neutral_r) * t)
        g = int(neutral_g + (target_g - neutral_g) * t)
        b = int(neutral_b + (target_b - neutral_b) * t)
    else:
        # Negative: interpolate from black to red
        # Target: red (200, 30, 30)
        target_r, target_g, target_b = 200, 30, 30
        t = abs(sentiment)  # 0 = neutral, 1 = full red
        r = int(neutral_r + (target_r - neutral_r) * t)
        g = int(neutral_g + (target_g - neutral_g) * t)
        b = int(neutral_b + (target_b - neutral_b) * t)
    
    return f"rgba({r}, {g}, {b}, {opacity})"


def create_chord_diagram(
    external_counts: dict,
    external_sentiments: dict,
    total_links: int,
    event_name: str,
    num_event_subs: int,
    output_path: str,
    notebook: bool = False
):
    """
    Create an interactive chord diagram using Plotly.
    
    Each node is an ARC SEGMENT on the circle perimeter.
    Arc length is proportional to the number of connections.
    """
    event_label = f"Event Subreddits ({num_event_subs})"
    external_subs = list(external_counts.keys())
    
    # Node colors
    event_color = "rgba(31, 119, 180, 0.9)"  # Blue
    external_color = "rgba(255, 127, 14, 0.8)"  # Orange
    
    # Calculate weights for arc sizes
    # Event weight = connections normalized by sqrt of subreddits (balanced scaling)
    event_weight = total_links / np.sqrt(num_event_subs)
    print(f"Event weight: {event_weight}")
    print(f"Total links: {total_links}")
    external_weights = list(external_counts.values())
    total_weight = event_weight + sum(external_weights)
    
    # Gap between segments (in radians)
    gap = 0.02
    total_gap = gap * (len(external_subs) + 1)  # gaps between all segments
    available_angle = 2 * np.pi - total_gap
    
    # Calculate arc angles proportional to weight
    event_arc = available_angle * (event_weight / total_weight)
    external_arcs = [available_angle * (w / total_weight) for w in external_weights]
    
    # Position segments around the circle
    # Start event at top (90°) and go clockwise
    radius = 1.0
    inner_radius = 0.85  # For arc thickness
    
    # Event segment centered at top
    event_start = np.pi/2 - event_arc/2
    event_end = np.pi/2 + event_arc/2
    event_mid = np.pi/2
    
    # External segments fill the rest, going clockwise from event
    segments = []  # (start_angle, end_angle, mid_angle, label, count, is_event)
    segments.append((event_start, event_end, event_mid, event_label, total_links, True))
    
    current_angle = event_end + gap
    for i, (sub, count) in enumerate(external_counts.items()):
        arc_len = external_arcs[i]
        seg_start = current_angle
        seg_end = current_angle + arc_len
        seg_mid = (seg_start + seg_end) / 2
        segments.append((seg_start, seg_end, seg_mid, sub, count, False))
        current_angle = seg_end + gap
    
    # Create figure
    fig = go.Figure()
    
    # Helper to draw an arc segment
    def draw_arc(start_a, end_a, r_outer, r_inner, color, label, count, is_event):
        # Create filled arc using polygon
        theta = np.linspace(start_a, end_a, 50)
        
        # Outer arc
        x_outer = r_outer * np.cos(theta)
        y_outer = r_outer * np.sin(theta)
        
        # Inner arc (reversed for closed polygon)
        x_inner = r_inner * np.cos(theta[::-1])
        y_inner = r_inner * np.sin(theta[::-1])
        
        # Combine into closed shape
        x_shape = np.concatenate([x_outer, x_inner, [x_outer[0]]])
        y_shape = np.concatenate([y_outer, y_inner, [y_outer[0]]])
        
        hover_text = f"{label}<br>{count:,} links" if not is_event else f"{label}<br>Total: {count:,} links"
        
        fig.add_trace(go.Scatter(
            x=x_shape, y=y_shape,
            fill='toself',
            fillcolor=color,
            line=dict(color='white', width=1),
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False,
            mode='lines'
        ))
        
        # Add label outside the arc
        mid_a = (start_a + end_a) / 2
        arc_size = end_a - start_a  # Size of this arc in radians
        
        # Skip labels for only the tiniest arcs (would overlap)
        if not is_event and arc_size < 0.01:
            return
        
        # Event label gets extra offset above
        if is_event:
            label_r = r_outer + 0.20
        else:
            label_r = r_outer + 0.08
            
        label_x = label_r * np.cos(mid_a)
        label_y = label_r * np.sin(mid_a)
        
        # Rotate text based on position
        if is_event:
            display_label = label
            # Event label: keep HORIZONTAL
            text_rotation = 0
        else:
            # Truncate long subreddit names
            if len(label) > 15:
                display_label = f"r/{label[:12]}..."
            else:
                display_label = f"r/{label}"
            
            # External labels: PERPENDICULAR (radial - pointing outward)
            # Normalize angle to 0-360 range
            angle_deg = np.degrees(mid_a) % 360
            
            # Radial text: labels should read from inside-out
            # Left side (90 to 270): flip to keep text readable
            if 90 < angle_deg < 270:
                text_rotation = angle_deg - 180
            else:
                text_rotation = angle_deg
        
        fig.add_annotation(
            x=label_x, y=label_y,
            text=display_label,
            showarrow=False,
            font=dict(size=7 if not is_event else 11, color='#333'),
            textangle=-text_rotation,
            xanchor="center",
            yanchor="middle"
        )
    
    # Draw all arc segments
    for seg in segments:
        start_a, end_a, mid_a, label, count, is_event = seg
        color = event_color if is_event else external_color
        draw_arc(start_a, end_a, radius, inner_radius, color, label, count, is_event)
    
    # Draw chord connections (curved lines from event to each external)
    event_seg = segments[0]
    for ext_seg in segments[1:]:
        sub_name = ext_seg[3]
        count = ext_seg[4]
        max_count = max(external_counts.values())
        
        # Get sentiment for this subreddit
        sentiment = external_sentiments.get(sub_name, 0)
        
        # Scale opacity and width by link count
        opacity = min(0.75, 0.3 + 0.45 * (count / max_count))
        width = max(1, min(8, count / max_count * 8))
        
        # Connection points at middle of inner radius
        conn_r = (radius + inner_radius) / 2
        
        # Event connection point
        ev_x = conn_r * np.cos(event_seg[2])
        ev_y = conn_r * np.sin(event_seg[2])
        
        # External connection point
        ext_x = conn_r * np.cos(ext_seg[2])
        ext_y = conn_r * np.sin(ext_seg[2])
        
        # Bezier curve through center
        t = np.linspace(0, 1, 50)
        cx, cy = 0, 0
        bx = (1-t)**2 * ev_x + 2*(1-t)*t * cx + t**2 * ext_x
        by = (1-t)**2 * ev_y + 2*(1-t)*t * cy + t**2 * ext_y
        
        # Color based on sentiment
        arc_color = sentiment_to_color(sentiment, opacity)
        
        # Format sentiment for hover
        sentiment_label = "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral"
        sentiment_pct = f"{abs(sentiment)*100:.0f}%"
        
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode='lines',
            line=dict(width=width, color=arc_color),
            hoverinfo='text',
            hovertext=f"Event ↔ r/{sub_name}: {count:,} links<br>Sentiment: {sentiment_label} ({sentiment_pct})",
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{event_name}</b><br><sub>Arc size ∝ link count | Chord color: <span style='color:green'>positive</span> / <span style='color:red'>negative</span> sentiment</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.6, 1.6], scaleanchor="y"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.6, 1.6]),
        width=950,
        height=950,
        plot_bgcolor='white',
        hovermode='closest',
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    if notebook:
        # Disable all interactivity for static-like display (no save)
        fig.update_layout(
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            dragmode=False
        )
        fig.show(config={'displayModeBar': False, 'staticPlot': True})
        return None
    
    # Save as HTML (only when not in notebook mode)
    fig.write_html(output_path)
    print(f"  Saved: {output_path}")
    
    return fig


def generate_chord_for_event(
    df: pd.DataFrame,
    event_name: str,
    event_data: dict,
    output_dir: str = "img/chord_diagrams",
    top_n: int = 25,
    pre_days: int = 7,
    post_days: int = 14,
    notebook: bool = False,
    force: bool = False,   # optional override
):
    """Generate chord diagram for a single event."""
    print(f"\nProcessing: {event_name}")

    # ---- FIX 1: use output_dir correctly ----
    chords_dir = Path(output_dir)
    chords_dir.mkdir(parents=True, exist_ok=True)

    # ---- FIX 2: use event_name (not event_title) ----
    existing = chord_already_exists(event_name, chords_dir=chords_dir)
    if existing and not force:
        print(f"[skip] chord already exists for '{event_name}': {existing.name}")
        return existing

    # ---- rest of your original logic ----
    event_subs = set(event_data.get("matched_subreddits", []))
    start_date = event_data.get("start_date")
    end_date = event_data.get("end_date")

    if not event_subs:
        print(f"  Skipping: No matched subreddits")
        return False

    if not start_date or not end_date:
        print(f"  Skipping: Missing date range")
        return False

    # Extend the timeframe
    original_start = start_date
    original_end = end_date
    start_date = (pd.to_datetime(start_date) - timedelta(days=pre_days)).strftime("%Y-%m-%d")
    end_date = (pd.to_datetime(end_date) + timedelta(days=post_days)).strftime("%Y-%m-%d")

    print(f"  Original period: {original_start} to {original_end}")
    print(f"  Extended period: {start_date} to {end_date} (pre: {pre_days}d, post: {post_days}d)")
    print(f"  Event subreddits: {len(event_subs)}")

    external_counts, external_sentiments, total_links = get_event_external_links(
        df, event_subs, start_date, end_date, top_n=top_n
    )

    if not external_counts:
        print(f"  Skipping: No external links found in time window")
        return False

    avg_sentiment = sum(external_sentiments.values()) / len(external_sentiments)
    positive_count = sum(1 for s in external_sentiments.values() if s > 0)
    negative_count = sum(1 for s in external_sentiments.values() if s < 0)

    print(f"  Total external links: {total_links:,}")
    print(f"  Top {len(external_counts)} external subs shown")
    print(f"  Sentiment: {positive_count} positive, {negative_count} negative (avg: {avg_sentiment:.2f})")

    # ---- FIX 3: filename generation should match canonical logic ----
    safe_name = canonical_key(event_name)
    output_path = chords_dir / f"chord_{safe_name}.html"

    fig = create_chord_diagram(
        external_counts,
        external_sentiments,
        total_links,
        event_name,
        len(event_subs),
        str(output_path),
        notebook=notebook,
    )

    return fig



def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive chord diagrams for event-subreddit links"
    )
    parser.add_argument(
        "--event", type=str, default=None,
        help="Name of specific event to generate diagram for"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate diagrams for all events"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available events"
    )
    parser.add_argument(
        "--top-n", type=int, default=25,
        help="Number of top external subreddits to include (default: 25)"
    )
    parser.add_argument(
        "--pre-days", type=int, default=7,
        help="Number of days before event start to include (default: 7)"
    )
    parser.add_argument(
        "--post-days", type=int, default=14,
        help="Number of days after event end to include (default: 14)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="img/chord_diagrams",
        help="Output directory for HTML files"
    )
    
    args = parser.parse_args()
    
    # Load event lexicon
    print("Loading event lexicon...")
    events = load_event_lexicon()
    print(f"Found {len(events)} events")
    
    if args.list:
        print("\nAvailable events:")
        for name, data in events.items():
            subs = len(data.get("matched_subreddits", []))
            dates = f"{data.get('start_date')} to {data.get('end_date')}"
            print(f"  - {name} ({dates}, {subs} subreddits)")
        return
    
    # Load hyperlinks data
    df = load_hyperlinks_data()
    
    if args.all:
        # Generate for all events
        success_count = 0
        for event_name, event_data in events.items():
            if generate_chord_for_event(df, event_name, event_data, args.output_dir, args.top_n, args.pre_days, args.post_days):
                success_count += 1
        print(f"\nGenerated {success_count}/{len(events)} chord diagrams")
        print(f"Output directory: {args.output_dir}")
        
    elif args.event:
        # Generate for specific event
        if args.event not in events:
            print(f"Error: Event '{args.event}' not found")
            print("Use --list to see available events")
            return
        generate_chord_for_event(df, args.event, events[args.event], args.output_dir, args.top_n, args.pre_days, args.post_days)
        
    else:
        # No event specified - show help
        parser.print_help()
        print("\nTip: Use --list to see available events")


if __name__ == "__main__":
    main()

"""
Script to update results.ipynb with proper structure and event-based analysis.
Uses standard json library instead of nbformat.
"""

import json
import uuid

def make_markdown_cell(source):
    """Create a markdown cell dictionary."""
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": source.split('\n') if isinstance(source, str) else source
    }

def make_code_cell(source):
    """Create a code cell dictionary."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": source.split('\n') if isinstance(source, str) else source
    }

def format_source(text):
    """Format source text as list of lines with proper newlines."""
    lines = text.strip().split('\n')
    # Add newline to all but the last line
    return [line + '\n' if i < len(lines) - 1 else line for i, line in enumerate(lines)]

# Load the existing notebook
with open("results.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Define new cells to insert at the beginning
new_cells = []

# Section 1: Title and Introduction
new_cells.append(make_markdown_cell(format_source("""# Reddit Political Network Analysis: Information Flow During Global Events

## Introduction

This notebook analyzes the Reddit Hyperlink Network, examining how information flows between communities during major global events from 2014-2017. Using hyperlinks as a proxy for information propagation, we investigate:

**Research Questions:**
1. **Cross-Community Propagation**: To what extent do discussions about major global events spread beyond their original communities?
2. **Community Alignment**: Which communities amplify or oppose narratives about particular global issues?
3. **Temporal Dynamics**: How do patterns of discussion and hyperlinking vary across different events?

The dataset captures directed hyperlinks between subreddits, with sentiment labels indicating whether the linking community references another in a positive or negative context.""")))

# Section 1.1: Setup and Imports
new_cells.append(make_markdown_cell(format_source("""---
## 1. Setup & Data Loading

First, we load all necessary libraries and the Reddit hyperlink datasets.""")))

# Consolidated imports cell
new_cells.append(make_code_cell(format_source("""# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Project modules - Data utilities
from src.utils.data_utils import get_links_within_period, properties_into_dataframe_columns
from src.data.data_loader import load_data
from src.data.unique_subs import get_all_unique_subreddits
from src.data.sub_inter_emb import get_subreddit_intersection
from src.data.sentiment_analysis import categorization

# Project modules - Analysis functions
from src.scripts.extended_analysis import (
    compute_dataset_stats, print_dataset_overview, get_top_subreddits, plot_top_subreddits,
    compute_monthly_activity, plot_temporal_activity, analyze_political_interactions,
    plot_political_interactions, analyze_all_events, plot_event_analysis,
    build_subreddit_network, compute_network_metrics, print_network_summary,
    compute_subreddit_roles, plot_subreddit_roles, analyze_hourly_patterns,
    plot_hourly_patterns, find_controversial_pairs, plot_controversial_pairs,
    plot_vader_distribution, compute_link_type_stats, print_link_type_summary,
    print_event_summary
)

from src.scripts.advanced_visualizations import (
    plot_sentiment_heatmap, plot_day_of_week_patterns, plot_sentiment_over_time,
    plot_link_volume_distribution, plot_yearly_comparison, plot_reciprocal_links,
    plot_political_activity_timeline
)

from src.scripts.event_analysis import (
    compute_event_propagation_metrics, plot_event_propagation,
    compute_category_interactions, plot_category_heatmap,
    compute_event_aligned_activity, plot_temporal_event_comparison,
    compute_event_lift, compute_event_sentiment_profile, plot_event_sentiment_comparison
)

# Project modules - Event data
from src.data.event_subreddits import (
    EVENT_PERIODS, EVENT_CATEGORIES,
    CRIMEA_UKRAINE, BREXIT, US_ELECTION_2016, TRUMP_INAUGURATION,
    EBOLA, NEPAL_EARTHQUAKE, FORT_MCMURRAY, MH370,
    IRAN_NUCLEAR_DEAL, COLOMBIA_FARC, ISIS, TURKISH_COUP, CLIMATE_PARIS_COP21
)

from src.models.knn import run_knn

# Process management
from src.scripts.process_political_subbredit import turn_txt_file_to_lowercase""")))

# Data loading cell
new_cells.append(make_markdown_cell(format_source("""### Loading the Reddit Hyperlink Dataset

The dataset contains two types of hyperlinks:
- **Body links**: Hyperlinks found in the body text of Reddit posts
- **Title links**: Hyperlinks found in post titles

Each link records the source subreddit, target subreddit, timestamp, sentiment, and various text properties.""")))

new_cells.append(make_code_cell(format_source("""# Load the datasets
body, title = load_data("data/soc-redditHyperlinks-body.tsv", "data/soc-redditHyperlinks-title.tsv")

# Load political subreddits list
pol_sb = pd.read_csv("data/pol_sb.txt", header=None)[0].str.lower().unique()

# Load subreddit embeddings
subreddits_emb = pd.read_csv("data/web-redditEmbeddings-subreddits.csv", header=None)[0].values

print(f"Body links loaded: {len(body):,}")
print(f"Title links loaded: {len(title):,}")
print(f"Political subreddits: {len(pol_sb):,}")
print(f"Subreddits with embeddings: {len(subreddits_emb):,}")
print(f"\\nDate range: {body['TIMESTAMP'].min()} to {body['TIMESTAMP'].max()}")""")))

# Dataset overview
new_cells.append(make_markdown_cell(format_source("""### Dataset Overview

Let's examine the key statistics of our dataset to understand its scope and characteristics.""")))

new_cells.append(make_code_cell(format_source("""# Compute and display comprehensive dataset statistics
stats = compute_dataset_stats(body, title, pol_sb)
print_dataset_overview(stats)""")))

# Timeline of events
new_cells.append(make_markdown_cell(format_source("""### Major Events Timeline (2014-2017)

Our analysis focuses on these major global events that occurred during the dataset period:

| Category | Event | Period |
|----------|-------|--------|
| **Political** | Crimea Annexation | Feb-Apr 2014 |
| **Political** | Brexit Referendum | May-Jul 2016 |
| **Political** | US Election 2016 | Sep-Nov 2016 |
| **Political** | Trump Inauguration | Jan-Feb 2017 |
| **Health** | Ebola Peak | Aug 2014 - Jan 2015 |
| **Disasters** | MH370 Disappearance | Mar-May 2014 |
| **Disasters** | Nepal Earthquake | Apr-Jun 2015 |
| **Disasters** | Fort McMurray Fire | May-Jun 2016 |
| **Diplomacy** | Iran Nuclear Deal | Jul-Aug 2015 |
| **Conflict** | ISIS Caliphate | Jun-Sep 2014 |
| **Conflict** | Turkish Coup Attempt | Jul-Aug 2016 |
| **Climate** | Paris Climate COP21 | Nov-Dec 2015 |""")))

# Insert new cells at the beginning of the notebook
notebook['cells'] = new_cells + notebook['cells']

# Save the updated notebook
with open("results.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully!")
print(f"Added {len(new_cells)} new cells at the beginning.")

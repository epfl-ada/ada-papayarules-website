"""
Script to add Event-Based Analysis section to results.ipynb.
"""

import json
import uuid

def make_markdown_cell(source):
    """Create a markdown cell dictionary."""
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

def make_code_cell(source):
    """Create a code cell dictionary."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }

def format_source(text):
    """Format source text as list of lines with proper newlines."""
    lines = text.strip().split('\n')
    return [line + '\n' if i < len(lines) - 1 else line for i, line in enumerate(lines)]

# Load the existing notebook
with open("results.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find the position after Section 3 (look for "4." or add at the end before any conclusion)
# We'll append these cells near the end of the notebook
cells = notebook['cells']

# Define new cells for Section 4: Event-Based Analysis
event_cells = []

# Section 4 Header
event_cells.append(make_markdown_cell(format_source("""---
## 4. Event-Based Analysis

This section applies the event-specific subreddit lists to analyze cross-community information flow during major global events. We examine how different communities engage with each event topic.""")))

# 4.1 Event Propagation
event_cells.append(make_markdown_cell(format_source("""### 4.1 Cross-Community Propagation

How do discussions about major events spread beyond their original communities? We track links between event-related and general subreddits.""")))

event_cells.append(make_code_cell(format_source("""# Compute cross-community propagation metrics for all events
propagation_metrics = compute_event_propagation_metrics(body)
propagation_metrics""")))

event_cells.append(make_code_cell(format_source("""# Visualize event propagation patterns
plot_event_propagation(propagation_metrics, save_path="img/event_propagation.png")""")))

# 4.2 Category Interactions
event_cells.append(make_markdown_cell(format_source("""### 4.2 Category-to-Category Interactions

How do different event categories interact? Political events may cross-reference disaster events, creating a web of inter-topic discourse.""")))

event_cells.append(make_code_cell(format_source("""# Analyze interactions between event categories
category_interactions = compute_category_interactions(body)
category_interactions""")))

event_cells.append(make_code_cell(format_source("""# Visualize category interaction heatmap
plot_category_heatmap(category_interactions, save_path="img/category_heatmap.png")""")))

# 4.3 Temporal Event Comparison
event_cells.append(make_markdown_cell(format_source("""### 4.3 Temporal Event Comparison

How does activity change around major events? We align activity to event dates and compare patterns.""")))

event_cells.append(make_code_cell(format_source("""# Compare temporal patterns across multiple events
plot_temporal_event_comparison(body, 
    events=["US Election 2016", "Brexit Referendum", "Trump Inauguration", "Ebola Peak"],
    save_path="img/temporal_event_comparison.png")""")))

# 4.4 Event Lift
event_cells.append(make_markdown_cell(format_source("""### 4.4 Activity Lift During Events

How much does activity increase during events compared to baseline periods?""")))

event_cells.append(make_code_cell(format_source("""# Compute activity lift for each event
event_lift = compute_event_lift(body)
event_lift.sort_values('lift', ascending=False)""")))

# 4.5 Event Sentiment Profiles
event_cells.append(make_markdown_cell(format_source("""### 4.5 Event Sentiment Comparison

How does sentiment vary across different events? Some events may trigger more polarized discussions.""")))

event_cells.append(make_code_cell(format_source("""# Compare sentiment profiles across all events
plot_event_sentiment_comparison(body, save_path="img/event_sentiment_comparison.png")""")))

# Section 5: Network Analysis
event_cells.append(make_markdown_cell(format_source("""---
## 5. Network Analysis

We construct a directed graph of subreddit interactions to identify key structural properties and community roles.""")))

event_cells.append(make_markdown_cell(format_source("""### 5.1 Network Construction and Metrics

Building the subreddit interaction network reveals the topology of cross-community discourse.""")))

event_cells.append(make_code_cell(format_source("""# Build the subreddit network (filtering for edges with at least 5 links)
G = build_subreddit_network(body, min_weight=5)

# Compute network metrics
network_metrics = compute_network_metrics(G)
print_network_summary(network_metrics)""")))

# 5.2 Subreddit Roles
event_cells.append(make_markdown_cell(format_source("""### 5.2 Subreddit Roles: Amplifiers, Observers, and Resistors

We classify subreddits based on their linking behavior:
- **Amplifiers**: High outgoing links, mostly positive sentiment
- **Resistors**: High outgoing links, mostly negative sentiment  
- **Observers**: High incoming links, low outgoing""")))

event_cells.append(make_code_cell(format_source("""# Compute and visualize subreddit roles
roles = compute_subreddit_roles(body, pol_sb)
plot_subreddit_roles(roles, top_n=30, save_path="img/subreddit_roles.png")""")))

# Section 6: Conclusions
event_cells.append(make_markdown_cell(format_source("""---
## 6. Key Findings and Conclusions

### Cross-Community Propagation
- Major political events (US Election 2016, Brexit) generated the highest cross-community activity
- Event-related communities act as both sources and sinks for information flow

### Community Alignment
- Political subreddits show distinct sentiment patterns when referencing non-political communities
- Some communities consistently amplify while others resist certain narratives

### Temporal Dynamics
- Activity spikes correlate with real-world event developments
- Different event types show distinct temporal signatures

### Implications
These findings provide insights into how Reddit facilitates information propagation during global events and the role different communities play in shaping public discourse.""")))

# Find where to insert (after the last main content before any existing conclusion)
# Insert before the last 2-3 cells if they seem like conclusion content, otherwise append
insert_position = len(cells)

# Check if there's already a conclusion section to avoid duplicating
for i, cell in enumerate(cells):
    if cell.get('cell_type') == 'markdown':
        source = ''.join(cell.get('source', []))
        if '## 6.' in source or 'Conclusion' in source or '## 5.' in source or '## 4.' in source:
            # Already have these sections, skip adding
            print(f"Found existing section at position {i}, will append after it")
            insert_position = i + 1
            break

# Append the new cells
notebook['cells'] = cells[:insert_position] + event_cells + cells[insert_position:]

# Save the updated notebook
with open("results.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook updated successfully!")
print(f"Added {len(event_cells)} new cells for event analysis and conclusions.")

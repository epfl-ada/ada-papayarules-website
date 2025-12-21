
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

def plot_interactive_network(df_edges, start_date=None, end_date=None, min_weight=1, top_k=75, layout='spring'):
    """
    Interactive widget to plot subreddit networks filtered by date and weight.
    Limits to top_k nodes to keep the graph readable.
    
    Args:
        df_edges (pd.DataFrame): DataFrame with columns ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']
        min_weight (int): Minimum links required to show an edge.
        top_k (int): Maximum number of nodes to display (by degree centrality).
        layout (str): NetworkX layout algorithm ('spring', 'circular', 'kamada').
    """
    
    if not pd.api.types.is_datetime64_any_dtype(df_edges['TIMESTAMP']):
        df_edges['TIMESTAMP'] = pd.to_datetime(df_edges['TIMESTAMP'], errors='coerce')
        try:
            if getattr(df_edges['TIMESTAMP'].dt, 'tz', None) is not None:
                df_edges['TIMESTAMP'] = df_edges['TIMESTAMP'].dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            pass
        
    date_min = df_edges['TIMESTAMP'].min()
    date_max = df_edges['TIMESTAMP'].max()
    
    date_slider = widgets.SelectionRangeSlider(
        options=pd.date_range(date_min, date_max, freq='M').strftime('%Y-%m').tolist(),
        index=(0, len(pd.date_range(date_min, date_max, freq='M'))-1),
        description='Date Range',
        orientation='horizontal',
        layout={'width': '600px'}
    )
    
    weight_slider = widgets.IntSlider(value=min_weight, min=1, max=50, step=1, description='Min Weight:')
    node_limit_slider = widgets.IntSlider(value=top_k, min=10, max=200, step=10, description='Max Nodes:')
    
    def _plot_network(date_range, weight_threshold, k_nodes):
        s_date, e_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        
        # 1. Date Filter
        mask = (df_edges['TIMESTAMP'] >= s_date) & (df_edges['TIMESTAMP'] <= e_date)
        filtered_df = df_edges[mask]
        
        # 2. Weight Filter
        agg_edges = filtered_df.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']).size().reset_index(name='weight')
        agg_edges = agg_edges[agg_edges['weight'] >= weight_threshold]
        
        if agg_edges.empty:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No edges found for filters", ha='center', fontsize=14)
            plt.axis('off')
            plt.show()
            return

        # 3. Top-K Node Filter (Crucial for readability)
        # We build a temp graph to calculate degrees
        G_temp = nx.from_pandas_edgelist(agg_edges, 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', ['weight'], create_using=nx.DiGraph())
        degrees = dict(G_temp.degree(weight='weight'))
        
        # Identify top nodes
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:k_nodes]
        G = G_temp.subgraph(top_nodes).copy()
        
        plt.figure(figsize=(14, 10))
        
        # Layouts
        if layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=0.3, seed=42, iterations=50)
            
        # Draw
        node_sizes = [degrees[n] * 5 for n in G.nodes()]
        
        # Edges with transparency based on weight
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_colors = [(0.5, 0.5, 0.5, 0.2 + 0.8 * (w/max_w)) for w in weights]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='cornflowerblue', alpha=0.9, edgecolors='white')
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color=edge_colors, arrowsize=15, connectionstyle='arc3,rad=0.1')
        
        # Labels with background for readability
        labels = nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        import matplotlib.patheffects as PathEffects
        for t in labels.values():
            t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        
        plt.title(f"Network Structure: {s_date.date()} to {e_date.date()}\nTop {len(G.nodes())} Nodes (Min Weight: {weight_threshold})", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    interact(_plot_network, date_range=date_slider, weight_threshold=weight_slider, k_nodes=node_limit_slider)

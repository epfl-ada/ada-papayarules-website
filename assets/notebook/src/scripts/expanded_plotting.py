
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_core_vs_expanded(metrics_df: pd.DataFrame, save_path: str = None):
    """
    Plots comparison of metrics between core and expanded subreddits.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame returned by compute_core_vs_expanded_metrics.
        save_path (str, optional): Path to save the figure.
    """
    if metrics_df.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 1. Volume Comparison
    # Melt dataframe for grouped bar chart
    vol_df = metrics_df[["event", "seed_volume", "added_volume"]].melt(
        id_vars="event", 
        var_name="Group", 
        value_name="Volume"
    )
    # Rename for cleaner legend
    vol_df["Group"] = vol_df["Group"].map({"seed_volume": "Core (Seed)", "added_volume": "Expanded (Added)"})
    
    sns.barplot(data=vol_df, x="event", y="Volume", hue="Group", ax=axes[0], palette="viridis")
    axes[0].set_title("Link Volume: Core vs. Expanded Subreddits", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Event")
    axes[0].set_ylabel("Number of Links")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title="Subreddit Group")
    
    # 2. Sentiment Difference (Expanded - Core)
    # Use color to indicate positive (green) vs negative (red) difference
    metrics_df["sent_diff_color"] = metrics_df["sentiment_diff"].apply(lambda x: "green" if x > 0 else "red")
    
    sns.barplot(
        data=metrics_df, 
        x="event", 
        y="sentiment_diff", 
        ax=axes[1], 
        palette=metrics_df["sent_diff_color"].tolist() # Pass list of colors directly to avoid seaborn warning with categorical palette
    )
    
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title("Sentiment Shift: (Expanded - Core)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Event")
    axes[1].set_ylabel("Sentiment Difference")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        
    plt.show()

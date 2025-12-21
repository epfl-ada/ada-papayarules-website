
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detect_bursts(df, metric_col='count', window=7, threshold=2.0):
    """
    Detect anomalies/bursts in a time series using rolling z-score.
    
    Args:
        df (pd.DataFrame): DataFrame with a datetime index or 'date' column.
        metric_col (str): Column to analyze.
        window (int): Rolling window size.
        threshold (float): Z-score threshold for anomaly.
        
    Returns:
        pd.DataFrame: Original df with 'z_score' and 'is_anomaly' columns.
    """
    df = df.copy()
    
    # Ensure it's sorted
    df = df.sort_index()
    
    # Calculate rolling stats
    rolling_mean = df[metric_col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[metric_col].rolling(window=window, min_periods=1).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, 1)
    
    df['rolling_mean'] = rolling_mean
    df['z_score'] = (df[metric_col] - rolling_mean) / rolling_std
    df['is_anomaly'] = df['z_score'].abs() > threshold
    
    return df

def plot_anomalies(df, metric_col, title="Time Series Anomalies"):
    """Plot the time series with anomalies highlighted."""
    plt.figure(figsize=(14, 6))
    
    plt.plot(df.index, df[metric_col], label='Actual', color='blue', alpha=0.6)
    plt.plot(df.index, df['rolling_mean'], label='Rolling Mean', color='orange', linestyle='--')
    
    anomalies = df[df['is_anomaly']]
    plt.scatter(anomalies.index, anomalies[metric_col], color='red', label='Anomaly', zorder=5)
    
    plt.title(title)
    plt.legend()
    plt.ylabel(metric_col)
    plt.show()

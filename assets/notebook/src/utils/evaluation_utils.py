import numpy as np
import pandas as pd

def positive_sentiment_ratio(df):
    """Calculate the ratio of positive sentiment links in the dataframe."""
    num_pos = np.sum(df["LINK_SENTIMENT"] == 1)
    num_neg = np.sum(df["LINK_SENTIMENT"] == -1)
    ratio = num_pos / (num_pos + num_neg) if (num_pos + num_neg) > 0 else 0
    return ratio
import numpy as np
import pandas as pd

def categorization(df, filter, xor = True):  

    if xor:
        source_mask = df['SOURCE_SUBREDDIT'].isin(filter["item"])
        target_mask = df['TARGET_SUBREDDIT'].isin(filter["item"])

        xor_mask = source_mask ^ target_mask

        source_filtered = df[source_mask]
        target_filtered = df[target_mask]

        df_xor = df[xor_mask]

        return source_filtered, target_filtered, df_xor
    
    
    df_mask = df[0].isin(filter["item"])
    df_filtered = df[df_mask]        
    return df_filtered

def sentiment_selection(df, sentiment_value):
    """Select rows from the dataframe based on the LINK_SENTIMENT value."""
    match sentiment_value:
        case "positive":
            return df[df["LINK_SENTIMENT"] == 1]
        case "negative":
            return df[df["LINK_SENTIMENT"] == -1]
        case "pos":
            return df[df["LINK_SENTIMENT"] == 1]
        case "neg":
            return df[df["LINK_SENTIMENT"] == -1]
        case 1:
            return df[df["LINK_SENTIMENT"] == 1]
        case -1:
            return df[df["LINK_SENTIMENT"] == -1]
        case _:
            return df
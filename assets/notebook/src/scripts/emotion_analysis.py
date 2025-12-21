
import pandas as pd
import numpy as np
import re

# Simplified Lexicon for Demo purposes (can be expanded with NRCLex)
EMOTION_LEXICON = {
    'anger': ['bad', 'hate', 'kill', 'angry', 'mad', 'fury', 'rage', 'stupid', 'idiot', 'worst'],
    'fear': ['scared', 'fear', 'afraid', 'risk', 'danger', 'threat', 'worry', 'anxious', 'crisis'],
    'joy': ['good', 'love', 'happy', 'great', 'best', 'win', 'success', 'safe', 'relief', 'hope']
}

def analyze_emotions(text_series):
    """
    Score text for basic emotions (Anger, Fear, Joy).
    
    Args:
        text_series (pd.Series): Series of text strings.
        
    Returns:
        pd.DataFrame: DataFrame with emotion scores.
    """
    
    results = []
    
    for text in text_series:
        scores = {'anger': 0, 'fear': 0, 'joy': 0}
        
        if not isinstance(text, str):
            results.append(scores)
            continue
            
        words = re.findall(r'\w+', text.lower())
        word_count = len(words) if len(words) > 0 else 1
        
        for word in words:
            for emotion, keywords in EMOTION_LEXICON.items():
                if word in keywords:
                    scores[emotion] += 1
        
        # Normalize by length? (Optional, here just raw counts or frequency)
        # scores = {k: v / word_count for k, v in scores.items()}
        results.append(scores)
        
    return pd.DataFrame(results, index=text_series.index)

def aggregate_subreddit_emotions(df, text_col='clean_text'):
    """
    Aggregate emotions by subreddit.
    Needs a dataframe with 'SOURCE_SUBREDDIT' and text.
    """
    if text_col not in df.columns:
        # Fallback to creating dummy text if not present, or checking properties
        return None

    emotions = analyze_emotions(df[text_col])
    combined = pd.concat([df, emotions], axis=1)
    
    return combined.groupby('SOURCE_SUBREDDIT')[['anger', 'fear', 'joy']].mean()

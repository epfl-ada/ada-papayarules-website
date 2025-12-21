
import json
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from src.data.data_loader import load_data
from src.data.event_subreddits import EVENT_PERIODS

def expand_and_export_subreddits():
    """
    Expands the list of related subreddits for each event by finding 
    heavily interacting subreddits during the event window.
    Exports to JSON.
    """
    print("Loading data...")
    body, title, _, _, pol_sb = load_data()
    
    # Combine relevant columns for checking interactions
    # We only need source, target, timestamp
    cols = ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']
    df = pd.concat([body[cols], title[cols]])
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    # Political subreddit set for filtering (optional, keeping it broad for now)
    pol_set = set(pol_sb['item'].str.lower())
    
    expanded_data = {}
    
    print("Expanding event subreddit lists...")
    for event_name, (start, end, seed_subs) in EVENT_PERIODS.items():
        print(f"Processing {event_name}...")
        
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        # 1. Filter by time window
        mask = (df['TIMESTAMP'] >= start_dt) & (df['TIMESTAMP'] <= end_dt)
        event_window_df = df[mask]
        
        # 2. Identify links involving seed subreddits
        seed_set = set(s.lower() for s in seed_subs)
        
        # Links where Source is in Seed
        src_mask = event_window_df['SOURCE_SUBREDDIT'].str.lower().isin(seed_set)
        # Links where Target is in Seed
        tgt_mask = event_window_df['TARGET_SUBREDDIT'].str.lower().isin(seed_set)
        
        relevant_links = event_window_df[src_mask | tgt_mask]
        
        # 3. Count neighbors
        # Collect all sources interacting with seed targets, and all targets interacting with seed sources
        # excluding the seed subs themselves
        
        candidate_counts = {}
        
        for _, row in relevant_links.iterrows():
            src = row['SOURCE_SUBREDDIT'].lower()
            tgt = row['TARGET_SUBREDDIT'].lower()
            
            if src not in seed_set:
                candidate_counts[src] = candidate_counts.get(src, 0) + 1
            if tgt not in seed_set:
                candidate_counts[tgt] = candidate_counts.get(tgt, 0) + 1
                
        # 4. Select top candidates
        # Convert to list of (sub, count)
        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 10 new subreddits
        top_new = [sub for sub, count in sorted_candidates[:10]]
        
        # Combine seed and new
        # Use a list to maintain some order (seeds first) then alpha for new? or just append
        combined_subs = list(seed_subs) + top_new
        
        # Remove duplicates while preserving order
        seen = set()
        final_list = []
        for s in combined_subs:
            if s.lower() not in seen:
                final_list.append(s)
                seen.add(s.lower())
                
        expanded_data[event_name] = {
            "seed": seed_subs,
            "added": top_new,
            "all": final_list
        }
        
    output_path = os.path.join("data", "event_related_subreddits_expanded.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(expanded_data, f, indent=4)
        
    print(f"Successfully exported expanded event subreddits to {output_path}")

if __name__ == "__main__":
    expand_and_export_subreddits()

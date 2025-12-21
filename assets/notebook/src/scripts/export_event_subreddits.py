
import json
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.data.event_subreddits import EVENT_PERIODS

def export_event_subreddits():
    """
    Exports the list of related subreddits for each event from EVENT_PERIODS
    to a JSON file.
    """
    output_data = {}
    
    for event_name, event_data in EVENT_PERIODS.items():
        # event_data is (start_date, end_date, subreddit_list)
        subreddits = event_data[2]
        output_data[event_name] = subreddits
        
    output_path = os.path.join("data", "event_related_subreddits.json")
    
    # Ensure data dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Successfully exported event subreddits to {output_path}")

if __name__ == "__main__":
    export_event_subreddits()

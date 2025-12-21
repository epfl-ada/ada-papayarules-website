import numpy as np

def get_all_unique_subreddits(body, title):
    all_subreddits = np.unique(
    np.concatenate([
        np.unique(body["SOURCE_SUBREDDIT"]),
        np.unique(body["TARGET_SUBREDDIT"]),
        np.unique(title["SOURCE_SUBREDDIT"]),
        np.unique(title["TARGET_SUBREDDIT"])
    ])
    )

    print("Total number of unique subreddits:", len(all_subreddits))
    print("Sample subreddits:", all_subreddits[:20])

    # Save to file
    np.savetxt("src/data/all_subreddits.txt", all_subreddits, fmt="%s")

    return all_subreddits
import numpy as np

def get_subreddit_intersection(all_subreddits, subreddits_emb):
    inter = np.intersect1d(all_subreddits, subreddits_emb[0])
    print("Number of common subreddits:", len(inter))
    print("Common subreddits:", inter[:20])
    return inter

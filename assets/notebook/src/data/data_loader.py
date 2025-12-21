import pandas as pd


def load_data(body_path=None, title_path=None):
    """
    Load Reddit hyperlink datasets.
    
    Args:
        body_path: Optional path to body hyperlinks TSV file. 
                   If provided along with title_path, returns only (body, title).
        title_path: Optional path to title hyperlinks TSV file.
                    If provided along with body_path, returns only (body, title).
    
    Returns:
        If paths provided: (body, title) DataFrames
        Otherwise: (body, title, users_emb, subreddits_emb, pol_sb)
    """
    # Use provided paths or defaults
    body_file = body_path or "data/soc-redditHyperlinks-body.tsv"
    title_file = title_path or "data/soc-redditHyperlinks-title.tsv"
    
    body = pd.read_csv(body_file, sep="\t")
    title = pd.read_csv(title_file, sep="\t")
    
    # If custom paths were provided, return only body and title
    if body_path is not None and title_path is not None:
        return body, title
    
    # Otherwise load and return all data
    users_emb = pd.read_csv("data/web-redditEmbeddings-users.csv", header=None)
    subreddits_emb = pd.read_csv("data/web-redditEmbeddings-subreddits.csv", header=None)
    pol_sb = pd.read_csv("data/pol_sb.txt", header=None, names=["item"])
    return body, title, users_emb, subreddits_emb, pol_sb
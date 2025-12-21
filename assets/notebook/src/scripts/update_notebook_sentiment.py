
import json
import os

notebook_path = r"c:\Users\ifare\OneDrive\Documents\EPFL\MA1\ADA\ada-2025-project-papayarules\resultsP3_extended.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The code we want to find
old_code_snippet = "sns.histplot(body['LINK_SENTIMENT'], bins=30, kde=True, color='purple')"

# The new code to insert
new_source = [
    "# Advanced Sentiment Comparison\n",
    "from src.scripts.advanced_visualizations import plot_comparative_sentiment_distribution\n",
    "plot_comparative_sentiment_distribution(body, pol_sb)"
]

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Check if the cell contains the old plotting code
        source_str = "".join(cell['source'])
        if old_code_snippet in source_str:
            print("Found target cell. updating...")
            cell['source'] = new_source
            cell['outputs'] = [] # Clear outputs as they will be invalid
            cell['execution_count'] = None
            found = True
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")

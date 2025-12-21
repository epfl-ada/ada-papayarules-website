"""Verify notebook structure."""
import json

with open("results.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== Notebook Structure Verification ===")
print(f"Total cells: {len(nb['cells'])}")
print()
print("Main sections found:")

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        source = ''.join(c['source'])
        # Check for main section headers
        lines = source.split('\n')
        for line in lines[:5]:
            if line.startswith('# ') or line.startswith('## ') or line.startswith('### '):
                indent = "  " if line.startswith('# ') else ("    " if line.startswith('## ') else "      ")
                print(f"{indent}Cell {i}: {line[:65]}")
                break

print()
print("=== Verification Complete ===")

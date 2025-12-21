import os
import re

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAYOUT_PATH = os.path.join(BASE_DIR, '_layouts', 'full.html')
INDEX_PATH = os.path.join(BASE_DIR, 'index.md')
OUTPUT_PATH = os.path.join(BASE_DIR, 'preview.html')

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def build_page(layout_content, content_path, output_path, apply_markdown=True, body_class=""):
    print(f"Building {output_path}...")
    content = read_file(content_path)
    
    # Remove Front Matter
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            content = parts[2]
            
    # Simple Replacements (Markdown-style) — skip for raw HTML files like map.html
    if apply_markdown:
        content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
        content = re.sub(r'\*(.*?)\*', r'<i>\1</i>', content)
        content = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', content)
    
    # Fix links to map.html to point to preview_map.html for local preview
    content = content.replace('href="map.html"', 'href="preview_map.html"')
    content = content.replace('href="index.html"', 'href="preview.html"')

    final_html = layout_content.replace('{{ content }}', content)
    
    # Fix Liquid tags
    final_html = final_html.replace("{{ site.title | default: site.github.repository_name }}", "The Reddit Political Network")
    final_html = final_html.replace("{{ site.github.owner_name }}", "ADA Team")
    final_html = final_html.replace("{{ page.body_class | default: '' }}", body_class)
    
    # Fix assets
    final_html = re.sub(r'\{\{\s*\'/assets/css/styles.css.*?\s*\}\}', 'assets/css/styles.css', final_html)
    final_html = re.sub(r'\{\{\s*\'/assets/js/scale.fix.js.*?\s*\}\}', 'assets/js/scale.fix.js', final_html)
    final_html = re.sub(r'\{% if site.google_analytics %\}.*?\{% endif %\}', '', final_html, flags=re.DOTALL)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)

def main():
    layout_content = read_file(LAYOUT_PATH)
    
    # Build Index
    build_page(layout_content, INDEX_PATH, OUTPUT_PATH, apply_markdown=True, body_class="")
    
    # Build Map
    map_path = os.path.join(BASE_DIR, 'map.html')
    map_output = os.path.join(BASE_DIR, 'preview_map.html')
    if os.path.exists(map_path):
        build_page(layout_content, map_path, map_output, apply_markdown=False, body_class="map-page")
        
    print(f"Preview generated at: {OUTPUT_PATH}")
    print("Open this file in your browser to see the website.")

if __name__ == "__main__":
    main()

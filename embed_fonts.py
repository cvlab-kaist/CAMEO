import os
import base64
import re

# Configuration
font_path_regular = 'assets/fonts/carlito/Carlito-Regular.ttf'
font_path_bold = 'assets/fonts/carlito/Carlito-Bold.ttf'
target_files = [
    'assets/analysis_1.svg',
    'assets/analysis_2.svg',
    'assets/analysis_3a.svg',
    'assets/analysis_3b.svg',
    'assets/analysis_3c.svg',
    'assets/exp_co3d_graph.svg',
    'assets/exp_re10k_graph.svg',
    'assets/teaser_qual.svg'
]

def get_base64_font(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def embed_font(svg_path, regular_b64, bold_b64):
    if not os.path.exists(svg_path):
        print(f"File not found: {svg_path}")
        return

    with open(svg_path, 'r') as f:
        content = f.read()

    # Create CSS style block
    style_block = f"""
    <defs>
        <style>
            @font-face {{
                font-family: 'Carlito';
                src: url('data:font/ttf;base64,{regular_b64}') format('truetype');
                font-weight: normal;
                font-style: normal;
            }}
            @font-face {{
                font-family: 'Carlito';
                src: url('data:font/ttf;base64,{bold_b64}') format('truetype');
                font-weight: bold;
                font-style: normal;
            }}
        </style>
    </defs>
    """

    # Inject style block after <svg ...> tag
    # Find the end of the opening <svg ...> tag
    match = re.search(r'<svg[^>]*>', content)
    if match:
        insert_pos = match.end()
        # Check if <defs> already exists to avoid duplication or malformed SVG if possible, 
        # but simply appending after <svg> is usually safe if we wrap in our own <defs> or just <style>
        # To be safer, let's just insert our block.
        new_content = content[:insert_pos] + style_block + content[insert_pos:]
        
        # Replace font-family
        # We want to replace "Calibri,Calibri_MSFontService,sans-serif" or similar with "Carlito, Calibri, sans-serif"
        # The regex should be flexible enough to catch variations
        new_content = re.sub(r'font-family="[^"]*Calibri[^"]*"', 'font-family="Carlito, Calibri, sans-serif"', new_content)
        
        with open(svg_path, 'w') as f:
            f.write(new_content)
        print(f"Processed: {svg_path}")
    else:
        print(f"Could not find <svg> tag in {svg_path}")

# Main execution
try:
    regular_b64 = get_base64_font(font_path_regular)
    bold_b64 = get_base64_font(font_path_bold)
    
    for svg_file in target_files:
        embed_font(svg_file, regular_b64, bold_b64)
        
except Exception as e:
    print(f"Error: {e}")

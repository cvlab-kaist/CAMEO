import os
import glob
import xml.etree.ElementTree as ET
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
import matplotlib as mpl

# Set backend to Agg to avoid display issues
mpl.use('Agg')

def codes_to_svg_d(path):
    """
    Converts a matplotlib Path to an SVG d string.
    """
    parts = []
    for vertices, code in path.iter_segments():
        if code == Path.MOVETO:
            parts.append(f"M {vertices[0]:.4f} {vertices[1]:.4f}")
        elif code == Path.LINETO:
            parts.append(f"L {vertices[0]:.4f} {vertices[1]:.4f}")
        elif code == Path.CURVE3:
            parts.append(f"Q {vertices[0]:.4f} {vertices[1]:.4f} {vertices[2]:.4f} {vertices[3]:.4f}")
        elif code == Path.CURVE4:
            parts.append(f"C {vertices[0]:.4f} {vertices[1]:.4f} {vertices[2]:.4f} {vertices[3]:.4f} {vertices[4]:.4f} {vertices[5]:.4f}")
        elif code == Path.CLOSEPOLY:
            parts.append("Z")
            
    return " ".join(parts)

def convert_svg_text_to_paths(svg_path, font_path):
    print(f"Processing {svg_path}...")
    
    try:
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        ET.register_namespace('xlink', "http://www.w3.org/1999/xlink")
        tree = ET.parse(svg_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {svg_path}: {e}")
        return

    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    # Load font
    fp = FontProperties(fname=font_path)
    
    # Find all text elements recursively
    # Note: ET doesn't support parent traversal easily, so we might need a different approach
    # or just iterate and replace in a second pass if we can find parents.
    # A simple way to replace is to build a mapping of parent -> children and replace.
    
    parent_map = {c: p for p in tree.iter() for c in p}
    text_elements = list(root.findall(".//svg:text", ns))
    
    if not text_elements:
        print(f"No text elements found in {svg_path}")
        return

    print(f"Found {len(text_elements)} text elements.")

    for text_elem in text_elements:
        text_content = text_elem.text
        if not text_content:
            continue
            
        try:
            x = float(text_elem.get('x', 0))
            y = float(text_elem.get('y', 0))
            font_size = float(text_elem.get('font-size', 12)) # Default 12 if missing
        except ValueError:
            print(f"Skipping text element with invalid attributes: {text_elem.attrib}")
            continue

        # Create TextPath
        # Note: SVG y is usually top-down, but TextPath might be bottom-up?
        # SVG text (x,y) is the baseline. TextPath((x,y)) is also baseline.
        # However, Matplotlib might flip Y if not careful.
        # But TextPath generates raw coordinates.
        # Let's verify coordinate system. SVG usually has y increasing downwards.
        # Matplotlib usually has y increasing upwards.
        # BUT TextPath just generates coordinates relative to (0,0) + offset.
        # If we use the same values, it should be fine IF the font glyphs are defined correctly.
        # TTF fonts usually have y up. SVG has y down.
        # So we might need to flip the path?
        # Wait, SVG coordinate system has (0,0) at top-left, y increases down.
        # Font glyphs (in TTF) have (0,0) at baseline, y increases UP.
        # So simply putting TTF glyphs into SVG will result in upside-down text if we don't flip it.
        # Matplotlib TextPath handles this? 
        # TextPath generates vertices. If I plot them in MPL, they look right.
        # If I put them in SVG, I might need `transform="scale(1, -1)"` on the path?
        # Or I can flip the coordinates myself.
        
        # Actually, let's try generating it.
        # If I use `ismath=False`, it uses the font directly.
        
        tp = TextPath((x, y), text_content, size=font_size, prop=fp)
        
        # We need to flip the Y coordinates of the path relative to the baseline (y).
        # The vertices are (vx, vy). The baseline is y.
        # In TTF, a point above baseline has vy > y.
        # In SVG, a point above baseline should have vy < y.
        # So we need to mirror around the baseline y.
        # New vy = y - (vy - y) = 2y - vy.
        # Let's apply this transformation to vertices.
        
        # Wait, TextPath output:
        # If I ask for text at (0,0), a point at top of 'A' might be (5, 10).
        # In SVG, if I want 'A' at (0,0), top should be (5, -10).
        # So yes, we need to flip Y.
        # But wait, `TextPath` takes (x,y) as position.
        # If I pass (x,y), the vertices will be around (x,y).
        # Let's generate at (0,0) and translate/flip manually to be safe.
        
        tp_zero = TextPath((0, 0), text_content, size=font_size, prop=fp)
        vertices = tp_zero.vertices
        codes = tp_zero.codes
        
        # Flip Y coordinates
        # SVG Y is down. Font Y is up.
        # So v_svg_y = - v_font_y
        # And then translate to (x, y).
        # So final_x = v_font_x + x
        # final_y = - v_font_y + y
        
        new_vertices = []
        for vx, vy in vertices:
            new_vertices.append([vx + x, y - vy])
            
        # Create a new Path with transformed vertices
        transformed_path = Path(new_vertices, codes)
        
        d_string = codes_to_svg_d(transformed_path)
        
        # Create path element
        path_elem = ET.Element('path')
        path_elem.set('d', d_string)
        
        # Copy attributes
        if 'transform' in text_elem.attrib:
            path_elem.set('transform', text_elem.attrib['transform'])
        if 'style' in text_elem.attrib:
            path_elem.set('style', text_elem.attrib['style'])
        if 'fill' in text_elem.attrib:
            path_elem.set('fill', text_elem.attrib['fill'])
        else:
            # Default fill if not specified? Or inherit?
            # Text usually defaults to black.
            # Let's check if style has fill.
            pass
            
        # Replace in parent
        parent = parent_map[text_elem]
        # Find index
        # This is a bit tricky in ET if there are multiple identical children, but object identity should work?
        # No, ET doesn't support index by object.
        # We have to iterate parent children.
        
        for i, child in enumerate(parent):
            if child is text_elem:
                parent[i] = path_elem
                break
                
    tree.write(svg_path)
    print(f"Saved {svg_path}")

def main():
    font_path = '/home/CAMEO/calibri.ttf'
    if not os.path.exists(font_path):
        print(f"Font file not found: {font_path}")
        return

    svg_files = glob.glob('/home/CAMEO/assets/*.svg')
    for svg_file in svg_files:
        convert_svg_text_to_paths(svg_file, font_path)

if __name__ == "__main__":
    main()

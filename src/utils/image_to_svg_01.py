import re
import xml.etree.ElementTree as ET
from io import BytesIO
from itertools import product
import numpy as np
import cairosvg
import io

import vtracer
from PIL import Image
from scour import scour
from skimage.metrics import structural_similarity as ssim

default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""

def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
         The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
         The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
         The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if "viewBox" not in svg_code:
        svg_code = svg_code.replace("<svg", f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB").resize(size)

def compare_pil_images(img1: Image.Image, img2: Image.Image, size=(384, 384)):
    """
    So sánh hai ảnh PIL sau khi resize về kích thước cố định bằng SSIM và MSE.
    
    Parameters:
        img1 (PIL.Image.Image): Ảnh thứ nhất.
        img2 (PIL.Image.Image): Ảnh thứ hai.
        size (tuple): Kích thước resize (mặc định: 384x384).
    
    Returns:
        dict: {'ssim': ..., 'mse': ...}
    """
    # Resize và chuyển về grayscale
    img1_gray = img1.resize(size, Image.Resampling.LANCZOS).convert('L')
    img2_gray = img2.resize(size, Image.Resampling.LANCZOS).convert('L')

    # Chuyển sang mảng numpy
    arr1 = np.array(img1_gray)
    arr2 = np.array(img2_gray)

    # Tính SSIM
    score_ssim, _ = ssim(arr1, arr2, full=True)

    return score_ssim

def remove_version_attribute(svg_str: str) -> str:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.ElementTree(ET.fromstring(svg_str))
    root = tree.getroot()

    if "version" in root.attrib:
        del root.attrib["version"]

    output = BytesIO()
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output.getvalue().decode("utf-8")


def add_ocr_decoy_svg(svg_code: str) -> str:
    """
    Adds nested circles with second darkest and second brightest colors from the existing SVG,
    positioned in one of the four corners (randomly selected) but positioned to avoid being
    cropped out during image processing.

    Parameters:
    -----------
    svg_code : str
        The original SVG string

    Returns:
    --------
    str
        Modified SVG with the nested circles added
    """
    import random
    from colorsys import rgb_to_hls

    # Check if SVG has a closing tag
    if "</svg>" not in svg_code:
        return svg_code

    # Extract viewBox if it exists to understand the dimensions
    viewbox_match = re.search(r'viewBox=["\'](.*?)["\']', svg_code)
    if viewbox_match:
        viewbox = viewbox_match.group(1).split()
        try:
            x, y, width, height = map(float, viewbox)
        except ValueError:
            # Default dimensions if we can't parse viewBox
            width, height = 384, 384
    else:
        # Default dimensions if viewBox not found
        width, height = 384, 384

    # Function to convert hex color to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join([c * 2 for c in hex_color])
        return tuple(int(hex_color[i: i + 2], 16) / 255 for i in (0, 2, 4))

    # Function to convert RGB to hex
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

    # Function to calculate color lightness
    def get_lightness(color):
        # Handle different color formats
        if color.startswith("#"):
            rgb = hex_to_rgb(color)
            return rgb_to_hls(*rgb)[1]  # Lightness is the second value in HLS
        elif color.startswith("rgb"):
            rgb_match = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color)
            if rgb_match:
                r, g, b = map(lambda x: int(x) / 255, rgb_match.groups())
                return rgb_to_hls(r, g, b)[1]
        return 0.5  # Default lightness if we can't parse

    # Extract all colors from the SVG
    color_matches = re.findall(
        r'(?:fill|stroke)="(#[0-9A-Fa-f]{3,6}|rgb\(\d+,\s*\d+,\s*\d+\))"', svg_code
    )

    # Default colors in case we don't find enough
    second_darkest_color = "#333333"  # Default to dark gray
    second_brightest_color = "#CCCCCC"  # Default to light gray

    if color_matches:
        # Remove duplicates and get unique colors
        unique_colors = list(set(color_matches))

        # Calculate lightness for each unique color
        colors_with_lightness = [
            (color, get_lightness(color)) for color in unique_colors
        ]

        # Sort by lightness (brightness)
        sorted_colors = sorted(colors_with_lightness, key=lambda x: x[1])

        # Handle different scenarios based on number of unique colors
        if len(sorted_colors) >= 4:
            # We have at least 4 unique colors - use 2nd darkest and 2nd brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[-2][0]
        elif len(sorted_colors) == 3:
            # We have 3 unique colors - use 2nd darkest and brightest
            second_darkest_color = sorted_colors[1][0]
            second_brightest_color = sorted_colors[2][0]
        elif len(sorted_colors) == 2:
            # We have only 2 unique colors - use the darkest and brightest
            second_darkest_color = sorted_colors[0][0]
            second_brightest_color = sorted_colors[1][0]
        elif len(sorted_colors) == 1:
            # Only one color - use it for second_darkest and a derived lighter version
            base_color = sorted_colors[0][0]
            base_lightness = sorted_colors[0][1]
            second_darkest_color = base_color

            # Create a lighter color variant if the base is dark, or darker if base is light
            if base_lightness < 0.5:
                # Base is dark, create lighter variant
                second_brightest_color = "#CCCCCC"
            else:
                # Base is light, create darker variant
                second_darkest_color = "#333333"

    # Ensure the colors are different
    if second_darkest_color == second_brightest_color:
        # If they ended up the same, modify one of them
        if get_lightness(second_darkest_color) < 0.5:
            # It's a dark color, make the bright one lighter
            second_brightest_color = "#CCCCCC"
        else:
            # It's a light color, make the dark one darker
            second_darkest_color = "#333333"

    # Base size for the outer circle
    base_outer_radius = width * 0.023

    # Randomize size by ±10%
    size_variation = base_outer_radius * 0.1
    outer_radius = base_outer_radius + \
        random.uniform(-size_variation, size_variation)

    # Define radii for inner circles based on outer radius
    middle_radius = outer_radius * 0.80
    inner_radius = middle_radius * 0.65

    # Calculate the maximum crop margin based on the image processing (5% of dimensions)
    # Add 20% extra margin for safety
    crop_margin_w = int(width * 0.05 * 1.2)
    crop_margin_h = int(height * 0.05 * 1.2)

    # Calculate center point based on the outer radius to ensure the entire circle stays visible
    safe_offset = outer_radius + max(crop_margin_w, crop_margin_h)

    # Choose a random corner (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
    corner = random.randint(0, 3)

    # Position the circle in the chosen corner, accounting for crop margin
    if corner == 0:  # Top-left
        center_x = safe_offset
        center_y = safe_offset
    elif corner == 1:  # Top-right
        center_x = width - safe_offset
        center_y = safe_offset
    elif corner == 2:  # Bottom-left
        center_x = safe_offset
        center_y = height - safe_offset
    else:  # Bottom-right
        center_x = width - safe_offset
        center_y = height - safe_offset

    # Add a small random offset (±10% of safe_offset) to make positioning less predictable
    random_offset = safe_offset * 0.1
    center_x += random.uniform(-random_offset, random_offset)
    center_y += random.uniform(-random_offset, random_offset)

    # Round to 1 decimal place to keep file size down
    outer_radius = round(outer_radius, 1)
    middle_radius = round(middle_radius, 1)
    inner_radius = round(inner_radius, 1)
    center_x = round(center_x, 1)
    center_y = round(center_y, 1)

    # Create the nested circles
    outer_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{outer_radius}" fill="{second_darkest_color}" />'
    middle_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{middle_radius}" fill="{second_brightest_color}" />'
    inner_circle = f'<circle cx="{center_x}" cy="{center_y}" r="{inner_radius}" fill="{second_darkest_color}" />'

    # Create a group element that contains all three circles
    group_element = f"<g>{outer_circle}{middle_circle}{inner_circle}</g>"

    # Insert the group element just before the closing SVG tag
    modified_svg = svg_code.replace("</svg>", f"{group_element}</svg>")

    # Calculate and add a comment with the byte size information
    outer_bytes = len(outer_circle.encode("utf-8"))
    middle_bytes = len(middle_circle.encode("utf-8"))
    inner_bytes = len(inner_circle.encode("utf-8"))
    total_bytes = outer_bytes + middle_bytes + inner_bytes

    corner_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    byte_info = (
        f"<!-- Circle bytes: outer={outer_bytes}, middle={middle_bytes}, "
        f"inner={inner_bytes}, total={total_bytes}, "
        f"colors: dark={second_darkest_color}, light={second_brightest_color}, "
        f"position: {corner_names[corner]} -->"
    )

    # modified_svg = modified_svg.replace("</svg>", f"{byte_info}</svg>")

    return modified_svg


def image_to_svg(image: Image, max_size: int = 10000) -> str:
    image_rgb = image.convert("RGBA")
    resized_img = image_rgb.resize((384, 384), Image.Resampling.LANCZOS)
    pixels = list(resized_img.getdata())

    speckle_values = [10, 20, 40]
    layer_diff_values = [64, 128]
    color_precision_values = [4, 5, 6, 8]
    # speckle_values = [10, 40, 60]
    # layer_diff_values = [64, 124]
    # color_precision_values = [4, 6, 8]

    best_svg = None
    best_params = {}
    best_similarity = 0.0
    best_size = 0  # theo dõi kích thước tốt nhất nhỏ hơn max_size
    best_aesthetic = 0.0

    for filter_speckle, layer_difference, color_precision in product(speckle_values, layer_diff_values, color_precision_values):
        svg_str = vtracer.convert_pixels_to_svg(
            rgba_pixels=pixels,
            size=resized_img.size,
            colormode="color",        # ["color"] or "binary"
            hierarchical="stacked",     # ["stacked"] or "cutout"
            mode="polygon",             # ["spline"], "polygon", "none"
            filter_speckle=filter_speckle,   # default: 4
            color_precision=color_precision,  # default: 6
            layer_difference=layer_difference,  # default: 16
            corner_threshold=60,  # default: 60
            length_threshold=4.0,  # in [3.5, 10] default: 4.0
            max_iterations=10,   # default: 10
            splice_threshold=45,  # default: 45
            path_precision=8,   # default: 8
        )

        options = scour.sanitizeOptions({
            'enable_comment_stripping': True,
            'remove_metadata': True,
            'remove_descriptions': True,
            'set_precision': 1,
            'remove_descriptive_elements': True,
            'strip_xml_prolog': True
        })

        add_o = add_ocr_decoy_svg(svg_str)
        optimized_svg = scour.scourString(add_o, options)
        byte_len = len(optimized_svg.encode("utf-8"))

        ssim = compare_pil_images(image, svg_to_png(svg_str))
        
        if byte_len <= max_size and byte_len > best_size:
            if best_similarity <= ssim:
                best_similarity = ssim
                best_svg = remove_version_attribute(optimized_svg)
                best_svg = re.sub(r"<\?xml[^>]+\?>\s*", "", best_svg)
                best_params = {
                    "filter_speckle": filter_speckle,
                    "layer_difference": layer_difference,
                    "color_precision": color_precision,
                    "ssim": ssim,
                }
                best_size = byte_len
    print(f"{best_params}")

    if best_svg:
        return best_svg
    else:
        return default_svg
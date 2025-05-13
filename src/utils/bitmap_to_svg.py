import re

import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


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
    import re
    from colorsys import rgb_to_hls, hls_to_rgb

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
        return tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))

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
    outer_radius = base_outer_radius + random.uniform(-size_variation, size_variation)

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


def add_manual_i_to_svg_no_group(
    svg_content: str,
    x: int = 50,  # Tọa độ X góc trên bên trái của chữ I
    y: int = 50,  # Tọa độ Y góc trên bên trái của chữ I
    width: int = 10,  # Chiều rộng của chữ I (thường hẹp)
    height: int = 40,  # Chiều cao của chữ I
    color: str = "black",  # Màu sắc của chữ I
    # stroke_width không cần thiết nếu dùng fill
) -> str:
    """
    Thêm chữ 'I' được vẽ thủ công bằng một <path> duy nhất vào nội dung SVG,
    không sử dụng thẻ <g>.

    Args:
        svg_content: Chuỗi chứa nội dung của file SVG gốc.
        x: Tọa độ X cho góc trên bên trái của chữ I.
        y: Tọa độ Y cho góc trên bên trái của chữ I.
        width: Chiều rộng của chữ I.
        height: Chiều cao của chữ I.
        color: Màu sắc của chữ I.

    Returns:
        Chuỗi chứa nội dung SVG mới với chữ I đã được thêm vào.

    Raises:
        ValueError: Nếu không tìm thấy thẻ đóng </svg> trong nội dung đầu vào.
    """

    # --- Định nghĩa hình dạng chữ I bằng một path hình chữ nhật ---
    # M = MoveTo, H = Horizontal LineTo, V = Vertical LineTo, Z = ClosePath
    path_d = f"M {x} {y} H {x + width} V {y + height} H {x} Z"

    # --- Tạo phần tử SVG cho chữ I ---
    # Tạo trực tiếp thẻ <path> với các thuộc tính
    # Sử dụng fill để tô đặc, stroke="none" để không có viền.
    # Nếu muốn chỉ có viền: fill="none" stroke="{color}" stroke-width="<value>"
    attributes = f'fill="{color}" stroke="none"'
    manual_i_element = f'\n  <path id="manual_I_{x}_{y}" d="{path_d}" {attributes} />\n'
    # Thêm \n để định dạng dễ nhìn hơn trong file SVG kết quả

    # --- Tìm vị trí chèn ---
    # Tìm vị trí của thẻ đóng </svg> (không phân biệt chữ hoa/thường)
    match = re.search(r"</svg>", svg_content, re.IGNORECASE)
    if not match:
        raise ValueError("Không tìm thấy thẻ đóng </svg> trong nội dung SVG.")

    insert_pos = match.start()

    # --- Chèn phần tử chữ I vào trước thẻ đóng ---
    modified_svg_content = (
        svg_content[:insert_pos] + manual_i_element + svg_content[insert_pos:]
    )

    return modified_svg_content


def high_score_svg_resize(
    svg_code: str,
    new_size: int = 384,
    padding_ratio: float = 0.1,
    min_stroke: float = 1.5,
    max_stroke: float = 16,
    preserve_aspect: bool = True,
) -> str:
    """
    Enhances the SVG code by adjusting the stroke width, font size, and scaling it to the new size.

    Parameters:
    svg_code (str): The original SVG code.
    new_size (int): The target size for the SVG (default 384).
    padding_ratio (float): Padding to apply around the SVG (default 0.1).
    min_stroke (float): Minimum stroke width (default 1.5).
    max_stroke (float): Maximum stroke width (default 16).
    preserve_aspect (bool): Whether to preserve the aspect ratio (default True).

    Returns:
    str: The enhanced SVG code.
    """
    root = ET.fromstring(svg_code)
    viewBox = root.get("viewBox")
    if viewBox is None:
        viewBox = "0 0 96 96"
        root.set("viewBox", viewBox)
    vb_x, vb_y, vb_w, vb_h = map(float, viewBox.strip().split())

    scale = (1 - 2 * padding_ratio) * new_size / max(vb_w, vb_h)
    translate_x = (new_size - vb_w * scale) / 2
    translate_y = (new_size - vb_h * scale) / 2

    # Transform the SVG content to adjust scaling and positioning
    g = ET.Element("g")
    transform = f"translate({translate_x:.2f},{translate_y:.2f}) scale({scale:.4f}) translate({-vb_x:.6f},{-vb_y:.6f})"
    g.set("transform", transform)

    # Remove structural elements and append to the main group
    structural_tags = {
        ET.QName("http://www.w3.org/2000/svg", ln)
        for ln in ["defs", "style", "title", "metadata", "script"]
    }
    for child in list(root):
        qname = ET.QName(child.tag)
        if qname in structural_tags:
            continue
        g.append(child)
        root.remove(child)
    root.append(g)

    # Update width, height, and viewBox to the new size
    root.set("width", str(new_size))
    root.set("height", str(new_size))
    root.set("viewBox", f"0 0 {new_size} {new_size}")
    if preserve_aspect:
        root.set("preserveAspectRatio", "xMidYMid meet")

    # Adjust stroke-width and font-size to fit the new scale
    def scale_visuals(el, scale_factor):
        for attr in ("stroke-width", "font-size"):
            if attr in el.attrib:
                try:
                    original = float(el.attrib[attr])
                    effective = original * scale_factor
                    clamped = max(min_stroke, min(max_stroke, effective))
                    new_val = clamped / scale_factor
                    el.attrib[attr] = f"{new_val:.2f}"
                except ValueError:
                    pass
        for child in el:
            scale_visuals(child, scale_factor)

    scale_visuals(g, scale)
    return ET.tostring(root, encoding="unicode")


def compress_hex_color(hex_color):
    """Convert hex color to shortest possible representation"""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        return f"#{r // 17:x}{g // 17:x}{b // 17:x}"
    return hex_color


def extract_features_by_scale(img_np, num_colors=16):
    """
    Extract image features hierarchically by scale

    Args:
        img_np (np.ndarray): Input image
        num_colors (int): Number of colors to quantize

    Returns:
        list: Hierarchical features sorted by importance
    """
    # Convert to RGB if needed
    if len(img_np.shape) == 3 and img_np.shape[2] > 1:
        img_rgb = img_np
    else:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    # Perform color quantization
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Quantized image
    palette = centers.astype(np.uint8)
    quantized = palette[labels.flatten()].reshape(img_rgb.shape)

    # Hierarchical feature extraction
    hierarchical_features = []

    # Sort colors by frequency
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_colors = [palette[i] for i in sorted_indices]

    # Center point for importance calculations
    center_x, center_y = width / 2, height / 2

    for color in sorted_colors:
        # Create color mask
        color_mask = cv2.inRange(quantized, color, color)

        # Find contours
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Convert RGB to compressed hex
        hex_color = compress_hex_color(f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")

        color_features = []
        for contour in contours:
            # Skip tiny contours
            area = cv2.contourArea(contour)
            # if area < 10:
            #     continue

            # Calculate contour center
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue

            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])

            # Distance from image center (normalized)
            dist_from_center = np.sqrt(
                ((cx - center_x) / width) ** 2 + ((cy - center_y) / height) ** 2
            )

            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Generate points string
            points = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])

            # Calculate importance (area, proximity to center, complexity)
            importance = area * (1 - dist_from_center) * (1 / (len(approx) + 1))

            color_features.append(
                {
                    "points": points,
                    "color": hex_color,
                    "area": area,
                    "importance": importance,
                    "point_count": len(approx),
                    "original_contour": approx,  # Store original contour for adaptive simplification
                }
            )

        # Sort features by importance within this color
        color_features.sort(key=lambda x: x["importance"], reverse=True)
        hierarchical_features.extend(color_features)

    # Final sorting by overall importance
    hierarchical_features.sort(key=lambda x: x["importance"], reverse=True)

    return hierarchical_features


def simplify_polygon(points_str, simplification_level):
    """
    Simplify a polygon by reducing coordinate precision or number of points

    Args:
        points_str (str): Space-separated "x,y" coordinates
        simplification_level (int): Level of simplification (0-3)

    Returns:
        str: Simplified points string
    """
    if simplification_level == 0:
        return points_str

    points = points_str.split()

    # Level 1: Round to 1 decimal place
    if simplification_level == 1:
        return " ".join(
            [
                f"{float(p.split(',')[0]):.1f},{float(p.split(',')[1]):.1f}"
                for p in points
            ]
        )

    # Level 2: Round to integer
    if simplification_level == 2:
        return " ".join(
            [
                f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                for p in points
            ]
        )

    # Level 3: Reduce number of points (keep every other point, but ensure at least 3 points)
    if simplification_level == 3:
        if len(points) <= 4:
            # If 4 or fewer points, just round to integer
            return " ".join(
                [
                    f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                    for p in points
                ]
            )
        else:
            # Keep approximately half the points, but maintain at least 3
            step = min(2, len(points) // 3)
            reduced_points = [points[i] for i in range(0, len(points), step)]
            # Ensure we keep at least 3 points and the last point
            if len(reduced_points) < 3:
                reduced_points = points[:3]
            if points[-1] not in reduced_points:
                reduced_points.append(points[-1])
            return " ".join(
                [
                    f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                    for p in reduced_points
                ]
            )

    return points_str


def bitmap_to_svg_layered(
    image,
    max_size_bytes=10000,
    resize=True,
    target_size=(384, 384),
    adaptive_fill=True,
    num_colors=None,
):
    """
    Convert bitmap to SVG using layered feature extraction with optimized space usage

    Args:
        image: Input image (PIL.Image)
        max_size_bytes (int): Maximum SVG size
        resize (bool): Whether to resize the image before processing
        target_size (tuple): Target size for resizing (width, height)
        adaptive_fill (bool): Whether to adaptively fill available space
        num_colors (int): Number of colors to quantize, if None uses adaptive selection

    Returns:
        str: SVG representation
    """
    # Adaptive color selection based on image complexity
    if num_colors is None:
        # Simple heuristic: more colors for complex images
        if resize:
            pixel_count = target_size[0] * target_size[1]
        else:
            pixel_count = image.size[0] * image.size[1]

        if pixel_count < 65536:  # 256x256
            num_colors = 8
        elif pixel_count < 262144:  # 512x512
            num_colors = 12
        else:
            num_colors = 16

    # Resize the image if requested
    if resize:
        original_size = image.size
        image = image.resize(target_size, Image.LANCZOS)
    else:
        original_size = image.size

    # Convert to numpy array
    img_np = np.array(image)

    # Get image dimensions
    height, width = img_np.shape[:2]

    # Calculate average background color
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        avg_bg_color = np.mean(img_np, axis=(0, 1)).astype(int)
        bg_hex_color = compress_hex_color(
            f"#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}"
        )
    else:
        bg_hex_color = "#fff"

    # Start building SVG
    # Use original dimensions in viewBox for proper scaling when displayed
    orig_width, orig_height = original_size
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}">\n'
    svg_bg = f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>\n'
    svg_base = svg_header + svg_bg
    svg_footer = "</svg>"

    # Calculate base size
    base_size = len((svg_base + svg_footer).encode("utf-8"))
    available_bytes = max_size_bytes - base_size

    # Extract hierarchical features
    features = extract_features_by_scale(img_np, num_colors=num_colors)

    # If not using adaptive fill, just add features until we hit the limit
    if not adaptive_fill:
        svg = svg_base
        for feature in features:
            # Try adding the feature
            feature_svg = (
                f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
            )

            # Check if adding this feature exceeds size limit
            if len((svg + feature_svg + svg_footer).encode("utf-8")) > max_size_bytes:
                break

            # Add the feature
            svg += feature_svg

        # Close SVG
        svg += svg_footer
        return svg

    # For adaptive fill, use binary search to find optimal simplification level

    # First attempt: calculate size of all features at different simplification levels
    feature_sizes = []
    for feature in features:
        feature_sizes.append(
            {
                "original": len(
                    f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'.encode(
                        "utf-8"
                    )
                ),
                "level1": len(
                    f'<polygon points="{simplify_polygon(feature["points"], 1)}" fill="{feature["color"]}" />\n'.encode(
                        "utf-8"
                    )
                ),
                "level2": len(
                    f'<polygon points="{simplify_polygon(feature["points"], 2)}" fill="{feature["color"]}" />\n'.encode(
                        "utf-8"
                    )
                ),
                "level3": len(
                    f'<polygon points="{simplify_polygon(feature["points"], 3)}" fill="{feature["color"]}" />\n'.encode(
                        "utf-8"
                    )
                ),
            }
        )

    # Two-pass approach: first add most important features, then fill remaining space
    svg = svg_base
    bytes_used = base_size
    added_features = set()

    # Pass 1: Add most important features at original quality
    for i, feature in enumerate(features):
        feature_svg = (
            f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
        )
        feature_size = feature_sizes[i]["original"]

        if bytes_used + feature_size <= max_size_bytes:
            svg += feature_svg
            bytes_used += feature_size
            added_features.add(i)

    # Pass 2: Try to add remaining features with progressive simplification
    for level in range(1, 4):  # Try simplification levels 1-3
        for i, feature in enumerate(features):
            if i in added_features:
                continue

            feature_size = feature_sizes[i][f"level{level}"]
            if bytes_used + feature_size <= max_size_bytes:
                feature_svg = f'<polygon points="{simplify_polygon(feature["points"], level)}" fill="{feature["color"]}" />\n'
                svg += feature_svg
                bytes_used += feature_size
                added_features.add(i)

    # Finalize SVG
    svg += svg_footer

    # Double check we didn't exceed limit
    final_size = len(svg.encode("utf-8"))
    if final_size > max_size_bytes:
        # If we somehow went over, return basic SVG
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>'

    # Calculate space utilization
    utilization = (final_size / max_size_bytes) * 100

    modified_svg = add_ocr_decoy_svg(svg)
    # Return the SVG with efficient space utilization
    return modified_svg

import re

import cv2
import numpy as np
from PIL import Image


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
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
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
            if area < 20:
                continue

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
    and adds a small "1" vector shape (rectangle) at the top-left.

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
        # Sử dụng RESIZE thay vì LANCZOS nếu phiên bản Pillow cũ hơn
        # Hoặc Image.Resampling.LANCZOS nếu Pillow mới hơn
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS  # Fallback cho Pillow cũ
        image = image.resize(target_size, resample_filter)
    else:
        original_size = image.size

    # Convert to numpy array, ensure RGB
    image_rgb = image.convert("RGB")
    img_np = np.array(image_rgb)

    # Get image dimensions (after potential resize) for viewBox
    height, width = img_np.shape[:2]

    # Calculate average background color
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        avg_bg_color = np.mean(img_np, axis=(0, 1)).astype(int)
        bg_hex_color = compress_hex_color(
            f"#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}"
        )
    else:  # Fallback for grayscale or unexpected formats
        bg_hex_color = "#fff"  # Default white

    # Start building SVG
    # Use original dimensions in width/height attributes for display size
    # Use potentially resized dimensions in viewBox for coordinate system
    orig_width, orig_height = original_size
    # Thêm xmlns namespace là thực hành tốt
    svg_header = f'<svg width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    # Dùng 100% thay vì width, height cố định
    svg_bg = f'<rect width="100%" height="100%" fill="{bg_hex_color}"/>\n'
    svg_base = svg_header + svg_bg
    svg_footer = "</svg>"

    # Calculate base size (bao gồm cả hình số 1)
    base_size = len((svg_base + svg_footer).encode("utf-8"))
    available_bytes = max_size_bytes - base_size
    if available_bytes < 0:  # Nếu base đã vượt quá giới hạn
        print(
            "Warning: Base SVG size exceeds max_size_bytes. Only base SVG will be returned."
        )
        # Có thể trả về SVG chỉ có nền và số 1 nếu muốn
        return svg_base + svg_footer  # Hoặc chỉ trả về SVG nền
        # return svg_header + svg_bg + svg_footer

    # Extract hierarchical features
    # Đảm bảo hàm này tồn tại và hoạt động đúng
    try:
        features = extract_features_by_scale(img_np, num_colors=num_colors)
    except NameError:
        print("Error: 'extract_features_by_scale' function is not defined.")
        return (
            svg_base + svg_footer
        )  # Trả về SVG cơ bản nếu không thể trích xuất feature

    # --- Phần còn lại của hàm (logic adaptive_fill / non-adaptive) giữ nguyên ---
    # Chỉ cần đảm bảo rằng `svg_base` đã bao gồm `svg_shape_1`
    # và `base_size` đã được tính toán chính xác.

    # If not using adaptive fill, just add features until we hit the limit
    if not adaptive_fill:
        svg = svg_base
        for feature in features:
            # Try adding the feature
            try:
                feature_svg = f'<polygon points="{feature["points"]}" fill="{compress_hex_color(feature["color"])}" />\n'
            except KeyError:
                print(
                    f"Warning: Skipping feature due to missing 'points' or 'color'. Feature: {feature}"
                )
                continue
            except NameError:
                print("Error: 'compress_hex_color' function is not defined.")
                # Xử lý lỗi, có thể dùng màu gốc hoặc bỏ qua
                feature_svg = f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'

            # Check if adding this feature exceeds size limit
            if len((svg + feature_svg + svg_footer).encode("utf-8")) > max_size_bytes:
                break

            # Add the feature
            svg += feature_svg

        # Close SVG
        svg += svg_footer
        return svg

    # --- Logic adaptive_fill (giữ nguyên như code gốc của bạn) ---
    # Đảm bảo sử dụng các hàm simplify_polygon, compress_hex_color thực tế
    feature_sizes = []
    for feature in features:
        try:
            # Use compress_hex_color here as well for consistency
            color = compress_hex_color(feature["color"])
            original_points = feature["points"]
            level1_points = simplify_polygon(original_points, 1)
            level2_points = simplify_polygon(original_points, 2)
            level3_points = simplify_polygon(original_points, 3)

            feature_sizes.append(
                {
                    "original": len(
                        f'<polygon points="{original_points}" fill="{color}" />\n'.encode(
                            "utf-8"
                        )
                    ),
                    "level1": len(
                        f'<polygon points="{level1_points}" fill="{color}" />\n'.encode(
                            "utf-8"
                        )
                    ),
                    "level2": len(
                        f'<polygon points="{level2_points}" fill="{color}" />\n'.encode(
                            "utf-8"
                        )
                    ),
                    "level3": len(
                        f'<polygon points="{level3_points}" fill="{color}" />\n'.encode(
                            "utf-8"
                        )
                    ),
                    # Lưu trữ points và color để sử dụng lại
                    "points": original_points,
                    "color": color,
                }
            )
        except KeyError:
            print(
                f"Warning: Skipping feature sizing due to missing 'points' or 'color'. Feature: {feature}"
            )
            continue
        except NameError:
            print(
                "Error: 'compress_hex_color' or 'simplify_polygon' function is not defined."
            )
            # Handle error appropriately, maybe skip feature or use defaults
            return svg_base + svg_footer  # Exit early if core functions missing

    # Two-pass approach
    svg = svg_base
    bytes_used = base_size
    added_features_indices = set()  # Theo dõi index của feature đã thêm

    # Pass 1: Add most important features at original quality (using pre-calculated/stored data)
    for i, size_info in enumerate(feature_sizes):
        feature_size = size_info["original"]
        if bytes_used + feature_size <= max_size_bytes:
            feature_svg = f'<polygon points="{size_info["points"]}" fill="{size_info["color"]}" />\n'
            svg += feature_svg
            bytes_used += feature_size
            added_features_indices.add(i)

    # Pass 2: Try to add remaining features with progressive simplification
    for level in range(1, 4):
        for i, size_info in enumerate(feature_sizes):
            if i in added_features_indices:
                continue

            level_key = f"level{level}"
            feature_size = size_info[level_key]

            if bytes_used + feature_size <= max_size_bytes:
                # Cần gọi lại simplify_polygon vì chỉ lưu kích thước, không lưu points đã simplify
                try:
                    simplified_points = simplify_polygon(size_info["points"], level)
                except NameError:
                    print("Error: 'simplify_polygon' function is not defined.")
                    # Handle error: skip or exit
                    continue  # Skip this feature/level if function missing

                feature_svg = f'<polygon points="{simplified_points}" fill="{size_info["color"]}" />\n'
                svg += feature_svg
                bytes_used += feature_size
                added_features_indices.add(i)

    # Finalize SVG
    svg += svg_footer

    # Double check we didn't exceed limit (should be less likely now with careful checks)
    final_size = len(svg.encode("utf-8"))
    if final_size > max_size_bytes:
        print(
            f"Warning: Final SVG size {final_size} slightly exceeded limit {max_size_bytes}. Returning base SVG."
        )
        # Fallback an toàn nhất là trả về SVG cơ bản
        return f'<svg width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="{bg_hex_color}"/><rect x="2" y="2" width="1" height="2" fill="#000"/></svg>'

    # Calculate space utilization (optional)
    # utilization = (final_size / max_size_bytes) * 100
    # print(f"SVG generated. Size: {final_size} bytes. Utilization: {utilization:.2f}%")

    return add_manual_i_to_svg_no_group(svg)
    # return svg

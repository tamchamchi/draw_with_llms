import re

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Any, Dict, List


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


def extract_features_by_scale(img_np, num_colors=16) -> List[Dict[str, Any]]:
    """
    Trích xuất features dạng hình tròn (circle) hoặc đường dẫn (path) từ ảnh.

    Returns:
        List[Dict[str, Any]]: Danh sách các feature, mỗi feature là dict chứa:
            'type': 'circle' hoặc 'path'
            'color': Màu hex (đã nén)
            'importance': Điểm số tầm quan trọng
            Các thuộc tính khác tùy theo 'type' ('cx', 'cy', 'r' cho circle; 'd' cho path)
    """
    # --- Phần đầu hàm: chuyển sang RGB, grayscale, KMeans (giữ nguyên) ---
    if len(img_np.shape) == 3 and img_np.shape[2] > 1:
        img_rgb = img_np
    else:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape

    pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    palette = centers.astype(np.uint8)

    quantized = palette[labels.flatten()].reshape(img_rgb.shape)

    unique_labels, counts = np.unique(labels, return_counts=True)

    sorted_indices = np.argsort(-counts)

    sorted_colors = [palette[i] for i in sorted_indices]

    center_x, center_y = width / 2, height / 2
    # -----------------------------------------------------------------------------

    all_features = []

    for color_index, color in enumerate(sorted_colors):
        # Đảm bảo hàm compress_hex_color tồn tại và hoạt động
        try:
            hex_color = compress_hex_color(
                f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            )
        except NameError:
            # Fallback nếu thiếu hàm
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            print("Warning: compress_hex_color not defined.")

        color_mask = cv2.inRange(quantized, color, color)
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            MIN_AREA_THRESHOLD = 20
            if contour_area < MIN_AREA_THRESHOLD:
                continue

            feature = None  # Khởi tạo

            # --- BƯỚC PHÂN TÍCH HÌNH DẠNG CONTOUR ---

            # 1. Thử khớp Hình tròn (Circle)
            try:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * (radius**2)
                area_ratio_circle = (
                    contour_area / circle_area if circle_area > 1e-6 else 0
                )
                CIRCLE_AREA_RATIO_THRESHOLD = 0.95  # Ngưỡng có thể điều chỉnh
                if area_ratio_circle > CIRCLE_AREA_RATIO_THRESHOLD:
                    feature = {
                        "type": "circle",
                        "cx": float(cx),
                        "cy": float(cy),
                        "r": float(radius),
                        "color": hex_color,
                        "area": contour_area,
                    }
                    # print(f"    Detected Circle (Area ratio: {area_ratio_circle:.2f})")
            except Exception as e_circle:
                print(f"    Error checking circle: {e_circle}")

            # 2. Nếu không phải hình tròn -> Mặc định dùng Path
            if feature is None:
                try:
                    # Dùng epsilon để xấp xỉ đa giác trước khi tạo path
                    # Điều chỉnh epsilon nếu cần
                    epsilon_path = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon_path, True)

                    if len(approx) > 0:
                        # Chuyển đổi approx thành chuỗi 'd' cho <path> (M = MoveTo, L = LineTo, Z = ClosePath)
                        path_d_list = [
                            f"M {p[0][0]:.1f},{p[0][1]:.1f}"
                            for i, p in enumerate(approx)
                            if i == 0
                        ]
                        path_d_list.extend(
                            [
                                f"L {p[0][0]:.1f},{p[0][1]:.1f}"
                                for i, p in enumerate(approx)
                                if i > 0
                            ]
                        )
                        path_d_list.append("Z")  # Đóng path lại
                        path_d_string = " ".join(path_d_list)

                        feature = {
                            "type": "path",
                            "d": path_d_string,
                            "color": hex_color,
                            "area": contour_area,
                            # Số điểm sau khi xấp xỉ
                            "point_count": len(approx),
                        }
                        # print(f"    Defaulted to Path (Points: {len(approx)})")
                    else:
                        print("    Warning: approxPolyDP resulted in empty points.")

                except Exception as e_path:
                    print(f"    Error creating path: {e_path}")

            # --- Tính Importance và Thêm Feature ---
            if feature is not None:
                m = cv2.moments(contour)
                cx_moment = (m["m10"] / m["m00"]) if m["m00"] != 0 else center_x
                cy_moment = (m["m01"] / m["m00"]) if m["m00"] != 0 else center_y
                dist_from_center = np.sqrt(
                    ((cx_moment - center_x) / width) ** 2
                    + ((cy_moment - center_y) / height) ** 2
                )
                # Ước lượng độ phức tạp: hình tròn có thể coi là đơn giản (ít điểm ảo)
                # Ví dụ coi hình tròn như 3-4 điểm
                complexity_factor = (
                    1.0 / (feature.get("point_count", 4) + 1)
                    if feature["type"] == "path"
                    else 1.0 / (3 + 1)
                )

                feature["importance"] = (
                    feature.get("area", 0) * (1 - dist_from_center) * complexity_factor
                )
                all_features.append(feature)

    # --- Sắp xếp features theo importance ---
    all_features.sort(key=lambda x: x.get("importance", 0), reverse=True)
    print(f"--- Extracted {len(all_features)} features (Circle/Path only) ---")
    return all_features


def bitmap_to_svg_layered(
    image: Image.Image,
    max_size_bytes: int = 10000,
    resize: bool = True,
    target_size: Tuple[int, int] = (384, 384),
    adaptive_fill: bool = True,
    num_colors: Optional[int] = None,  # num_colors giờ là tùy chọn hơn
    # Thêm các tham số cho việc khớp primitive nếu cần tinh chỉnh
    circle_threshold: float = 0.95,
    path_epsilon_factor: float = 0.01,
) -> str:
    """
    Chuyển đổi bitmap sang SVG sử dụng primitive fitting và path/polygon fallback.
    """
    # --- Phần đầu hàm: adaptive num_colors, resize, tính bg_color (giữ nguyên) ---
    if num_colors is None:
        # ... (logic xác định num_colors dựa trên kích thước như cũ) ...
        if image.size[0] * image.size[1] < 65536:
            num_colors = 8
        elif image.size[0] * image.size[1] < 262144:
            num_colors = 12
        else:
            num_colors = 16

    if resize:
        original_size = image.size
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        image = image.resize(target_size, resample_filter)
    else:
        original_size = image.size

    image_rgb = image.convert("RGB")
    img_np = np.array(image_rgb)
    height, width = img_np.shape[:2]
    # ... (logic tính bg_hex_color như cũ) ...
    try:
        avg_bg_color = np.mean(img_np, axis=(0, 1)).astype(int)
        bg_hex_color = compress_hex_color(
            f"#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}"
        )
    except NameError:
        bg_hex_color = "#fff"
        print("Warning: compress_hex_color not defined, using #fff background")
    except Exception:
        bg_hex_color = "#fff"  # Fallback

    orig_width, orig_height = original_size
    svg_header = f'<svg width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">\n'
    svg_bg = f'<rect width="100%" height="100%" fill="{bg_hex_color}"/>\n'
    svg_base = svg_header + svg_bg
    svg_footer = "</svg>"
    base_size = len((svg_base + svg_footer).encode("utf-8"))
    available_bytes = max_size_bytes - base_size
    if available_bytes < 0:
        print("Warning: Base SVG size exceeds limit.")
        return svg_base + svg_footer
    # -----------------------------------------------------------------------------

    # --- Gọi hàm trích xuất feature đã sửa đổi ---
    try:
        # Truyền thêm các ngưỡng nếu muốn hàm extract tùy biến được
        features = extract_features_by_scale(img_np, num_colors=num_colors)
    except NameError:
        print("Error: 'extract_features_by_scale' is not defined.")
        return svg_base + svg_footer
    except Exception as e_feat:
        print(f"Error during feature extraction: {e_feat}")
        return svg_base + svg_footer

    # --- Xây dựng nội dung SVG từ features ---
    svg_elements = ""
    current_bytes = base_size

    # Format số float (ví dụ: 1 chữ số thập phân)
    FMT = ".1f"

    for feature in features:
        feature_type = feature.get("type")
        color = feature.get("color", "#000")  # Màu mặc định nếu thiếu
        element_str = ""

        try:
            # --- Tạo chuỗi SVG element dựa trên type ---
            if feature_type == "circle":
                element_str = f'<circle cx="{feature["cx"]:{FMT}}" cy="{feature["cy"]:{FMT}}" r="{feature["r"]:{FMT}}" fill="{color}" />\n'
            elif feature_type == "path":
                element_str = f'<path d="{feature["d"]}" fill="{color}" />\n'
            elif feature_type == "polygon":
                element_str = (
                    f'<polygon points="{feature["points"]}" fill="{color}" />\n'
                )
            else:
                print(f"Warning: Unknown feature type '{feature_type}'. Skipping.")
                continue

            # --- Kiểm tra dung lượng ---
            element_size = len(element_str.encode("utf-8"))
            if current_bytes + element_size <= max_size_bytes:
                svg_elements += element_str
                current_bytes += element_size
            else:
                # Nếu vượt quá dung lượng, dừng thêm feature
                print(
                    f"--- Reached size limit ({current_bytes} + {element_size} > {max_size_bytes}). Stopping feature addition. ---"
                )
                break  # Dừng vòng lặp

        except KeyError as ke:
            print(
                f"Warning: Skipping feature due to missing key {ke}. Feature data: {feature}"
            )
            continue
        except Exception as e_build:
            print(
                f"Warning: Error building SVG element for feature: {e_build}. Feature data: {feature}"
            )
            continue

    # --- Kết hợp và trả về SVG cuối cùng (chưa có logic adaptive fill phức tạp) ---
    final_svg = svg_base + svg_elements + svg_footer

    # Tùy chọn: Thêm chữ 'I' vào cuối nếu vẫn muốn
    final_svg = add_manual_i_to_svg_no_group(final_svg)

    print(f"--- SVG generated. Final size: {len(final_svg.encode('utf-8'))} bytes ---")
    return final_svg

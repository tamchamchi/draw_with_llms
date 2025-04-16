import os
import json


def save_results_to_json(results, filename="generated_outputs.json"):
    """Lưu kết quả vào file JSON dưới dạng danh sách và in ra kết quả trung bình của cột total_score."""

    # Kiểm tra nếu file đã tồn tại
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
                if not isinstance(data, list):  # Đảm bảo data là một danh sách
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Thêm kết quả mới vào danh sách
    data.extend(results)  # Nếu results là list(dict()), dùng extend thay vì append

    # Ghi lại vào file JSON
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

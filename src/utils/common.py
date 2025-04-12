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


def naming_template(t, attempt, quality):
    return f"{t} - {attempt} - {quality:.4f}.png"


def score_caption(total, vqa, aesthetic, ocr):
    return f"TOTAL:{total:.4f} VQA:{vqa:.4f} Aesthetic:{aesthetic:.4f} OCR:{ocr:.4f}"


def vqa_ocr_caption(vqa, ocr, num_char):
    return f"VQA:{vqa:.4f} OCR:{ocr:.4f} Num_char:{num_char}"

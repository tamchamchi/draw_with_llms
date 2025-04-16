def naming_template(t, attempt, quality):
    return f"{t} - {attempt} - {quality:.4f}.png"


def score_caption(total, vqa, aesthetic, ocr):
    return f"TOTAL:{total:.4f} VQA:{vqa:.4f} Aesthetic:{aesthetic:.4f} OCR:{ocr:.4f}"


def vqa_ocr_caption(vqa, ocr, num_char):
    return f"VQA:{vqa:.4f} OCR:{ocr:.4f} Num_char:{num_char}"

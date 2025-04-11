import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.configs import INTERNAL_DATA_DIR
from src.models.global_models import aesthetic_evaluator, my_model
from src.prompt import version_11, version_12
from src.utils.global_variable import data

PREFIX_PROMPT = version_11.get("PREFIX_PROMPT")
SUFFIX_PROMPT = version_11.get("SUFFIX_PROMPT")
NEGATIVE_PROMPT = version_11.get("NEGATIVE_PROMPT")
VERSION = version_11.get("VERSION")

TEST_SIMILARITY_DIR = os.path.join(INTERNAL_DATA_DIR, "test_similarity")
os.makedirs(TEST_SIMILARITY_DIR, exist_ok=True)

train_data = data.get_train_csv()

for i, (idx, desc) in enumerate(train_data.itertuples(index=False)):
    prompt = f"{PREFIX_PROMPT} {desc} {SUFFIX_PROMPT}"

    bit_map = my_model.generate(
        prompt=prompt,
        height=512,
        width=512,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=25,
        guidance_scale=10,
        seed=42,
    )

    clip_similarity_score = aesthetic_evaluator.compute_clip_similarity(
        image=bit_map, text=prompt
    )

    # Tạo folder con tương ứng với mỗi idx (nếu chưa tồn tại)
    subfolder_path = os.path.join(TEST_SIMILARITY_DIR, str(idx))
    os.makedirs(subfolder_path, exist_ok=True)  # Tạo nếu chưa có

    # Tạo tên file ảnh và đường dẫn đầy đủ
    image_file_name = f"{clip_similarity_score:.4f} - {VERSION}.png"
    image_path = os.path.join(subfolder_path, image_file_name)

    bit_map.save(image_path, format="png")

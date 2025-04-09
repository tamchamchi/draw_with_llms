import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.vqa_model import VQAEvaluator
from src.models.stable_diffusion_v2 import StableDiffusionV2
from src.data.make_dataset import Data
from build_png2svg import bitmap_to_svg_layered
from build_image_processor import ImageProcessor, svg_to_png
from src.configs import RAW_DATA_DIR, RESULTS_DIR


TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/train.csv")
QUESTION_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/questions.parquet")
IMAGE_DIR = os.path.join(RESULTS_DIR, "image")

PREFIX_PROMPT = "Simple, Classic, Minimal illustration, ((flat vector icon)) of"
SUFFIX_PROMPT = "with flat color blocks, high contrast, beautiful, outline only, no details, solid color only."
NEGATIVE_PROMPT = "photorealistic, blurry, noisy, complex background, detailed texture, shadows, gradients"


def evaluation_ocr_score():
    sd_model = StableDiffusionV2()
    vqa_evaluator = VQAEvaluator()

    data = Data(TRAIN_DATA_PATH, QUESTION_DATA_PATH)
    RAW_IMAGE_DIR = os.path.join(IMAGE_DIR, "raw_image")
    PROCESSED_IMAGE_DIR = os.path.join(IMAGE_DIR, "processed_image")
    RESET_IMAGE_DIR = os.path.join(IMAGE_DIR, "reset_image")

    train_data = data.get_train_csv()
    rng = np.random.RandomState(42)

    for row in train_data.itertuples(index=False):
        prompt = f"{PREFIX_PROMPT} {row.description} {SUFFIX_PROMPT}"
        solution = data.get_solution(row.id)
        questions = solution["question"].tolist()  # list[str]
        choices_list = (
            solution["choices"]
            .apply(eval)
            .apply(lambda x: x[0] if isinstance(x[0], list) else x)
            .tolist()
        )  # list[list[str]]
        answers = solution["answer"].tolist()  # list[str]

        bitmap = sd_model.generate(
            prompt=prompt,
            height=256,
            width=256,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=25,
            guidance_scale=15,
            seed=42,
        )

        bitmap.save(f"{RAW_IMAGE_DIR}/image_{row.id}_raw.png", format="PNG")

        svg = bitmap_to_svg_layered(
            image=bitmap,
            max_size_bytes=10000,
            resize=True,
            target_size=(256, 256),
            adaptive_fill=True,
            num_colors=5,
        )

        group_seed = rng.randint(0, np.iinfo(np.int32).max)
        image_processor = ImageProcessor(image=svg_to_png(svg), seed=group_seed).apply()
        image_processor.image.save(
            f"{PROCESSED_IMAGE_DIR}/image_{row.id}_processed.png", format="PNG"
        )

        image = image_processor.image.copy()
     
        vqa_score = vqa_evaluator.score_batch(
            image=image,
            questions=questions,
            choices_list=choices_list,
            answers=answers,
        )

        image_processor.reset().apply_random_crop_resize().apply_jpeg_compression(
            quality=90
        )
        image_processor.image.save(
            f"{RESET_IMAGE_DIR}/image_{row.id}_reset.png", format="PNG"
        )
        ocr_score, num_char = vqa_evaluator.ocr(
            image_processor.image, free_chars=4, use_num_char=True
        )
        print("-" * 50)
        print(f"Description: {row.description}")
        print(f"VAQ_SCORE: {vqa_score}")
        print(f"OCR_SCORE: {ocr_score}")
        print(f"Number char: {num_char}")


evaluation_ocr_score()

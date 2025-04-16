import json
import os

from configs.configs import RAW_DATA_DIR
from src.data.data_loader import Data
from src.evaluators.aesthetic import AestheticEvaluator
from src.evaluators.score_evaluator import ScoreEvaluator
from src.evaluators.vqa import VQAEvaluator
from src.generators.stable_diffusion_v2 import StableDiffusionV2

TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/train.csv")
QUESTION_DATA_PATH = os.path.join(
    RAW_DATA_DIR, "drawing-with-llms/questions.parquet")

svg_config = {
    'num_colors': 12,  # Make sure num_color variable exists here
    'max_size_bytes': 10000,  # Make sure these exist too
    'resize': True,
    'target_size': (384, 384),
    'adaptive_fill': True
}
if __name__ == "__main__":
    vqa_evaluator = VQAEvaluator()
    aesthetic_evaluator = AestheticEvaluator()
    generator = StableDiffusionV2()
    data = Data(TRAIN_DATA_PATH, QUESTION_DATA_PATH)

    evaluator = ScoreEvaluator(
        vqa_evaluator=vqa_evaluator,
        aesthetic_evluator=aesthetic_evaluator,
        generator=generator,
        data=data
    )

    results_no_compress = evaluator.generate_and_evaluate(
        id_prompt="02d892",
        prefix_prompt="Watercolor painting of",
        suffix_prompt="on textured paper",
        negative_prompt="photorealistic",
        bitmap2svg_configs=svg_config,
        use_image_compression=False,
        compression_k=16,  # Giá trị này bị bỏ qua
        num_attempts=1,
        verbose=True,
        version="exp_v3_k_param",
        random_seed=42
    )
    print("\n--- Final Results (No Compression) ---")
    print(json.dumps(results_no_compress, indent=4))

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.build_evaluation import ScoreEvaluation
from src.utils.save_results import save_results_to_json
from src.configs import SCORE_DIR, RAW_DATA_DIR
from src.data.make_dataset import Data
from src.models.global_models import vqa_evaluator, aesthetic_evaluator, my_model
from src.prompt import version_11

PREFIX_PROMPT = version_11.get("PREFIX_PROMPT")
SUFFIX_PROMPT = version_11.get("SUFFIX_PROMPT")
NEGATIVE_PROMPT = version_11.get("NEGATIVE_PROMPT")
VERSION = version_11.get("VERSION")

TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/train.csv")
QUESTION_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/questions.parquet")

data = Data(TRAIN_DATA_PATH, QUESTION_DATA_PATH)


def run_generate_and_evaluation():
    use_image_compression = True
    if use_image_compression:
        res_path = os.path.join(
            SCORE_DIR,
            f"result_have_compression_image_{version_11.get('id_prompt')}.json",
        )
    else:
        res_path = os.path.join(
            SCORE_DIR, f"result_no_compression_image_{version_11.get('id_prompt')}.json"
        )

    evaluation = ScoreEvaluation(
        vqa_evaluator=vqa_evaluator,
        aesthetic_evaluator=aesthetic_evaluator,
        my_model=my_model,
        data=data,
    )

    train_data = data.get_train_csv()

    results = []

    for i, (idx, desc) in enumerate(train_data.itertuples(index=False)):
        result = evaluation.generate_and_evaluation(
            id_prompt=idx,
            prefix_prompt=PREFIX_PROMPT,
            suffix_prompt=SUFFIX_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=512,
            height=512,
            num_color=12,
            num_inference_steps=25,
            guidance_scale=15,
            use_image_compression=use_image_compression,
            version=VERSION,
            num_attempts=1,
            verbose=True,
            random_seed=42,
        )
        output = {"id_result": i, "id_prompt": version_11.get("id_prompt"), **result}

        results.append(output)

    save_results_to_json(results=results, filename=res_path)


if __name__ == "__main__":
    run_generate_and_evaluation()

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.build_evaluation import ScoreEvaluation
from src.utils.common import save_results_to_json
from src.configs import SCORE_DIR, RAW_DATA_DIR
from src.data.make_dataset import Data
from src.models.global_models import vqa_evaluator, aesthetic_evaluator, my_model
from src.versions import versions


TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/train.csv")
QUESTION_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/questions.parquet")

data = Data(TRAIN_DATA_PATH, QUESTION_DATA_PATH)

version = "version_16"
VERSION = versions.get(version).copy()


def run_generate_and_evaluation():
    use_image_compression = True
    if use_image_compression:
        res_path = os.path.join(
            SCORE_DIR,
            f"result_have_compression_image_{VERSION.get('id_prompt')}.json",
        )
    else:
        res_path = os.path.join(
            SCORE_DIR, f"result_no_compression_image_{VERSION.get('id_prompt')}.json"
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
            prefix_prompt=VERSION.get("prefix_prompt"),
            suffix_prompt=VERSION.get("suffix_prompt"),
            negative_prompt=VERSION.get("negative_prompt"),
            width=VERSION.get("width"),
            height=VERSION.get("height"),
            num_color=VERSION.get("num_color"),
            num_inference_steps=VERSION.get("num_inference_steps"),
            guidance_scale=VERSION.get("guidance_scale"),
            use_image_compression=VERSION.get("use_image_compression"),
            version=f"{version}-{VERSION.get('version_description')}",
            num_attempts=VERSION.get("num_attempts"),
            verbose=VERSION.get("verbose"),
            random_seed=VERSION.get("random_seed"),
        )
        output = {"id_result": i, "id_prompt": VERSION.get("id_prompt"), **result}

        results.append(output)

    save_results_to_json(results=results, filename=res_path)


if __name__ == "__main__":
    run_generate_and_evaluation()

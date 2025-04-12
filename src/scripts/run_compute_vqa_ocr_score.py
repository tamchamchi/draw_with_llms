import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.build_evaluation import ScoreEvaluation
from src.utils.common import save_results_to_json, vqa_ocr_caption
from src.configs import RAW_DATA_DIR, INTERNAL_DATA_DIR
from src.data.make_dataset import Data
from src.models.global_models import vqa_evaluator, aesthetic_evaluator, my_model
from src.versions import versions

PREFIX_PROMPT = versions.get("version_12").get("PREFIX_PROMPT")
SUFFIX_PROMPT = versions.get("version_12").get("SUFFIX_PROMPT")
NEGATIVE_PROMPT = versions.get("version_12").get("NEGATIVE_PROMPT")
VERSION = versions.get("version_12").get("VERSION")

TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/train.csv")
QUESTION_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/questions.parquet")

data = Data(TRAIN_DATA_PATH, QUESTION_DATA_PATH)

def run_compute_vqa_ocr_score():
    pass
from models.vqa_model import VQAEvaluator
from models.aesthetic_model import AestheticEvaluator
from models.stable_diffusion_v2 import StableDiffusionV2
from features.build_calc_total_score import score
from features.build_image_processor import ImageProcessor, svg_to_png
from features.build_png2svg import bitmap_to_svg_layered
from src.data.make_dataset import Data
from PIL import Image
import os
from configs import RESULTS_DIR, RAW_DATA_DIR

# Khai báo các biến toàn cục để lưu mô hình
vqa_evaluator = None
aesthetic_evaluator = None
pipe = None

# Hàm để tải mô hình nếu chưa được tải
def load_models():
     global vqa_evaluator, aesthetic_evaluator, pipe
     
     if vqa_evaluator is None:
          vqa_evaluator = VQAEvaluator()
     if aesthetic_evaluator is None:
          aesthetic_evaluator = AestheticEvaluator()
     if pipe is None:
          pipe = StableDiffusionV2()

def main():
     pass


if __name__ == "__main__":
     main()
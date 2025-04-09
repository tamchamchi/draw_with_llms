from src.models.aesthetic_model import AestheticEvaluator
from src.models.stable_diffusion_v2 import StableDiffusionV2
from src.models.vqa_model import VQAEvaluator

vqa_evaluator = VQAEvaluator()
aesthetic_evaluator = AestheticEvaluator()
my_model = StableDiffusionV2()
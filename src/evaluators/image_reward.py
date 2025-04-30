import ImageReward as RM
from PIL import Image


class IREvaluator:
    def __init__(self):
        self.model = RM.load("ImageReward-v1.0")

    def score(self, prompt: str, image: Image.Image) -> float:
        return self.model.score(prompt=prompt, image=image)

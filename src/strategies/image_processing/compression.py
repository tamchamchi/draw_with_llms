from PIL import Image

from src.utils.image_utils import image_compression

from ..base import ImageProcessingStrategy


class CompressionStrategy(ImageProcessingStrategy):
    """Image compression strategy"""

    def __init__(self, k: int = 8) -> None:
        print(f"--- Strategy: Init CompressionStrategy with k={k} ---")
        if k <= 0:
            raise ValueError("Parameter k for compression must be positive.")
        self.k = k

    def process(self, image: Image.Image) -> Image.Image:
        print("--- Strategy: Image compression ---")
        return image_compression(image, k=self.k)

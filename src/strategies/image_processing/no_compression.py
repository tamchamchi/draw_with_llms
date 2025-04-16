from PIL import Image

from ..base import ImageProcessingStrategy


class NoCompressionStrategy(ImageProcessingStrategy):
    """No Image Compression Strategy"""

    def process(self, image: Image.Image) -> Image.Image:
        print("--- Strategy: No Image Compression ---")
        return image

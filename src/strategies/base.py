from abc import ABC, abstractmethod

from PIL import Image


class ImageProcessingStrategy(ABC):
    """Inferface to image processing strartegy before SVG convertion."""
    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        pass

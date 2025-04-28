from abc import ABC, abstractmethod
from PIL import Image

from typing import Dict, Optional


class ImageProcessingStrategy(ABC):
    """Interface to image processing strartegy before SVG convertion."""
    @abstractmethod
    def process(self, image: Image.Image) -> Image.Image:
        pass


class PromptBuildingStrategy(ABC):
    @abstractmethod
    # Bỏ các tham số prompt gốc khỏi signature nếu strategy này không dùng đến
    def build(self, description: str, **kwargs) -> Dict[str, Optional[str]]:
        pass



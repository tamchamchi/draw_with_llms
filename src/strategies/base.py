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

class SimilarityRewardStrategy(ABC):
    """Interface cho các chiến lược tính điểm tương đồng hoặc điểm thưởng."""
    @abstractmethod
    def calculate(self, image: Image.Image, description: Optional[str] = None, **kwargs) -> Optional[float]:
        """
        Tính toán điểm số.

        Args:
            image (Image.Image): Ảnh cần đánh giá.
            description (Optional[str]): Mô tả text (cần cho CLIP).
            **kwargs: Các tham số bổ sung.

        Returns:
            Optional[float]: Điểm số tính được hoặc None nếu lỗi.
        """
        pass


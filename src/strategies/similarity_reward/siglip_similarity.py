from ..base import SimilarityRewardStrategy
from PIL import Image
from typing import Any, Optional

# --- Concrete Strategy cho CLIP Similarity ---


class SIGLIPSimilarityStrategy(SimilarityRewardStrategy):
    """Sử dụng AestheticEvaluator để tính SIGLIP Similarity."""

    # Dùng Any để tránh import vòng tròn nếu cần
    def __init__(self, aesthetic_evaluator: Any):
        """Khởi tạo với đối tượng aesthetic_evaluator."""
        self.aesthetic_evaluator = aesthetic_evaluator
        print("--- Similarity/Reward Strategy: Initialized SIGLIPSimilarityStrategy ---")
        if not hasattr(self.aesthetic_evaluator, "compute_siglip_similarity"):
            raise TypeError(
                "Provided aesthetic_evaluator does not have 'compute_siglip_similarity' method."
            )

    def calculate(
        self, image: Image.Image, description: Optional[str] = None, **kwargs
    ) -> Optional[float]:
        if description is None:
            print(
                "Warning: SIGLIPSimilarityStrategy requires a description, none provided."
            )
            return None
        try:
            print("  Calculating SIGLIP Similarity...")
            print(f"   Prompt: {description}")
            # Giả sử hàm này trả về một số float
            score = self.aesthetic_evaluator.compute_siglip_similarity(image, description)
            # Chuyển đổi kiểu nếu cần và kiểm tra không phải NaN/Inf
            score = float(score)
            print(f"  SIGLIP Similarity Score: {score:.4f}")
            return score
        except Exception as e:
            print(f"  Error calculating SIGLIP similarity: {e}")
            # import traceback; traceback.print_exc() # Bỏ comment để debug sâu
            return None

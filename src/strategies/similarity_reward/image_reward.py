from typing import Any, Optional

from PIL import Image

from ..base import SimilarityRewardStrategy


# --- Concrete Strategy cho ImageReward ---
class ImageRewardStrategy(SimilarityRewardStrategy):
    """Sử dụng ImageRewardEvaluator (giả định) để tính điểm thưởng."""

    def __init__(self, image_reward_evaluator: Any):  # Dùng Any
        """Khởi tạo với đối tượng image_reward_evaluator."""
        self.image_reward_evaluator = image_reward_evaluator
        print("--- Similarity/Reward Strategy: Initialized ImageRewardStrategy ---")
        # !!! Quan trọng: Đảm bảo image_reward_evaluator có phương thức 'score' hoặc tương tự !!!
        if not hasattr(self.image_reward_evaluator, "score"):
            raise TypeError(
                "Provided image_reward_evaluator does not have 'score' method."
            )

    def calculate(
        self, image: Image.Image, description: Optional[str] = None, **kwargs
    ) -> Optional[float]:
        try:
            print("  Calculating ImageReward Score...")
            # !!! Gọi phương thức tính điểm của ImageRewardEvaluator !!!
            # Giả sử tên phương thức là 'score' và chỉ nhận 'image'
            score = self.image_reward_evaluator.score(
                prompt=description, image=image
            )  # Sửa lại nếu tên/tham số khác
            # Chuyển đổi kiểu và kiểm tra
            score = float(score)
            print(f"  ImageReward Score: {score:.4f}")
            return score
        except Exception as e:
            print(f"  Error calculating ImageReward score: {e}")
            # import traceback; traceback.print_exc()
            return None

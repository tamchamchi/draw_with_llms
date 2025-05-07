from ..base import SimilarityRewardStrategy
from PIL import Image
from typing import Any, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class CaptionStrategy(SimilarityRewardStrategy):

    # Dùng Any để tránh import vòng tròn nếu cần
    def __init__(self, vqa_evaluator: Any):
        """Khởi tạo với đối tượng vqa_evaluator."""
        self.vqa_evaluator = vqa_evaluator
        print("--- Similarity/Reward Strategy: Initialized CaptionStrategy ---")
        if not hasattr(self.vqa_evaluator, "caption"):
            raise TypeError(
                "Provided vqa_evaluator does not have 'caption' method."
            )
        
    def calculate(
        self, image: Image.Image, description: Optional[str] = None, **kwargs
    ) -> Optional[float]:
        if description is None:
            print("Warning: CaptionStrategy requires a description, none provided.")
            return None

        try:
            print("  Calculating Caption Similarity...")
            print(f"   Prompt: {description}")

            # Caption do mô hình sinh ra từ ảnh
            caption = self.vqa_evaluator.caption(image)
            print(f"   Generated Caption: {caption}")

            # Tiền xử lý: tách từ
            reference = [description.strip().split()]   # reference phải là list các list
            candidate = caption.strip().split()

            # Tính BLEU với smoothing
            smoothie = SmoothingFunction().method4
            score = sentence_bleu(reference, candidate, smoothing_function=smoothie)

            print(f"   Caption Similarity Score: {score:.4f}")
            return score

        except Exception as e:
            print(f"  Error calculating Caption similarity: {e}")
            return None


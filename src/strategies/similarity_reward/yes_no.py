from ..base import SimilarityRewardStrategy
from PIL import Image
from typing import Any, Optional

class YesNoStrategy(SimilarityRewardStrategy):
        # Dùng Any để tránh import vòng tròn nếu cần
    def __init__(self, vqa_evaluator: Any):
        """Khởi tạo với đối tượng vqa_evaluator."""
        self.vqa_evaluator = vqa_evaluator
        self.questions_template = {
                'fidelity': 'Does <image> portray "SVG illustration of {}"? Answer yes or no.'                
        }
        print("--- Similarity/Reward Strategy: Initialized YesNo---")
        if not hasattr(self.vqa_evaluator, "score_yes_no"):
            raise TypeError(
                "Provided vqa_evaluator does not have 'score_yes_no' method."
            )
    
    def calculate(
        self, image: Image.Image, description: Optional[str] = None, **kwargs
    ) -> Optional[float]:
        if description is None:
            print("Warning: CaptionStrategy requires a description, none provided.")
            return None

        try:
            print("  Yes-No Caption Similarity...")
            print(f"   Prompt: {description}")

            p_fidelity = self.vqa_evaluator.score_yes_no(self.questions_template['fidelity'].format(description), image)
            score = p_fidelity
        
            print(f"   Caption Similarity Score: {score:.4f}")
            return score
        
        except Exception as e:
            print(f"  Error calculating YesNo similarity: {e}")
            return None
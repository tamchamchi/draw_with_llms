import json
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image

from configs.configs import RESULTS_DIR

from ..data.data_loader import Data
from ..data.image_processor import ImageProcessor, svg_to_png
from ..scoring.scoring import score, harmonic_mean
from ..strategies.base import (
    ImageProcessingStrategy,
    PromptBuildingStrategy,
    SimilarityRewardStrategy,
)
from ..strategies.image_processing.compression import CompressionStrategy
from ..strategies.image_processing.no_compression import NoCompressionStrategy
from ..utils.bitmap_to_svg import bitmap_to_svg_layered
# from ..utils.image_to_svg import bitmap_to_svg_layered
from ..utils.formatting_utils import naming_template, score_caption
from ..utils.image_utils import add_caption_to_image
from .aesthetic import AestheticEvaluator
from .vqa import VQAEvaluator


class ScoreEvaluator:
    def __init__(
        self,
        vqa_evaluator: VQAEvaluator,
        aesthetic_evaluator: AestheticEvaluator,
        generator: None,
        data: Data,
        prompt_builder: PromptBuildingStrategy,
        similarity_reward_strategy: SimilarityRewardStrategy,
        seed: int = 42,
    ):
        # Dependency Injection
        self.vqa_evaluator = vqa_evaluator
        self.aesthetic_evaluator = aesthetic_evaluator
        self.generator = generator
        self.data = data
        self.prompt_builder = prompt_builder
        self.similarity_reward_strategy = similarity_reward_strategy
        self._set_random_seed(seed)

    def _set_random_seed(self, seed: int = 42) -> None:
        """Set up random seeds for libraries."""
        print(f"--- Set up Random Seeds: {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_directories(
        self, version: str, id_prompt, processing_strategy: ImageProcessingStrategy
    ) -> str:
        """Create a output folder structure."""

        if isinstance(processing_strategy, CompressionStrategy):
            strategy_suffix = "-image_compression"
        elif isinstance(processing_strategy, NoCompressionStrategy):
            strategy_suffix = "-no_image_compression"
        else:
            strategy_suffix = "-unknown_processing"

        version_folder = os.path.join(RESULTS_DIR, f"{version}-{strategy_suffix}")
        id_folder = os.path.join(version_folder, str(id_prompt))

        os.makedirs(id_folder, exist_ok=True)
        return id_folder

    def _build_prompt(
        self,
        prefix_prompt: str = None,
        description: str = None,
        suffix_prompt: str = None,
        negative_prompt: str = None,
    ) -> str:
        return f"{prefix_prompt} {description} {suffix_prompt}".strip()

    def _generate_bitmap(
        self,
        prompt: str = None,
        negative: str = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 15,
        seed: int = 42,
    ) -> Image.Image:
        """Generates an image based on the given prompt using Stable Diffusion v2"""
        return self.generator.generate(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
        )

    def _process_image(
        self, bitmap: Image.Image, processing_strategy: ImageProcessingStrategy
    ) -> Image.Image:
        print("--- Template Step: Process Image ---")
        return processing_strategy.process(bitmap)

    def _convert_to_svg(self, processed_image: Image.Image, **kwargs) -> str:
        print("--- Template Step: Convert to SVG ---")
        return bitmap_to_svg_layered(
            image=processed_image,
            max_size_bytes=kwargs.get("max_size_bytes", 9900),
            resize=kwargs.get("resize", True),
            target_size=kwargs.get("target_size", (384, 384)),
            adaptive_fill=kwargs.get("adaptive_fill", True),
            num_colors=kwargs.get("num_colors", 12),
        )

    def _evaluate_results(
        self,
        id_prompt: str,
        svg_content: str,
        bitmap: Image.Image,
        processed_image: Image.Image,
        description: str,
        processing_strategy: ImageProcessingStrategy,
    ) -> dict:
        print("--- Template Step: Evaluate Results ---")
        results = {}
        solution = self.data.get_solution(id_prompt)
        # Kiểm tra loại strategy để lấy điểm compressed_quality
        if isinstance(processing_strategy, CompressionStrategy):
            # Giả sử aesthetic_evaluator có thể đánh giá ảnh đã xử lý
            results["compressed_quality"] = self.aesthetic_evaluator.score(
                processed_image
            )
            print(
                f"Compressed Quality (k={processing_strategy.k}): {results['compressed_quality']:.4f}"
            )
        else:
            results["compressed_quality"] = None

        (
            results["total_score"],
            results["vqa_score"],
            results["aesthetic_score"],
            results["ocr_score"],
        ) = score(
            solution=solution,
            submission=pd.DataFrame({"id": [id_prompt], "svg": [svg_content]}),
            row_id_column_name="row_id",
            vqa_evaluator=self.vqa_evaluator,
            aesthetic_evaluator=self.aesthetic_evaluator,
            random_seed=42,
        )
        results["bitmap_quality"] = self.aesthetic_evaluator.score(bitmap)
        eval_vqa_bitmap = self._vqa_ocr_evaluation_helper(
            id_prompt=id_prompt, image=bitmap
        )
        results["vqa_bitmap_scores"] = eval_vqa_bitmap.get("vqa_score", [])
        results["num_char_bitmap"] = eval_vqa_bitmap.get("num_char", 0)
        image_submit = svg_to_png(svg_content)
        results["image_submit"] = image_submit
        eval_vqa_submit = self._vqa_ocr_evaluation_helper(
            id_prompt=id_prompt, image=image_submit
        )
        results["vqa_submit_scores"] = eval_vqa_submit.get("vqa_score", [])
        results["num_char_submit"] = eval_vqa_submit.get("num_char", 0)

        # --- !!! SỬ DỤNG STRATEGY ĐỂ TÍNH SIMILARITY/REWARD !!! ---
        print(
            f"--- Calculating Similarity/Reward using: {type(self.similarity_reward_strategy).__name__} ---"
        )
        similarity_reward_score = self.similarity_reward_strategy.calculate(
            image=bitmap,  # Hoặc image_submit tùy bạn muốn đánh giá ảnh nào
            description=description,  # Cần cho CLIP
        )
        # Lưu kết quả (có thể là None)
        results["text_alignment_score"] = similarity_reward_score
        results["vqa_origin"] = results["vqa_bitmap_scores"]
        results["aesthetic_origin"] = results["bitmap_quality"]
        # >>>---------------------------------------------------->>>

        print(
            f"Scores (SVG-based): Total={results['total_score']:.4f}, VQA={results['vqa_score']:.4f}, Aesthetic={results['aesthetic_score']:.4f}, OCR={results['ocr_score']:.4f}"
        )
        # print(f"Bitmap Quality: {results['bitmap_quality']:.4f}")
        print(
            f"Bitmap VQA: {results['vqa_bitmap_scores']}, Chars: {results['num_char_bitmap']}"
        )
        print(
            f"Submit Img VQA: {results['vqa_submit_scores']}, Chars: {results['num_char_submit']}"
        )
        # print(f"Similarity/Reward Score: {results['text_alignment_score']:.4f}")
        return results

    def _save_artifacts(
        self,
        id_folder: str,
        attempt: int,
        bitmap: Image.Image,
        processed_image: Image.Image,
        image_submit: Image.Image,
        svg_content: str,
        evaluation_results: dict,
        description: str,
        processing_strategy: ImageProcessingStrategy,
    ) -> None:
        print("--- Template Step: Save Artifacts ---")
        # ... (Logic lưu file như code trước) ...
        # Lấy các điểm số cần thiết
        bitmap_quality = evaluation_results["bitmap_quality"]
        aesthetic_score_submit = evaluation_results["aesthetic_score"]
        compressed_quality = evaluation_results.get(
            "compressed_quality"
        )  # Có thể là None

        # Chuỗi caption
        vqa_bitmap_str = " ".join(
            f"{s:.2f}" for s in evaluation_results["vqa_bitmap_scores"]
        )
        char_bitmap_str = f"Text: {evaluation_results['num_char_bitmap']}"
        vqa_submit_str = " ".join(
            f"{s:.2f}" for s in evaluation_results["vqa_submit_scores"]
        )
        char_submit_str = f"Text: {evaluation_results['num_char_submit']}"
        main_scores_str = score_caption(
            evaluation_results["total_score"],
            evaluation_results["vqa_score"],
            evaluation_results["aesthetic_score"],
            evaluation_results["ocr_score"],
        )
        k_value = (
            f"k={processing_strategy.k}"
            if isinstance(processing_strategy, CompressionStrategy)
            else ""
        )

        # Lưu ảnh đã nén (nếu có)
        if (
            isinstance(processing_strategy, CompressionStrategy)
            and compressed_quality is not None
        ):
            captioned_processed = add_caption_to_image(
                processed_image, [description, k_value]
            )
            save_path = os.path.join(
                id_folder,
                naming_template(f"compressed_{k_value}", attempt, compressed_quality),
            )
            captioned_processed.save(save_path)
            print(f"    Saved: {save_path}")
        # ... (Lưu các ảnh bitmap, submit có/không caption, svg như trước) ...
        # Bitmap
        captioned_bitmap = add_caption_to_image(
            bitmap, [description, f"VQA: {vqa_bitmap_str}", char_bitmap_str]
        )
        save_path = os.path.join(
            id_folder, naming_template("raw", attempt, bitmap_quality)
        )
        captioned_bitmap.save(save_path)
        print(f"    Saved: {save_path}")

        # Submit
        captioned_submit = add_caption_to_image(
            image_submit,
            [description, main_scores_str, f"VQA: {vqa_submit_str}", char_submit_str],
        )
        save_path = os.path.join(
            id_folder, naming_template("submit", attempt, aesthetic_score_submit)
        )
        captioned_submit.save(save_path)
        print(f"Saved: {save_path}")

        save_path = os.path.join(
            id_folder, naming_template("no_cap", attempt, aesthetic_score_submit)
        )
        image_submit.save(save_path)
        print(f"Saved: {save_path}")

        # Eval Image
        image_processor = ImageProcessor(image=svg_to_png(svg_content), seed=42).apply()
        eval_image = image_processor.image.copy()
        save_path = os.path.join(
            id_folder, naming_template("eval", attempt, aesthetic_score_submit)
        )
        eval_image.save(save_path)
        print(f"Saved: {save_path}")

        # SVG file
        svg_filename = f"output_attempt_{attempt + 1}.svg"
        svg_path = os.path.join(id_folder, svg_filename)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

        print(f"Saved: {svg_path} (Size: {len(svg_content.encode('utf-8'))} bytes)")

    def _update_best_score(
        self, current_eval_results: dict, best_scores_tracking: dict, verbose: bool
    ) -> bool:
        print("--- Template Step: Update Best Score ---")
        # ... (Logic cập nhật điểm như code trước) ...
        current_total = current_eval_results["total_score"]
        current_similarity_reward_score = current_eval_results["text_alignment_score"]
        current_aesthetic = current_eval_results["aesthetic_score"]
        is_better = False
        current_val_score = (
            harmonic_mean(current_similarity_reward_score, current_aesthetic)
        )
        best_val_score = (
            harmonic_mean(
                best_scores_tracking["best_text_alignment_score"],
                best_scores_tracking["best_aesthetic_score"],
            )
        )
        print(f"Current Score: {current_val_score}")
        print(f"Best Score: {best_val_score}")
        if (
            # current_total > best_scores_tracking["best_total_score"]
            # current_similarity_reward_score > best_scores_tracking["best_text_alignment_score"]
            # and current_aesthetic > best_scores_tracking["best_aesthetic_score"]
            current_val_score > best_val_score
            # current_aesthetic > best_scores_tracking["best_aesthetic_score"]
        ):
            best_scores_tracking["best_total_score"] = current_total
            best_scores_tracking["best_vqa_score"] = current_eval_results["vqa_score"]
            best_scores_tracking["best_aesthetic_score"] = current_eval_results[
                "aesthetic_score"
            ]
            best_scores_tracking["best_ocr_score"] = current_eval_results["ocr_score"]
            best_scores_tracking["best_text_alignment_score"] = (
                current_similarity_reward_score
            )
            best_scores_tracking["best_aesthetic_origin"] = current_eval_results["aesthetic_origin"]
            best_scores_tracking["best_vqa_origin"] = sum(current_eval_results["vqa_origin"]) / len(current_eval_results["vqa_origin"])
            is_better = True
            if verbose:
                print("✅ New best result found!")
        elif verbose:
            print("❌ Score not improved based on criteria.")

        return is_better

    def _vqa_ocr_evaluation_helper(self, id_prompt: str, image: Image.Image) -> dict:
        print(
            f"--- Helper: Evaluating VQA/OCR for image ({image.width}x{image.height}) ---"
        )
        results = {"vqa_score": None, "ocr_score": None, "num_char": None}
        try:
            solution = self.data.get_solution(idx=id_prompt)
            question = json.loads(solution.loc[0, "question"])
            choices = json.loads(solution.loc[0, "choices"])
            answers = json.loads(solution.loc[0, "answer"])
        except Exception as e:
            print(f"Error parsing solution JSON for {id_prompt}: {e}")
            return results
        rng = np.random.RandomState(42)
        group_seed = rng.randint(0, np.iinfo(np.int32).max)
        try:
            image_processor = ImageProcessor(image=image, seed=group_seed).apply()
            processed_image_vqa = image_processor.image.copy()
            vqa_score = self.vqa_evaluator.score_batch(
                image=processed_image_vqa,
                questions=question,
                choices_list=choices,
                answers=answers,
            )
            reset_image_ocr = (
                image_processor.reset()
                .apply_random_crop_resize()
                .apply_jpeg_compression(quality=90)
                .image
            )
            ocr_score, num_char = self.vqa_evaluator.ocr(
                reset_image_ocr, free_chars=4, use_num_char=True
            )
            results["vqa_score"] = vqa_score
            results["ocr_score"] = ocr_score
            results["num_char"] = num_char
        except Exception as e:
            print(f"Error during VQA/OCR evaluation for {id_prompt}: {e}")
        return results

    # --- Phương thức chính (Public Interface - Cập nhật tham số) ---
    def generate_and_evaluate(
        self,
        id_prompt: str = None,
        prefix_prompt: str = None,
        suffix_prompt: str = None,
        negative_prompt: str = None,
        width: int = 512,
        height: int = 512,
        bitmap2svg_config: dict = None,  # Số màu cho SVG conversion
        num_inference_steps: int = 25,
        guidance_scale: float = 15,
        use_image_compression: bool = False,
        use_prompt_builder: bool = False,
        compression_k: int = 8,  # !!! THAM SỐ MỚI CHO KHI NÉN !!!
        version: str = "v1.0",
        num_attempts: int = 1,
        verbose: bool = False,
        random_seed: int = 42,
    ) -> dict:
        """
        Orchestrates the image generation and evaluation process.

        Args:
            id_prompt (str): Unique identifier for the prompt configuration
            prefix_prompt (str): Text prepended to the main description
            suffix_prompt (str): Text appended to the main description
            negative_prompt (str): Negative guidance for image generation
            width (int): Output image width in pixels (default: 512)
            height (int): Output image height in pixels (default: 512)
            num_color (int): Color quantization for SVG conversion (default: 8)
            num_inference_steps (int): Diffusion process iterations (default: 25)
            guidance_scale (float): Prompt guidance strength (default: 15)
            use_image_compression (bool): If True, use CompressionStrategy.
            compression_k (int): The 'k' parameter for image compression (used only if use_image_compression is True). Default: 8
            version (str): Experiment version identifier
            num_attempts (int): Generation attempts for best result (default: 1)
            verbose (bool): Enable detailed logging (default: False)
            random_seed (int): Reproducibility seed (default: 42)

        Returns:
            dict: Evaluation results with best scores and metadata.
        """
        print(f"\n===== Starting Evaluation for ID: {id_prompt} =====")
        self._set_random_seed(random_seed)

        # --- Chọn và khởi tạo Strategy ---
        if use_image_compression:
            # !!! Tạo CompressionStrategy với tham số k !!!
            processing_strategy = CompressionStrategy(k=compression_k)
        else:
            processing_strategy = NoCompressionStrategy()
        print(f"--- Using Strategy: {type(processing_strategy).__name__} ---")

        # --- Setup (sử dụng _setup_directories đã cập nhật) ---
        description = self.data.get_description_by_id(id_prompt)
        id_folder = self._setup_directories(version, f"{id_prompt}-{description}", processing_strategy)
        # prompt = self._build_prompt(prefix_prompt, description, suffix_prompt, negative_prompt)

        # --- !!! Xây dựng Prompt bằng Strategy !!! ---
        if use_prompt_builder:
            print(
                f"--- Building prompt using strategy: {type(self.prompt_builder).__name__} ---"
            )
            try:
                prompt_data = self.prompt_builder.build(
                    description=description,
                )
                final_prompt = prompt_data.get("prompt")
                final_negative_prompt = prompt_data.get("negative_prompt")
                if not final_prompt:
                    raise ValueError(
                        "Prompt building strategy did not return a valid 'prompt'."
                    )
            except Exception as e_prompt:
                print(f"🚨 Error during prompt building for ID {id_prompt}: {e_prompt}")
                return {
                    "status": "FAILED",
                    "error": f"Prompt building error: {e_prompt}",
                    "id_desc": id_prompt,
                    "description": description,
                }
        else:
            final_prompt = self._build_prompt(
                prefix_prompt, description, suffix_prompt, negative_prompt
            )
            final_negative_prompt = negative_prompt

        if verbose:
            print(f"Full Prompt: {final_prompt}")
            print(f"Negative Prompt: {final_negative_prompt}")
            print(
                f"Parameters: W={width}, H={height}, SVG Colors={bitmap2svg_config.get('num_color', 12)}, Steps={num_inference_steps}, Scale={guidance_scale}"
            )
            if use_image_compression:
                print(f"Compression Enabled: k={compression_k}")
            else:
                print("Compression Disabled")
            print(f"Attempts: {num_attempts}, Seed: {random_seed}")

        # --- Tracking Best Results (như cũ) ---
        best_scores_tracking = {
            "best_total_score": float("-inf"),
            "best_vqa_score": 0.0,
            "best_vqa_origin": 0.0,
            "best_aesthetic_score": 0.0,
            "best_aesthetic_origin": 0.0,
            "best_ocr_score": 0.0,
            "best_text_alignment_score": 0.0,
        }

        # --- Vòng lặp chính (gọi Template Method - như cũ) ---
        for attempt in range(num_attempts):
            print(f"\n--- Attempt {attempt + 1}/{num_attempts} ---")
            try:
                bitmap = self._generate_bitmap(
                    final_prompt,
                    final_negative_prompt,
                    width,
                    height,
                    num_inference_steps,
                    guidance_scale,
                    seed=random_seed + attempt,
                )
                # !!! Gọi _process_image không cần truyền k vì strategy đã giữ nó !!!
                processed_image = self._process_image(bitmap, processing_strategy)
                svg_content = self._convert_to_svg(processed_image, **bitmap2svg_config)
                evaluation_results = self._evaluate_results(
                    id_prompt,
                    svg_content,
                    bitmap,
                    processed_image,
                    description,
                    processing_strategy,
                )
                # !!! image_submit được lấy từ evaluation_results để dùng trong _save_artifacts !!!
                image_submit = evaluation_results.get("image_submit")
                if image_submit:  # Đảm bảo image_submit tồn tại trước khi lưu
                    self._save_artifacts(
                        id_folder,
                        attempt,
                        bitmap,
                        processed_image,
                        image_submit,
                        svg_content,
                        evaluation_results,
                        description,
                        processing_strategy,
                    )
                else:
                    print(
                        "⚠️ Warning: image_submit not found in evaluation results, skipping saving artifacts dependent on it."
                    )
                self._update_best_score(
                    evaluation_results, best_scores_tracking, verbose
                )
            except Exception as e:
                print(f"🚨 Error during attempt {attempt + 1}: {e}")
                import traceback

                traceback.print_exc()  # In chi tiết lỗi

        # --- Final Results Packaging (cập nhật method name) ---
        print(f"\n===== Evaluation Finished for ID: {id_prompt} =====")
        method_name = type(processing_strategy).__name__.replace("Strategy", "")
        if isinstance(processing_strategy, CompressionStrategy):
            method_name += f"_k{processing_strategy.k}"

        final_results = {
            "method": method_name,
            "model": "SDv2",  # Hoặc tên model thực tế
            "id_desc": id_prompt,
            "description": description,
            "total_score": best_scores_tracking["best_total_score"]
            if best_scores_tracking["best_total_score"] > -1
            else 0.0,
            "vqa_score": best_scores_tracking["best_vqa_score"],
            "vqa_origin": best_scores_tracking["best_vqa_origin"],
            "aesthetic_score": best_scores_tracking["best_aesthetic_score"],
            "aesthetic_origin": best_scores_tracking["best_aesthetic_origin"],
            "ocr_score": best_scores_tracking["best_ocr_score"],
            "text_alignment_score": best_scores_tracking["best_text_alignment_score"]
            if best_scores_tracking["best_text_alignment_score"] > -1
            else 0.0,
            "size": len(svg_content.encode("utf-8"))
        }
        print(
            f"Best Scores Found: Total={final_results['total_score']:.4f}, Text Alignment={final_results['text_alignment_score']:.4f}"
        )
        return final_results

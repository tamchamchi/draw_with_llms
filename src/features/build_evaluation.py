import torch
import numpy as np
import random
import os
import pandas as pd


from src.utils.image_compression import image_compression
from src.models.model_metrics import score
from src.data.image_processor import svg_to_png
from src.utils.bitmap_to_svg import bitmap_to_svg_layered


from src.configs import RESULTS_DIR


class ScoreEvaluation:
    def __init__(self, vqa_evaluator, aesthetic_evaluator, my_model, data):
        self.vqa_evaluator = vqa_evaluator
        self.aesthetic_evaluator = aesthetic_evaluator
        self.my_model = my_model
        self.data = data
        self.set_random_seed(42)

    def set_random_seed(self, random_seed: int = 42) -> None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    def generate_and_evaluation(
        self,
        id_prompt: str = None,
        prefix_prompt: str = None,
        suffix_prompt: str = None,
        negative_prompt: str = None,
        width: int = 512,
        height: int = 512,
        num_color: int = 8,
        num_inference_steps: int = 25,
        guidance_scale: float = 15,
        use_image_compression: bool = False,
        version: str = None,
        num_attempts: int = 1,
        verbose: bool = False,
        random_seed: int = 42,
    ) -> dict:
        """
        Generates and evaluates multiple image variants from text prompts using a diffusion model.
        Produces SVG conversions with optional compression and tracks best results across attempts.

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
            use_image_compression (bool): Enable pre-SVG compression (default: False)
            version (str): Experiment version identifier
            num_attempts (int): Generation attempts for best result (default: 1)
            verbose (bool): Enable detailed logging (default: False)
            random_seed (int): Reproducibility seed (default: 42)

        Returns:
            dict: Evaluation results with scores and metadata
        """

        # Setup output directory structure
        # ------------------------------
        # Create version-specific folder with compression status flag
        compression_suffix = (
            "-have_image_compression"
            if use_image_compression
            else "-no_image_compression"
        )
        version_folder = os.path.join(RESULTS_DIR, f"{version}{compression_suffix}")

        # Create ID-specific subfolder
        id_folder = os.path.join(version_folder, str(id_prompt))

        # Ensure directories exist
        os.makedirs(id_folder, exist_ok=True)  # Nested creation with exist_ok

        # Initialize experiment environment
        # --------------------------------
        self.set_random_seed(random_seed)  # Ensure reproducibility

        # Track best performing attempt
        best_scores = {"total": 0, "vqa": 0, "aesthetic": 0, "ocr": 0}

        # Get dataset references
        # ----------------------
        solution = self.data.get_solution(id_prompt)  # Ground truth data
        description = self.data.get_description_by_id(id_prompt)  # Core prompt content

        # Construct generation prompt
        # ---------------------------
        prompt = f"{prefix_prompt} {description} {suffix_prompt}".strip()

        # Main generation loop
        # --------------------
        for attempt in range(num_attempts):
            # Generate initial bitmap
            bitmap = self.my_model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            # SVG Conversion pipeline
            # -----------------------
            if use_image_compression:
                # Apply lossy compression before vectorization
                processed_image = image_compression(bitmap)
            else:
                processed_image = bitmap  # Use original image

            # Convert to layered SVG with size constraints
            svg = bitmap_to_svg_layered(
                image=processed_image,
                max_size_bytes=10000,  # 10KB size limit
                resize=True,
                target_size=(width, height),
                adaptive_fill=True,
                num_colors=num_color,  # Color quantization
            )

            # Evaluation metrics
            # ------------------
            # Create evaluation dataframe
            submission = pd.DataFrame({"id": [id_prompt], "svg": [svg]})

            # Calculate scores
            total_score, vqa_score, aesthetic_score, ocr_score = score(
                solution, submission, "row_id", random_seed=42
            )

            # Additional quality metrics
            bitmap_quality = self.aesthetic_evaluator.score(bitmap)
            if use_image_compression:
                compressed_quality = self.aesthetic_evaluator.score(processed_image)

            # File persistence
            # ----------------
            # Save all image variants with quality scores in filenames
            def naming_template(t, quality):
                return f"{t} - {attempt} - {quality:.4f}.png"

            if use_image_compression:
                processed_image.save(
                    os.path.join(
                        id_folder, naming_template("compressed", compressed_quality)
                    ),
                )

            bitmap.save(os.path.join(id_folder, naming_template("raw", bitmap_quality)))

            svg_to_png(svg).save(
                os.path.join(id_folder, naming_template("submit", aesthetic_score)),
            )

            # Score tracking
            # --------------
            if total_score > best_scores["total"]:
                best_scores.update(
                    {
                        "total_score": total_score,
                        "vqa_score": vqa_score,
                        "aesthetic_score": aesthetic_score,
                        "ocr_score": ocr_score,
                    }
                )
                if verbose:
                    print("✅ New best result")
            elif verbose:
                print("❌ Score not improved")

            # Progress logging
            # ----------------
            if verbose:
                print(f"\nAttempt {attempt + 1}/{num_attempts}")
                print(f"Prompt ID: {id_prompt}")
                print(f"Description: {description}")
                print(f"SVG Size: {len(svg.encode('utf-8'))} bytes")
                print(f"- Total Score: {total_score:.4f}")
                print(f"- VQA Score: {vqa_score:.4f}")
                print(f"- Aesthetic Score: {aesthetic_score:.4f}")
                print(f"- OCR Score: {ocr_score:.4f}")
                print("-" * 40)

        # Final results packaging
        # -----------------------
        return {
            "method": "I->KMean->SVG" if use_image_compression else "I->SVG",
            "model": "SDv2",
            "id_desc": id_prompt,
            "description": description,
            **{f"{k}": v for k, v in best_scores.items()},
        }

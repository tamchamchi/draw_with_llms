import torch
import numpy as np
import random
import os
import pandas as pd


from features.build_image_compression import image_compression
from features.build_image_processor import svg_to_png
from features.build_png2svg import bitmap_to_svg_layered
from features.build_calc_total_score import score

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
        if not use_image_compression:
            version_folder = os.path.join(
                RESULTS_DIR, f"{version}-no_image_compression "
            )
        else:
            version_folder = os.path.join(
                RESULTS_DIR, f"{version}-have_image_compression"
            )

        id_folder = os.path.join(version_folder, f"{id_prompt}")

        os.makedirs(version_folder, exist_ok=True)
        os.makedirs(id_folder, exist_ok=True)

        self.set_random_seed(random_seed=random_seed)

        best_vqa_score = 0
        best_aesthetic_score = 0
        best_ocr_score = 0
        best_total_score = 0

        solution = self.data.get_solution(idx=id_prompt)
        description = self.data.get_description_by_id(idx=id_prompt)

        prompt = f"{prefix_prompt} {description} {suffix_prompt}"

        for i in range(num_attempts):
            if verbose:
                print(f"\n=== Attempt {i + 1}/{num_attempts} ===")
                print(f"Id: {id_prompt}")
                print(f"Description: {description}")

                bitmap = self.my_model.generate(
                    prompt=prompt,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

                if verbose:
                    print("Converting to SVG... \n")

                if use_image_compression:
                    compressed_image = image_compression(image_input=bitmap)

                    svg = bitmap_to_svg_layered(
                        image=compressed_image,
                        max_size_bytes=1000,
                        resize=True,
                        target_size=(width, height),
                        adaptive_fill=True,
                        num_colors=num_color,
                    )
                else:
                    svg = bitmap_to_svg_layered(
                        image=bitmap,
                        max_size_bytes=1000,
                        resize=True,
                        target_size=(width, height),
                        adaptive_fill=True,
                        num_colors=num_color,
                    )
                    

                svg_size = len(svg.encode("utf-8"))
                if verbose:
                    print(f"SVG size: {svg_size} bytes\n")

                submission = pd.DataFrame({"id": [id_prompt], "svg": [svg]})

                total_score, vqa_score, aesthetic_score, ocr_score = score(
                    solution, submission, "row_id", random_seed=42
                )
                aesthetic_score_original = self.aesthetic_evaluator.score(bitmap)

                if use_image_compression:
                    aesthetic_score_compressed = self.aesthetic_evaluator.score(
                        compressed_image
                    )
                    compressed_image_path = os.path.join(
                        id_folder,
                        f"compressed - {aesthetic_score_compressed:.4f}.png",
                    )
                    compressed_image.save(compressed_image_path, format="PNG")

                raw_image_path = os.path.join(
                    id_folder, f"raw - {aesthetic_score_original:.4f}.png"
                )
                bitmap.save(raw_image_path, format="PNG")

                final_image_path = os.path.join(
                    id_folder, f"final - {aesthetic_score:.4f}.png"
                )
                svg_to_png(svg).save(final_image_path, format="PNG")

                if verbose:
                    print(f"SVG VQA Score: {vqa_score:.4f}")
                    print(f"SVG Aesthetic Score: {aesthetic_score:.4f}")
                    print(f"SVG Ocr Score: {ocr_score:.4f}")
                    print(f"SVG Total Score: {total_score:.4f}")

                if total_score > best_total_score:
                    best_total_score = total_score
                    best_vqa_score = vqa_score
                    best_aesthetic_score = aesthetic_score
                    best_ocr_score = ocr_score
                    if verbose:
                        print("✅ New best result")
                else:
                    if verbose:
                        print("❌ Not better than current best")

                print("-" * 40)
                print("\n")

        return {
            "method": "I->SVG" if not use_image_compression else "I->KMean->SVG",
            "model": "SDv2",
            "id_desc": id_prompt,
            "description": description,
            "total_score": best_total_score,
            "vqa_score": best_vqa_score,
            "aesthetic_score": best_aesthetic_score,
            "ocr_score": best_ocr_score,
        }

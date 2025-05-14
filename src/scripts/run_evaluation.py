#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json
import yaml
from dotenv import load_dotenv

load_dotenv()
# --- Import các lớp/hàm thực tế ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    os.environ["HF_HOME"] = os.getenv("HF_HOME")

    from src.evaluators.score_evaluator import ScoreEvaluator
    from src.evaluators.aesthetic import AestheticEvaluator
    from src.evaluators.siglip import Siglip
    from src.evaluators.vqa import VQAEvaluator
    from src.generators.stable_diffusion_v2 import StableDiffusionV2
    from src.generators.sdxl_turbo import StableDiffusionXL_Turbo
    from src.generators.sdxl_vetor import SDXL_Vector
    from src.data.data_loader import Data  # Lớp xử lý data
    from src.strategies.build_prompt.categorized_prompt_strategy import (
        CategorizedPromptStrategy,
    )
    from src.strategies.similarity_reward.captioning import CaptionStrategy
    from src.strategies.similarity_reward.clip_similarity import ClipSimilarityStrategy
    from src.strategies.similarity_reward.siglip_similarity import (
        SIGLIPSimilarityStrategy,
    )

    # Import đường dẫn dữ liệu thô
    from configs.configs import RAW_DATA_DIR, YAML_CONFIG_FILE

    TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "drawing-with-llms/train.csv")
    QUESTION_DATA_PATH = os.path.join(
        RAW_DATA_DIR, "drawing-with-llms/questions.parquet"
    )
    print(f"--- Using Train Data: {TRAIN_DATA_PATH} ---")
    print(f"--- Using Question Data: {QUESTION_DATA_PATH} ---")

except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Kiểm tra lại cấu trúc thư mục, PYTHONPATH và các file __init__.py.")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"Lỗi không tìm thấy file/thư mục khi import hoặc định nghĩa đường dẫn: {e}")
    sys.exit(1)


def load_config_from_yaml(version_name: str) -> dict:
    """Tải cấu hình cho một version cụ thể từ file YAML."""
    if not os.path.exists(YAML_CONFIG_FILE):
        print(f"Lỗi: Không tìm thấy file cấu hình YAML tại '{YAML_CONFIG_FILE}'")
        sys.exit(1)
    try:
        with open(YAML_CONFIG_FILE, "r") as f:
            all_configs = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Lỗi khi đọc file YAML '{YAML_CONFIG_FILE}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi không xác định khi mở file YAML: {e}")
        sys.exit(1)

    if not isinstance(all_configs, dict) or version_name not in all_configs:
        print(
            f"Lỗi: Không tìm thấy key version '{version_name}' trong file '{YAML_CONFIG_FILE}'"
        )
        print(
            f"Các version có sẵn: {list(all_configs.keys()) if isinstance(all_configs, dict) else 'Không có'}"
        )
        sys.exit(1)

    print(f"--- Đã tải cấu hình cho version: '{version_name}' từ YAML ---")
    return all_configs[version_name]


def main():
    parser = argparse.ArgumentParser(
        description="Chạy đánh giá ảnh từ config/versions.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Tên version cấu hình trong versions.yaml.",
    )
    # --- Tham số ghi đè cấp cao ---
    # ... (giữ nguyên các args prefix, suffix, neg_prompt, height, width, steps, scale, compress, k, attempts, seed, verbose) ...
    parser.add_argument("--prefix_prompt", type=str, default=None)
    parser.add_argument("--suffix_prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument(
        "--use_image_compression", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--use_prompt_builder", action=argparse.BooleanOptionalAction, default=None
    )
    # <<<--- Thêm Argument chọn Similarity/Reward Strategy ---<<<
    parser.add_argument(
        "--similarity_strategy",
        type=str,
        default="clip",  # Mặc định dùng CLIP
        choices=["clip", "siglip", "caption"],
        help="Chiến lược tính điểm tương đồng/thưởng ('clip' hoặc 'siglip').",
    )
    # >>>-------------------------------------------------->>>

    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--num_attempts", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--output_version_name", type=str, default=None)
    # --- Thêm args để override method và model ---
    parser.add_argument(
        "--method", type=str, default=None, help="Ghi đè tên 'method' trong YAML."
    )
    parser.add_argument(
        "--model", type=str, default=None, choices=["StableDiffusionV2", "SDXL-Turbo", "SDXL-Vector"],help="Ghi đè tên 'model' trong YAML."
    )

    # --- Tham số ghi đè SVG config (như cũ) ---
    parser.add_argument("--svg_num_colors", type=int, default=None)
    # ... (giữ nguyên các args svg_max_bytes, svg_resize, svg_target_w/h, svg_adaptive_fill) ...
    parser.add_argument("--svg_max_bytes", type=int, default=None)
    parser.add_argument(
        "--svg_resize", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--svg_target_w", type=int, default=None)
    parser.add_argument("--svg_target_h", type=int, default=None)
    parser.add_argument(
        "--svg_adaptive_fill", action=argparse.BooleanOptionalAction, default=None
    )

    # --- Tham số output tổng hợp ---
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Đường dẫn file JSON để lưu tất cả kết quả.",
    )

    # --- Load YAML và Parse Args ---
    known_args, _ = parser.parse_known_args()
    yaml_params = load_config_from_yaml(known_args.version)
    # id_prompt và base_bitmap2svg_config sẽ được xử lý trong vòng lặp nếu có
    base_bitmap2svg_config = yaml_params.get(
        "bitmap2svg_config", {}
    )  # Lấy hoặc dùng dict rỗng nếu thiếu

    # --- Đặt YAML làm defaults ---
    yaml_defaults = {
        k: v
        for k, v in yaml_params.items()
        if hasattr(parser, "get_default")
        and hasattr(parser.get_default(k), "__class__")
    }
    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args()

    # --- Khởi tạo Dependencies và ScoreEvaluator (như cũ) ---
    print("--- Khởi tạo Dependencies Thực Tế ---")
    try:
        vqa_evaluator = VQAEvaluator()
        aesthetic_evaluator = AestheticEvaluator()

        if args.model == "StableDiffusionV2":
            print("   Use Stable Diffusion v2")
            generator = StableDiffusionV2()
        elif args.model == "SDXL-Turbo":
            print("   Use Stable Diffusion XL Turbo")
            generator = StableDiffusionXL_Turbo()
        elif args.model == "SDXL-Vector":
            print("   Use Stable Diffusion XL Vector")
            generator = SDXL_Vector()

        data = Data(TRAIN_DATA_PATH, QUESTION_DATA_PATH)
        prompt_builder = CategorizedPromptStrategy()

        # <<<--- Khởi tạo Similarity/Reward Strategy được chọn ---<<<
        if args.similarity_strategy == "clip":
            similarity_reward_strategy = ClipSimilarityStrategy(
                aesthetic_evaluator=aesthetic_evaluator
            )
        elif args.similarity_strategy == "caption":
            similarity_reward_strategy = CaptionStrategy(vqa_evaluator=vqa_evaluator)
        elif args.similarity_strategy == "siglip":
            similarity_reward_strategy = SIGLIPSimilarityStrategy(Siglip())
        else:
            print(
                f"Lỗi: Similarity strategy '{args.similarity_strategy}' không hợp lệ."
            )
            sys.exit(1)
        print(
            f"--- Sử dụng Similarity/Reward Strategy: {type(similarity_reward_strategy).__name__} ---"
        )
        # >>>------------------------------------------------------>>>

    except Exception as e:
        print(f"Lỗi khởi tạo dependencies: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("--- Dependencies Initialized ---")
    print("--- Khởi tạo ScoreEvaluator ---")
    try:
        evaluator = ScoreEvaluator(
            vqa_evaluator,
            aesthetic_evaluator,
            generator,
            data,
            prompt_builder,
            similarity_reward_strategy,
        )
    except Exception as e:
        print(f"Lỗi khởi tạo ScoreEvaluator: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("--- ScoreEvaluator Initialized ---")

    # --- Lấy dữ liệu train ---
    print("--- Lấy dữ liệu train từ Data object ---")
    try:
        train_df = data.get_train_csv()
        id_column_name = "id"  # Hoặc 'id_prompt', 'row_id'... -> Sửa nếu cần
        if id_column_name not in train_df.columns:
            print(f"Lỗi: Không tìm thấy cột ID '{id_column_name}'.")
            sys.exit(1)
        print(f"--- Tìm thấy {len(train_df)} prompts trong train data ---")
    except Exception as e:
        print(f"Lỗi khi gọi data.get_train_csv(): {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # --- Chuẩn bị các tham số cố định và xử lý override 1 lần ---
    output_version_name = (
        args.output_version_name
        if args.output_version_name
        else yaml_params.get("version_description", args.version)
    )
    final_verbose = (
        args.verbose if args.verbose is not None else yaml_params.get("verbose", False)
    )
    final_use_prompt_builder = (
        args.use_prompt_builder
        if args.use_prompt_builder is not None
        else yaml_params.get("use_prompt_builder", False)
    )
    final_use_compression = (
        args.use_image_compression
        if args.use_image_compression is not None
        else yaml_params.get("use_image_compression", False)
    )
    final_k = args.k if args.k is not None else yaml_params.get("k")
    if final_use_compression and final_k is None:
        print("Warning: compress=true nhưng k không xác định. Dùng default k=8.")
        final_k = 8
    final_method = (
        args.method
        if args.method is not None
        else yaml_params.get("method", "UnknownMethod")
    )  # Lấy method
    final_model = (
        args.model
        if args.model is not None
        else yaml_params.get("model", "UnknownModel")
    )  # Lấy model

    # Xử lý override cho SVG config
    final_bitmap2svg_config = base_bitmap2svg_config.copy()
    if args.svg_num_colors is not None:
        final_bitmap2svg_config["num_colors"] = args.svg_num_colors
    # ... (xử lý các override svg khác như cũ) ...
    if args.svg_max_bytes is not None:
        final_bitmap2svg_config["max_size_bytes"] = args.svg_max_bytes
    if args.svg_resize is not None:
        final_bitmap2svg_config["resize"] = args.svg_resize
    target_w, target_h = args.svg_target_w, args.svg_target_h
    if target_w is not None or target_h is not None:
        current_target_size = tuple(
            base_bitmap2svg_config.get("target_size", [384, 384])
        )
        new_w = target_w if target_w is not None else current_target_size[0]
        new_h = target_h if target_h is not None else current_target_size[1]
        final_bitmap2svg_config["target_size"] = [new_w, new_h]
    if args.svg_adaptive_fill is not None:
        final_bitmap2svg_config["adaptive_fill"] = args.svg_adaptive_fill
    if "num_colors" not in final_bitmap2svg_config:
        print(
            "Warning: 'num_colors' không có trong bitmap2svg_config. Dùng default=12."
        )
        final_bitmap2svg_config["num_colors"] = 12

    base_evaluation_params = {
        "prefix_prompt": args.prefix_prompt,
        "suffix_prompt": args.suffix_prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "use_image_compression": final_use_compression,
        "use_prompt_builder": final_use_prompt_builder,
        "compression_k": final_k,
        "bitmap2svg_config": final_bitmap2svg_config,
        "version": output_version_name,  # Dùng cho thư mục output bên trong ScoreEvaluator
        "num_attempts": args.num_attempts if args.num_attempts is not None else 1,
        "verbose": final_verbose,
        "random_seed": args.random_seed if args.random_seed is not None else 42,
    }
    # Kiểm tra None cho các giá trị bắt buộc
    required_keys = [
        "prefix_prompt",
        "suffix_prompt",
        "negative_prompt",
        "height",
        "width",
        "num_inference_steps",
        "guidance_scale",
        "use_image_compression",
        "use_prompt_builder",
        "num_attempts",
        "random_seed",
    ]
    missing_keys = [k for k in required_keys if base_evaluation_params.get(k) is None]
    if missing_keys:
        print(f"Lỗi: Các tham số cơ bản sau bị thiếu: {missing_keys}")
        sys.exit(1)

    # --- Vòng lặp xử lý ---
    all_results = []
    success_count = 0
    fail_count = 0
    total_prompts = len(train_df)

    print(
        f"\n--- Bắt đầu xử lý {total_prompts} prompts với cấu hình version '{args.version}' ---"
    )
    print(f"--- Method: {final_method}, Model: {final_model} ---")
    print("-" * 30)

    for index, row in train_df.iterrows():
        current_id_prompt = row[id_column_name]
        print(
            f"\n--- [{index + 1}/{total_prompts}] Processing ID: {current_id_prompt} ---"
        )

        current_evaluation_params = base_evaluation_params.copy()
        current_evaluation_params["id_prompt"] = current_id_prompt

        try:
            # Gọi hàm generate_and_evaluate (không cần truyền method, model vào đây)
            result_core = evaluator.generate_and_evaluate(**current_evaluation_params)

            # !!! Thêm method và model vào kết quả TRƯỚC KHI lưu !!!
            # Tạo bản sao để không ảnh hưởng lần lặp sau nếu result_core là mutable reference
            result_final = result_core.copy()
            result_final["method"] = final_method
            result_final["model"] = final_model

            # Thêm kết quả đã bổ sung metadata
            all_results.append(result_final)
            success_count += 1
            print(f"--- Hoàn thành ID: {current_id_prompt} ---")
            if final_verbose:
                print(json.dumps(result_final, indent=2))

        except Exception as e:
            fail_count += 1
            print(f"🚨 Lỗi khi xử lý ID {current_id_prompt}: {e} ---")
            error_result = {
                "id_desc": current_id_prompt,  # Đổi tên key cho khớp với kết quả thành công nếu muốn
                "status": "FAILED",
                "error": str(e),
                "method": final_method,
                "model": final_model,  # Thêm metadata vào cả lỗi
                "config_version": args.version,
                "output_version": output_version_name,
            }
            all_results.append(error_result)

    # --- Tổng kết và Lưu kết quả ---
    print("\n" + "=" * 50)
    print("--- Xử lý hoàn tất ---")
    print(
        f"Tổng số prompts: {total_prompts}, Thành công: {success_count}, Thất bại: {fail_count}"
    )
    print("=" * 50)

    if args.output_json:
        output_file_path = args.output_json
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            print(f"--- Đã lưu tổng hợp kết quả vào: {output_file_path} ---")
        except Exception as e:
            print(f"Lỗi khi lưu file kết quả JSON: {e}")
    else:
        print("--- Kết quả tổng hợp không được lưu vào file (dùng --output_json). ---")


if __name__ == "__main__":
    main()

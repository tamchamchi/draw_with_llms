# #!/bin/bash

# echo "Running version 1_00 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_sdv2_00.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_1_sdv2_00\
#   --image_to_svg "contour"\
#   --model StableDiffusionV2\
#   --similarity_strategy yes-no\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_1_sdv2_00.txt
# echo "Complete!!!"

# echo "Running version 1_01 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_sdv2_01.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_1_sdv2_01\
#   --image_to_svg "vtracer"\
#   --model StableDiffusionV2\
#   --similarity_strategy yes-no\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_1_sdv2_01.txt
# echo "Complete!!!"

# echo "Running version 2_00 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_2\
#   --output_json data/results/score/json/version_2_sdv2_00.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_2_sdv2_00\
#   --image_to_svg "contour"\
#   --model StableDiffusionV2\
#   --similarity_strategy yes-no\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_2_sdv2_00.txt
# echo "Complete!!!"

# echo "Running version 2_01 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_2\
#   --output_json data/results/score/json/version_2_sdv2_01.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_2_sdv2_01\
#   --image_to_svg "vtracer"\
#   --model StableDiffusionV2\
#   --similarity_strategy yes-no\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_2_sdv2_01.txt
# echo "Complete!!!"

# echo "Running version 3_00 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdv2_00.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_3_sdv2_00\
#   --image_to_svg "contour"\
#   --model StableDiffusionV2\
#   --similarity_strategy yes-no\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_3_sdv2_00.txt
# echo "Complete!!!"

# echo "Running version 3_01 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdv2_01.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_3_sdv2_01\
#   --image_to_svg "vtracer"\
#   --model StableDiffusionV2\
#   --similarity_strategy yes-no\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_3_sdv2_01.txt
# echo "Complete!!!"

# echo "Running version 4_00 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_00.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_4_sdxl_turbo_00\
#   --image_to_svg "contour"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_00.txt
# echo "Complete!!!"

# echo "Running version 4_15 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_15.json\
#   --output_version_name version_4_sdxl_turbo_15\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --model SDXL-Turbo\
#   --image_to_svg "contour"\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 6\
#    > report/score_version_4_sdxl_turbo_15.txt
# echo "Complete!!!"

# echo "Running version 4_03 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_03.json\
#   --output_version_name version_4_sdxl_turbo_03\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --model SDXL-Turbo\
#   --image_to_svg "contour"\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_03.txt
# echo "Complete!!!"

# echo "Running version 4_01 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_01.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_4_sdxl_turbo_01\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_01.txt
# echo "Complete!!!"

# echo "Running version 5_00 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_5\
#   --output_json data/results/score/json/version_5_sdxl_turbo_00.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_5_sdxl_turbo_00\
#   --image_to_svg "contour"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_5_sdxl_turbo_00.txt
# echo "Complete!!!"

# echo "Running version 5_01 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_5\
#   --output_json data/results/score/json/version_5_sdxl_turbo_01.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_5_sdxl_turbo_01\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_5_sdxl_turbo_01.txt
# echo "Complete!!!"

# echo "Running version 4_02 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_02.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_4_sdxl_turbo_02\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 5\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_02.txt
# echo "Complete!!!"

# echo "Running version 6_00 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_6\
#   --output_json data/results/score/json/version_6_sdxl_turbo_00.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_6_sdxl_turbo_00\
#   --image_to_svg "contour"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_6_sdxl_turbo_00.txt
# echo "Complete!!!"

# echo "Running version 6_01 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_6\
#   --output_json data/results/score/json/version_6_sdxl_turbo_01.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_6_sdxl_turbo_01\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_6_sdxl_turbo_01.txt
# echo "Complete!!!"



# echo "Running version 5_02 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_5\
#   --output_json data/results/score/json/version_5_sdxl_turbo_02.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_5_sdxl_turbo_02\
#   --image_to_svg "contour"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_5_sdxl_turbo_02.txt
# echo "Complete!!!"

# echo "Running version 5_03 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_5\
#   --output_json data/results/score/json/version_5_sdxl_turbo_03.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_5_sdxl_turbo_03\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_5_sdxl_turbo_03.txt
# echo "Complete!!!"

# echo "Running version 1_02 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_sdv2_02.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_1_sdv2_02\
#   --image_to_svg "vtracer"\
#   --model StableDiffusionV2\
#   --similarity_strategy yes-no\
#   --num_inference_steps 25\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_1_sdv2_02.txt
# echo "Complete!!!"

# echo "Running version 4_04 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_04.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_4_sdxl_turbo_04\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 8\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_04.txt
# echo "Complete!!!"

# echo "Running version 4_05 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_05.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_4_sdxl_turbo_05\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_05.txt
# echo "Complete!!!"

# echo "Running version 4_06 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_06.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_4_sdxl_turbo_06\
#   --image_to_svg "contour"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 6\
#    > report/score_version_4_sdxl_turbo_06.txt
# echo "Complete!!!"

# echo "Running version 4_07 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_07.json\
#   --method "I->KMean->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_4_sdxl_turbo_07\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --use_image_compression \
#   --k 8\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_07.txt
# echo "Complete!!!"

# echo "Running version 5_07 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_5\
#   --output_json data/results/score/json/version_5_sdxl_turbo_07.json\
#   --method "I->KMean->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_5_sdxl_turbo_07\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --use_image_compression \
#   --k 8\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_5_sdxl_turbo_07.txt
# echo "Complete!!!"

# echo "Running version 6_07 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_6\
#   --output_json data/results/score/json/version_6_sdxl_turbo_07.json\
#   --method "I->KMean->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_6_sdxl_turbo_07\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --use_image_compression \
#   --k 8\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_6_sdxl_turbo_07.txt
# echo "Complete!!!"


# echo "Running version 4_08 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_08.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_4_sdxl_turbo_08\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 6\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_08.txt
# echo "Complete!!!"

# echo "Running version 4_09 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_09.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_4_sdxl_turbo_09\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 5\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_09.txt
# echo "Complete!!!"

# echo "Running version 7_00 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_7\
#   --output_json data/results/score/json/version_7_sdxl_turbo_00.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_7_sdxl_turbo_00\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 5\
#   --svg_num_colors 12\
#    > report/score_version_7_sdxl_turbo_00.txt
# echo "Complete!!!"

# echo "Running version 7_01 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_7\
#   --output_json data/results/score/json/version_7_sdxl_turbo_01.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_7_sdxl_turbo_01\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 6\
#   --svg_num_colors 12\
#    > report/score_version_7_sdxl_turbo_01.txt
# echo "Complete!!!"

# echo "Running version 7_02 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_7\
#   --output_json data/results/score/json/version_7_sdxl_turbo_02.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_7_sdxl_turbo_02\
#   --image_to_svg "contour"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 5\
#   --svg_num_colors 12\
#    > report/score_version_7_sdxl_turbo_02.txt
# echo "Complete!!!"

# echo "Running version 7_03 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_7\
#   --output_json data/results/score/json/version_7_sdxl_turbo_03.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_7_sdxl_turbo_03\
#   --image_to_svg "contour"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 6\
#   --svg_num_colors 12\
#    > report/score_version_7_sdxl_turbo_03.txt
# echo "Complete!!!"

# echo "Running version 7_04 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_7\
#   --output_json data/results/score/json/version_7_sdxl_turbo_04.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_7_sdxl_turbo_04\
#   --image_to_svg "contour"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_7_sdxl_turbo_04.txt
# echo "Complete!!!"

# echo "Running version 7_05 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_7\
#   --output_json data/results/score/json/version_7_sdxl_turbo_05.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_7_sdxl_turbo_05\
#   --image_to_svg "vtracer+ssim"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 5\
#   --svg_num_colors 12\
#    > report/score_version_7_sdxl_turbo_05.txt
# echo "Complete!!!"

# echo "Running version 7_05 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_7\
#   --output_json data/results/score/json/version_7_sdxl_turbo_05.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_7_sdxl_turbo_05\
#   --image_to_svg "vtracer+ssim"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 5\
#   --svg_num_colors 12\
#    > report/score_version_7_sdxl_turbo_05.txt
# echo "Complete!!!"

# echo "Running version 7_06 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_7\
#   --output_json data/results/score/json/version_7_sdxl_turbo_06.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_7_sdxl_turbo_06\
#   --image_to_svg "vtracer+ssim"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 6\
#   --svg_num_colors 12\
#    > report/score_version_7_sdxl_turbo_06.txt
# echo "Complete!!!"

# echo "Running version 6_08 with sdxl_turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_6\
#   --output_json data/results/score/json/version_6_sdxl_turbo_08.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
#   --output_version_name version_6_sdxl_turbo_08\
#   --image_to_svg "vtracer+ssim"\
#   --model SDXL-Turbo\
#   --num_inference_steps 4\
#   --guidance_scale 0\
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 6\
#   --svg_num_colors 12\
#    > report/score_version_6_sdxl_turbo_08.txt
# echo "Complete!!!"

echo "Running version 7_07 with sdxl_turbo..."
python src/scripts/run_evaluation.py\
 --version version_7\
  --output_json data/results/score/json/version_7_sdxl_turbo_07.json\
  --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
  --output_version_name version_7_sdxl_turbo_07\
  --image_to_svg "vtracer+ssim"\
  --model SDXL-Turbo\
  --num_inference_steps 4\
  --guidance_scale 0\
  --similarity_strategy yes-no\
  --height 512\
  --width 512\
  --svg_target_w 384\
  --svg_target_h 384\
  --num_attempts 6\
  --svg_num_colors 12\
   > report/score_version_7_sdxl_turbo_07.txt
echo "Complete!!!"

echo "Running version 6_09 with sdxl_turbo..."
python src/scripts/run_evaluation.py\
 --version version_6\
  --output_json data/results/score/json/version_6_sdxl_turbo_09.json\
  --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG(1)"\
  --output_version_name version_6_sdxl_turbo_09\
  --image_to_svg "vtracer+ssim"\
  --model SDXL-Turbo\
  --num_inference_steps 4\
  --guidance_scale 0\
  --similarity_strategy yes-no\
  --height 512\
  --width 512\
  --svg_target_w 384\
  --svg_target_h 384\
  --num_attempts 6\
  --svg_num_colors 12\
   > report/score_version_6_sdxl_turbo_09.txt
echo "Complete!!!"

echo "Running Report..."
python src/scripts/generate_report.py\
 --json_dir data/results/score/json\
  --output_per_file_csv data/results/score/per_file_averages.csv\
   > report/score.txt
echo "Complete Report"
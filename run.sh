#!/bin/bash

# echo "Running version 1 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_sdv2.json\
#   --model StableDiffusionV2\
#   --num_attempts 1\
#    > report/score_version_1_sdv2.txt
# echo "Complete!!!"

# echo "Running version 2 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_2\
#   --output_json data/results/score/json/version_2_sdv2.json\
#   --model StableDiffusionV2\
#   --num_attempts 1\
#    > report/score_version_2_sdv2.txt
# echo "Complete!!!"

# echo "Running version 3_01 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_01_sdv2.json\
#   --model StableDiffusionV2\
#   --num_attempts 3\
#    > report/score_version_3_01_sdv2.txt
# echo "Complete!!!"

# echo "Running version 2_01 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_2\
#   --output_json data/results/score/json/version_2_01_sdv2.json\
#   --model StableDiffusionV2\
#   --num_attempts 3\
#    > report/score_version_2_01_sdv2.txt
# echo "Complete!!!"

# echo "Running version 1_01 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_01_sdv2.json\
#   --model StableDiffusionV2\
#   --num_attempts 3\
#    > report/score_version_1_01_sdv2.txt
# echo "Complete!!!"

# echo "Running version 3_03 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_03_sdv2.json\
#   --model StableDiffusionV2\
#   --num_attempts 1\
#    > report/score_version_3_03_sdv2.txt
# echo "Complete!!!"

# echo "Running version 3 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo.json\
#   --output_version_name version_3_sdxl_turbo\
#   --model SDXL-Turbo\
#   --num_inference_steps 2\
#   --height 768\
#   --width 768\
#   --guidance_scale 0\
#   --num_attempts 5\
#    > report/score_version_3_sdxl_turbo.txt
# echo "Complete!!!" 

# echo "Running version 3_01 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_01_sdxl_turbo.json\
#   --output_version_name version_3_01_sdxl_turbo\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --height 512\
#   --width 512\
#   --guidance_scale 0\
#   --num_attempts 5\
#    > report/score_version_3_01_sdxl_turbo.txt
# echo "Complete!!!" 

# echo "Running version 4 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo.json\
#   --output_version_name version_4_sdxl_turbo\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --height 512\
#   --width 512\
#   --num_attempts 5\
#   --svg_num_colors 6\
#    > report/score_version_4_sdxl_turbo.txt
# echo "Complete!!!" 

# echo "Running version 1_02 with SD..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_02_sd.json\
#   --output_version_name version_1_02_sd\
#   --model StableDiffusionV2\
#   --num_inference_steps 15\
#   --guidance_scale 7.5\
#   --height 768\
#   --width 768\
#   --num_attempts 1\
#   --svg_num_colors 12\
#    > report/score_version_1_02_sd.txt
# echo "Complete!!!" 

# echo "Running version 1_03 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --prefix_prompt "subject at the center"\
#   --output_json data/results/score/json/version_1_03_sdv2.json\
#   --output_version_name version_1_03_sdv2\
#   --model StableDiffusionV2\
#   --num_inference_steps 15\
#   --guidance_scale 7.5\
#   --height 768\
#   --width 768\
#   --num_attempts 1\
#   --svg_num_colors 12\
#    > report/score_version_1_03_sdv2.txt
# echo "Complete!!!" 

echo "Running version 1_04 with SDv2..."
python src/scripts/run_evaluation.py\
 --version version_1\
 --output_version_name version_1_04_sdv2\
  --output_json data/results/score/json/version_1_04_sdv2.json\
  --model StableDiffusionV2\
  --svg_target_w 256\
  --svg_target_h 256\
  --num_attempts 3\
   > report/score_version_1_04_sdv2.txt
echo "Complete!!!"

echo "Running version 1_05 with SDv2..."
python src/scripts/run_evaluation.py\
 --version version_1\
 --output_version_name version_1_05_sdv2\
  --output_json data/results/score/json/version_1_05_sdv2.json\
  --model StableDiffusionV2\
  --svg_target_w 128\
  --svg_target_h 128\
  --num_attempts 3\
   > report/score_version_1_05_sdv2.txt
echo "Complete!!!"

echo "Running version 1_06 with SDv2..."
python src/scripts/run_evaluation.py\
 --version version_1\
  --output_version_name version_1_06_sdv2\
  --output_json data/results/score/json/version_1_06_sdv2.json\
  --model StableDiffusionV2\
  --svg_target_w 64\
  --svg_target_h 64\
  --num_attempts 3\
   > report/score_version_1_06_sdv2.txt
echo "Complete!!!"

echo "Running version 3_02 with SDXL-Turbo..."
python src/scripts/run_evaluation.py\
 --version version_3\
  --output_json data/results/score/json/version_3_02_sdxl_turbo.json\
  --output_version_name version_3_02_sdxl_turbo\
  --model SDXL-Turbo\
  --num_inference_steps 1\
  --guidance_scale 0\
  --height 512\
  --width 512\
   --svg_target_w 256\
  --svg_target_h 256\
  --num_attempts 5\
   > report/score_version_3_02_sdxl_turbo.txt
echo "Complete!!!" 

echo "Running version 3_03 with SDXL-Turbo..."
python src/scripts/run_evaluation.py\
 --version version_3\
  --output_json data/results/score/json/version_3_03_sdxl_turbo.json\
  --output_version_name version_3_03_sdxl_turbo\
  --model SDXL-Turbo\
  --num_inference_steps 1\
  --guidance_scale 0\
  --height 512\
  --width 512\
   --svg_target_w 128\
  --svg_target_h 128\
  --num_attempts 5\
   > report/score_version_3_03_sdxl_turbo.txt
echo "Complete!!!" 

# echo "Running version 1_best with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_best_sdv2.json\
#   --model StableDiffusionV2\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 3\
#    > report/score_version_1_best_sdv2.txt
# echo "Complete!!!"

echo "Running Report..."
python src/scripts/generate_report.py\
 --json_dir data/results/score/json\
  --output_per_file_csv data/results/score/per_file_averages.csv\
   > report/score.txt
echo "Complete Report"
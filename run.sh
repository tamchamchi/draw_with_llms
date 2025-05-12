#!/bin/bash

echo "Running version 1 with SDv2..."
python src/scripts/run_evaluation.py\
 --version version_1\
  --output_json data/results/score/json/version_1_sdv2.json\
  --model StableDiffusionV2\
  --num_attempts 1\
   > report/score_version_1_sdv2.txt
echo "Complete!!!"

echo "Running version 2 with SDv2..."
python src/scripts/run_evaluation.py\
 --version version_2\
  --output_json data/results/score/json/version_2_sdv2.json\
  --model StableDiffusionV2\
  --num_attempts 1\
   > report/score_version_2_sdv2.txt
echo "Complete!!!"

echo "Running version 3_01 with SDv2..."
python src/scripts/run_evaluation.py\
 --version version_3\
  --output_json data/results/score/json/version_3_01_sdv2.json\
  --model StableDiffusionV2\
  --num_attempts 3\
   > report/score_version_3_01_sdv2.txt
echo "Complete!!!"

echo "Running version 2_01 with SDv2..."
python src/scripts/run_evaluation.py\
 --version version_2\
  --output_json data/results/score/json/version_2_01_sdv2.json\
  --model StableDiffusionV2\
  --num_attempts 3\
   > report/score_version_2_01_sdv2.txt
echo "Complete!!!"

echo "Running version 1_01 with SDv2..."
python src/scripts/run_evaluation.py\
 --version version_1\
  --output_json data/results/score/json/version_1_01_sdv2.json\
  --model StableDiffusionV2\
  --num_attempts 3\
   > report/score_version_1_01_sdv2.txt
echo "Complete!!!"

# echo "Running version 3 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdv2.json\
#   --model StableDiffusionV2\
#   --num_attempts 1\
#    > report/score_version_3_sdv2.txt
# echo "Complete!!!"

# echo "Running version 3 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo.json\
#   --model SDXL-Turbo\
#   --num_inference_steps 2\
#   --height 512\
#   --width 512\
#   --guidance_scale 0\
#   --num_attempts 6\
#    > report/score_version_3_sdxl_turbo.txt
# echo "Complete!!!" 


echo "Running Report..."
python src/scripts/generate_report.py\
 --json_dir data/results/score/json\
  --output_per_file_csv data/results/score/per_file_averages.csv\
   > report/score.txt
echo "Complete Report"
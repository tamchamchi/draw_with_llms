#!/bin/bash

# ------------Stabel Diffusion v2---------------
#---------Version 1-----------------------------

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

# echo "Running version 1_00 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_sdv2_00.json\
#   --model StableDiffusionV2\
#   --num_attempts 1\
#    > report/score_version_1_sdv2_00.txt
# echo "Complete!!!"

# echo "Running version 1_01 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_sdv2_01.json\
#   --model StableDiffusionV2\
#   --num_attempts 3\
#    > report/score_version_1_sdv2_01.txt
# echo "Complete!!!"

# echo "Running version 1_03 with SD..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_sdv2_03.json\
#   --output_version_name version_1_sdv2_03\
#   --model StableDiffusionV2\
#   --num_inference_steps 15\
#   --guidance_scale 7.5\
#   --height 768\
#   --width 768\
#   --num_attempts 1\
#   --svg_num_colors 12\
#    > report/score_version_1_sdv2_03.txt
# echo "Complete!!!" 

# echo "Running version 1_04 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#  --output_version_name version_1_sdv2_04\
#   --output_json data/results/score/json/version_1_sdv2_04.json\
#   --model StableDiffusionV2\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 3\
#    > report/score_version_1_sdv2_04.txt
# echo "Complete!!!"

# echo "Running version 1_05 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#  --output_version_name version_1_sdv2_05\
#   --output_json data/results/score/json/version_1_sdv2_05.json\
#   --model StableDiffusionV2\
#   --svg_target_w 128\
#   --svg_target_h 128\
#   --num_attempts 3\
#    > report/score_version_1_sdv2_05.txt
# echo "Complete!!!"

# echo "Running version 1_06 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_version_name version_1_sdv2_06\
#   --output_json data/results/score/json/version_1_sdv2_06.json\
#   --model StableDiffusionV2\
#   --svg_target_w 64\
#   --svg_target_h 64\
#   --num_attempts 3\
#    > report/score_version_1_sdv2_06.txt
# echo "Complete!!!"

# echo "Running version 1_07 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#  --method "I->SVG + CLIP + Text_Insert + Siglip"\
#  --similarity_strategy siglip\
#  --output_version_name version_1_sdv2_07\
#   --output_json data/results/score/json/version_1_sdv2_07.json\
#   --model StableDiffusionV2\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 3\
#    > report/score_version_1_sdv2_07.txt
# echo "Complete!!!"

# echo "Running version 1_61 with SDv2..." #submit = 0.603
# python src/scripts/run_evaluation.py\
#  --version version_1\
#   --output_json data/results/score/json/version_1_sdv2_61.json\
#   --output_version_name version_1_sdv2_61\
#   --model StableDiffusionV2\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_1_sdv2_61.txt
# echo "Complete!!!"

#---------Version 2-----------------------------

# echo "Running version 2_00 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_2\
#   --output_json data/results/score/json/version_2_sdv2_00.json\
#   --model StableDiffusionV2\
#   --num_attempts 1\
#    > report/score_version_2_sdv2_00.txt
# echo "Complete!!!"

# echo "Running version 2_01 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_2\
#   --output_json data/results/score/json/version_2_sdv2_01.json\
#   --model StableDiffusionV2\
#   --num_attempts 3\
#    > report/score_version_2_sdv2_01.txt
# echo "Complete!!!"

#---------Version 3-----------------------------

# echo "Running version 3_00 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_00_sdv2.json\
#   --model StableDiffusionV2\
#   --num_attempts 3\
#    > report/score_version_3_00_sdv2.txt
# echo "Complete!!!"

# echo "Running version 3_01 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdv2_01.json\
#   --model StableDiffusionV2\
#   --num_attempts 1\
#    > report/score_version_3_sdv2_01.txt
# echo "Complete!!!"

# echo "Running version 3_62 with SDv2..." #submit = 0.598
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdv2_62.json\
#   --output_version_name version_3_sdv2_62\
#   --model StableDiffusionV2\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 1\
#   --svg_num_colors 12\
#    > report/score_version_3_sdv2_62.txt
# echo "Complete!!!"

# echo "Running version 3_63 with SDv2..." #submit = 0.608
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdv2_63.json\
#   --output_version_name version_3_sdv2_63\
#   --model StableDiffusionV2\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_3_sdv2_63.txt
# echo "Complete!!!"

# echo "Running version 3_08 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdv2_08.json\
#   --output_version_name version_3_sdv2_08\
#   --model StableDiffusionV2\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --height 768\
#   --width 768\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_3_sdv2_08.txt
# echo "Complete!!!"


# ------------SDXL-Turbo------------------------
#---------Version 1-----------------------------
#---------Version 2-----------------------------
#---------Version 3-----------------------------

# echo "Running version 3_01 with SDXL-Turbo..." #submit = 0.582
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo_01.json\
#   --output_version_name version_3_sdxl_turbo_01\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#    > report/score_version_3_sdxl_turbo_01.txt
# echo "Complete!!!" 

# echo "Running version 3_04 with SDXL-Turbo..." #submit = 0.602
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo_04.json\
#   --output_version_name version_3_sdxl_turbo_04\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 6\
#    > report/score_version_3_sdxl_turbo_04.txt
# echo "Complete!!!" 


# echo "Running version 3_05 with SDXL-Turbo..." #submit = 0.574
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo_05.json\
#   --output_version_name version_3_sdxl_turbo_05\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --height 768\
#   --width 768\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 6\
#    > report/score_version_3_sdxl_turbo_05.txt
# echo "Complete!!!" 

# echo "Running version 3_08 with SDXL-Turbo..." #submit = 0.609
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo_08.json\
#   --output_version_name version_3_sdxl_turbo_08\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_3_sdxl_turbo_08.txt
# echo "Complete!!!" 

echo "Running version 3_09 with SDXL-Turbo..."
python src/scripts/run_evaluation.py\
 --version version_3\
  --output_json data/results/score/json/version_3_sdxl_turbo_09.json\
  --output_version_name version_3_sdxl_turbo_09\
  --model SDXL-Turbo\
  --num_inference_steps 1\
  --guidance_scale 0\
  --height 768\
  --width 768\
  --svg_target_w 256\
  --svg_target_h 256\
  --num_attempts 10\
  --svg_num_colors 12\
   > report/score_version_3_sdxl_turbo_09.txt
echo "Complete!!!" 

echo "Running version 3_10 with SDXL-Turbo..."
python src/scripts/run_evaluation.py\
 --version version_3\
  --output_json data/results/score/json/version_3_sdxl_turbo_10.json\
  --output_version_name version_3_sdxl_turbo_10\
  --model SDXL-Turbo\
  --num_inference_steps 1\
  --guidance_scale 0\
  --height 768\
  --width 768\
  --svg_target_w 256\
  --svg_target_h 256\
  --num_attempts 20\
  --svg_num_colors 12\
   > report/score_version_3_sdxl_turbo_10.txt
echo "Complete!!!"

# ------------SDXL-Vector------------------------
#---------Version 1-----------------------------

# echo "Running version 1_00 with SDXL-Vector..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#  --method "I->SVG + CLIP + Text_Insert + Aesthetic"\
#  --prefix_prompt ""\
#  --suffix_prompt ""\
#  --similarity_strategy clip\
#  --output_version_name version_1_sdxl_vector_00\
#   --output_json data/results/score/json/version_1_sdxl_vector_00.json\
#   --model SDXL-Vector\
#   --num_inference_steps 20\
#   --guidance_scale 10\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 3\
#    > report/score_version_1_sdxl_vector_00.txt
# echo "Complete!!!"

# echo "Running version 1_01 with SDXL-Vector..."
# python src/scripts/run_evaluation.py\
#  --version version_1\
#  --method "I->SVG + CLIP + Text_Insert + Aesthetic"\
#  --similarity_strategy clip\
#  --output_version_name version_1_sdxl_vector_01\
#   --output_json data/results/score/json/version_1_sdxl_vector_01.json\
#   --model SDXL-Vector\
#   --num_inference_steps 15\
#   --guidance_scale 7.5\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 3\
#    > report/score_version_1_sdxl_vector_01.txt
# echo "Complete!!!"

#---------Version 2-----------------------------
#---------Version 3-----------------------------

# ========================================================


echo "Running Report..."
python src/scripts/generate_report.py\
 --json_dir data/results/score/json\
  --output_per_file_csv data/results/score/per_file_averages.csv\
   > report/score.txt
echo "Complete Report"
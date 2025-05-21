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

# echo "Running version 3_08 with SDv2..." #submit=0.610
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

# echo "Running version 3_09 with SDv2..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdv2_09.json\
#   --output_version_name version_3_sdv2_09\
#   --model StableDiffusionV2\
#   --num_inference_steps 15\
#   --guidance_scale 7.5\
#   --height 768\
#   --width 768\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 5\
#   --svg_num_colors 12\
#    > report/score_version_3_sdv2_09.txt
# echo "Complete!!!"

# echo "Running version 3_10 with SDv2..." #submit=
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdv2_10.json\
#   --output_version_name version_3_sdv2_10\
#   --model StableDiffusionV2\
#   --num_inference_steps 35\
#   --guidance_scale 15\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 768\
#   --width 768\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_3_sdv2_10.txt
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

# echo "Running version 3_09 with SDXL-Turbo..." #submit = 0.569
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo_09.json\
#   --output_version_name version_3_sdxl_turbo_09\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --height 768\
#   --width 768\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 10\
#   --svg_num_colors 12\
#    > report/score_version_3_sdxl_turbo_09.txt
# echo "Complete!!!" 

# echo "Running version 3_10 with SDXL-Turbo..." # submit = error
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo_10.json\
#   --output_version_name version_3_sdxl_turbo_10\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --height 768\
#   --width 768\
#   --svg_target_w 256\
#   --svg_target_h 256\
#   --num_attempts 20\
#   --svg_num_colors 12\
#    > report/score_version_3_sdxl_turbo_10.txt
# echo "Complete!!!"

# echo "Running version 3_11 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo_11.json\
#   --output_version_name version_3_sdxl_turbo_11\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 6\
#    > report/score_version_3_sdxl_turbo_11.txt
# echo "Complete!!!"

# echo "Running version 3_12 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#   --output_json data/results/score/json/version_3_sdxl_turbo_12.json\
#   --output_version_name version_3_sdxl_turbo_12\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 6\
#    > report/score_version_3_sdxl_turbo_12.txt
# echo "Complete!!!"

# ---------Version 4-----------------------------

# echo "Running version 4_13 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_13.json\
#   --output_version_name version_4_sdxl_turbo_13\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 10\
#   --svg_num_colors 6\
#    > report/score_version_4_sdxl_turbo_13.txt
# echo "Complete!!!"

# new I -> SVG algorthm
# echo "Running version 4_14 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_14.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_4_sdxl_turbo_14\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 6\
#    > report/score_version_4_sdxl_turbo_14.txt
# echo "Complete!!!"

# old I -> SVG algorthm
# echo "Running version 4_15 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_15.json\
#   --output_version_name version_4_sdxl_turbo_15\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 6\
#    > report/score_version_4_sdxl_turbo_15.txt
# echo "Complete!!!"

echo "Running version 4_15_01 with SDXL-Turbo..."
python src/scripts/run_evaluation.py\
 --version version_4\
  --output_json data/results/score/json/version_4_sdxl_turbo_15_01.json\
  --output_version_name version_4_sdxl_turbo_15\
  --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
  --image_to_svg "vtracer"\
  --model SDXL-Turbo\
  --num_inference_steps 1\
  --guidance_scale 0\
  --no-use_image_compression \
  --similarity_strategy yes-no\
  --height 512\
  --width 512\
  --svg_target_w 384\
  --svg_target_h 384\
  --num_attempts 3\
  --svg_num_colors 6\
   > report/score_version_4_sdxl_turbo_15_01.txt
echo "Complete!!!"

# old I -> SVG algorthm 
# echo "Running version 4_16 with SDXL-Turbo..." #submit = 0.635 (new metrix, new prompt)
# python src/scripts/run_evaluation.py\
#  --version version_4\
#  --method "I->SVG + Yes-No + Text_Insert + Aesthetic"\
#   --output_json data/results/score/json/version_4_sdxl_turbo_16.json\
#   --output_version_name version_4_sdxl_turbo_16\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 6\
#    > report/score_version_4_sdxl_turbo_16.txt
# echo "Complete!!!"

# echo "Running version 4_17 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_4\
#   --output_json data/results/score/json/version_4_sdxl_turbo_17.json\
#   --output_version_name version_4_sdxl_turbo_17\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 7\
#   --svg_num_colors 12\
#    > report/score_version_4_sdxl_turbo_17.txt
# echo "Complete!!!"

# new I -> SVG algorthm
# echo "Running version 5_15 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_5\
#   --output_json data/results/score/json/version_5_sdxl_turbo_15.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_5_sdxl_turbo_15\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_5_sdxl_turbo_15.txt
# echo "Complete!!!"

# echo "Running version 5_15 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_5\
#   --output_json data/results/score/json/version_5_sdxl_turbo_15_01.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_5_sdxl_turbo_15\
#   --image_to_svg "vtracer"\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_5_sdxl_turbo_15_01.txt
# echo "Complete!!!"

# new I -> SVG algorthm
# echo "Running version 5_16 with SDXL-Turbo..."
# python src/scripts/run_evaluation.py\
#  --version version_5\
#   --output_json data/results/score/json/version_5_sdxl_turbo_16.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_5_sdxl_turbo_16\
#   --model SDXL-Turbo\
#   --num_inference_steps 1\
#   --guidance_scale 0\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_5_sdxl_turbo_16.txt
# echo "Complete!!!"



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

# echo "Running version 2_02 with SDXL-Vector..."
# python src/scripts/run_evaluation.py\
#  --version version_2\
#  --method "I->SVG + CLIP + Text_Insert + Aesthetic"\
#  --similarity_strategy clip\
#  --output_version_name version_2_sdxl_vector_02\
#   --output_json data/results/score/json/version_2_sdxl_vector_02.json\
#   --model SDXL-Vector\
#   --num_inference_steps 20\
#   --guidance_scale 15\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#    > report/score_version_2_sdxl_vector_02.txt
# echo "Complete!!!"

#---------Version 3-----------------------------
# echo "Running version 3_03 with SDXL-Vector..."
# python src/scripts/run_evaluation.py\
#  --version version_3\
#  --method "I->SVG + CLIP + Text_Insert + Aesthetic"\
#  --similarity_strategy clip\
#  --output_version_name version_3_sdxl_vector_03\
#   --output_json data/results/score/json/version_3_sdxl_vector_03.json\
#   --model SDXL-Vector\
#   --num_inference_steps 20\
#   --guidance_scale 15\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#    > report/score_version_3_sdxl_vector_03.txt
# echo "Complete!!!"

# new I->SVG
# echo "Running version 5_16 with SDXL-vector..."
# python src/scripts/run_evaluation.py\
#  --version version_5\
#   --output_json data/results/score/json/version_5_sdxl_vector_16.json\
#   --method "I->SVG + Yes-No + Text_Insert + Aesthetic + New I->SVG"\
#   --output_version_name version_5_sdxl_vector_16\
#   --model SDXL-Vector\
#   --num_inference_steps 20\
#   --guidance_scale 10\
#   --no-use_image_compression \
#   --similarity_strategy yes-no\
#   --height 512\
#   --width 512\
#   --svg_target_w 384\
#   --svg_target_h 384\
#   --num_attempts 3\
#   --svg_num_colors 12\
#    > report/score_version_5_sdxl_vector_16.txt
# echo "Complete!!!"

# ========================================================


echo "Running Report..."
python src/scripts/generate_report.py\
 --json_dir data/results/score/json\
  --output_per_file_csv data/results/score/per_file_averages.csv\
   > report/score.txt
echo "Complete Report"
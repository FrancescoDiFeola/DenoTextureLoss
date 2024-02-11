#!/usr/bin/env bash

# Train HERE YOU RUN YOUR PROGRAM
python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name baseline_diff01_1 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name baseline_diff01_2 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name baseline_diff01_3 --windowing True

python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_max_diff0001_1 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_max_diff0001_2 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_max_diff0001_3 --windowing True

python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_avg_diff0001_1 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_avg_diff0001_2 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_avg_diff0001_3  --windowing True

python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_Frob_diff0001_1 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_Frob_diff0001_2 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_Frob_diff0001_3 --windowing True

python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_att_diff0001_1 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_att_diff0001_2 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name texture_att_diff0001_3 --windowing True

python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name perceptual_diff_4 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name perceptual_diff_5 --windowing True
# python3 ./test.py --dataset_mode mayo --name baseline_cycleGAN --model cycle_gan --text_file ./data/mayo_training_9pat.csv --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --experiment_name perceptual_diff_6 --windowing True

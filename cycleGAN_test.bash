#!/usr/bin/env bash

# Train HERE YOU RUN YOUR PROGRAM
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_baseline_10 --experiment_name baseline_10 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_baseline_11 --experiment_name baseline_11 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_baseline_12 --experiment_name baseline_12 --windowing True

python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metric_perceptual_10 --experiment_name perceptual_10 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metric_perceptual_11 --experiment_name perceptual_11 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metric_perceptual_12 --experiment_name perceptual_12 --windowing True

python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_max_10 --experiment_name texture_max_10 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_max_11 --experiment_name texture_max_11 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_max_12 --experiment_name texture_max_12 --windowing True

python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_avg_10 --experiment_name texture_avg_10 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_avg_11 --experiment_name texture_avg_11 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_avg_12 --experiment_name texture_avg_12 --windowing True

python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_Frob_10 --experiment_name texture_Frob_10 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_Frob_11 --experiment_name texture_Frob_11 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_Frob_12 --experiment_name texture_Frob_12 --windowing True

python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_att_13 --experiment_name texture_att_13 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_att_14 --experiment_name texture_att_14 --windowing True
python3 ./test.py --dataset_mode LIDC_IDRI --name baseline_cycleGAN --model cycle_gan --dataroot False --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --metric_folder metrics_texture_att_15 --experiment_name texture_att_15 --windowing True

#!/usr/bin/env bash

# HERE YOU RUN YOUR PROGRAM
# python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_ssim_1 --experiment_name ssim_1 --windowing True
# python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_ssim_2 --experiment_name ssim_2 --windowing True
# python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_ssim_3 --experiment_name ssim_3 --windowing True

# python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_edge_1 --experiment_name edge_1 --windowing True
# python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_edge_2 --experiment_name edge_2 --windowing True
# python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_edge_3 --experiment_name edge_3 --windowing True

# python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_autoencoder_1 --experiment_name autoencoder_1 --windowing True
# python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_autoencoder_2 --experiment_name autoencoder_2 --windowing True
# python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_autoencoder_3 --experiment_name autoencoder_3 --windowing True

python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 3 --output_nc 3 --load_size 256 --channels 3 --img_height 256 --img_width 256 --metric_folder metrics_baseline_1 --experiment_name baseline_1 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 3 --output_nc 3 --load_size 256 --channels 3 --img_height 256 --img_width 256 --metric_folder metrics_baseline_2 --experiment_name baseline_2 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 3 --output_nc 3 --load_size 256 --channels 3 --img_height 256 --img_width 256 --metric_folder metrics_baseline_3 --experiment_name baseline_3 --windowing True

"""python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_max_diff_1 --experiment_name texture_max_diff_1 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_max_diff_2 --experiment_name texture_max_diff_2 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_max_diff_3 --experiment_name texture_max_diff_3 --windowing True

python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_avg_diff_1 --experiment_name texture_avg_diff_1 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_avg_diff_2 --experiment_name texture_avg_diff_2 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_avg_diff_3 --experiment_name texture_avg_diff_3 --windowing True

python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_Frob_diff_1 --experiment_name texture_Frob_diff_1 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_Frob_diff_2 --experiment_name texture_Frob_diff_2 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_Frob_diff_3 --experiment_name texture_Frob_diff_3 --windowing True

python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_attention_diff_piqe_niqe_1 --experiment_name texture_attention_diff_1 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_attention_diff_piqe_niqe_2 --experiment_name texture_attention_diff_2 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 1 --output_nc 1 --load_size 256 --channels 1 --img_height 256 --img_width 256 --metric_folder metrics_texture_attention_diff_piqe_niqe_3 --experiment_name texture_attention_diff_3 --windowing True"""

python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 3 --output_nc 3 --load_size 256 --channels 3 --img_height 256 --img_width 256 --metric_folder metrics_perceptual_1 --experiment_name perceptual_1 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 3 --output_nc 3 --load_size 256 --channels 3 --img_height 256 --img_width 256 --metric_folder metrics_perceptual_2 --experiment_name perceptual_2 --windowing True
python3 ./unit_test.py --dataset_mode mayo --name unit --model unit --dataroot False --gpu_ids -1 --input_nc 3 --output_nc 3 --load_size 256 --channels 3 --img_height 256 --img_width 256 --metric_folder metrics_perceptual_3 --experiment_name perceptual_3 --windowing True

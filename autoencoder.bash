#!/usr/bin/env bash

# Train HERE YOU RUN YOUR PROGRAM
python3 ./autoencoder_main.py --dataset_mode LIDC_IDRI --text_file ./data/autoencoder_training.csv --dataroot False --batch_size 16 --gpu_ids -1 --load_size 256 --input_nc 1 --output_nc 1 --no_html --display_id 0 --windowing True

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Fréchet Inception Distance (FID). Adapted from Karras et al.:
   https://github.com/NVlabs/stylegan2/blob/master/metrics/frechet_inception_distance.py"""

import scipy.linalg
import numpy as np
import os
import pickle
import requests
import time
import tqdm
import dnnlib
import torch_utils
from typing import Any
import torch

_URL_TO_PKL_NAME = {'https://drive.google.com/uc?id=1j3pS3bdTXIYL56kpcpdMrrvPJVT90IY0': 'clip-vit_b32.pkl',
                    'https://drive.google.com/uc?id=119HvnQ5nwHl0_vUTEFWQNk4bwYjoXTrC': 'ffhq-fid50k_5.30-snapshot-022608.pkl',
                    'https://drive.google.com/uc?id=1yDD9iqw3YYbkn2d7N8ciYu81widI-uEL': 'inception_v3-tf-2015-12-05.pkl'}


def compute_fid(real_features: np.ndarray,
                gen_features: np.ndarray) -> float:
    """Computes the Fréchet Inception Distance."""
    assert real_features.ndim == 2 and gen_features.ndim == 2
    assert real_features.shape[0] == gen_features.shape[0]

    # Feature statistics.
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # FID.
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return fid


def download_pickle(url: str,
                    pkl_name: str,
                    pickle_dir: str = './.pickles/',
                    num_attempts: int = 10,
                    chunk_size: int = 512 * 1024,  # 512 KB.
                    retry_delay: int = 2) -> str:
    """Downloads network pickle file from an URL."""
    os.makedirs(pickle_dir, exist_ok=True)

    def _is_successful(response: requests.models.Response) -> bool:
        return response.status_code == 200

    print(pickle_dir)
    print(pkl_name)
    # Download file from Google Drive URL.
    network_path = os.path.join(pickle_dir, pkl_name)
    print(network_path)
    if not os.path.exists(network_path):
        print(f'Downloading network pickle ({pkl_name})...')
        for attempts_left in reversed(range(num_attempts)):
            try:
                with requests.Session() as session:
                    with session.get(f'{url}&confirm=t', stream=True) as response:
                        assert _is_successful(response), \
                            f'Downloading network pickle ({pkl_name}) from URL {url} failed.'

                        # Save network pickle.
                        with open(network_path, 'wb') as f:
                            total = response.headers['Content-Length']
                            total = int(total)
                            pbar = tqdm.tqdm(total=total, unit="B", unit_scale=True)
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                f.write(chunk)
                                pbar.update(len(chunk))
                        break
            except KeyboardInterrupt:
                raise
            except:
                print(f'Failed. Retrying in {retry_delay}s (attempts left {attempts_left})...')
                time.sleep(retry_delay)
            else:
                print(f'Downloading {pkl_name} skipped; already exists in {pickle_dir}')

    return network_path


def load_feature_network(network_name: str) -> Any:
    """Loads a pre-trained feature network."""
    _network_urls = {  # 'clip': 'https://drive.google.com/uc?id=1VF0xYAfGEPH0bhNYLFS_yTEoVT2rkFFG',
        'clip': 'https://drive.google.com/uc?id=1j3pS3bdTXIYL56kpcpdMrrvPJVT90IY0',
        'inception_v3_tf': 'https://drive.google.com/uc?id=1yDD9iqw3YYbkn2d7N8ciYu81widI-uEL'}
    assert network_name in _network_urls.keys(), \
        f"Unknown feature network name {network_name}."
    url = _network_urls[network_name]
    network_path = download_pickle(url=url,
                                   pkl_name=_URL_TO_PKL_NAME[url])
    print(network_path)
    with open(network_path, 'rb') as f:
        network = pickle.load(f)
    return network

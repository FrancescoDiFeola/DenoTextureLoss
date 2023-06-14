import os
import random
import pandas as pd
import pydicom
from tqdm import tqdm
import shutil


def create_annotation_file(path_folder, domain):
    info = {
        'path_slice': [],
        'partial_path': [],
        'patient': [],
        'domain': [],
        'tube_current': [],
        'bit_stored': []

    }
    patients_list = sorted(os.listdir(path_folder))
    if '.DS_Store' in patients_list:
        patients_list.remove('.DS_Store')
    for t in patients_list:
        slices_list = sorted(os.listdir(os.path.join(path_folder, t)))  # , dose_name (for mayo dataset)
        if '.DS_Store' in slices_list:
            slices_list.remove('.DS_Store')
        for j in slices_list:
            partial_path = os.path.join(t, j)
            # print(partial_path)
            total_path = os.path.join(path_folder, partial_path)
            dicom = pydicom.read_file(total_path)
            info['path_slice'].append(total_path)
            info['partial_path'].append(partial_path)
            info['patient'].append(t)
            info['domain'].append(domain)
            try:
                info['tube_current'].append(dicom['XRayTubeCurrent'].value)
            except KeyError:
                info['tube_current'].append('-')
            try:
                info['bit_stored'].append(dicom['BitsStored'].value)
            except KeyError:
                info['bit_stored'].append('-')
    return pd.DataFrame(info)


if __name__ == "__main__":



    
    interim_dir = './data/'
    df_ld = create_annotation_file(
        path_folder='./CT_data/interim/test_mayo_ext/HDCT',
        domain='HD',
    )
    df_hd = create_annotation_file(
        path_folder='./CT_data/interim/test_mayo_ext/LDCT',
        domain='LD',
    )

    df = pd.concat([df_ld, df_hd], axis=0, ignore_index=True)
    df.to_csv(os.path.join(interim_dir, 'mayo_test_ext.csv'))

    print('May be the force with you.')

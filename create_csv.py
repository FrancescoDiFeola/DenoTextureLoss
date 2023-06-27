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
        'bit_stored': [],
        'exposure_time': [],
        'exposure': [],
        'slice_thickness': [],
        'protocol': [],
        'tension': [],
    }
    patients_list = sorted(os.listdir(path_folder))
    if '.DS_Store' in patients_list:
        patients_list.remove('.DS_Store')
    for t in tqdm(patients_list):
        slices_list = sorted(os.listdir(os.path.join(path_folder, t)))  # , dose_name (for mayo dataset)
        if '.DS_Store' in slices_list:
            slices_list.remove('.DS_Store')
        for j in slices_list:
            partial_path = os.path.join(t, j)
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
            try:
                info['exposure_time'].append(dicom['ExposureTime'].value)
            except KeyError:
                info['exposure_time'].append('-')
            try:
                info['exposure'].append(dicom['Exposure'].value)
            except KeyError:
                info['exposure'].append('-')
            try:
                info['slice_thickness'].append(dicom['SliceThickness'].value)
            except KeyError:
                info['slice_thickness'].append('-')
            try:
                info['protocol'].append(dicom['ProtocolName'].value)
            except KeyError:
                info['protocol'].append('-')
            try:
                info['tension'].append(dicom['KVP'].value)
            except KeyError:
                info['tension'].append('-')

    return pd.DataFrame(info)


def create_annotation_file_2(path_folder):
    info = {
        'path_slice': [],
        'partial_path': [],
        'patient': [],
        'domain': [],
        'tube_current': [],
        'bit_stored': [],
        'exposure_time': [],
        'exposure': [],
        'slice_thickness': [],
        'protocol': [],
        'tension': [],
    }
    patients_list = os.listdir(path_folder)
    if '.DS_Store' in patients_list:
        patients_list.remove('.DS_Store')
    if 'LICENSE' in patients_list:
        patients_list.remove('LICENSE')
    for t in tqdm(patients_list):
        path_int = os.listdir(os.path.join(path_folder, t))  # , dose_name (for mayo dataset)
        if '.DS_Store' in path_int:
            path_int.remove('.DS_Store')
        for j in path_int:
            path_int_int = os.listdir(os.path.join(path_folder, t, j))
            for k in path_int_int:
                total_path = os.listdir(os.path.join(path_folder, t, j, k))
                for s in total_path:
                    dicom = pydicom.read_file(os.path.join(path_folder, t, j, k, s), force=True)

                    info['path_slice'].append('-')
                    info['partial_path'].append('-')
                    info['patient'].append(t)
                    info['domain'].append('-')
                    try:
                        info['tube_current'].append(dicom['XRayTubeCurrent'].value)
                    except KeyError:
                        info['tube_current'].append('-')
                    try:
                        info['bit_stored'].append(dicom['BitsStored'].value)
                    except KeyError:
                        info['bit_stored'].append('-')
                    try:
                        info['exposure_time'].append(dicom['ExposureTime'].value)
                    except KeyError:
                        info['exposure_time'].append('-')
                    try:
                        info['exposure'].append(dicom['Exposure'].value)
                    except KeyError:
                        info['exposure'].append('-')
                    try:
                        info['slice_thickness'].append(dicom['SliceThickness'].value)
                    except KeyError:
                        info['slice_thickness'].append('-')
                    try:
                        info['protocol'].append(dicom['ProtocolName'].value)
                    except KeyError:
                        info['protocol'].append('-')
                    try:
                        info['tension'].append(dicom['KVP'].value)
                    except KeyError:
                        info['tension'].append('-')

    return pd.DataFrame(info)


if __name__ == "__main__":
    interim_dir = './data/'
    df = create_annotation_file(
        path_folder='CT_data/interim/test_LIDC',
        domain='LD',
    )
    '''df_hd = create_annotation_file(
        path_folder='/Volumes/sandisk/LIDC_IDRI_unpaired_balanced/LDCT',
        domain='LD',
    )'''

    '''df = create_annotation_file_2(
         path_folder='/Volumes/UmuAILab-backup/manifest-1600709154662/LIDC-IDRI'
     )'''

    # df = pd.concat([df_ld, df_hd], axis=0, ignore_index=True)
    df.to_csv(os.path.join(interim_dir, 'LIDC_test.csv'))

    print('May be the force with you.')

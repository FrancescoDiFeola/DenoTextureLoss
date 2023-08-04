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
        'XRayTubeCurrent': [],
        'BitsStored': [],
        'ExposureTime': [],
        'Exposure': [],
        'SliceThickness': [],
        'ProtocolName': [],
        'KVP': [],
        'ConvolutionKernel': [],
        'BodyPartExamined': [],
        'ScanOptions': [],
        'ReconstructionDiameter': [],
        'DistanceSourceToDetector': [],
        'DistanceSourceToPatient': [],
        'Manufacturer': [],
        'Modality': [],
        'ManufacturerModelName': [],
        'PatientID': [],
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
                info['XRayTubeCurrent'].append(dicom['XRayTubeCurrent'].value)
            except KeyError:
                info['XRayTubeCurrent'].append('-')
            try:
                info['BitsStored'].append(dicom['BitsStored'].value)
            except KeyError:
                info['BitsStored'].append('-')
            try:
                info['ExposureTime'].append(dicom['ExposureTime'].value)
            except KeyError:
                info['ExposureTime'].append('-')
            try:
                info['Exposure'].append(dicom['Exposure'].value)
            except KeyError:
                info['Exposure'].append('-')
            try:
                info['SliceThickness'].append(dicom['SliceThickness'].value)
            except KeyError:
                info['SliceThickness'].append('-')
            try:
                info['ProtocolName'].append(dicom['ProtocolName'].value)
            except KeyError:
                info['ProtocolName'].append('-')
            try:
                info['KVP'].append(dicom['KVP'].value)
            except KeyError:
                info['KVP'].append('-')
            try:
                info['ConvolutionKernel'].append(dicom['ConvolutionKernel'].value)
            except KeyError:
                info['ConvolutionKernel'].append('-')
            try:
                info['BodyPartExamined'].append(dicom['BodyPartExamined'].value)
            except KeyError:
                info['BodyPartExamined'].append('-')
            try:
                info['ScanOptions'].append(dicom['ScanOptions'].value)
            except KeyError:
                info['ScanOptions'].append('-')
            try:
                info['ReconstructionDiameter'].append(dicom['ReconstructionDiameter'].value)
            except KeyError:
                info['ReconstructionDiameter'].append('-')
            try:
                info['DistanceSourceToDetector'].append(dicom['DistanceSourceToDetector'].value)
            except KeyError:
                info['DistanceSourceToDetector'].append('-')
            try:
                info['DistanceSourceToPatient'].append(dicom['DistanceSourceToPatient'].value)
            except KeyError:
                info['DistanceSourceToPatient'].append('-')
            try:
                info['Manufacturer'].append(dicom['Manufacturer'].value)
            except KeyError:
                info['Manufacturer'].append('-')
            try:
                info['Modality'].append(dicom['Modality'].value)
            except KeyError:
                info['Modality'].append('-')
            try:
                info['ManufacturerModelName'].append(dicom['ManufacturerModelName'].value)
            except KeyError:
                info['ManufacturerModelName'].append('-')
            try:
                info['PatientID'].append(dicom['PatientID'].value)
            except KeyError:
                info['PatientID'].append('-')


    return pd.DataFrame(info)

# for LIDC/IDRI
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
    df_ld = create_annotation_file(
        path_folder='./CT_data/interim/ELCAP',
        domain='LD',
    )
    '''df_hd = create_annotation_file(
        path_folder='./CT_data/interim/ELCAP/HDCT',
        domain='HD',
    )'''

    '''df = create_annotation_file_2(
         path_folder='/Volumes/UmuAILab-backup/manifest-1600709154662/LIDC-IDRI'
     )'''

    # df = pd.concat([df_ld, df_hd], axis=0, ignore_index=True)
    df_ld.to_csv(os.path.join(interim_dir, 'ELCAP.csv'))

    print('May be the force with you.')

import os
import pickle
from collections import defaultdict, OrderedDict
from enum import Enum
from glob import glob
import shutil
from scipy.ndimage import zoom
import nibabel as nib
import numpy as np
import scipy.ndimage
import torch
from torchvision import transforms

class Modalities(Enum):
    FLAIR = 'FLAIR'
    T2 = 'T2'


class Phase(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Views(Enum):
    AXIAL = 0
    SAGITTAL = 1
    CORONAL = 2


class Mode(Enum):
    STATIC = 'static'
    LONGITUDINAL = 'longitudinal'


class Dataset(Enum):
    MSSEG = 'msseg'
    ISBI = 'isbi'
    INHOUSE = 'inhouse'


class Evaluate(Enum):
    TRAINING = 'training'
    TEST = 'test'

def load_preprocessed_files(root_dir, datalist, evaluate, base_path='data'):

    patients = datalist

    empty_slices = []
    non_positive_slices = []

    for patient in patients:
        empty_slices_patient , non_positive_slices_patient = [], []
        print(f'Processing patient {patient}')
        patient_path = os.path.join(root_dir, patient)

        with open(os.path.join(patient_path, 'empty_slices.pckl'), 'rb') as f:
            empty_slices_patient = pickle.load(f)
        with open(os.path.join(patient_path, 'non_positive_slices.pckl'), 'rb') as f:
            non_positive_slices_patient = pickle.load(f)

        empty_slices += empty_slices_patient
        non_positive_slices += non_positive_slices_patient

    return empty_slices, non_positive_slices

def retrieve_data_dir_paths_3D(data_dir, datalist, phase, preprocess, mode, view=None):

    if preprocess:
        print('Preprocessing files...')
        preprocess_files_3D(data_dir, datalist, phase)
        print('Files preprocessed.')

    if mode == Mode.LONGITUDINAL:
        data_dir_paths = retrieve_paths_longitudinal3D(get_patient_paths3D(data_dir, datalist)).items()
    else:
        data_dir_paths = retrieve_paths_static3D(get_patient_paths3D(data_dir, datalist)).items()
    data_dir_paths = OrderedDict(sorted(data_dir_paths))
    _data_dir_paths = []
    patient_keys = [key for key in data_dir_paths.keys()]

    if mode == Mode.STATIC:
        for patient in patient_keys:
                _data_dir_paths += data_dir_paths[patient]

        if view:
            _data_dir_paths = list(filter(lambda path: int(path.split(os.sep)[-2]) == view.value, _data_dir_paths))

    if mode == Mode.LONGITUDINAL:
        for patient in patient_keys:
            _data_dir_paths.append(tuple(data_dir_paths[patient]))

    return _data_dir_paths

def create_folders(data, label, timestep_path, modality=None):

    if not os.path.exists(timestep_path):
        os.makedirs(timestep_path)

    #save npy files
    if modality is not None:
        np.save(os.path.join(timestep_path, f'img_{modality}.npy'), np.array(data))
    else:
        np.save(os.path.join(timestep_path, f'img.npy'), np.array(data))
    if label is not None:
        np.save(os.path.join(timestep_path, f'lbl_{modality}.npy'), np.array(label))

def preprocess_files_3D(root_dir, datalist, phase, base_path='data3D'):
    patients = datalist
    for patient in patients:
        print(f'Processing patient {patient}')
        patient_path = os.path.join(root_dir, patient)
        if os.path.exists(os.path.join(patient_path, base_path)):
            #continue
            shutil.rmtree(os.path.join(patient_path, base_path), ignore_errors=True)
        timestep_limit = os.listdir(patient_path)

        for timestep in timestep_limit:
            data = [] 
            for modality in list(Modalities):
                _, value = modality.name, modality.value
                data_path = f'{patient_path}/{timestep}/{patient}_{timestep}_{value}.nii.gz'
                #transformed_data = resize_image(data_path)
                transformed_data = pad_to_largest_dimension(data_path)
                normalized_data = (transformed_data - np.min(transformed_data)) / (np.max(transformed_data) - np.min(transformed_data))

                data.append(normalized_data)

            # create folders and save images and labels
            create_folders(np.array(data), timestep_path = os.path.join(patient_path, base_path, str(timestep)), label = None)
            label_path = f'{patient_path}/{timestep}/{patient}_{timestep}_MASK.nii.gz'
            if os.path.exists(label_path):
                #transformed_labels = resize_image(label_path)
                transformed_labels = pad_to_largest_dimension(label_path)
            else:
                transformed_labels = np.zeros(normalized_data.shape)
            # Save label
            np.save(os.path.join(patient_path, base_path, str(timestep), 'label.npy'), np.array(transformed_labels))


def get_padding(max_dim, dim):
    pad_before = (max_dim - dim) // 2
    pad_after = max_dim - dim - pad_before
    return (pad_before, pad_after)

def crop_to_smallest_dimension(data_path):
    # Load the NIfTI file
    data = nib.load(data_path).get_fdata()
    
    # Original dimensions of the data
    original_x_dim, original_y_dim, original_z_dim = data.shape
    
    # Desired dimensions after resizing
    desired_x_dim, desired_y_dim, desired_z_dim = 176, 224, 176
    
    # Calculate padding amounts for each dimension
    pad_x = max(desired_x_dim - original_x_dim, 0)
    pad_y = max(desired_y_dim - original_y_dim, 0)
    pad_z = max(desired_z_dim - original_z_dim, 0)
    
    # Calculate cropping coordinates after padding
    x_start = pad_x // 2
    y_start = pad_y // 2
    z_start = pad_z // 2
    
    x_end = x_start + original_x_dim
    y_end = y_start + original_y_dim
    z_end = z_start + original_z_dim
    
    # Pad the data
    padded_data = np.pad(data, ((x_start, pad_x - x_start), (y_start, pad_y - y_start), (z_start, pad_z - z_start)), mode='constant')
    
    # Crop the padded data to desired dimensions
    cropped_data = padded_data[x_start:x_start + desired_x_dim, y_start:y_start + desired_y_dim, z_start:z_start + desired_z_dim]
    
    return cropped_data

def reverse_resize_image(data, target_size=(182, 218, 182)):
    """
    Resizes the image back to the target size.
    
    Parameters:
    data (np.ndarray): The input data array.
    target_size (tuple): The target size to resize back to.
    
    Returns:
    np.ndarray: The resized data array.
    """
    x_target, y_target, z_target = target_size
    x_dim, y_dim, z_dim = data.shape

    # Pad x and z dimensions to match target size
    pad_x = (x_target - x_dim) // 2
    pad_z = (z_target - z_dim) // 2

    padded_data = np.pad(data, ((pad_x, pad_x), (0, 0), (pad_z, pad_z)), mode='constant')

    # Crop y dimension to match target size
    y_start = (padded_data.shape[1] - y_target) // 2
    y_end = y_start + y_target

    resized_data = padded_data[:, y_start:y_end, :]

    return resized_data

def resize_image(data_path, target_size=(176, 224, 176)):
    # Load the NIfTI file
    data = nib.load(data_path).get_fdata()
    x_target, y_target, z_target = target_size
    # Pad y dimension to match target size
    x_dim, y_dim, z_dim = data.shape
    pad_y = y_target - y_dim

    data = np.pad(data, ((0, 0), (pad_y//2, pad_y//2), (0, 0)), mode='constant')

    # Crop x and z dimensions to match target size
    x_start = (x_dim - x_target) // 2
    z_start = (z_dim - z_target) // 2

    x_end = x_start + x_target
    z_end = z_start + z_target

    cropped_data = data[x_start:x_end, :, z_start:z_end]

    return cropped_data

def pad_to_largest_dimension(data_path):
    # Load the NIfTI file
    data = nib.load(data_path).get_fdata()
    
    # Get the dimensions of the data
    x_dim, y_dim, z_dim = data.shape
    
    # Find the largest dimension
    max_dim = max(x_dim, y_dim, z_dim)
    
    # Calculate the padding required for each axis to center the data
    x_pad_before = (max_dim - x_dim) // 2
    x_pad_after = max_dim - x_dim - x_pad_before
    
    y_pad_before = (max_dim - y_dim) // 2
    y_pad_after = max_dim - y_dim - y_pad_before
    
    z_pad_before = (max_dim - z_dim) // 2
    z_pad_after = max_dim - z_dim - z_pad_before
    
    # Pad the data to the largest dimension
    padded_data = np.pad(data, 
                         ((x_pad_before, x_pad_after), 
                          (y_pad_before, y_pad_after), 
                          (z_pad_before, z_pad_after)), 
                         mode='constant', constant_values=0)
    
    return padded_data

def crop_to_smallest_dimension(data_path):
    # Load the NIfTI file
    data = nib.load(data_path).get_fdata()
    
    # Get the dimensions of the data
    x_dim, y_dim, z_dim = data.shape
    
    # Find the minimum dimension
    #min_dim = min(x_dim, y_dim, z_dim)
    min_dim = 96
    
    # Calculate the cropping coordinates for each axis to center the crop
    x_start = (x_dim - min_dim) // 2
    y_start = (y_dim - min_dim) // 2
    z_start = (z_dim - min_dim) // 2
    
    x_end = x_start + min_dim
    y_end = y_start + min_dim
    z_end = z_start + min_dim
    
    # Crop the data to the smallest dimension
    cropped_data = data[x_start:x_end, y_start:y_end, z_start:z_end]
    
    return cropped_data

def transform_data(data_path):
    data = nib.load(data_path).get_fdata()
    
    x_dim, y_dim, z_dim = data.shape
   # downsampled_data = zoom(data, (0.5, 0.5, 0.5), order=1)
    
    x_dim, y_dim, z_dim = data.shape
    max_dim = max(x_dim, y_dim, z_dim)
    
    x_pad = get_padding(max_dim, x_dim)
    y_pad = get_padding(max_dim, y_dim)
    z_pad = get_padding(max_dim, z_dim)
    
    padded_data = np.pad(data, (x_pad, y_pad, z_pad), 'constant')
    
    return padded_data


def get_padding(max_dim, current_dim):
    diff = max_dim - current_dim
    pad = diff // 2
    if diff % 2 == 0:
        return pad, pad
    else:
        return pad, pad + 1


def retrieve_paths_static(patient_paths):
    data_dir_paths = defaultdict(list)
    for patient_path in patient_paths:
        if not os.path.isdir(patient_path):
            continue
        patient = patient_path.split(os.sep)[-2]
        for timestep in filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path)):
            timestep_path = os.path.join(patient_path, timestep)
            for axis in filter(lambda x: os.path.isdir(os.path.join(timestep_path, x)), os.listdir(timestep_path)):
                axis_path = os.path.join(timestep_path, axis)
                slice_paths = filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path)))
                data_dir_paths[patient] += slice_paths

    return data_dir_paths

def retrieve_paths_static3D(patient_paths):
    data_dir_paths = defaultdict(list)
    for patient_path in patient_paths:
        if not os.path.isdir(patient_path):
            continue
        patient = patient_path.split(os.sep)[-2]
        for timestep in filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path)):
            timestep_path = os.path.join(patient_path, timestep)
            data_dir_paths[patient].append(timestep_path)

    return data_dir_paths

# def retrieve_paths_longitudinal3D(patient_paths):
#     data_dir_paths = defaultdict(list)
    
#     for patient_path in patient_paths:
#         if not os.path.isdir(patient_path):
#             continue
        
#         patient = patient_path.split(os.sep)[-2]
#         timepoints = [tp for tp in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, tp)) and tp.startswith('T')]
#         timepoints.sort()  # Ensure timepoints are in order
        
#         full_timepoint_paths = [os.path.join(patient_path, tp) for tp in timepoints]

#         if len(full_timepoint_paths) == 4:
#             data_dir_paths[patient] += full_timepoint_paths 
#         elif len(full_timepoint_paths) == 3:
#             data_dir_paths[patient] += full_timepoint_paths + [full_timepoint_paths[-1]]
#         elif len(full_timepoint_paths) == 2:
#             data_dir_paths[patient] += full_timepoint_paths + [full_timepoint_paths[-1]] * 2
#         elif len(full_timepoint_paths) == 1:
#             data_dir_paths[patient] += full_timepoint_paths * 4

#     return data_dir_paths

def retrieve_paths_longitudinal3D(patient_paths):
    data_dir_paths = {}

    for patient_path in patient_paths:
        if not os.path.isdir(patient_path):
            continue

        patient = patient_path.split(os.sep)[-2]
        timepoints = [tp for tp in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, tp)) and tp.startswith('T')]
        timepoints.sort()  # Ensure timepoints are in order

        full_timepoint_paths = [os.path.join(patient_path, tp) for tp in timepoints]

        if len(full_timepoint_paths) >= 2:
            for i in range(len(full_timepoint_paths)):
                for j in range(i + 1, len(full_timepoint_paths)):
                    pair_key = f"{patient}_T{i+1}_T{j+1}"
                    data_dir_paths[pair_key] = (full_timepoint_paths[i], full_timepoint_paths[j])

    return data_dir_paths

def retrieve_paths_longitudinal(patient_paths):
    data_dir_paths = defaultdict(list)
    for patient_path in patient_paths:
        if not os.path.isdir(patient_path):
            continue
        patient = patient_path.split(os.sep)[-2]
        baseline_timepoint = ['0']
        follow_up_timepoint = ['1']
        #for timestep_x in sorted(filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path))):
        for timestep_x in baseline_timepoint:
            x_timestep = defaultdict(list)
            timestep_x_int = int(timestep_x)
            timestep_x_path = os.path.join(patient_path, timestep_x)
            for axis in sorted(filter(lambda x: os.path.isdir(os.path.join(timestep_x_path, x)), os.listdir(timestep_x_path))):
                axis_path = os.path.join(timestep_x_path, axis)
                slice_paths = sorted(filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                x_timestep[axis] = slice_paths

            #for timestep_x_ref in sorted(filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path))):
            for timestep_x_ref in follow_up_timepoint:
                x_ref_timestep = defaultdict(list)
                timestep_x_ref_int = int(timestep_x_ref)
                timestep_x_ref_path = os.path.join(patient_path, timestep_x_ref)
                for axis in sorted(filter(lambda x: os.path.isdir(os.path.join(timestep_x_ref_path, x)), os.listdir(timestep_x_ref_path))):
                    axis_path = os.path.join(timestep_x_ref_path, axis)
                    slice_paths = sorted(filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                    x_ref_timestep[axis] = slice_paths

                    if timestep_x_int != timestep_x_ref_int:
                        data_dir_paths[patient] += zip(x_ref_timestep[axis], x_timestep[axis])

    return data_dir_paths

def get_patient_paths3D(data_dir, datalist):
    #K_fold change
    #patient_paths = map(lambda name: os.path.join(name, 'data'),
                        # (filter(lambda name: (evaluate.value if phase == Phase.TEST else Evaluate.TRAINING.value) in name,
                        #         glob(os.path.join(data_dir, '*')))))
    patient_paths = list(map(lambda patient: os.path.join(patient, 'data3D'), filter(lambda name: os.path.basename(name) in datalist, glob(os.path.join(data_dir, '*')))))
                        
    return patient_paths

def get_patient_paths(data_dir, evaluate, phase):
    #K_fold change
    #patient_paths = map(lambda name: os.path.join(name, 'data'),
                        # (filter(lambda name: (evaluate.value if phase == Phase.TEST else Evaluate.TRAINING.value) in name,
                        #         glob(os.path.join(data_dir, '*')))))
    patient_paths = list(map(lambda patient: os.path.join(patient, 'data'), filter(lambda name: os.path.basename(name) in phase, glob(os.path.join(data_dir, '*')))))
                        
    return patient_paths


def retrieve_filtered_data_dir_paths(root_dir, phase, data_dir_paths, empty_slices, non_positive_slices, mode, val_patients, view: Views = None):

    data_dir_path = os.path.join(root_dir, f'data_dir_{mode.value}_{phase}_{val_patients}{f"_{view.name}" if view else ""}.pckl')
    print(f'Elements in data_dir_paths before filtering empty slices: {len(data_dir_paths)}')
    if mode == Mode.STATIC:
        data_dir_paths = [x for x in data_dir_paths if x not in set(empty_slices + non_positive_slices)]
    else:
        data_dir_paths = [(x_ref, x) for x_ref, x in data_dir_paths if x not in set(empty_slices + non_positive_slices)]

    print(f'Elements in data_dir_paths after filtering empty slices: {len(data_dir_paths)}')
    pickle.dump(data_dir_paths, open(data_dir_path, 'wb'))

    return data_dir_paths

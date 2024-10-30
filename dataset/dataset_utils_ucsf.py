import os
import pickle
from collections import defaultdict, OrderedDict
from enum import Enum
from glob import glob
import shutil
import h5py
import nibabel as nib
import numpy as np


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
    UCSF = 'isbi'


class Evaluate(Enum):
    TRAINING = 'training'
    TEST = 'test'


def retrieve_data_dir_paths(data_dir, datalist, modalities, phase, preprocess, mode, view=None):
    empty_slices = None
    non_positive_slices = None
    if preprocess:
        print('Preprocessing files...')
        empty_slices, non_positive_slices = preprocess_files(data_dir, datalist, modalities, phase)
        print('Files preprocessed.')
    if mode == Mode.LONGITUDINAL:
        data_dir_paths = retrieve_paths_longitudinal(get_patient_paths(data_dir, datalist, phase)).items()
    else:
        data_dir_paths = retrieve_paths_static(get_patient_paths(data_dir, datalist, phase)).items()
    data_dir_paths = OrderedDict(sorted(data_dir_paths))
    _data_dir_paths = []
    patient_keys = [key for key in data_dir_paths.keys()]

    for patient in patient_keys:
            _data_dir_paths += data_dir_paths[patient]
    
    if phase == 'train' or phase == 'val':
        _data_dir_paths = retrieve_filtered_data_dir_paths(data_dir, phase, _data_dir_paths, empty_slices, non_positive_slices,
                                                           mode, view)
    return _data_dir_paths


def preprocess_files(root_dir, datalist, modalities, phase, base_path='data'):
    patients = datalist
    empty_slices = []
    non_positive_slices = []
    i_patients = len(patients) + 1
    for patient in patients:
        print(f'Processing patient {patient}')
        patient_path = os.path.join(root_dir, patient)
        if os.path.exists(os.path.join(patient_path, base_path)):
            # delete old data
            # os.system(f'rm -r {os.path.join(patient_path, base_path)}')
            continue

        for mod in modalities:
            
            timepoints = ['time1', 'time2']
            for timestep in timepoints:
                data_path = f'{patient_path}/{patient}_{timestep}_{mod}.nii.gz'
                if not os.path.exists(data_path):
                    shutil.rmtree(os.path.join(patient_path, base_path), ignore_errors=True)
                    #continue
                transformed_data = pad_to_largest_dimension(data_path, label=False)
                normalized_data = (transformed_data - np.min(transformed_data)) / (np.max(transformed_data) - np.min(transformed_data))
                label_path = f'{patient_path}/{patient}_{timestep}_seg.nii.gz'
                if os.path.exists(label_path):
                    rotated_labels = pad_to_largest_dimension(label_path, label=True)
                else:
                    rotated_labels = np.zeros(normalized_data.shape)

                # create slices through all views
                temp_empty_slices, temp_non_positive_slices = create_slices(normalized_data, rotated_labels,
                                                                            os.path.join(patient_path, base_path, str(timestep)), mod)
                empty_slices += temp_empty_slices
                non_positive_slices += temp_non_positive_slices

        i_patients += 1
    return empty_slices, non_positive_slices

def pad_to_largest_dimension(data_path, label=True):
    # Load the NIfTI file
    data = nib.load(data_path).get_fdata()

    #if label is true, convert to binary
    if label:
        data = np.where(data > 0, 1, 0).astype(np.float64)
    
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


def get_padding(max_dim, current_dim):
    diff = max_dim - current_dim
    pad = diff // 2
    if diff % 2 == 0:
        return pad, pad
    else:
        return pad, pad + 1


def create_slices(data, label, timestep_path, modality):
    empty_slices = []
    non_positive_slices = []
    for view in list(Views):
        name, axis = view.name, view.value
        temp_data = np.moveaxis(data, axis, 0)
        temp_labels = np.moveaxis(label, axis, 0)
        for i, (data_slice, label_slice) in enumerate(zip(temp_data, temp_labels)):
            path = os.path.join(timestep_path, str(axis), f'{i:03}')
            full_path = os.path.join(path, f'{modality}.h5')
            if np.sum(data_slice) <= 1e-5:
                empty_slices.append(path)

            if np.sum(label_slice) <= 1e-5:
                non_positive_slices.append(path)

            while not os.path.exists(full_path):  # sometimes file is not created correctly => Just redo until it exists
                if not os.path.exists(path):
                    os.makedirs(path)
                with h5py.File(full_path, 'w') as data_file:
                    data_file.create_dataset('data', data=data_slice, dtype='f')
                    data_file.create_dataset('label', data=label_slice, dtype='i')

    return empty_slices, non_positive_slices


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


def retrieve_paths_longitudinal(patient_paths):
    data_dir_paths = defaultdict(list)
    for patient_path in patient_paths:
        if not os.path.isdir(patient_path):
            continue
        patient = patient_path.split(os.sep)[-2]
        # Get sorted list of timepoints
        timepoints = sorted(filter(lambda x: os.path.isdir(os.path.join(patient_path, x)), os.listdir(patient_path)))

        for i, timestep_x in enumerate(timepoints):
            x_timestep = defaultdict(list)
            timestep_x_path = os.path.join(patient_path, timestep_x)
            
            for axis in sorted(filter(lambda x: os.path.isdir(os.path.join(timestep_x_path, x)), os.listdir(timestep_x_path))):
                axis_path = os.path.join(timestep_x_path, axis)
                slice_paths = sorted(filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                x_timestep[axis] = slice_paths

            # Iterate over subsequent timepoints
            for j in range(i + 1, len(timepoints)):
                timestep_x_ref = timepoints[j]
                x_ref_timestep = defaultdict(list)
                timestep_x_ref_path = os.path.join(patient_path, timestep_x_ref)
                
                for axis in sorted(filter(lambda x: os.path.isdir(os.path.join(timestep_x_ref_path, x)), os.listdir(timestep_x_ref_path))):
                    axis_path = os.path.join(timestep_x_ref_path, axis)
                    slice_paths = sorted(filter(lambda x: os.path.isdir(x), map(lambda x: os.path.join(axis_path, x), os.listdir(axis_path))))
                    x_ref_timestep[axis] = slice_paths

                    # Combine slice paths for the current timepoint pair
                    data_dir_paths[patient] += zip(x_timestep[axis], x_ref_timestep[axis])

    return data_dir_paths


def get_patient_paths(data_dir, datalist, phase):
    
    patient_paths = list(map(lambda patient: os.path.join(patient, 'data'), filter(lambda name: os.path.basename(name) in datalist, glob(os.path.join(data_dir, '*')))))
     
    return patient_paths

def retrieve_filtered_data_dir_paths(root_dir, phase, data_dir_paths, empty_slices, non_positive_slices, mode, view: Views = None):
    empty_file_path = os.path.join(root_dir, f'empty_slices_{phase}.pckl')
    non_positive_slices_path = os.path.join(root_dir, f'non_positive_slices_{phase}.pckl')

    if empty_slices:
        pickle.dump(empty_slices, open(empty_file_path, 'wb'))
    if non_positive_slices:
        pickle.dump(non_positive_slices, open(non_positive_slices_path, 'wb'))

    data_dir_path = os.path.join(root_dir, f'data_dir_{mode.value}_{phase}_{f"_{view.name}" if view else ""}.pckl')
    if os.path.exists(data_dir_path):
        # means it has been preprocessed before -> directly load data_dir_paths
        data_dir_paths = pickle.load(open(data_dir_path, 'rb'))
        print(f'Elements in data_dir_paths: {len(data_dir_paths)}')
    else:
        if not empty_slices:
            empty_slices = pickle.load(open(empty_file_path, 'rb'))
        if not non_positive_slices:
            non_positive_slices = pickle.load(open(non_positive_slices_path, 'rb'))
        print(f'Elements in data_dir_paths before filtering empty slices: {len(data_dir_paths)}')
        if mode == Mode.STATIC:
            data_dir_paths = [x for x in data_dir_paths if x not in set(empty_slices + non_positive_slices)]
        else:
            data_dir_paths = [(x_ref, x) for x_ref, x in data_dir_paths if x not in set(empty_slices + non_positive_slices)]

        print(f'Elements in data_dir_paths after filtering empty slices: {len(data_dir_paths)}')
        pickle.dump(data_dir_paths, open(data_dir_path, 'wb'))

    return data_dir_paths

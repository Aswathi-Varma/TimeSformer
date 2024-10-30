from math import sqrt
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from dataset.dataset_utils_ucsf import retrieve_data_dir_paths, Modalities, Phase, Views, Mode
import torchio as tio
import torch.nn.functional as F
import h5py

class DatasetSeg2D(Dataset):

    def __init__(self, data_dir, datalist, batch_size, phase=Phase.TRAIN, modalities=(), dim=None, transforms = None, preprocess=True,
                 view: Views = None):

        #Hardcoded values
        self.mode = Mode.LONGITUDINAL
        self.dim = dim
        self.modalities = ['flair','t1', 't2', 't1ce', 't1ce-t1']
        self.return_mask = True

        self.use_z_score = True
        self.transforms = transforms
        self.batch_size = batch_size
        self.data_dir_paths = retrieve_data_dir_paths(data_dir, datalist, self.modalities, phase, preprocess, self.mode, view)

    def __len__(self):
        return len(self.data_dir_paths)
    
    def _normalize_data(self, volume):
        if self.use_z_score:
            return (volume - volume.mean()) / sqrt(volume.var())
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return 2 * volume - 1

    def _min_max_normalize_data(self, volume):
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return volume

    def _apply_transforms(self, img, label):
        """Helper function to apply transformations."""
        if self.transforms is not None:
            img_tio = tio.ScalarImage(tensor=img)
            lbl_tio = tio.LabelMap(tensor=label)
            
            subject = tio.Subject(
                image=img_tio,
                label=lbl_tio
            )
            
            subject = self.transforms(subject)
            
            img = subject.image.tensor
            label = subject.label.tensor
        
        return img, label

    def __getitem__(self, idx):

        x_1, x_2, label_1, label_2 = [], [], None, None
        x_1_path, x_2_path = self.data_dir_paths[idx]
        for i, modality in enumerate(self.modalities):
            with h5py.File(os.path.join(x_1_path, f'{modality}.h5'), 'r') as f:
                x_1.append(f['data'][()])
                if label_1 is None:
                    label_1 =torch.as_tensor(f['label'][()])

            with h5py.File(os.path.join(x_2_path, f'{modality}.h5'), 'r') as f:
                x_2.append(f['data'][()])
                if label_2 is None:
                    label_2 = torch.as_tensor(f['label'][()]).unsqueeze(0)

        # Concatenate x_1 and x_2 along a new dimension
        x_1 = torch.as_tensor(np.stack(x_1, axis=0)).unsqueeze(0)
        x_2 = torch.as_tensor(np.stack(x_2, axis=0)).unsqueeze(0)
        x = torch.cat([x_1, x_2], dim=0)

        if self.return_mask:
            label_1 = label_1.unsqueeze(0)
            return x.float(), label_1.float(), label_2.float()
        else:
            zeros = torch.zeros_like(label_1)
            label_1_stacked = torch.stack([zeros, label_1], dim=0).unsqueeze(1)
            x = torch.cat([x, label_1_stacked], dim=1)
            return x.float(), label_2.float()




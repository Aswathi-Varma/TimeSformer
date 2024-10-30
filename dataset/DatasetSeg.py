from math import sqrt
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from dataset.dataset_utils import Phase, Modalities, Views, Mode, retrieve_data_dir_paths_3D, Evaluate
from dataset.dataset_utils_2D import retrieve_data_dir_paths
import torchio as tio
class DatasetSeg(Dataset):

    def __init__(self, data_dir, datalist, batch_size, phase=Phase.TRAIN, modalities=(), dim=None, transforms = None, preprocess=True,
                 view: Views = None):
        self.modalities = list(map(lambda x: Modalities(x), modalities))

        #Hardcoded values
        self.mode = Mode.LONGITUDINAL
        self.dim = dim

        self.use_z_score = True
        self.transforms = transforms
        self.batch_size = batch_size
        self.data_dir_paths = retrieve_data_dir_paths_3D(data_dir, datalist, phase, preprocess, self.mode, view)

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

        if self.mode == Mode.STATIC:
            img_path = os.path.join(self.data_dir_paths[idx], 'img.npy')
            lbl_path = os.path.join(self.data_dir_paths[idx], 'label.npy')

            img = torch.as_tensor(np.load(img_path))
            lbl = torch.as_tensor(np.load(lbl_path), dtype=torch.int64)
            label = torch.nn.functional.one_hot(lbl, num_classes=2).permute(3, 0, 1, 2)

            img = self._normalize_data(img.clone())

            img, label = self._apply_transforms(img, label)

        elif self.mode == Mode.LONGITUDINAL:
            timepoint_paths = self.data_dir_paths[idx]
            images = []
            labels = []

            for tp_path in timepoint_paths:
                img_path = os.path.join(tp_path, 'img.npy')
                lbl_path = os.path.join(tp_path, 'label.npy')

                img = torch.as_tensor(np.load(img_path))
                lbl = torch.as_tensor(np.load(lbl_path), dtype=torch.int64).unsqueeze(0)
                #lbl = torch.nn.functional.one_hot(lbl, num_classes=2).permute(3, 0, 1, 2)

                # img = self._normalize_data(img.clone())

                # img, lbl = self._apply_transforms(img, lbl)

                images.append(img)
                labels.append(lbl)

            # Stack images and labels along a new dimension
            img = torch.stack(images, dim=0)
            label_in = labels[0]
            label_out = labels[-1]
        
            # Stack zeros of shape label_in to itself along dim=0
            zeros = torch.zeros_like(label_in)
            label_in_stacked = torch.stack([zeros, label_in], dim=0)

            # Concatenate label_in_stacked to img along dim=1
            img = torch.cat([img, label_in_stacked], dim=1)
        else:
            raise ValueError("Invalid mode")

        return img.float(), label_out.float()


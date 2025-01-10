import os
import numpy as np
import torch
from torch.utils.data import Dataset
from opendataset_offline import HeartDatasetProcessor


class HeartDataset(Dataset):
    """OpenDatasets
    https://www.ub.edu/mnms/
    """
    def __init__(self, root_path,
                 matrix_size,
                 sampling_fraction,
                 center_fraction,
                 split='train'):
        self.root_path = root_path
        self.matrix_size = matrix_size
        self.Nx = matrix_size[0]
        self.Ny = matrix_size[1]
        self.sampling_fraction = sampling_fraction
        self.center_fraction = center_fraction
        self.split = split.lower()
        self.data_files = []
        self.mask_files = []
        if self.split == 'train':
            data_list = os.path.join(root_path,'train')
        elif self.split == 'test':
            data_list = os.path.join(root_path,'test')
        elif self.split == 'val':
            data_list = os.path.join(root_path,'val')
        else:
            raise RuntimeError("Incorrect split input.")

        # Call HeartDatasetProcessor to process the dataset
        self.processor = HeartDatasetProcessor(self.root_path,
                                               self.Nx,
                                               self.Ny,
                                               self.sampling_fraction,
                                               self.center_fraction,
                                               self.split)

        # Ensure the {split} data and {split} ground truth files are matched one-to-one
        for file_name in sorted(os.listdir(data_list)):
            file_path = os.path.join(data_list, file_name)
            if file_name.endswith('_gt.npy'):
                self.mask_files.append(file_path)
            elif file_name.endswith('.npy'):
                # Exclude cases with _gt.npy
                if not file_name.endswith('_gt.npy'):
                    self.data_files.append(file_path)
        data_base_names = [os.path.basename(f).replace('.npy', '')
                           for f in self.data_files]
        mask_base_names = [os.path.basename(f).replace('_gt.npy', '')
                           for f in self.mask_files]
        assert data_base_names == mask_base_names, "Mismatch between the data files and ground truth files"

    def __getitem__(self, index):
        # Load the {split} data and ground truth
        data = np.load(self.data_files[index])
        data_mask = np.load(self.mask_files[index])
        # label = int(np.any(data_mask == 1))

        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=0)
        # data = np.repeat(data, 3, axis=0)
        # Convert numpy array to PyTorch Tensor
        data = torch.from_numpy(data).to(torch.float)
        data_mask = torch.from_numpy(data_mask).long()
        return data, data_mask

    def __len__(self):

        return len(self.data_files)
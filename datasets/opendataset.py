import os
import numpy as np
import torch
from torch.utils.data import Dataset
from opendataset_offline import HeartDatasetProcessor
import torchio as tio
from torchio import SubjectsDataset
class HeartDataset(SubjectsDataset):
    """OpenDatasets
    https://www.ub.edu/mnms/
    """
    def __init__(self, root_path,
                 matrix_size,
                 sampling_fraction,
                 center_fraction,
                 split='train',
                 transform=None):

        self.root_path = root_path
        self.matrix_size = matrix_size
        self.Nx = matrix_size[0]
        self.Ny = matrix_size[1]
        self.sampling_fraction = sampling_fraction
        self.center_fraction = center_fraction
        self.split = split.lower()
        self.transform = transform
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

        # Create a list of Subject objects
        self.subjects = []
        for i in range(len(self.data_files)):
            data = np.load(self.data_files[i])
            data_mask = np.load(self.mask_files[i])

            # If it's a 2D image, add channel and depth dimensions
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=0)  # Add channel dimension -> (1, height, width)
                data = np.expand_dims(data, axis=-1)  # Add depth dimension -> (1, height, width, depth)

            if len(data_mask.shape) == 2:
                data_mask = np.expand_dims(data_mask, axis=0)  # Add channel dimension
                data_mask = np.expand_dims(data_mask, axis=-1)  # Add depth dimension

            # Create Subject for each data-mask pair
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=torch.from_numpy(data).float()),
                label=tio.LabelMap(tensor=torch.from_numpy(data_mask).long())
            )

            self.subjects.append(subject)

        # Call the parent class constructor
        super().__init__(subjects=self.subjects, transform=self.transform)

    def __getitem__(self, index):
        if self.transform is not None:
            subject = self.transform(self.subjects[index])
        # In this implementation, __getitem__ is not strictly necessary since the subjects are already preloaded
        return subject

        # data = np.repeat(data, 3, axis=0)
        # Convert numpy array to PyTorch Tensor
        # data = torch.from_numpy(data).to(torch.float)
        # data_mask = torch.from_numpy(data_mask).long()
        # return data, data_mask

    def __len__(self):

        return len(self.subjects)
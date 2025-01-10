import os
import yaml
import glob
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk
"""
    opendatasets_offline processor,including three class which are responsible for this task.
    
    - HeartDatasetProcessor: According to {split}, slice the raw data for the corresponding test,
    train, and val datasets and perform dimensionality reduction by selecting the ed and es time points.
    Convert the 4D MRI nii.gz image file (height, width, slice, time point)
    into 2D normalized k-space NumPy data (height, width).
    For the mask images, only select the ed and es time points and slices, 
    converting them into corresponding NumPy data. Discard slices without masks.
    - ProcessingUtils: Provide static methods for HeartDatasetProcessor to process the data.
    The main functions are gaussian_under_sampling_mask and heart_data_offline.
    - YmlCheck: Provide a validation method for HeartDatasetProcessor to determine whether 
    to regenerate the data based on new parameters or skip the data processing
    if the data has already been processed.
"""
class HeartDatasetProcessor:
    """
    According to {split}, slice the raw data for the corresponding test, train, and val datasets
    and perform dimensionality reduction by selecting the ed and es time points.
    Convert the 4D MRI nii.gz image file (height, width, slice, time point)
    into 2D normalized k-space NumPy data (height, width).

    The training, testing, and validation sets are named 'train','test', and 'val', respectively.
    The file naming format is “External code_slice{slice number}_(ed/es)_{time frame}(_gt/.).npy”.
    For example: A1K2P5_slice0_ed33_gt.npy, where External code = A1K2P5, slice number = 0, ed refers to time frame 33.

    Instantiating HeartDatasetProcessor, it first checks whether the {split}_params.yml
    file corresponding to the split is consistent with the instantiation parameters.
    If they are inconsistent, it clears the previous contents of the split folder.
    If the file does not exist, this step is skipped.
    If {split}_params.yml is consistent with the instantiation parameters, the data processing step is skipped.
    """
    def __init__(self, root_path, Nx, Ny, sampling_fraction, center_fraction, split):
        self.root_path = root_path
        self.Nx = Nx
        self.Ny = Ny
        self.matrix_size = (Nx, Ny)
        self.sampling_fraction = sampling_fraction
        self.split = split
        self.center_fraction = center_fraction

        # Parameters for comparison
        self.checker = YmlCheck(self.root_path, self.Nx, self.Ny, self.sampling_fraction, self.center_fraction, self.split)
        if  self.checker.result:
            print(f"The {split} parameters have changed. This will remove the existing {split} data and regenerate it.")
            self.checker.clear_split_data()
            self.process_split_datasets()
            # Save the new parameters to the yml file
            self.checker.save_params_to_yml(self.checker.params, self.checker.yml_file_path)
        else:
            print(f"The {split} parameters have not changed, using existing data.")
            self.checker.load_existing_data()

    # Get the list of folders
    def list_folders(self, path):
        folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
        folders.sort()
        return folders

    # Process the dataset and save the data
    def dataset(self, new_path, file_list, mask):
        # Read the csv file
        csv_path = os.path.join(self.root_path, '211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
        df = pd.read_csv(csv_path)
        if 'gt' in file_list[0]:
            gt_file_path = file_list[0]
            file_path = file_list[1]
        else:
            file_path = file_list[0]
            gt_file_path = file_list[1]

        External_code = os.path.basename(file_path).split('_')[0]
        matching_row = df[df['External code'] == External_code]

        if not matching_row.empty:
            # Retrieve the relevant information for that row, such as ed, es, etc.
            vendor = matching_row['VendorName'].values[0]
            centre = matching_row['Centre'].values[0]
            ed = matching_row['ED'].values[0]
            es = matching_row['ES'].values[0]

            # Load image data
            data = nib.load(file_path)
            data = data.get_fdata()
            gt_data = nib.load(gt_file_path)
            gt_data = gt_data.get_fdata()

            # Process the image data for each slice
            for i in range(data.shape[2]):
                slice_data_ed = data[:, :, i, ed]
                slice_data_es = data[:, :, i, es]
                slice_gt_data_ed = gt_data[:, :, i, ed]
                slice_gt_data_es = gt_data[:, :, i, es]

                ed_has_label = np.any(slice_gt_data_ed)  # Check if slice_gt_data_ed has labels
                es_has_label = np.any(slice_gt_data_es)  # Check if slice_gt_data_es has labels

                # Skip the entire slice if there are no labels
                if not ed_has_label and not es_has_label:
                    continue  # Skip the entire slice

                # Process the ED slice and ground truth
                if ed_has_label:
                    new_gt_file_name_ed = f"{External_code}_slice{i}_ed{ed}_gt"
                    new_file_name_ed = f"{External_code}_slice{i}_ed{ed}"

                    # Process the ground truth data
                    slice_gt_data_ed = ProcessingUtils.resample_image(slice_gt_data_ed, self.matrix_size)

                    # Process the raw data
                    slice_data_ed = ProcessingUtils.resample_image(slice_data_ed, self.matrix_size)

                    # Save the ED slice
                    np.save(os.path.join(new_path, new_gt_file_name_ed), slice_gt_data_ed)
                    np.save(os.path.join(new_path, new_file_name_ed), slice_data_ed)

                # Process the ES slice and ground truth
                if es_has_label:
                    new_gt_file_name_es = f"{External_code}_slice{i}_es{es}_gt"
                    new_file_name_es = f"{External_code}_slice{i}_es{es}"

                    # Process the ground truth data
                    slice_gt_data_es = ProcessingUtils.resample_image(slice_gt_data_es, self.matrix_size)

                    # Process the raw data
                    slice_data_es = ProcessingUtils.resample_image(slice_data_es, self.matrix_size)

                    # Save the ES slice
                    np.save(os.path.join(new_path, new_gt_file_name_es), slice_gt_data_es)
                    np.save(os.path.join(new_path, new_file_name_es), slice_data_es)

    # Main function to process all datasets
    def process_split_datasets(self):
        self.mask = ProcessingUtils.gaussian_under_sampling_mask(Nx=self.Nx,
                                                sampling_fraction=self.sampling_fraction,
                                                center_fraction=self.center_fraction)

        # The variable names to the corresponding file lists
        if self.split == 'test':
            folder_path = os.path.join(self.root_path, "Testing")
            folder_list = self.list_folders(folder_path)

        elif self.split == 'train':
            folder_path = os.path.join(self.root_path, "Training/Labeled")
            folder_list = self.list_folders(folder_path)
        elif self.split == 'val':
            folder_path = os.path.join(self.root_path, "Validation")
            folder_list = self.list_folders(folder_path)


        total_files = len(folder_list)
        # Create a total progress bar for all datasets
        with tqdm(total=total_files, desc=f"Processing {self.split} datasets") as pbar:
            # Process each file list sequentially
            for folder in folder_list:
                folder_full_path = os.path.join(folder_path, folder)
                new_path = os.path.join(self.root_path, f"{self.split}")
                os.makedirs(new_path, exist_ok=True)
                # Get all .nii.gz files in the current folder
                files_in_folder = glob.glob(os.path.join(folder_full_path, '*.nii.gz'))
                self.dataset(new_path, files_in_folder, self.mask)
                pbar.update(1)  # Process each file in the folder

class ProcessingUtils:
    """
    Provide static methods for HeartDatasetProcessor to process the data.
    The main functions are gaussian_under_sampling_mask and heart_data_offline.
    - gaussian_under_sampling_mask:
    - heart_data_offline:
    """
    def __init__(self):
        pass

    @staticmethod
    def gaussian_under_sampling_mask(Nx, sampling_fraction, center_fraction):
        """
    Generate a 1D Gaussian distribution-based under sampling mask with total sampling constraint.
    The total sampling rate is sampling_fraction, center_fraction forced to retain in the low-frequency center.
    For example, if sampling_fraction = 0.3 and center_fraction = 0.1, this means 10% is retained in the low-frequency center,
    while the remaining 20% is sampled using a Gaussian distribution, resulting in a total sampling rate of 30%.

    Parameters:
    - length: Integer representing the length of the mask (number of rows to undersample).
    - sampling_fraction: The desired total sampling ratio, including the central region.
    - center_fraction: The proportion of the central region retained to preserve low-frequency components.

    Returns:
    - mask: 1D NumPy array, where 1 indicates keeping the point and 0 indicates discarding it.
        """
        length = Nx
        # Calculate the number of points in the central region
        center_length = int(center_fraction * length)

        # Check if the center_fraction is too large for the given sampling_fraction
        if center_length > int(sampling_fraction * length):
            raise ValueError("Center region is larger than the total sampling fraction."
                             " Adjust center_fraction or sampling_fraction.")

        # Create the initial mask with zeros
        mask = np.zeros(length, dtype=int)

        # Force the center region to be retained
        center_start = (length - center_length) // 2
        mask[center_start:center_start + center_length] = 1

        # Calculate the number of remaining points we need to sample
        remaining_length = length - center_length
        remaining_points_to_sample = int(sampling_fraction * length) - center_length

        # Create a Gaussian distribution along the remaining points
        x = np.linspace(-1, 1, remaining_length)
        gaussian = np.exp(-x ** 2 / 0.5)  # Gaussian distribution with sigma=0.5
        gaussian /= gaussian.max()  # Normalize the Gaussian distribution

        # Adjust sampling fraction for the remaining points
        adjusted_sampling_fraction = remaining_points_to_sample / remaining_length

        # Create a random sampling mask for the remaining points based on adjusted fraction
        remaining_mask = np.random.rand(remaining_length) < (gaussian * adjusted_sampling_fraction)

        # Fill the mask with the remaining sampled points
        mask[:center_start] = remaining_mask[:center_start]
        mask[center_start + center_length:] = remaining_mask[center_start:]

        return mask

    @staticmethod
    def pad_or_crop_image(image_np, matrix_size):
        """
        Pad or crop the input numpy array to match the target 2D size.

        Parameters:
        - image_np: Input numpy array.
        - matrix_size: Target size, format (height, width).

        Returns:
        - padded_or_cropped_image_np: Padded or cropped numpy array.
        """
        height, width = image_np.shape
        target_height, target_width = matrix_size

        pad_y = max((target_height - height) // 2, 0)
        pad_x = max((target_width - width) // 2, 0)

        crop_y = max((height - target_height) // 2, 0)
        crop_x = max((width - target_width) // 2, 0)

        padded_image = np.pad(image_np, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
        cropped_image = padded_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

        return cropped_image
    @staticmethod
    def resample_image(image, matrix_size, target_spacing=(1.25, 1.25)):
        """
        Resample a 2D image to the specified resolution and target size.

        :param image: Input image (numpy array or SimpleITK Image).
        :param target_spacing: Target pixel spacing (resolution), format: (spacing_x, spacing_y), unit: mm.
        :param matrix_size: Target image size (number of pixels), format: (width, height). If None, it is calculated based on the target resolution.
        :return: Resampled image array (Image Array).
        """
        # If the input image is a numpy array, convert it to a SimpleITK image
        if isinstance(image, np.ndarray):
            image = sitk.GetImageFromArray(image)

        # Retrieve the original image information
        original_spacing = image.GetSpacing()  # Original resolution (pixel size)
        original_size = image.GetSize()  # Original size (number of pixels)

        # If target size is not specified, calculate it based on target resolution and original size
        if matrix_size is None:
            matrix_size = [
                int(np.round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / target_spacing[1])))
            ]

        # Configure the resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)  # Set the target resolution (pixel spacing)
        resampler.SetSize(matrix_size)  # Set the target size
        resampler.SetOutputOrigin(image.GetOrigin())  # Keep the origin unchanged
        resampler.SetOutputDirection(image.GetDirection())  # Keep the direction unchanged
        resampler.SetInterpolator(
            sitk.sitkLinear)  # Set the interpolation method to linear (can be changed to other methods)

        # Perform resampling
        resampled_image = resampler.Execute(image)
        resampled_array = sitk.GetArrayFromImage(resampled_image)
        return resampled_array

    @staticmethod
    def apply_under_sampling(image, mask):
        """
        Apply a 1D mask along the rows of a 2D image for under_sampling.

        Parameters:
        - image: numpy array representing the input image.
        - mask: numpy array representing the mask to be applied along the rows.

        Returns:
        - masked_image: The 2D image after applying the 1D mask along the rows.
        """
        assert image.shape[0] == len(mask), "The length of the mask must match the number of rows in the image."

        masked_image = image * mask[:, np.newaxis]
        return masked_image

    @staticmethod
    def normalize_k_space(k_space: np.ndarray) -> np.ndarray:
        """
        Normalize k-space data by the global maximum value.

        Parameters:
        - k_space: np.array, under sampled k-space data.

        Returns:
        - normalized_k_space: np.array, normalized k-space data.
        """
        max_value = np.max(np.abs(k_space))
        if max_value > 0:
            normalized_k_space = k_space / max_value
        else:
            normalized_k_space = k_space
        return normalized_k_space

    @staticmethod
    def heart_data_offline(image_np: np.ndarray, matrix_size, mask):
        """
        Main function that processes the input 2D MRI numpy image and returns the normalized k-space data.

        Parameters:
        - image_np: 2D NumPy array representing the input 2D MRI image.
        - matrix_size: Target size, format (height, width).
        - mask: The Gaussian under sampling mask to apply.

        Returns:
        - Under sampled k-space data (2D NumPy array).
        """

        # Pad or crop the input numpy array to match the target 2D size.
        image_padded = ProcessingUtils.pad_or_crop_image(image_np, matrix_size)
        # Convert 2D MRI image data to k-space.
        k_space_data = np.fft.fft2(image_padded)
        # Apply fft shift to a 2D numpy array (image).
        fft_shift_data = np.fft.fftshift(k_space_data)
        # Normalize k-space data by the global maximum value.
        under_sampled_k_space_data = ProcessingUtils.apply_under_sampling(fft_shift_data, mask)

        return under_sampled_k_space_data # normalized_k_space_data

class YmlCheck:
    """
    Provide a validation method for HeartDatasetProcessor to determine whether
    to regenerate the data based on new parameters or skip the data processing
    if the data has already been processed.
    """
    def __init__(self, root_path, Nx, Ny, sampling_fraction, center_fraction, split):
        self.root_path = root_path
        self.Nx = Nx
        self.Ny = Ny
        self.sampling_fraction = sampling_fraction
        self.split = split
        self.center_fraction = center_fraction

        # Parameters for comparison
        self.params = {
            'root_path': self.root_path,
            'sampling_fraction': self.sampling_fraction,
            'center_fraction': self.center_fraction,
            'Nx': self.Nx,
            'Ny': self.Ny,
        }

        # Define the path to the yml file
        self.yml_file_path = os.path.join(self.root_path, f"{split}_params.yml")

        # Load old parameters from the yml file
        old_params = self.load_params_from_yml(self.yml_file_path)

        # Check if the parameters have changed
        self.result = self.check_params_changed(self.params, old_params)

    def clear_split_data(self):
        """

        :return:
        """
        split_path = os.path.join(self.root_path, self.split)
        # Remove the data folder if it exists
        if os.path.exists(split_path) and os.path.isdir(split_path):
            print(f"Removing existing {self.split} data...")
            shutil.rmtree(split_path)  # Remove the directory and all its contents
        yml_file_path = self.yml_file_path
        # Remove the yml file if it exists
        if os.path.exists(yml_file_path) and os.path.isfile(yml_file_path):
            os.remove(yml_file_path)
        # Recreate an empty directory for new data
        os.makedirs(split_path, exist_ok=True)

    # Save parameters to a yml file
    def save_params_to_yml(self, params, file_path):
        with open(file_path, 'w') as file:
            yaml.dump(params, file)

    # Load parameters from a yml file
    def load_params_from_yml(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        return None

    # Check if the parameters have changed
    def check_params_changed(self, new_params, old_params):
        if old_params is None:
            return True  # If the file does not exist, it is considered necessary to regenerate the data
        return new_params != old_params  # If the new and old parameters are different, an update is required

    # Placeholder for loading existing data
    def load_existing_data(self):
        print("Loading existing processed data...")

    # Placeholder for processing all datasets
    def process_all_datasets(self):
        print(f"Processing {self.split} datasets...")


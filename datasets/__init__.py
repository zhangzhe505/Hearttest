import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from torch.utils.data import DataLoader
from opendataset import HeartDataset


def create_data(data_config, root_path, matrix_size, batch_size, sampling_fraction, center_fraction):
    """
    C                                                                                                                                                                              reate a dataset for training, testing and validation.
    :param data_config: [str, str], 1st for train/valid config, 2nd for test.
        Currently, only the OpenDatasets dataset is available.
    :param root_path: The absolute path to the dataset.
    :param matrix_size: (int,int), the required length and width for the model input.
    :param batch_size: (int or list), set internally if None.
    :param sampling_fraction:float，the undersampling ratio, for example, 0.3 means a 30% sampling ratio
    :param center_fraction:float,the sampling ratio retained in the center,
        for example, 0.1 means a 10%  retained sampling ratio
    :return:
        train_loader: a DataLoader object for train data
        valid_loader: a DataLoader object for valid data
        test_loader: a DataLoader object for test data

    """
    train_config, test_config = data_config
    Nx, Ny = matrix_size
    bs_dict = {"OpenDatasets": 1,
               }
    if isinstance(batch_size, list):
        if len(batch_size) == 2:
            bs_train, bs_test = batch_size
        elif len(batch_size) == 1:
            bs_train, bs_test = batch_size[0], batch_size[0]
        else:
            raise RuntimeError("batch_size should have one or two values")
    elif isinstance(batch_size, int):
        bs_train, bs_test = batch_size, batch_size
    else:
        bs_train = bs_dict[train_config]
        bs_test = bs_dict[test_config]

    if train_config == "OpenDatasets":
        train_data = HeartDataset(
            root_path=root_path,
            split="train",
            matrix_size=(Nx, Ny),
            sampling_fraction=sampling_fraction,
            center_fraction=center_fraction,
        )

        test_data = HeartDataset(
            root_path=root_path,
            split="val",
            matrix_size=(Nx, Ny),
            sampling_fraction=sampling_fraction,
            center_fraction=center_fraction,
        )
    if test_config == "OpenDatasets":
        val_data = HeartDataset(
            root_path=root_path,
            split="test",
            matrix_size=(Nx, Ny),
            sampling_fraction=sampling_fraction,
            center_fraction=center_fraction,
        )

    train_loader = DataLoader(train_data, batch_size=bs_train, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=bs_train, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=bs_test, shuffle=True)

    return train_loader, valid_loader, test_loader




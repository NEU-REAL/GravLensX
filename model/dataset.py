import os
import torch
import numpy as np

class GeodesicDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        paths = os.listdir(data_dir)
        paths.sort()
        self.data = [os.path.join(data_dir, path) for path in paths]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = np.load(self.data[item])
        return torch.from_numpy(data)

class GeodesicTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, labels_tensor):
        """
        Initialize the dataset with data tensors and label tensors.
        Args:
        data_tensor (Tensor): The input data tensor.
        labels_tensor (Tensor): The corresponding label tensor.
        """
        assert data_tensor.shape[0] == labels_tensor.shape[0], "Mismatched data and labels"
        self.data_tensor = data_tensor
        self.labels_tensor = labels_tensor

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.data_tensor.shape[0]

    def __getitem__(self, index):
        """
        Generate one sample of data.
        Args:
        index (int): The index of the sample.
        """
        # Fetch the data and label at the specified index
        return self.data_tensor[index], self.labels_tensor[index]
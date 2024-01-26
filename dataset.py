import numpy as np
import pandas as pd
import torch

class MovieLens1MDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        data = pd.read_csv(data_dir).to_numpy()
        self.field = data[:,:-1]
        self.label = data[:,-1].astype(np.float32)
        self.field_dims = np.array([3706,301,81,6040,21,7,2,3402])
    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        field = self.field[item]
        label = self.label[item]
        return field, label

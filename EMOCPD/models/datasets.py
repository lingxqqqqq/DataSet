import numpy as np
import torch.utils.data as Data


class MyDataset(Data.Dataset):
    def __init__(self, file):
        self.data = file.root.data.data
        self.label = file.root.data.label

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.label)


class NpzDataset(Data.Dataset):
    def __init__(self, data_file, label_file):
        self.data = np.load(data_file)['arr']
        self.label = np.load(label_file)['arr']

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.label)


class ValDataset(Data.Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file)['a']
        self.label = np.load(data_file)['b']

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y

    def __len__(self):
        return len(self.label)



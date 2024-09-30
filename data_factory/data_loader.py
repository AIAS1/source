import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(data_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(data_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(data_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test' or self.mode == 'thre':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return 0  # Or handle other modes if necessary

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
            labels = np.zeros(self.win_size)  # No labels for training data
        elif self.mode == 'val':
            data = self.val[index:index + self.win_size]
            labels = np.zeros(self.win_size)  # Assuming no labels for validation
        elif self.mode == 'test' or self.mode == 'thre':
            data = self.test[index:index + self.win_size]
            labels = self.test_labels[index:index + self.win_size]
        else:
            data = None
            labels = None

        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).float()

        return data, labels

def get_loader_segment(data_path, batch_size, win_size=100, step=1, mode='train', dataset='MSL'):
    if dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    else:
        raise ValueError("Dataset not supported: {}".format(dataset))

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
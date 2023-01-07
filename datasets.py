import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


class datasetMusic(Dataset):
    def __init__(self, path=None, sliced=None) -> None:
        super().__init__()
        self.path = path
        self.sliced = sliced
        with open(self.path) as f:
            self.data = np.genfromtxt(self.path, delimiter=',')
            self.data = self.data[1:, :].astype(np.float64)
        self.len = len(self.data)
        
    def __getitem__(self, index: int):
        inData  = self.data[index][:-1]
        outData = self.data[index][-1:]

        if self.sliced != None and len(self.sliced) != 0:
            inData = inData[self.sliced]    

        return inData, outData

    def __len__(self) -> int:
        return self.len
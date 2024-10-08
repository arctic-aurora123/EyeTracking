import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset

class EyeData():
    def __init__(self):
        self.pos_x = []
        self.pos_y = []
        self.start_cnt, self.end_cnt = 0, 0
        self.blink_time = []

class EyeDataset(Dataset):
    def __init__(self, data_path, transform=None, overlap = True, window_size = 25, step_size = 10, ignore_first_sec = 5):
        self.data = pd.read_csv(data_path).values
        self.transform = transform
        if overlap:
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.data = self.data.unfold(0, window_size, step_size)
            ignore_first_n = int(7.5 * ignore_first_sec * ignore_first_sec + 7.5 * ignore_first_sec + 10)
            self.data = self.data[ignore_first_n:, :, :]
            self.data = torch.permute(self.data, (0, 2, 1))
            
        else:
            self.data = np.split(self.data, int(len(self.data)/25))
            ignore_first_n = ignore_first_sec * 10
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.data = self.data[ignore_first_n:, :, :]
            
        print(f"dataset shape: {self.data.shape}")
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        std_signal = self.data[idx, :, 0:4]
        # diff_signal = self.data[idx, :, 4:6]
        screen_pos = self.data[idx, :, 6:8]
        if self.transform:
            signal = self.transform(signal)
        
        return std_signal, screen_pos
    
class EyeDataset_continuous(Dataset):
    def __init__(self, data_path, transform=None, overlap = True, window_size = 25, step_size = 10, ignore_first_sec = 5):
        self.data = pd.read_csv(data_path).values
        self.transform = transform
        if overlap:
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.data = self.data.unfold(0, window_size, step_size)
            ignore_first_n = int(7.5 * ignore_first_sec * ignore_first_sec + 7.5 * ignore_first_sec + 10)
            self.data = self.data[ignore_first_n:, :, :]
            self.data = torch.permute(self.data, (0, 2, 1))
            
        else:
            self.data = np.split(self.data, int(len(self.data)/25))
            ignore_first_n = ignore_first_sec * 10
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.data = self.data[ignore_first_n:, :, :]
            
        print(f"dataset shape: {self.data.shape}")
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        std_signal = self.data[idx, :, 0:4]
        screen_pos = self.data[idx, :, 4:6]
        blink_label = self.data[idx, :, 6]
        if self.transform:
            signal = self.transform(signal)
        return std_signal, screen_pos, blink_label
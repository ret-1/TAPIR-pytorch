import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, root):
        pass

    def __len__(self):
        return 16

    def __getitem__(self, idx):
        rgbs = torch.ones((S, H, W, 3))
        trajs = torch.ones((S, N, 2))
        visibs = torch.ones((S, N))
        return rgbs, trajs, visibs
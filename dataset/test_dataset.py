from torch.utils.data import Dataset
import os
import pickle
from tqdm import tqdm
import numpy as np
from PIL import Image
import io
import torch
import torchvision.transforms.functional as F

class TapVidTestDataset(Dataset):
    def __init__(self, root, resize=[256,256]):
        self.gt = []
        self.resize = resize
        if 'tapvid_davis' in root:
            with open(os.path.join(root, 'tapvid_davis.pkl'), 'rb') as f:
                self.vid_gt = pickle.load(f)  # type: dict
            
            for vid, vid_gt in self.vid_gt.items():
                self.gt.append(self.get_gt(vid, vid_gt))
        elif 'tapvid_rgb_stacking' in root:
            with open(os.path.join(root, 'tapvid_rgb_stacking.pkl'), 'rb') as f:
                self.vid_gt = pickle.load(f)  # type: list
            
            for i, vid_gt in enumerate(self.vid_gt):
                self.gt.append(self.get_gt(str(i), vid_gt))
        else:
            raise NotImplementedError
        
    def get_gt(self, vid, vid_gt):
        # adapted from pips2: tapviddataset_fullseq
        rgbs = vid_gt['video'] # list of H,W,C uint8 images
        if isinstance(rgbs[0], bytes):
            rgbs = [np.array(Image.open(io.BytesIO(rgb))) for rgb in rgbs]
        rgbs = torch.from_numpy(np.stack(rgbs)).float()
        rgbs = (2 * (rgbs / 255.0) - 1.0)
        rgbs = F.resize(rgbs.permute(0, 3, 1, 2), self.resize, antialias=True).permute(0, 2, 3, 1)
        
        trajs = vid_gt['points'] # N,S,2 array normalized coordinates
        valids = 1-vid_gt['occluded'] # N,S array
        # note the annotations are only valid when not occluded

        vis_ok = valids[:, 0] > 0

        sample = {
            'vid': vid,
            'rgbs': rgbs, # S,H,W,C
            'trajs': torch.from_numpy(trajs[vis_ok]), # N,S,2
            'visibs': torch.from_numpy(valids[vis_ok]), # N,S
        }
        return sample
    
    def __len__(self):
        return len(self.gt)
    
    def __getitem__(self, idx):
        return self.gt[idx]
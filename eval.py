from model.tapir_model import Tapir
import torch
from dataset.test_dataset import TapVidTestDataset
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from utils.evaluate import compute_metrics
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser("Evaluation script for Tapir")
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--checkpoint", type=str, default="tapir.pth")
parser.add_argument("--dataset", type=str, default="tapvid_rgb_stacking")
parser.add_argument("--resize", type=int, nargs=2, default=[256, 256])
args = parser.parse_args()

model = Tapir(pyramid_level=0, extra_convs=False)
model.load_state_dict(torch.load(args.checkpoint))
model.eval()

os.makedirs(args.output_dir, exist_ok=True)

dataset = TapVidTestDataset(args.dataset, resize=args.resize)
df = []

for data in tqdm(dataset):
    rgbs = data["rgbs"].unsqueeze(0).cuda() # B, S, H, W, C
    B, S, H, W, C = rgbs.shape

    trajs_g = data["trajs"].unsqueeze(0).cuda() # B, N, S, 2, normalized coordinates
    visibs_g = data["visibs"].unsqueeze(0).cuda() # B, N, S

    # taking all points from frame 0
    points_0_xy = trajs_g[:, :, 0, :]  # B, N, 2 format:(x,y)
    points_0 = points_0_xy[:, :, [1, 0]]  # format:(y,x)
    points_0[..., 0] *= H - 1
    points_0[..., 1] *= W - 1
    # preparing the time dimension to be concatenated
    time_dim = torch.zeros((points_0.shape[0], points_0.shape[1], 1)).cuda()
    # prepending a column to be -> (B, N, 3)
    points_0 = torch.concatenate(
        (time_dim, points_0), axis=-1
    )  # format:(t,y,x)

    output, _ = model(rgbs, points_0)
    trajs_e = output['tracks']
    visibs_e = (1 - F.sigmoid(output['occlusion'])) * (1 - F.sigmoid(output['expected_dist'])) > 0.5

    trajs_e[...,0] /= W - 1
    trajs_e[...,1] /= H - 1

    kwargs = {
        'query_points': points_0_xy,
        'trajs_g': trajs_g,
        'visibs_g': visibs_g,
        'trajs_e': trajs_e,
        'visibs_e': visibs_e
    } # all coordinates are normalized

    np.savez(os.path.join(args.output_dir, f"{data['vid']}.npz"), **kwargs)
    metrics = compute_metrics(**kwargs)
    metrics.update({
        'vid': data['vid'],
        'num_frames': S,
        'num_points': trajs_g.shape[1]
    })
    df.append(metrics)
    
df = pd.DataFrame(df).set_index('vid')
df.to_csv(os.path.join(args.output_dir, 'metrics.csv'))
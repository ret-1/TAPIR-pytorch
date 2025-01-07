import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from model.tapir_model import Tapir
from dataset.test_dataset import TapVidTestDataset
from utils.evaluate import compute_metrics

parser = argparse.ArgumentParser("Evaluation script for Tapir")
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--ckpt", type=str, default="tapir.pth")
parser.add_argument("--dataset", type=str, default="tapvid_rgb_stacking")
parser.add_argument("--resize", type=int, nargs=2, default=[256, 256])
parser.add_argument(
    "--num_points", type=int, default=15, help="number of points per run"
)
args = parser.parse_args()

model = Tapir(pyramid_level=0, extra_convs=False).cuda()
model.load_state_dict(torch.load(args.ckpt))
model.eval()

os.makedirs(args.output_dir, exist_ok=True)

dataset = TapVidTestDataset(args.dataset, resize=args.resize)
df = []
n = args.num_points

for data in tqdm(dataset):
    rgbs = data["rgbs"].unsqueeze(0).cuda()  # B, S, H, W, C
    B, S, H, W, C = rgbs.shape

    trajs_g = data["trajs"].unsqueeze(0).cuda()  # B, N, S, 2, normalized coordinates
    visibs_g = data["visibs"].unsqueeze(0).cuda()  # B, N, S
    N = trajs_g.shape[1]

    # taking all points from frame 0
    points_0_xy = trajs_g[:, :, 0, :]  # B, N, 2 format:(x,y)
    points_0 = points_0_xy[:, :, [1, 0]]  # format:(y,x)
    points_0[..., 0] *= H - 1
    points_0[..., 1] *= W - 1
    # preparing the time dimension to be concatenated
    time_dim = torch.zeros((points_0.shape[0], points_0.shape[1], 1)).cuda()
    # prepending a column to be -> (B, N, 3)
    points_0 = torch.concatenate((time_dim, points_0), axis=-1)  # format:(t,y,x)

    trajs_e = []
    visibs_e = []

    with torch.no_grad():
        # to avoid memory issues, we run the model in chunks
        for i in range(0, N, n):
            output, _ = model(rgbs, points_0[:, i : i + n])
            trajs_e.append(output["tracks"])  # B, n, S, 2
            visibs_e_i = (1 - F.sigmoid(output["occlusion"])) * (
                1 - F.sigmoid(output["expected_dist"])
            ) > 0.5
            visibs_e.append(visibs_e_i)  # B, n, S

    trajs_e = torch.cat(trajs_e, dim=1)
    visibs_e = torch.cat(visibs_e, dim=1)

    trajs_e[..., 0] /= W - 1
    trajs_e[..., 1] /= H - 1

    kwargs = {
        "query_points": points_0_xy.cpu().numpy(),
        "trajs_g": trajs_g.cpu().numpy(),
        "visibs_g": visibs_g.cpu().numpy(),
        "trajs_e": trajs_e.cpu().numpy(),
        "visibs_e": visibs_e.cpu().numpy(),
    }  # all coordinates are normalized

    np.savez(os.path.join(args.output_dir, f"{data['vid']}.npz"), **kwargs)
    metrics = compute_metrics(**kwargs)
    metrics.update({"vid": data["vid"], "num_frames": S, "num_points": N})
    df.append(metrics)

df = pd.DataFrame(df).set_index("vid")
df.loc['avg'] = df.mean()
df.to_csv(os.path.join(args.output_dir, "metrics.csv"))

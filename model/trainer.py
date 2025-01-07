import os, pathlib
import torch
import torch.nn.functional as F
from model.tapir_model import Tapir
from utils.evaluate import compute_metrics, getMetricsDict
from utils.trainer import requires_grad, fetch_optimizer

class Trainer:
    def __init__(self, config, steps_per_epoch):
        self.model = Tapir(
            pyramid_level=config['MODEL']['PYRAMID_LEVEL'],
            extra_convs=config['MODEL']['EXTRA_CONVS'],
        ).cuda()
        self.optimizer, self.scheduler = fetch_optimizer(
            lr=config['TRAIN']['LR'],
            wdecay=config['TRAIN']['WEIGHT_DECAY'],
            epsilon=config['TRAIN']['EPSILON'],
            epochs=config['TRAIN']['EPOCHS'],
            steps_per_epoch=steps_per_epoch,
            params=self.model.parameters(),
        )

        if config['CHECKPOINT']['LOAD_CHECKPOINT'] is not None:
            self.load_ckpt(config['CHECKPOINT']['LOAD_CHECKPOINT'])
        if config['CHECKPOINT']['LOAD_NETWORK'] is not None:
            self.load_network(config['CHECKPOINT']['LOAD_NETWORK'])

    def do_pass(
        self,
        rgbs: list[torch.Tensor],  # [(S, H, W, C)]
        trajs_g: list[torch.Tensor],  # [(S, N, 2)]
        visibs_g: list[torch.Tensor],  # [(S, N)]
        is_train: bool = True,
    ):
        metrics = getMetricsDict()
        metrics["loss"] = 0.0
        batch_loss = []

        for rgb, traj_g, visib_g in zip(rgbs, trajs_g, visibs_g):
            # forward pass
            rgb, traj_g, visib_g = (
                rgb.unsqueeze(0),
                traj_g.unsqueeze(0),
                visib_g.unsqueeze(0),
            )
            rgb = rgb.float().cuda()  # B, S, H, W, C
            rgb = 2 * (rgb / 255.0) - 1.0  # normalizing [-1 1]
            traj_g = traj_g.permute(0, 2, 1, 3).cuda()  # B, N, S, 2 format:(x,y)
            visib_g = visib_g.permute(0, 2, 1).cuda()  # B, N, S

            B, S, H, W, C = rgb.shape
            traj_g[..., 0] *= W - 1
            traj_g[..., 1] *= H - 1

            # taking all points from frame 0
            points_0_xy = traj_g[:, :, 0, :]  # B, N, 2 format:(x,y)
            points_0 = points_0_xy[:, :, [1, 0]]  # format:(y,x)
            # preparing the time dimension to be concatenated
            time_dim = torch.zeros((points_0.shape[0], points_0.shape[1], 1)).cuda()
            # prepending a column to be -> (B, N, 3)
            points_0 = torch.concatenate(
                (time_dim, points_0), axis=-1
            )  # format:(t,y,x)

            output, loss = self.model(
                video=rgb, query_points=points_0, points_gt=traj_g, visibs_gt=visib_g
            )
            batch_loss.append(loss.mean()) # mean() is for parallel GPU computing 

            traj_e = output["tracks"]  # B, N, S, 2 format:(x,y)
            visib_e = (1 - F.sigmoid(output["occlusion"])) * (
                1 - F.sigmoid(output["expected_dist"])
            ) > 0.5
            # normalize for evaluation
            traj_e[..., 0] /= W - 1
            traj_e[..., 1] /= H - 1
            traj_g[..., 0] /= W - 1
            traj_g[..., 1] /= H - 1
            points_0_xy[..., 0] /= W - 1
            points_0_xy[..., 1] /= H - 1

            output = compute_metrics(
                query_points=points_0_xy.cpu().numpy(),
                trajs_g=traj_g.cpu().numpy(),
                visibs_g=visib_g.cpu().numpy(),
                trajs_e=traj_e.detach().cpu().numpy(),
                visibs_e=visib_e.cpu().numpy(),
            )

            for k, v in output.items():
                metrics[k] += v

            # mean() is for parallel GPU computing
            metrics["loss"] += loss.mean().item()  

        if is_train:
            torch.stack(batch_loss).mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        batch_size = len(rgbs)
        return {k: v / batch_size for k, v in metrics.items()}

    def save(self, ckpt_dir, global_step, keep_latest=2, model_name='model'):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        prev_ckpts = list(pathlib.Path(ckpt_dir).glob('%s-*' % model_name))
        prev_ckpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
        if len(prev_ckpts) > keep_latest-1:
            for f in prev_ckpts[keep_latest-1:]:
                f.unlink()
        model_path = '%s/%s-%09d.pth' % (ckpt_dir, model_name, global_step)
        
        ckpt = {'optimizer_state_dict': self.optimizer.state_dict()}
        ckpt['model_state_dict'] = self.model.state_dict()
        ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(ckpt, model_path)
        print("saved a checkpoint: %s" % (model_path))

    def load_ckpt(self, ckpt_dir, step=0, model_name='model', ignore_load=None):
        print('reading ckpt from %s' % ckpt_dir)
        if not os.path.exists(ckpt_dir):
            print('...there is no full checkpoint here!')
            print('-- note this function no longer appends "saved_checkpoints/" before the ckpt_dir --')
        else:
            ckpt_names = os.listdir(ckpt_dir)
            steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
            if len(ckpt_names) > 0:
                if step==0:
                    step = max(steps)
                model_name = '%s-%09d.pth' % (model_name, step)
                path = os.path.join(ckpt_dir, model_name)
                print('...found checkpoint %s'%(path))

                if ignore_load is not None:
                    
                    print('ignoring', ignore_load)
                    checkpoint = torch.load(path)['model_state_dict']

                    model_dict = self.model.state_dict()

                    # 1. filter out ignored keys
                    pretrained_dict = {k: v for k, v in checkpoint.items()}
                    for ign in ignore_load:
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not ign in k}
                        
                    # 2. overwrite entries in the existing state dict
                    model_dict.update(pretrained_dict)
                    # 3. load the new state dict
                    self.model.load_state_dict(model_dict, strict=False)
                else:
                    checkpoint = torch.load(path, map_location="cpu")
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print('...there is no full checkpoint here!')
        return step
    
    def load_network(self, path):
        self.model.load_state_dict(torch.load(path))
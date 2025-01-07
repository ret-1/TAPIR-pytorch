import argparse
import yaml
import os
import torch
import torch.distributed as distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataset.train_dataset import TrainDataset
from model.trainer import Trainer
from utils.evaluate import getMetricsDict

parser = argparse.ArgumentParser(description='Train TAPIR model.')
parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")

args = parser.parse_args()
config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

"""
Initial setup
"""
if 'RANK' not in os.environ.keys():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '15678'
    print('one device.')
else:
    print('DDP training!')

# Init distributed environment
distributed.init_process_group(backend="nccl")
print(f'CUDA Device count: {torch.cuda.device_count()}')

local_rank = distributed.get_rank()
world_size = distributed.get_world_size()
torch.cuda.set_device(local_rank)
print(f'I am rank {local_rank} in this world of size {world_size}!')

if local_rank == 0 and config['LOGGER']['ENABLE']:
    print('I will take the role of logging!')
    import wandb
    run = wandb.init(entity=config['LOGGER']['ENTITY'], project=config['LOGGER']['PROJECT'], config=config)
else:
    run = None

def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]

def construct_loader(dataset_root, batch_size, num_workers):
    dataset = TrainDataset(dataset_root)
    sampler = DistributedSampler(dataset, rank=local_rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    return loader

train_loader = construct_loader(config['TRAIN']['DATASET_ROOT'], config['TRAIN']['BATCH_SIZE'], config['TRAIN']['NUM_WORKERS'])

if config['TRAIN']['VALIDATION']['ENABLE']:
    print('Validation is enabled!')
    val_loader = construct_loader(config['VALIDATION']['DATASET_ROOT'], config['TRAIN']['BATCH_SIZE'], config['TRAIN']['NUM_WORKERS'])

len_train_loader = len(train_loader)
len_val_loader = len(val_loader) if config['TRAIN']['VALIDATION']['ENABLE'] else 0
trainer = Trainer(config, len_train_loader)

for epoch in range(1, config['TRAIN']['EPOCHS']+1):
    best_loss = 100000.0
    trainer.model.train()
    for data in train_loader:
        metrics = getMetricsDict()
        metrics['loss'] = 0.0
        
        metrics_b = trainer.do_pass(*data, is_train=True)
        for k, v in metrics_b.items():
            metrics[k] += v

    if run is not None:
        run.log({f'train/{k}': v/len_train_loader for k, v in metrics.items()})

    if config['TRAIN']['VALIDATION']['ENABLE'] and epoch % config['TRAIN']['VALIDATION']['INTERVAL'] == 0:
        trainer.model.eval()
        metrics = getMetricsDict()
        metrics['loss'] = 0.0

        with torch.no_grad():
            for data in val_loader:
                metrics_b = trainer.do_pass(*data, is_train=False)
                for k, v in metrics_b.items():
                    metrics[k] += v

        if run is not None:
            run.log({f'val/{k}': v/len_val_loader for k, v in metrics.items()})

        trainer.save(os.path.join(config['CHECKPOINT']['SAVE_DIR'], 'lasts'), epoch)
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            trainer.save(os.path.join(config['CHECKPOINT']['SAVE_DIR'], 'best'), epoch)

distributed.destroy_process_group()

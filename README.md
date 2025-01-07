# TAPIR-pytorch
The pytorch training &amp; inference pipeline of [TAPIR](https://github.com/google-deepmind/tapnet), mostly based on the implementation in [EchoTracker](https://github.com/riponazad/echotracker). Thanks the authors for their great work!

## Installation
After cloning the repository, you can install the required packages by running the following commands:
```bash
conda create -n tapir python=3.11
conda activate tapir
pip install -r requirements.txt
```

## Training
1. Use your own dataset to replace the code in `dataset/train_dataset.py` and change 
the corresponding part in `train.py`.
2. Change the configs in `config/default.yaml` according to your needs.
3. Run the following command to start training:
```bash
python train.py --config config/default.yaml
```

## Inference
Currently, only TAP-Vid-DAVIS and TAP-Vid-RGB-Stacking are supported (because they are esay to implement). You could follow the [instructions](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid#downloading-tap-vid-davis-and-tap-vid-rgb-stacking) to download them.

To inference and get the results, you can run the following command:
```bash
python inference.py --ckpt /path/to/your/checkpoint.pth --dataset /path/to/dataset --output_dir /path/to/output
```

Then you can find the results in the `output_dir`.

## Results
Using the checkpoint provided [here](https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt), we 
could get the following results under 256x256 inference resolution:

dataset|AJ|$<\delta^x_{avg}$|OA|survival|MTE
:---:|:---:|:---:|:---:|:---:|:---:
DAVIS|57.4%|69.5%|86.9%|96.7%|4.31
RGB-Stacking|55.5%|71.5%|84.3%|96.7%|4.25

from config import get_config
import wandb
cfg = get_config()
cfg['batch_size'] = 20
cfg['preload'] = None
cfg['num_epochs'] = 100
cfg['seq_len']=30

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from train import train_model
train_model(cfg)
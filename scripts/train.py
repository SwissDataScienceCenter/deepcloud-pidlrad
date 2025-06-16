import re
import os
import sys
import glob
import yaml
import time
import pickle
import logging
from os import path
from datetime import datetime

import torch
import wandb
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms

sys.path.append(path.dirname(path.dirname(__file__)))
from scripts.argsconfig import get_argparser
from src.pidlrad.utils.smoothing_filters import exponential_sigma, gassuian_kernel
from src.pidlrad.utils.load_data import IconDataset, IconH5Metadata, HeightCutter
from src.pidlrad.nn import get_model
from src.pidlrad.utils.training_scripts import train_model, test_model

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Torch is using: {device}')

seed = 42
wandb_config = {'seed': seed}
torch.manual_seed(seed)
np.random.seed(seed)
prng = np.random.RandomState(seed)

args = get_argparser()

save_id = path.basename(path.normpath(args.save_dir))
checkpoint_path = path.join(args.save_dir, 'checkpoint/')
history_path = path.join(args.save_dir, 'history/')
plot_path = path.join(args.save_dir, 'plots/')
log_dir = path.join(args.save_dir, 'logs/')

models_path = path.join(args.save_dir, 'models/')

if args.smoothing is not None and args.exponential_decay:
    test_path = path.join(args.save_dir, f'test_smoothed_{args.smoothing}_exponential_decay/')
    serialized_model = path.join(models_path, f'{save_id}_smoothed_{args.smoothing}_exponential_decays.pt')
elif args.smoothing is not None:
    test_path = path.join(args.save_dir, f'test_smoothed_{args.smoothing}/')
    serialized_model = path.join(models_path, f'{save_id}_smoothed_{args.smoothing}.pt')
elif args.exponential_decay:
    test_path = path.join(args.save_dir, f'test_exponential_decay/')
    serialized_model = path.join(models_path, f'{save_id}_exponential_decay.pt')
else:
    test_path = path.join(args.save_dir, 'test/')
    serialized_model = path.join(models_path, f'{save_id}.pt')

os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(history_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

last_checkpoint = path.join(checkpoint_path, 'last_checkpoint.pth')
best_checkpoint = path.join(checkpoint_path, 'best_checkpoint.pth')

if args.smoothing == 'exponential_sigma':
    args.smoothing_kernel = gassuian_kernel(sigmas=exponential_sigma()).to(device)
elif args.smoothing == 'constant_sigma':
    args.smoothing_kernel = gassuian_kernel(
        sigmas=torch.full((args.cutoff_height,), args.smoothing_sigma),
        size=args.cutoff_height
    ).to(device)
elif args.smoothing is None:
    args.smoothing_kernel = None
else:
    raise NotImplementedError(f'{args.smoothing} not supported!')

def main():
    icon_metadata = IconH5Metadata(path.join(args.data_dir, 'metadata.h5'))
    mean_std = icon_metadata.get_mean_std(args.normalization)
    x3d_mean, x3d_std, x2d_mean, x2d_std = [a.to(device) for a in mean_std]

    y_mean, y_std = icon_metadata.y_mean_pfph, icon_metadata.y_std_pfph
    args.y_mean, args.y_std = y_mean.to(device), y_std.to(device)
    transform = [HeightCutter(height_in=args.height_in, height_out=args.height_out)]
    icon_dataset = IconDataset(
        data_dir=args.data_dir,
        spatial_subsample=args.spatial_subsample,
        temporal_subsample=args.temporal_subsample,
        cache_dir=args.cache_dir,
        transform=transforms.Compose(transform)
    )
    train_set, val_set, test_set = icon_dataset.split()

    train_dataloader = DataLoader(
        train_set, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers
    )
    val_dataloader = DataLoader(
        val_set, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers
    )
    test_dataloader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers
    )

    args.beta = icon_metadata.beta_constant.to(device) if args.exponential_decay else None

    model = get_model(x3d_mean, x3d_std, x2d_mean, x2d_std, args).to(device)
    if args.summary:
        summary(model, [(32, args.height_in, args.channel_3d), (32, args.channel_2d)])

    if args.train:
        model = train_model(
            model, train_dataloader, val_dataloader, logger,
            last_checkpoint, best_checkpoint, save_id, device, args
        )

    if args.test:
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        test_model(model, test_dataloader, test_path, logger, device, args)

    if args.serialize:
        logger.info(f'Serializing model: {serialized_model}')
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        frozen_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
        assert "batch_norm" not in str(frozen_model.graph)
        frozen_model.save(serialized_model)

    if args.test_serialized:
        assert os.path.exists(serialized_model), f'{serialized_model} NOT FOUND!'
        s_model = torch.jit.load(serialized_model, map_location=device)
        s_model.eval()

        with open(os.path.join(test_path, 'timing.log'), 'w') as file:
            file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \n')
            for i in range(50):
                start_time = time.time()  # Start time
                test_model(s_model, test_dataloader, test_path, logger, device, args)
                end_time = time.time()  # End time
                elapsed_time = end_time - start_time  # Calculate elapsed time
                file.write(f"Trial {i+1}: {elapsed_time:.6f} seconds\n")
                logger.info(f"Trial {i+1}: {elapsed_time:.6f} seconds\n")

    logger.info('Code finished.')

if __name__ == '__main__':
    main()
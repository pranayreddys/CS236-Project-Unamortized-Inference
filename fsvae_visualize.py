# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.fsvae import FSVAE
from codebase.train import train
from pprint import pprint
# from torchvision import datasets, transforms
import torchvision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=60000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=60000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
parser.add_argument('--overwrite', type=int, default=0,     help="Flag for overwriting")
args = parser.parse_args()
layout = [
    ('model={:s}',  'fsvae'),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, test_set = ut.get_svhn_data(device)
fsvae = FSVAE(name=model_name).to(device)
ut.load_model_by_name(fsvae, global_step=args.iter_max, device=device)
sampled_latent = fsvae.sample_z(20)
sampled_latent = ut.duplicate(sampled_latent, 10)
y = torch.ones(sampled_latent.shape[0], dtype=int)
for i in range(10):
    y[20*i:20*(i+1)] = i
y = y.new(np.eye(10)[y]).to(device).float()
x_recon_mean = fsvae.dec(sampled_latent, y)
sampled_images = x_recon_mean.reshape(200,3,32,32)
sampled_images = torch.clip(sampled_images, min=0, max=1)
# sampled_images = sampled_images.reshape(200,1,28,28)
torchvision.utils.save_image(sampled_images, 'fsvae_svhn_samples.png', nrow=20)
# ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=True)

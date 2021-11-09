# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
from torch import optim

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--overwrite', type=int, default=0,     help="Flag for overwriting")
args = parser.parse_args()
layout = [
    ('model={:s}',  'vae'),
    ('z={:02d}',  args.z),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
vae = VAE(z_dim=args.z, name=model_name).to(device)
ut.load_model_by_name(vae, global_step=args.iter_max, device=device)
x= labeled_subset[0]
with torch.no_grad():
    m,v = vae.enc(x)
# vae.train()
m.requires_grad = True
optimizer = optim.Adam([m], lr=1e-3)
for i in range(100):
    loss, summary = vae.loss_unamortized(x, m, v)
    print(vae.loss(x), "**")
    print(summary['train/loss'])
    print(loss, "************")
    loss.backward()
    vae.zero_grad()
    optimizer.step()
    optimizer.zero_grad()
    # m.zero_grad()

# ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=True)

# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.models.gmvae import GMVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
from torch import optim
from copy import deepcopy

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--overwrite', type=int, default=0,     help="Flag for overwriting")
parser.add_argument('--iter_run', type=int, default=500,     help="Number of training runs")
parser.add_argument('--model', type=str, help="Model type")
parser.add_argument('--k', type=int, default=500,   help="Number mixture components in MoG prior")
parser.add_argument('--inpainting', action='store_true',    help="Number mixture components in MoG prior")

args = parser.parse_args()
if args.model=='vae':
    layout = [
        ('model={:s}',  'vae'),
        ('z={:02d}',  args.z),
        ('run={:04d}', args.run)
    ]
else:
    layout = [
        ('model={:s}',  'gmvae'),
        ('z={:02d}',  args.z),
        ('k={:03d}',  args.k),
        ('run={:04d}', args.run)
    ]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _, data_len = ut.get_mnist_data(device, use_test_subset=True)
if args.model=='vae':
    model = VAE(z_dim=args.z, name=model_name).to(device)
else:
    model = GMVAE(z_dim=args.z, k=args.k, name=model_name).to(device)

ut.load_model_by_name(model, global_step=args.iter_max, device=device)
model.initialize_cache(train_loader, data_len)
x= labeled_subset[0].to(device)

writer = None
if not args.inpainting:
    train(model=model,
            train_loader=train_loader,
            labeled_subset=labeled_subset,
            device=device,
            tqdm=tqdm.tqdm,
            writer=writer,
            iter_max=args.iter_run,
            iter_save=args.iter_save)
    ut.evaluate_lower_bound(model, labeled_subset, run_iwae=False)
else:
    ut.infill(model, labeled_subset)

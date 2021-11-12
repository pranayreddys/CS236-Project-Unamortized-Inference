# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.fsvae import FSVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=1000000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000,   help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,       help="Run ID. In case you want to run replicates")
parser.add_argument('--overwrite', type=int, default=0,       help="Flag for overwriting")
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
model = FSVAE(name=model_name).to(device)
writer = ut.prepare_writer(model_name, overwrite_existing=args.overwrite)
ut.load_model_by_name(model, global_step=args.iter_max, device=device)
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

#*********************
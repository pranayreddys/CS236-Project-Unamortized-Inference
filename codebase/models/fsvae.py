# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class FSVAE(nn.Module):
    def __init__(self, nn='v2', name='fsvae'):
        super().__init__()
        self.name = name
        self.z_dim = 10
        self.y_dim = 0
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that we are interested in the ELBO of ln p(x | y)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        m,v = self.enc(x)
        kl_z = ut.kl_normal(m,v,self.z_prior_m, self.z_prior_v).mean()
        z = ut.sample_gaussian(m,v)
        mprime = self.dec(z)
        rec = -ut.log_normal(x, mprime, torch.tensor(0.1)).mean()
        nelbo = rec + kl_z
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, rec

    def recon(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # m,v = self.enc(x)
        m, v = ut.get_function(x, self, None)
        kl = ut.kl_normal(m,v,self.z_prior_m, self.z_prior_v).mean()
        z = ut.sample_gaussian(m,v)
        return self.compute_mean_given(z)
    
    def unamortized_inference(self, x, m, v):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        kl_z = ut.kl_normal(m,v,self.z_prior_m, self.z_prior_v).mean()
        z = ut.sample_gaussian(m,v)
        mprime = self.dec(z)
        rec = -ut.log_normal(x, mprime, torch.tensor(0.1)).mean()
        nelbo = rec + kl_z
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, rec
    
    def loss_unamortized(self, x, m, v):
        nelbo, kl, rec = self.unamortized_inference(x, m, v)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def loss(self, x, y):
        nelbo, kl_z, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', loss),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_mean_given(self, z):
        return self.dec(z)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

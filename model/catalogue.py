import pdb

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
import numpy as np


class RealNVP(nn.Module):
    def __init__(self, mask, prior, num_layers, nets, nett):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.num_layers = num_layers

        # self.t = torch.nn.ModuleList([nett() for _ in range(self.mask.shape[0])])
        # self.s = torch.nn.ModuleList([nets() for _ in range(self.mask.shape[0])])

        self.t = torch.nn.ModuleList([nett() for _ in range(self.num_layers)])
        self.s = torch.nn.ModuleList([nets() for _ in range(self.num_layers)])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        try:
            return self.prior.log_prob(z) + logp
        except ValueError as ve:
            print(ve)
            pdb.set_trace()

    def sample(self, batchSize):
        pdb.set_trace()
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x
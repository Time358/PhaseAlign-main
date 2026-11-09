import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def norm_stats(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        return mean, stdev

    def apply_norm(self, x, stats):
        mean, stdev = stats
        return (x - mean) / stdev

    def apply_denorm(self, x, stats):
        mean, stdev = stats
        return x * stdev + mean

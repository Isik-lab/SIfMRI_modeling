#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
import torch


def calculate_p(r_null_, r_true_, n_perm_, H0_):
    # Get the p-value depending on the type of test
    denominator = n_perm_ + 1
    if H0_ == 'two_tailed':
        numerator = np.sum(np.abs(r_null_) >= np.abs(r_true_), axis=0) + 1
        p_ = numerator / denominator
    elif H0_ == 'greater':
        numerator = np.sum(r_true_ > r_null_, axis=0) + 1
        p_ = 1 - (numerator / denominator)
    else:  # H0 == 'less':
        numerator = np.sum(r_true_ < r_null_, axis=0) + 1
        p_ = 1 - (numerator / denominator)
    return p_


# def bootstrap(a, b, n_perm=int(5e3)):
#     # Randomly sample and recompute r^2 n_perm times
#     r_var = np.zeros((n_perm, a.shape[-1]))
#     for i in tqdm(range(n_perm), total=n_perm):
#         inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
#                                                size=a.shape[0])
#         if a.ndim == 3:
#             a_sample = a[inds, ...].reshape(a.shape[0] * a.shape[1], a.shape[-1])
#             b_sample = b[inds, ...].reshape(b.shape[0] * b.shape[1], b.shape[-1])
#         else:  # a.ndim == 2:
#             a_sample = a[inds, :]
#             b_sample = b[inds, :]
#         r_var[i, :] = corr2d(a_sample, b_sample)
#     return r_var




def corr2d_gpu(x, y):
    import torch
    x_m = x - torch.nanmean(x, dim=0)
    y_m = y - torch.nanmean(y, dim=0)

    numer = torch.nansum((x_m * y_m), dim=0)
    denom = torch.sqrt(torch.nansum((x_m * x_m), dim=0) * torch.nansum((y_m * y_m), dim=0))
    denom[denom == 0] = float('nan')
    return numer / denom


def perm_gpu(a, b, n_perm=int(5e3), verbose=False):
    import torch
    g = torch.Generator()
    r = corr2d_gpu(a, b)

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Permutation testing')
    else:
        iterator = range(n_perm)

    r_null = torch.zeros((n_perm, a.shape[-1]))
    for i in iterator:
        g.manual_seed(i)
        inds = torch.randperm(a.shape[0], generator=g)
        a_shuffle = a[inds]
        r_null[i, :] = corr2d_gpu(a_shuffle, b)
    return r, r_null


def bootstrap_gpu(a, b, n_perm=int(5e3), verbose=False):
    import torch
    g = torch.Generator()

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Permutation testing')
    else:
        iterator = range(n_perm)

    r_var = torch.zeros((n_perm, a.shape[-1]))
    for i in iterator:
        g.manual_seed(i)
        inds = torch.squeeze(torch.randint(high=a.shape[0], size=(a.shape[0],1), generator=g))
        a_sample, b_sample = a[inds], b[inds]
        r_var[i, :] = corr2d_gpu(a_sample, b_sample)
    return r_var

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
import torch


def feature_scaler(train, test, axis=0):
    mean_ = torch.mean(train, dim=axis, keepdim=True)
    std_ = torch.std(train, dim=axis, keepdim=True)
    return (train-mean_)/std_, (test-mean_)/std_


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


def corr(x, y):
    x_m = x - np.nanmean(x)
    y_m = y - np.nanmean(y)
    numer = np.nansum(x_m * y_m)
    denom = np.sqrt(np.nansum(x_m * x_m) * np.nansum(y_m * y_m))
    if denom != 0:
        return numer / denom
    else:
        return np.nan

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
    return r_null


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

def spearman_corr_rdm(rdm1, rdm2):
    # Flatten the upper triangle of each RDM, excluding the diagonal
    triu_indices = np.triu_indices(rdm1.shape[0], k=1)
    rdm1_vector = rdm1[triu_indices]
    rdm2_vector = rdm2[triu_indices]

    # Convert to ranks
    rdm1_rank = torch.tensor(rdm1_vector.argsort().argsort(), dtype=torch.float)
    rdm2_rank = torch.tensor(rdm2_vector.argsort().argsort(), dtype=torch.float)

    # Compute Spearman correlation
    return corr2d_gpu(rdm1_rank, rdm2_rank)

def perm_test_rdm(rdm_neural, rdm_model, n_perm=1000):
    # Initial correlation
    original_corr = spearman_corr_rdm(rdm_neural, rdm_model).item()
    # Store permutation correlations
    perm_corrs = torch.zeros(n_perm)
    # Number of elements along one side of the RDM
    n = rdm_neural.shape[0]
    # Generate random permutations and compute correlations
    for i in range(n_perm):
        # Random permutation
        perm = torch.randperm(n)
        # Permuted RDM
        rdm_perm = rdm_neural[perm][:, perm]
        # Compute Spearman correlation for permuted RDM
        perm_corrs[i] = spearman_corr_rdm(rdm_perm, rdm_model)

    # Compute p-value
    p_value = (perm_corrs >= original_corr).float().mean().item()
    return original_corr, perm_corrs, p_value

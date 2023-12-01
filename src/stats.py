#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
import torch


def filter_r(rs, ps, p_crit=0.05, correct=True, threshold=True):
    rs_out = rs.copy()
    if correct:
        _, ps_corrected, _, _ = multipletests(ps, method='fdr_bh')
    else:
        ps_corrected = ps.copy()

    if threshold:
        rs_out[ps_corrected >= p_crit] = 0.
    else:
        rs_out[rs_out < 0.] = 0.
    return rs_out, ps_corrected


def corr(x, y):
    x_m = x - np.nanmean(x)
    y_m = y - np.nanmean(y)
    numer = np.nansum(x_m * y_m)
    denom = np.sqrt(np.nansum(x_m * x_m) * np.nansum(y_m * y_m))
    if denom != 0:
        return numer / denom
    else:
        return np.nan


def corr2d(x, y):
    x_m = x - np.nanmean(x, axis=0)
    y_m = y - np.nanmean(y, axis=0)

    numer = np.nansum((x_m * y_m), axis=0)
    denom = np.sqrt(np.nansum((x_m * x_m), axis=0) * np.nansum((y_m * y_m), axis=0))
    denom[denom == 0] = np.nan
    return numer / denom


def mantel_permutation(a, i):
    a = squareform(a)
    inds = np.random.permutation(a.shape[0])
    a_shuffle = a[inds][:, inds]
    return squareform(a_shuffle)


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


def bootstrap(a, b, n_perm=int(5e3)):
    # Randomly sample and recompute r^2 n_perm times
    r_var = np.zeros((n_perm, a.shape[-1]))
    for i in tqdm(range(n_perm), total=n_perm):
        inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                               size=a.shape[0])
        if a.ndim == 3:
            a_sample = a[inds, ...].reshape(a.shape[0] * a.shape[1], a.shape[-1])
            b_sample = b[inds, ...].reshape(b.shape[0] * b.shape[1], b.shape[-1])
        else:  # a.ndim == 2:
            a_sample = a[inds, :]
            b_sample = b[inds, :]
        r_var[i, :] = corr2d(a_sample, b_sample)
    return r_var


def bootstrap_unique_variance(a, b, c, n_perm=int(5e3)):
    # Randomly sample and recompute r^2 n_perm times
    r2_var = np.zeros((n_perm, a.shape[-1]))
    for i in tqdm(range(n_perm), total=n_perm):
        inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                               size=a.shape[0])
        if a.ndim == 3:
            a_sample = a[inds, ...].reshape(a.shape[0] * a.shape[1], a.shape[-1])
            b_sample = b[inds, ...].reshape(b.shape[0] * b.shape[1], b.shape[-1])
            c_sample = c[inds, ...].reshape(c.shape[0] * c.shape[1], c.shape[-1])
        else:  # a.ndim == 2:
            a_sample = a[inds, :]
            b_sample = b[inds, :]
            c_sample = c[inds, :]
        r2_var[i, :] = corr2d(a_sample, b_sample)**2 - corr2d(a_sample, c_sample)**2
    return r2_var


def perm(a, b, n_perm=int(5e3), H0='greater', verbose=False):
    if a.ndim == 3:
        a_not_shuffle = a.reshape(a.shape[0] * a.shape[1], a.shape[-1])
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[-1])
        r = corr2d(a_not_shuffle, b)
    else:
        r = corr2d(a, b)

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Permutation testing')
    else:
        iterator = range(n_perm)

    r_null = np.zeros((n_perm, a.shape[-1]))
    for i in iterator:
        inds = np.random.default_rng(i).permutation(a.shape[0])
        if a.ndim == 3:
            a_shuffle = a[inds, :, :].reshape(a.shape[0] * a.shape[1], a.shape[-1])
        else:  # a.ndim == 2:
            a_shuffle = a[inds, :]
        r_null[i, :] = corr2d(a_shuffle, b)

    # Get the p-value depending on the type of test
    p = calculate_p(r_null, r, n_perm, H0)
    return r, p, r_null


def corr2d_gpu(x, y):
    x_m = x - torch.nanmean(x, dim=0)
    y_m = y - torch.nanmean(y, dim=0)

    numer = torch.nansum((x_m * y_m), dim=0)
    denom = torch.sqrt(torch.nansum((x_m * x_m), dim=0) * torch.nansum((y_m * y_m), dim=0))
    denom[denom == 0] = float('nan')
    return numer / denom


def perm_gpu(a, b, n_perm=int(5e3), verbose=False):
    g = torch.Generator()
    r = corr2d(a, b)

    if verbose:
        iterator = tqdm(range(n_perm), total=n_perm, desc='Permutation testing')
    else:
        iterator = range(n_perm)

    r_null = np.zeros((n_perm, a.shape[-1]))
    for i in iterator:
        g.manual_seed(i)
        inds = torch.randperm(a.shape[0], generator=g)
        a_shuffle = a[inds]
        r_null[i, :] = corr2d(a_shuffle, b)
    return r, r_null


#!/usr/bin/env python
# coding: utf-8
import PIL
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import squareform
import math
from scipy.stats import spearmanr


def camera_switcher(hemi, view):
    if view == 'lateral':
        if hemi == 'lh':
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.5, y=0, z=0)
            )
        else:
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=0, z=0)
            )
    elif view == 'medial':
        if hemi == 'lh':
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.75, y=0, z=0)
            )
        else:
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.75, y=0, z=0)
            )
    elif view == 'ventral':
        if hemi == 'lh':
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=-2.5)
            )
        else:
            camera = dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=-2.5)
            )
    else:
        raise 'invalid view'
    return camera


def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def filter_r(rs, ps, p_crit=0.05, correct=True, threshold=True):
    from statsmodels.stats.multitest import fdrcorrection
    rs_out = rs.copy()
    if correct:
        _, ps_corrected = fdrcorrection(ps, method='p', alpha=0.05, is_sorted=False)
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
    x_m = x - np.mean(x, axis=0)
    y_m = y - np.mean(y, axis=0)

    numer = np.sum((x_m * y_m), axis=0)
    denom = np.sqrt(np.sum((x_m * x_m), axis=0) * np.sum((y_m * y_m), axis=0))
    denom[denom == 0] = np.nan
    return numer / denom


def mantel_permutation(a, i):
    a = squareform(a)
    inds = np.random.permutation(a.shape[0])
    a_shuffle = a[inds][:, inds]
    return squareform(a_shuffle)


def calculate_p(r_null_, r_true_, n_perm_, H0_):
    # Get the p-value depending on the type of test
    if H0_ == 'two_tailed':
        n_extreme = np.sum(np.abs(r_null_) >= np.abs(r_true_), axis=0)
        p_ = (n_extreme + 1) / (n_perm_ + 1)
    elif H0_ == 'greater':
        n_extreme = np.sum(r_true_ > r_null_, axis=0)
        p_ = 1 - (n_extreme / (n_perm_ + 1))
    else:  # H0 == 'less':
        n_extreme = np.sum(r_true_ < r_null_, axis=0)
        p_ = 1 - (n_extreme / (n_perm_ + 1))
    return p_


def bootstrap(a, b, n_perm=int(5e3), square=True, verbose=True):
    # Randomly sample and recompute r^2 n_perm times
    if verbose:
        iter_loop = tqdm(range(n_perm), total=n_perm)
    else:
        iter_loop = range(n_perm)

    if a.ndim > 1 :
        r2_var = np.zeros((n_perm, a.shape[-1]))
        for i in iter_loop:
            inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                                   size=a.shape[0])
            if a.ndim == 3:
                a_sample = a[inds, ...].reshape(a.shape[0] * a.shape[1], a.shape[-1])
                b_sample = b[inds, ...].reshape(b.shape[0] * b.shape[1], b.shape[-1])
            else:
                a_sample = a[inds, :]
                b_sample = b[inds, :]
            r = corr2d(a_sample, b_sample)
            if square:
                r = np.sign(r)*(r**2)
            r2_var[i, :] = r
    else:
        r2_var = np.zeros((n_perm,))
        for i in iter_loop:
            inds = np.random.default_rng(i).choice(np.arange(a.shape[0]),
                                                   size=a.shape[0])
            r = corr(a[inds], b[inds])
            if square:
                r = np.sign(r)*(r**2)
            r2_var[i] = r
    return r2_var


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


def spearmanr_perm(a, b, n_perm=int(5e3), H0='greater'):
    rho = spearmanr(a, b).statistic
    rho_null = np.zeros((n_perm,))

    for i in range(n_perm):
        inds = np.random.default_rng(i).permutation(a.shape[0])
        a_shuffle = a[inds]
        rho_null[i] = spearmanr(a_shuffle, b).statistic

    # Get the p-value depending on the type of test
    p = calculate_p(rho_null, rho, n_perm, H0)
    return rho, p, rho_null


def perm(a, b, n_perm=int(5e3), H0='greater', square=True, verbose=True):
    if a.ndim > 1:
        r2_null = np.zeros((n_perm, a.shape[-1]))
        if a.ndim == 3:
            a_not_shuffle = a.reshape(a.shape[0] * a.shape[1], a.shape[-1])
            b = b.reshape(b.shape[0] * b.shape[1], b.shape[-1])
            r = corr2d(a_not_shuffle, b)
        else: #a.ndim == 2:
            r = corr2d(a, b)
    else: #a.ndim == 1:
        r = corr(a, b)
        r2_null = np.zeros((n_perm,))

    if square:
        r_out = (r**2) * np.sign(r)
    else:
        r_out = r.copy()

    if verbose:
        iter_loop = tqdm(range(n_perm), total=n_perm)
    else:
        iter_loop = range(n_perm)

    for i in iter_loop:
        inds = np.random.default_rng(i).permutation(a.shape[0])
        if a.ndim == 3:
            a_shuffle = a[inds, :, :].reshape(a.shape[0] * a.shape[1], a.shape[-1])
        elif a.ndim == 2:
            a_shuffle = a[inds, :]
        else:# a.ndim == 1:
            a_shuffle = a[inds]

        if a.ndim > 1:
            r = corr2d(a_shuffle, b)
        else:
            r = corr(a_shuffle, b)

        if square:
            r2 = (r**2) * np.sign(r)
        else:
            r2 = r.copy()

        if a.ndim > 1:
            r2_null[i, :] = r
        else:
            r2_null[i] = r

    # Get the p-value depending on the type of test
    p = calculate_p(r2_null, r_out, n_perm, H0)
    return r_out, p, r2_null


def perm_unique_variance(a, b, c, n_perm=int(5e3), H0='greater'):
    if a.ndim == 3:
        a_not_shuffle = a.reshape(a.shape[0] * a.shape[1], a.shape[-1])
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[-1])
        c = c.reshape(c.shape[0] * c.shape[1], c.shape[-1])
        r_ab = corr2d(a_not_shuffle, b)
        r_ac = corr2d(a_not_shuffle, c)
    else:
        r_ab = corr2d(a, b)
        r_ac = corr2d(a, c)
    r2 = r_ab ** 2 - r_ac ** 2

    # Shuffle a and recompute r^2 n_perm times
    r2_null = np.zeros((n_perm, a.shape[-1]))
    for i in tqdm(range(n_perm), total=n_perm):
        inds = np.random.default_rng(i).permutation(a.shape[0])
        if a.ndim == 3:
            a_shuffle = a[inds, :, :].reshape(a.shape[0] * a.shape[1], a.shape[-1])
        else:  # a.ndim == 2:
            a_shuffle = a[inds, :]
        r_ab = corr2d(a_shuffle, b)
        r_ac = corr2d(a_shuffle, c)
        r2_null[i, :] = r_ab ** 2 - r_ac ** 2

    p = calculate_p(r2_null, r2, n_perm, H0)
    return r2, p, r2_null


def mask_img(img, mask, fill=0.):
    import nibabel as nib
    if type(img) is nib.nifti1.Nifti1Image:
        masked_img = np.array(img.dataobj)
        mask = np.array(mask.dataobj)
    else:
        masked_img = img.copy()
    mask = np.invert(mask.astype('bool'))
    i, j, k = np.where(mask)
    masked_img[i, j, k] = fill
    if type(img) is nib.nifti1.Nifti1Image:
        masked_img = nib.Nifti1Image(masked_img, img.affine, img.header)
    return masked_img


def add_svg(current_canvas, file, x, y, offset=50, scaling_factor=None, max_width=None):
    from reportlab.graphics import renderPDF
    from svglib.svglib import svg2rlg
    canvas_width = current_canvas._pagesize[0]
    drawing = svg2rlg(file)
    if scaling_factor is None:
        if max_width is None:
            scaling_factor = canvas_width / drawing.width
        else:
            scaling_factor = max_width / drawing.width
    drawing.scale(scaling_factor, scaling_factor)
    y_pos = y-drawing.height+offset
    renderPDF.draw(drawing, current_canvas,
                   x, y_pos,
                   showBoundary=False)
    return y_pos, scaling_factor


def add_img(current_canvas, file, x, y, scaling_factor=0.25, rotate=0):
    from reportlab.lib.utils import ImageReader
    pil_img = ImageReader(file)
    pil_img._image.rotate(rotate, PIL.Image.NEAREST, expand=1)
    img_width, img_height = pil_img._image._size
    new_width, new_height = int(img_width * scaling_factor), int(img_height * scaling_factor)
    current_canvas.drawImage(pil_img, x, y-new_height,
                             new_width, new_height, mask="auto")

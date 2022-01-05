# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from scipp.optimize import curve_fit

import pytest


def func(x, a, b):
    return a * sc.exp(-(b / x.unit) * x)


def array1d():
    size = 20
    x = sc.linspace(dim='xx', start=0.5, stop=2.0, num=size, unit='m')
    y = func(x, 1.2, 1.3)
    rng = np.random.default_rng()
    y.values += 0.1 * rng.normal(size)
    da = sc.DataArray(y, coords={'xx': x})
    return da


def test_should_not_raise_given_function_with_dimensionless_params_and_1d_array():
    curve_fit(func, array1d())


def test_should_raise_TypeError_when_xdata_given_as_param():
    with pytest.raises(TypeError):
        curve_fit(func, array1d(), xdata=np.arange(4))


def test_should_raise_TypeError_when_ydata_given_as_param():
    with pytest.raises(TypeError):
        curve_fit(func, array1d(), ydata=np.arange(4))


def test_should_raise_TypeError_when_sigma_given_as_param():
    with pytest.raises(TypeError):
        curve_fit(func, array1d(), sigma=np.arange(4))


def test_should_raise_NotFoundError_when_data_array_has_no_coord():
    da = array1d()
    del da.coords[da.dim]
    with pytest.raises(sc.NotFoundError):
        curve_fit(func, da)


def test_should_raise_BinEdgeError_when_data_array_is_histogram():
    da = array1d()
    hist = da[1:].copy()
    hist.coords[hist.dim] = da.coords[hist.dim]
    with pytest.raises(sc.BinEdgeError):
        curve_fit(func, hist)


def test_masks_are_not_ignored():
    da = array1d()
    unmasked, _ = curve_fit(func, da)
    da.masks['mask'] = sc.zeros(sizes=da.sizes, dtype=bool)
    da.masks['mask'][0] = True
    masked, _ = curve_fit(func, da)
    assert all(masked != unmasked)


@pytest.mark.parametrize("mask_pos", [0, 1, -3])
@pytest.mark.parametrize("mask_size", [1, 2])
def test_masked_points_are_treated_as_if_they_were_removed(mask_pos, mask_size):
    da = array1d()
    da.masks['mask'] = sc.zeros(sizes=da.sizes, dtype=bool)
    da.masks['mask'][mask_pos:mask_pos + mask_size] = sc.scalar(True)
    masked, _ = curve_fit(func, da)
    removed, _ = curve_fit(
        func, sc.concat([da[:mask_pos], da[mask_pos + mask_size:]], da.dim))
    assert all(masked == removed)


@pytest.mark.parametrize("variance,expected", [(1e9, 1.0), (1, 2.0), (1 / 3, 3.0),
                                               (1e-9, 5.0)],
                         ids=['disabled', 'equal', 'high', 'dominant'])
def test_variances_determine_weights(variance, expected):
    x = sc.array(dims=['x'], values=[1, 2, 3, 4])
    y = sc.array(dims=['x'], values=[1., 5., 1., 1.], variances=[1., 1., 1., 1.])
    da = sc.DataArray(data=y, coords={'x': x})
    da.variances[1] = variance
    # Fit a constant to highlight influence of weights
    popt, _ = curve_fit(lambda x, a: sc.scalar(a), da)
    assert popt[0] == pytest.approx(expected)

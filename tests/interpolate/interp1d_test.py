# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
import scipy.interpolate as theirs
from scipp.interpolate import interp1d

import pytest


def make_array():
    x = sc.geomspace(dim='xx', start=0.1, stop=0.4, num=10, unit='rad')
    y = sc.linspace(dim='yy', start=0.5, stop=2.0, num=6, unit='m')
    da = sc.DataArray(sc.sin(x) * y, coords={'xx': x})
    da.unit = 'K'
    return da


def check_metadata(out, da, x):
    assert out.unit == da.unit
    assert sc.identical(out.coords['xx'], x)


@pytest.mark.parametrize(
    "da", [make_array(),
           make_array().transpose(),
           make_array().transpose().copy()])
def test_metadata(da):
    f = interp1d(da, 'xx')
    x = sc.linspace(dim='xx', start=0.1, stop=0.4, num=10, unit='rad')
    check_metadata(f(x), da, x)
    check_metadata(f(x[:5]), da, x[:5])


def test_fail_variances():
    da = make_array()
    da.variances = da.values
    with pytest.raises(sc.VariancesError):
        interp1d(da, 'xx')


def test_fail_bin_edges():
    tmp = make_array()
    da = tmp['xx', 1:].copy()
    da.coords['xx'] = tmp.coords['xx']
    with pytest.raises(sc.BinEdgeError):
        interp1d(da, 'xx')


def test_fail_new_coord_unit():
    da = make_array()
    f = interp1d(da, 'xx')
    x = sc.linspace(dim='xx', start=0.1, stop=0.4, num=10, unit='deg')
    with pytest.raises(sc.UnitError):
        f(x)


def test_fail_new_coord_wrong_dim():
    da = make_array()
    f = interp1d(da, 'xx')
    x = sc.linspace(dim='x', start=0.1, stop=0.4, num=10, unit='rad')
    with pytest.raises(sc.DimensionError):
        f(x)
    x = sc.linspace(dim='yy', start=0.1, stop=0.4, num=da.sizes['yy'], unit='rad')
    with pytest.raises(sc.DimensionError):
        f(x)


def test_data():
    da = make_array()
    x = sc.linspace(dim='xx', start=0.1, stop=0.4, num=10, unit='rad')
    out = interp1d(da, 'xx')(x)
    assert np.array_equal(
        out.values,
        theirs.interp1d(x=da.coords['xx'].values, y=da.values, axis=0)(x.values))
    da = da.transpose()
    out = interp1d(da, 'xx')(x)
    assert np.array_equal(
        out.values,
        theirs.interp1d(x=da.coords['xx'].values, y=da.values, axis=1)(x.values))
    da = da.copy()
    out = interp1d(da, 'xx')(x)
    assert np.array_equal(
        out.values,
        theirs.interp1d(x=da.coords['xx'].values, y=da.values, axis=1)(x.values))


def test_data_datetime():
    da = make_array().rename_dims({'xx': 'time'})
    x = sc.arange(dim='time',
                  start=0,
                  stop=da.sizes['time'],
                  step=1,
                  unit='s',
                  dtype='datetime64')
    da.coords['time'] = x
    out = interp1d(da, 'time')(da.coords['time'])
    assert np.array_equal(
        out.values,
        theirs.interp1d(x=da.coords['time'].values.astype('int64'), y=da.values,
                        axis=0)(x.values))


def test_close():
    # Sanity check: are we using interp1d correctly? Remove points and interpolate
    da = make_array()
    da_missing_points = sc.concat([da['xx', :3], da['xx', 5:]], 'xx')
    out = interp1d(da_missing_points, 'xx')(da.coords['xx'])
    assert sc.allclose(da.data, out.data, rtol=sc.scalar(1e-3))


def test_fail_multidim_mask():
    da = make_array()
    da.masks['mask'] = da.data != da.data
    with pytest.raises(sc.DimensionError):
        interp1d(da, 'xx')


def test_masked():
    x = sc.linspace(dim='xx', start=0.0, stop=3.0, num=20, unit='rad')
    da = sc.DataArray(sc.sin(x), coords={'xx': x})
    da.masks['mask'] = da.data > sc.scalar(0.9)
    result = interp1d(da, 'xx', kind='cubic')(da.coords['xx'])
    assert sc.allclose(result.data, da.data, rtol=sc.scalar(3e-3))
    da.masks['mask'] = da.data > sc.scalar(0.8)
    result = interp1d(da, 'xx', kind='cubic')(da.coords['xx'])
    assert sc.allclose(result.data, da.data, rtol=sc.scalar(2e-2))


def test_midpoints():
    da = make_array()
    x = sc.linspace(dim='xx', start=0.1, stop=0.4, num=10, unit='rad')
    out = interp1d(da, 'xx')(x, midpoints=True)
    midpoints = (x[:-1] + 0.5 * (x[1:] - x[:-1])).values
    assert np.array_equal(
        out.values,
        theirs.interp1d(x=da.coords['xx'].values, y=da.values, axis=0)(midpoints))


def test_midpoints_datetime():
    da = make_array().rename_dims({'xx': 'time'})
    x = sc.arange(dim='time',
                  start=0,
                  stop=da.sizes['time'],
                  step=1,
                  unit='s',
                  dtype='datetime64')
    da.coords['time'] = x
    out = interp1d(da, 'time')(da.coords['time'], midpoints=True)
    midpoints = x[:-1].values + 0.5 * (x[1:].values - x[:-1].values)
    assert np.array_equal(
        out.values,
        theirs.interp1d(x=da.coords['time'].values.astype('int64'), y=da.values,
                        axis=0)(midpoints))


@pytest.mark.parametrize("kind", ['nearest', 'quadratic', 'cubic'])
@pytest.mark.parametrize("fill_value", [0.0, 'extrapolate'])
def test_options(kind, fill_value):
    da = make_array()
    x = sc.linspace(dim='xx', start=0.1, stop=0.4, num=10, unit='rad')
    out = interp1d(da, 'xx', kind=kind, fill_value=fill_value)(x)
    assert np.array_equal(
        out.values,
        theirs.interp1d(x=da.coords['xx'].values,
                        y=da.values,
                        axis=0,
                        kind=kind,
                        fill_value=fill_value)(x.values))

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest

import scipp as sc


def test_slice_bins_by_int_label():
    table = sc.data.table_xyz(100)
    table.coords['param'] = (table.coords.pop('y') * 10).to(dtype='int64')
    da = table.bin(x=10)
    param = sc.scalar(4, unit='m')
    result = da.bins['param', param]
    assert result.dims == da.dims
    assert sc.identical(result.attrs['param'], param)
    assert sc.identical(
        result,
        da.group(sc.array(dims=['param'], values=[4], dtype='int64',
                          unit='m')).squeeze())


def test_slice_bins_by_int_label_range():
    table = sc.data.table_xyz(100)
    table.coords['param'] = sc.arange(dim='row', start=0, stop=100, unit='s') // 10
    da = table.bin(x=10)
    start = sc.scalar(4, unit='s')
    stop = sc.scalar(6, unit='s')
    result = da.bins['param', start:stop]
    assert result.dims == da.dims
    assert sc.identical(
        result.attrs['param'],
        sc.array(dims=['param'], values=[4, 6], unit='s', dtype='int64'))
    assert result.bins.size().sum().value == 20  # 2x10 events
    assert sc.identical(
        result,
        da.bin(param=sc.array(dims=['param'], values=[4, 6], unit='s',
                              dtype='int64')).squeeze())


def test_slice_bins_by_float_label_range():
    table = sc.data.table_xyz(100)
    da = table.bin(x=10)
    start = sc.scalar(0.1, unit='m')
    stop = sc.scalar(0.2, unit='m')
    result = da.bins['z', start:stop]
    assert result.dims == da.dims
    assert sc.identical(result.attrs['z'],
                        sc.array(dims=['z'], values=[0.1, 0.2], unit='m'))
    assert sc.identical(
        result,
        da.bin(z=sc.array(dims=['z'], values=[0.1, 0.2], unit='m')).squeeze())


def test_slice_bins_by_open_range_includes_everything():
    table = sc.data.table_xyz(100)
    da = table.bin(x=10)
    result = da.bins['z', :]
    assert result.bins.size().sum().value == 100


def test_slice_bins_by_half_open_int_range_splits_without_duplication():
    table = sc.data.table_xyz(100)
    da = table.bin(x=10)
    split = sc.scalar(0.4, unit='m')
    left = da.bins['z', :split]
    right = da.bins['z', split:]
    assert left.bins.size().sum().value + right.bins.size().sum().value == 100


def test_slice_bins_by_half_open_float_range_splits_without_duplication():
    table = sc.data.table_xyz(100)
    table.coords['param'] = sc.arange(dim='row', start=0, stop=100, unit='s') // 10
    da = table.bin(x=10)
    split = sc.scalar(4, unit='s')
    left = da.bins['param', :split]
    right = da.bins['param', split:]
    assert left.bins.size().sum().value + right.bins.size().sum().value == 100


def test_slice_bins_with_step_raises():
    da = sc.data.table_xyz(100).bin(x=10)
    start = sc.scalar(0.1, unit='m')
    stop = sc.scalar(0.4, unit='m')
    step = sc.scalar(0.1, unit='m')
    with pytest.raises(ValueError):
        da.bins['z', start:stop:step]


def test_slice_bins_with_int_index_raises():
    da = sc.data.table_xyz(100).bin(x=10)
    with pytest.raises(ValueError):
        da.bins['z', 1:4]
    with pytest.raises(ValueError):
        da.bins['z', 1]

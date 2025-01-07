# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from inspect import getfullargspec, isfunction
from numbers import Real
from .core import (
    BinEdgeError,
    DataArray,
    DataGroup,
    Dataset,
    DimensionError,
    UnitError,
    array,
    scalar,
    stddevs,
)
from ._scipp import (
    DataArray as _DataArray,
    Dataset as _Dataset,
    zeros,
    curve_fit as _curve_fit,
)
import pickle
from multiprocessing import Pool


def _get_sigma(da):
    if da.variances is None:
        return None
    return stddevs(da).values


def _get_specific_bounds(bounds, k, unit):
    if k not in bounds:
        return scalar(-float('inf'), unit=unit), scalar(float('inf'), unit=unit)
    b = bounds[k]
    if isinstance(b, tuple):
        if len(b) != 2:
            raise ValueError(
                f'Bounds for parameter {k} must be a tuple of length 2, got {len(b)}'
            )
        return b[0], b[1]
    return scalar(-float('inf'), unit=unit), b


def _parse_bounds(bounds, p0):
    return {k: _get_specific_bounds(bounds or {}, k, v.unit) for k, v in p0.items()}


def _select_data_params_and_bounds(sel, da, p0, bounds):
    dim, i = sel
    return (
        da[sel],
        {k: v[sel] if dim in v.dims else v for k, v in p0.items()},
        {
            k: (le[sel] if dim in le.dims else le, ri[sel] if dim in ri.dims else ri)
            for k, (le, ri) in bounds.items()
        },
    )


def _datagroup_outputs(da, p0, map_over, pdata, covdata):
    variances = covdata.diagonal(dim1='param', dim2='param')
    dg = DataGroup(
        {
            p: DataArray(
                data=array(
                    dims=map_over,
                    values=pdata[p].data,
                    variances=variances[p].data,
                    unit=pdata[p].unit,
                ),
            )
            for i, (p, v0) in enumerate(p0.items())
        },
        coords={
            p: DataGroup(
                {
                    q: DataArray(
                        data=array(
                            dims=map_over,
                            values=covdata[p][q].data,
                            unit=covdata[p][q].unit,
                        ),
                    )
                    for j, (q, u0) in enumerate(p0.items())
                }
            )
            for i, (p, v0) in enumerate(p0.items())
        },
    )
    return dg


def _make_defaults(f, coords, p0):
    params = getfullargspec(f).args
    if p0 is None:
        p0 = {}
    params = getfullargspec(f).args
    for p in params:
        if p not in p0 and p not in coords:
            raise ValueError(
                f'Fit function parameter {p} not found in initial parameters or coords'
            )
    return {k: scalar(1.0, unit=scalar(1.0).unit) if k not in p0 else p0[k] for k in params}


def _curve_fit_chunk(
    coords,
    f,
    da,
    p0,
    bounds,
    map_over,
    unsafe_numpy_f,
    kwargs,
):
    # Create a dataarray with only the participating coords
    _da = DataArray(da.data, coords=coords, masks=da.masks)

    return _curve_fit(
        f=f,
        da=_DataArray(_da),
        p0=_Dataset({k: v for k, v in p0.items()}),
        bounds=_Dataset({k: _Dataset({'lower': le, 'upper': ri}) for k, (le, ri) in bounds.items()}),
        **kwargs,
    )


def curve_fit(
    f: Callable,
    da: DataArray,
    p0: Mapping[str, Real] = None,
    *,
    coords: Sequence[str] = None,
    bounds: Mapping[str, tuple[Real, Real]] = None,
    map_over: Sequence[str] = None,
    workers: int = 1,
    unsafe_numpy_f: bool = False,
    **kwargs,
) -> DataGroup:
    """
    Fit a function to data.

    Parameters
    ----------
    f:
        The function to fit.
    da:
        The data to fit to.
    p0:
        Initial guess for the parameters.
    coords:
        The coordinates to use for the fit.
    bounds:
        Bounds for the parameters.
    map_over:
        The dimensions to map over.
    workers:
        The number of workers to use for parallelization.
    unsafe_numpy_f:
        If True, the function `f` is assumed to be a numpy function and
        will be called with numpy arrays instead of scipp variables.
    kwargs:
        Additional keyword arguments to pass to the fitting function.

    Returns
    -------
    :
        A DataGroup containing the fit parameters and their covariance matrix.
    """
    if coords is None:
        coords = list(da.coords)
    if map_over is None:
        map_over = []
    if not isinstance(map_over, list):
        map_over = list(map_over)
    if not isinstance(coords, list):
        coords = list(coords)

    reduce_dims = [
        d
        for d in da.dims
        if d not in map_over and not any(d in c.dims for c in da.coords.values())
    ]

    if len(reduce_dims) > 0:
        da_reduced = da.group(reduce_dims)
    else:
        da_reduced = da

    p0 = _make_defaults(f, coords, p0)
    bounds = _parse_bounds(bounds, p0)
    popt, pcov = _curve_fit_chunk(coords, f, da_reduced, p0, bounds, map_over, unsafe_numpy_f, kwargs)
    return _datagroup_outputs(da_reduced, p0, map_over, popt, pcov)


__all__ = ['curve_fit']

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from inspect import getfullargspec, isfunction
from numbers import Real
import numpy as np
from .core import (
    BinEdgeError,
    DataArray,
    DataGroup,
    DimensionError,
    Variable,
    array,
    scalar,
    stddevs,
)
from ._scipp import (
    _curve_fit as cpp_curve_fit,
    DataArray as _DataArray,
    Dataset as _Dataset,
    zeros,
)


def _wrap_scipp_func(f, p0):
    p = {k: scalar(0.0, unit=v.unit) for k, v in p0.items()}

    def func(x, *args):
        for k, v in zip(p, args, strict=True):
            p[k].value = v
        return f(**x, **p).values

    return func


def _wrap_numpy_func(f, param_names, coord_names):
    def func(x, *args):
        # If there is only one predictor variable x might be a 1D array.
        # Make x 2D for consistency.
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        c = dict(zip(coord_names, x, strict=True))
        p = dict(zip(param_names, args, strict=True))
        return f(**c, **p)

    return func


def _get_sigma(da):
    if da.variances is None:
        return None

    sigma = stddevs(da).values
    if not sigma.all():
        raise ValueError(
            'There is a 0 in the input variances. This would break the optimizer. '
            'Mask the offending elements, remove them, or assign a meaningful '
            'variance if possible before calling curve_fit.'
        )
    return sigma


def _datagroup_outputs(da, p0, map_over, pdata, covdata):
    variances = covdata.diagonal(dim1='param', dim2='param')
    dg = DataGroup(
        {
            p: DataArray(
                dims=map_over,
                values=pdata[p].data,
                variances=variances[p].data,
                unit=pdata[p].unit,
            ),
        }
        for i, (p, v0) in enumerate(p0.items())
    )
    dgcov = DataGroup(
        {
            p: DataGroup(
                {
                    q: DataArray(
                        dims=map_over,
                        values=covdata[p][q].data,
                        unit=covdata[p][q].unit,
                    )
                    for j, (q, u0) in enumerate(p0.items())
                }
            )
            for i, (p, v0) in enumerate(p0.items())
        }
    )
    for m in da.masks:
        for p in dg:
            dg[p].masks[m] = da.masks[m]
        for p in dgcov:
            for q in dgcov[p]:
                dgcov[p][q].masks[m] = da.masks[m]
    return dg, dgcov


def _make_defaults(f, coords, p0):
    if p0 is None:
        p0 = {}
    params = getfullargspec(f).args
    if 'x' in params:
        params.remove('x')
    for c in coords:
        if c in params:
            params.remove(c)
    for p in params:
        if p not in p0:
            p0[p] = scalar(1.0)
    return p0


def _get_specific_bounds(bounds, k, unit):
    if k in bounds:
        b = bounds[k]
        if isinstance(b, tuple):
            return (scalar(b[0], unit=unit), scalar(b[1], unit=unit))
        else:
            raise ValueError(
                f'Bounds for parameter {k} must be a tuple of (lower, upper) bounds'
            )
    return (scalar(-np.inf, unit=unit), scalar(np.inf, unit=unit))


def _parse_bounds(bounds, p0):
    return {k: _get_specific_bounds(bounds or {}, k, v.unit) for k, v in p0.items()}


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
    f = (
        _wrap_scipp_func(f, p0)
        if not unsafe_numpy_f
        else _wrap_numpy_func(f, list(p0.keys()), list(coords.keys()))
    )
    # Create a dataarray with only the participating coords
    _da = DataArray(da.data, coords=coords, masks=da.masks)

    return cpp_curve_fit(
        f=f,
        da=_DataArray(_da),
        p0=_Dataset({k: v for k, v in p0.items()}),
        bounds=_Dataset(
            {
                k: _DataArray(array(dims=['bound'], values=[b[0].value, b[1].value], unit=b[0].unit))
                for k, b in bounds.items()
            }
        ),
        map_over=map_over,
        unsafe_numpy_f=unsafe_numpy_f,
        **kwargs,
    )


def _to_data_array_dict(dataset):
    return {name: DataArray(var) for name, var in dataset.items()}


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
) -> tuple[DataGroup, DataGroup]:
    """
    Fit a function to the data.

    Parameters
    ----------
    f:
        The function to fit.
    da:
        The data to fit to.
    p0:
        Initial guess for the parameters.
    coords:
        The coordinates to use as the independent variables.
    bounds:
        The bounds for the parameters.
    map_over:
        The dimensions to map over.
    workers:
        The number of workers to use for parallelization.
    unsafe_numpy_f:
        If True, the function f is assumed to be a numpy function.
        This avoids some overhead but is unsafe if f is not a numpy function.
    kwargs:
        Additional keyword arguments to pass to the optimizer.

    Returns
    -------
    :
        A tuple containing the fitted parameters and the covariance matrix.
    """
    if coords is None:
        coords = list(da.coords)
    coords = {c: da.coords[c] for c in coords}
    if map_over is None:
        map_over = []
    reduce_dims = [
        d
        for d in da.dims
        if d not in map_over and not any(d in c.dims for c in coords.values())
    ]

    if len(reduce_dims) > 0:
        da_reduced = da.group(reduce_dims)
    else:
        da_reduced = da

    p0 = _make_defaults(f, coords.keys(), p0)
    bounds = _parse_bounds(bounds, p0)
    popt, pcov = _curve_fit_chunk(coords, f, da_reduced, p0, bounds, map_over, unsafe_numpy_f, kwargs)
    return _datagroup_outputs(da_reduced, p0, map_over, popt, pcov)


__all__ = ['curve_fit']

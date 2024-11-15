# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from inspect import getfullargspec, isfunction
from numbers import Real
import warnings

import numpy as np
import dask
from dask.delayed import delayed
import dask.array as da

from .core import (
    BinEdgeError,
    DataArray,
    DataGroup,
    DimensionError,
    Variable,
    array,
    scalar,
    stddevs,
    zeros,
)
from .curve_fit import (
    _wrap_scipp_func,
    _wrap_numpy_func,
    _get_sigma,
    _make_defaults,
    _parse_bounds,
    _reshape_bounds,
    _datagroup_outputs,
)


def _parallel_curve_fit_chunk(
    f, data, coords, p0_values, bounds, sigma=None, **kwargs
):
    """Execute curve_fit on a single chunk of data"""
    import scipy.optimize as opt
    
    try:
        popt, pcov = opt.curve_fit(
            f,
            coords,
            data,
            p0_values,
            sigma=sigma,
            bounds=bounds,
            **kwargs,
        )
    except RuntimeError as err:
        if hasattr(err, 'message') and 'Optimal parameters not found:' in err.message:
            popt = np.array([np.nan for _ in p0_values])
            pcov = np.array([[np.nan for _ in p0_values] for _ in p0_values])
        else:
            raise err
            
    return popt, pcov


def _prepare_parallel_fit(
    f, da, p0, bounds, map_over, unsafe_numpy_f, chunks='auto'
):
    """Prepare data structures for parallel fitting"""
    if len(da.masks) > 0:
        _mask = zeros(dims=da.dims, shape=da.shape, dtype='bool')
        for mask in da.masks.values():
            _mask |= mask
        da = da[~_mask]

    if not unsafe_numpy_f:
        X = dict(da.coords)
    else:
        X = np.vstack([c.values for c in da.coords.values()], dtype='float')

    p0_values = [v.value for v in p0.values()]
    bounds = _reshape_bounds(bounds)
    sigma = _get_sigma(da)
    
    return X, p0_values, bounds, sigma


def curve_fit_parallel(
    coords: Sequence[str] | Mapping[str, str | Variable],
    f: Callable,
    da: DataArray,
    *,
    p0: dict[str, Variable | Real] | None = None,
    bounds: dict[str, tuple[Variable, Variable] | tuple[Real, Real]] | None = None,
    reduce_dims: Sequence[str] = (),
    unsafe_numpy_f: bool = False,
    n_workers: int | None = None,
    chunks: str | int = 'auto',
    **kwargs,
) -> tuple[DataGroup, DataGroup]:
    """Parallel version of curve_fit using Dask for distributed computation.
    
    This function provides the same interface as curve_fit but performs the fitting
    operation in parallel using Dask. All parameters are identical to curve_fit
    with the following additions:
    
    Parameters
    ----------
    n_workers:
        Number of worker processes to use. If None, Dask will use the number of
        CPU cores available.
    chunks:
        How to chunk the data for parallel processing. Can be 'auto' for automatic
        chunking, or an integer specifying the chunk size.
        
    See Also
    --------
    scipp.curve_fit:
        The serial version of this function with detailed documentation.
    """
    if 'jac' in kwargs:
        raise NotImplementedError(
            "The 'jac' argument is not yet supported. "
            "See https://github.com/scipp/scipp/issues/2544"
        )

    for arg in ['xdata', 'ydata', 'sigma']:
        if arg in kwargs:
            raise TypeError(
                f"Invalid argument '{arg}', already defined by the input data array."
            )

    for c in coords:
        if c in da.coords and da.coords.is_edges(c):
            raise BinEdgeError("Cannot fit data array with bin-edge coordinate.")

    if not isinstance(coords, dict):
        if not all(isinstance(c, str) for c in coords):
            raise TypeError(
                'Expected sequence of coords to only contain values of type `str`.'
            )
        coords = {c: c for c in coords}

    coords = {
        arg: da.coords[coord] if isinstance(coord, str) else coord
        for arg, coord in coords.items()
    }

    p0 = _make_defaults(f, coords.keys(), p0)
    f = (
        _wrap_scipp_func(f, p0)
        if not unsafe_numpy_f
        else _wrap_numpy_func(f, p0, coords.keys())
    )

    map_over = tuple(
        d
        for d in da.dims
        if d not in reduce_dims and not any(d in c.dims for c in coords.values())
    )

    _da = DataArray(da.data, coords=coords, masks=da.masks)
    
    X, p0_values, bounds, sigma = _prepare_parallel_fit(
        f, _da, p0, _parse_bounds(bounds, p0), map_over, unsafe_numpy_f, chunks
    )

    # Create delayed objects for parallel computation
    if len(map_over) > 0:
        shape = [da.sizes[d] for d in map_over]
        results = []
        
        for idx in np.ndindex(*shape):
            slice_dict = {dim: i for dim, i in zip(map_over, idx)}
            sliced_da = _da[slice_dict]
            
            if not unsafe_numpy_f:
                slice_X = {k: v[slice_dict] if any(d in v.dims for d in map_over) else v 
                          for k, v in X.items()}
            else:
                slice_X = np.vstack([c[slice_dict].values if any(d in c.dims for d in map_over) else c.values 
                                   for c in coords.values()], dtype='float')
            
            slice_sigma = sigma[slice_dict] if sigma is not None else None
            
            delayed_fit = delayed(_parallel_curve_fit_chunk)(
                f, sliced_da.data.values, slice_X, p0_values, 
                bounds, sigma=slice_sigma, **kwargs
            )
            results.append(delayed_fit)
        
        # Compute all fits in parallel
        with dask.config.set(num_workers=n_workers):
            computed_results = dask.compute(*results)
        
        # Reshape results into the original array shape
        popt_array = np.empty(shape + [len(p0)])
        pcov_array = np.empty(shape + [len(p0), len(p0)])
        
        for idx, (popt, pcov) in zip(np.ndindex(*shape), computed_results):
            popt_array[idx] = popt
            pcov_array[idx] = pcov
    else:
        # Single fit case
        popt, pcov = _parallel_curve_fit_chunk(
            f, _da.data.values, X, p0_values, bounds, 
            sigma=sigma, **kwargs
        )
        popt_array = popt
        pcov_array = pcov

    return _datagroup_outputs(da, p0, map_over, popt_array, pcov_array)

__all__ = ['curve_fit_parallel']

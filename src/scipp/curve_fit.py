# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Mapping, Sequence
from functools import partial
 from inspect import getfullargspec, isfunction
 from numbers import Real
-import numpy as np
 from .core import (
     BinEdgeError,
     DataArray,
@@ -27,7 +24,6 @@
     stddevs,
 )
 from ._scipp import (
-    _curve_fit as cpp_curve_fit,
     DataArray as _DataArray,
     Dataset as _Dataset,
     zeros,
@@ -66,7 +62,7 @@
 
 
 def _datagroup_outputs(da, p0, map_over, pdata, covdata):
-    variances = np.diagonal(covdata, axis1=-2, axis2=-1)
+    variances = covdata.diagonal(dim1='param', dim2='param')
     dg = DataGroup(
         {
             p: DataArray(
@@ -74,7 +70,7 @@
                     dims=map_over,
                     values=pdata[p].data,
                     variances=variances[p].data,
-                    unit=v0.unit,
+                    unit=pdata[p].unit,
                 ),
             )
             for i, (p, v0) in enumerate(p0.items())
@@ -87,9 +83,7 @@
                         data=array(
                             dims=map_over,
                             values=covdata[p][q].data,
-                            if covdata.ndim > 2
-                            else covdata[i, j],
-                            unit=v0.unit * u0.unit,
+                            unit=covdata[p][q].unit,
                         ),
                     )
                     for j, (q, u0) in enumerate(p0.items())
@@ -109,7 +103,7 @@
 
 
 def _make_defaults(f, coords, p0):
-    spec = getfullargspec(f)
+    params = getfullargspec(f).args
     if p0 is None:
         p0 = {}
     params = getfullargspec(f).args
@@ -175,10 +169,6 @@
     return {k: _get_specific_bounds(bounds or {}, k, v.unit) for k, v in p0.items()}
 
 
-def _reshape_bounds(bounds):
-    left, right = zip(*bounds.values(), strict=True)
-    left, right = [le.value for le in left], [ri.value for ri in right]
-    if all(le == -np.inf and ri == np.inf for le, ri in zip(left, right, strict=True)):
-        return -np.inf, np.inf
-    return left, right
-
-
 def _select_data_params_and_bounds(sel, da, p0, bounds):
     dim, i = sel
     return (
@@ -195,115 +185,6 @@
         },
     )
 
-
-def _serialize_variable(v):
-    return (v.dims, v.values, v.variances, str(v.unit) if v.unit is not None else None)
-
-
-def _serialize_mapping(v):
-    return (tuple(v.keys()), tuple(map(_serialize_variable, v.values())))
-
-
-def _serialize_bounds(v):
-    return (
-        tuple(v.keys()),
-        tuple(tuple(map(_serialize_variable, pair)) for pair in v.values()),
-    )
-
-
-def _serialize_data_array(da):
-    return (
-        _serialize_variable(da.data),
-        _serialize_mapping(da.coords),
-        _serialize_mapping(da.masks),
-    )
-
-
-def _deserialize_variable(t):
-    return array(dims=t[0], values=t[1], variances=t[2], unit=t[3])
-
-
-def _deserialize_data_array(t):
-    return DataArray(
-        _deserialize_variable(t[0]),
-        coords=_deserialize_mapping(t[1]),
-        masks=_deserialize_mapping(t[2]),
-    )
-
-
-def _deserialize_mapping(t):
-    return dict(zip(t[0], map(_deserialize_variable, t[1]), strict=True))
-
-
-def _deserialize_bounds(t):
-    return dict(
-        zip(
-            t[0],
-            (tuple(map(_deserialize_variable, pair)) for pair in t[1]),
-            strict=True,
-        )
-    )
-
-
-def _curve_fit(
-    f,
-    da,
-    p0,
-    bounds,
-    map_over,
-    unsafe_numpy_f,
-    out,
-    **kwargs,
-):
-    dg, dgcov = out
-
-    if len(map_over) > 0:
-        dim = map_over[0]
-        for i in range(da.sizes[dim]):
-            _curve_fit(
-                f,
-                *_select_data_params_and_bounds((dim, i), da, p0, bounds),
-                map_over[1:],
-                unsafe_numpy_f,
-                (dg[i], dgcov[i]),
-                **kwargs,
-            )
-
-        return
-
-    for k, v in p0.items():
-        if v.shape != ():
-            raise DimensionError(f'Parameter {k} has unexpected dimensions {v.dims}')
-
-    for k, (le, ri) in bounds.items():
-        if le.shape != ():
-            raise DimensionError(
-                f'Left bound of parameter {k} has unexpected dimensions {le.dims}'
-            )
-        if ri.shape != ():
-            raise DimensionError(
-                f'Right bound of parameter {k} has unexpected dimensions {ri.dims}'
-            )
-
-    fda = da.flatten(to='row')
-    if len(fda.masks) > 0:
-        _mask = zeros(dims=fda.dims, shape=fda.shape, dtype='bool')
-        for mask in fda.masks.values():
-            _mask |= mask
-        fda = fda[~_mask]
-
-    if not unsafe_numpy_f:
-        # Making the coords into a dict improves runtime,
-        # probably because of pybind overhead.
-        X = dict(fda.coords)
-    else:
-        X = np.vstack([c.values for c in fda.coords.values()], dtype='float')
-
-    import scipy.optimize as opt
-
-    if len(fda) < len(dg):
-        # More parameters than data points, unable to fit, abort.
-        dg[:] = np.nan
-        dgcov[:] = np.nan
-        return
-
-    try:
-        popt, pcov = opt.curve_fit(
-            f=f,
-            xdata=X,
-            ydata=fda.data.values,
-            p0=[v.value for v in p0.values()],
-            sigma=_get_sigma(fda),
-            bounds=_reshape_bounds(bounds),
-            **kwargs,
-        )
-    except RuntimeError as err:
-        if hasattr(err, 'message') and 'Optimal parameters not found:' in err.message:
-            popt = np.array([np.nan for p in p0])
-            pcov = np.array([[np.nan for q in p0] for p in p0])
-        else:
-            raise err
-
-    dg[:] = popt
-    dgcov[:] = pcov
-
-
 def _curve_fit_chunk(
     coords,
     f,
@@ -314,7 +195,7 @@
     # Create a dataarray with only the participating coords
     _da = DataArray(da.data, coords=coords, masks=da.masks)
 
-    return cpp_curve_fit(
+    return _scipp._curve_fit(
         f=f,
         da=_DataArray(_da),
         p0=_Dataset({k: v for k, v in p0.items()}),
@@ -360,37 +241,13 @@
         if d not in map_over and not any(d in c.dims for c in coords.values())
     ]
 
+
     if len(reduce_dims) > 0:
         da_reduced = da.group(reduce_dims)
     else:
         da_reduced = da
 
     p0 = _make_defaults(f, coords.keys(), p0)
-    bounds = _parse_bounds(bounds, p0)
-
-    pardim = None
-    if len(map_over) > 0:
-        max_size = max((da.sizes[dim] for dim in map_over))
-        max_size_dim = next((dim for dim in map_over if da.sizes[dim] == max_size))
-        # Parallelize over longest dim because that is most likely
-        # to give us a balanced workload over the workers.
-        pardim = max_size_dim if max_size > 1 else None
-
-    # Only parallelize if the user asked for more than one worker
-    # and a suitable dimension for parallelization was found.
-    if workers != 1 and pardim is not None:
-        try:
-            pickle.dumps(f)
-        except (AttributeError, pickle.PicklingError) as err:
-            raise ValueError(
-                'The provided fit function is not pickleable and can not be used '
-                'with the multiprocessing module. '
-                'Either provide a function that is compatible with pickle '
-                'or explicitly disable multiprocess parallelism by passing '
-                'workers=1.'
-            ) from err
-
-        chunksize = (da.sizes[pardim] // workers) + 1
-        args = []
-        for i in range(workers):
-            _da, _p0, _bounds = _select_data_params_and_bounds(
-                (pardim, slice(i * chunksize, (i + 1) * chunksize)), da, p0, bounds
-            )
-            args.append(
-                (
-                    _serialize_mapping(coords),
-                    f,
-                    _serialize_data_array(_da),
-                    _serialize_mapping(_p0),
-                    _serialize_bounds(_bounds),
-                    map_over,
-                    unsafe_numpy_f,
-                    kwargs,
-                )
-            )
-
-        with Pool(workers) as pool:
-            par, cov = zip(*pool.starmap(_curve_fit_chunk, args), strict=True)
-            concat_axis = map_over.index(pardim)
-            par = np.concatenate(par, axis=concat_axis)
-            cov = np.concatenate(cov, axis=concat_axis)
-    else:
-        par, cov = _curve_fit_chunk(
-            coords=coords,
-            f=f,
-            da=da,
-            p0=p0,
-            bounds=bounds,
-            map_over=map_over,
-            unsafe_numpy_f=unsafe_numpy_f,
-            kwargs=kwargs,
-        )
-
-    return _datagroup_outputs(da, p0, map_over, par, cov)
+    bounds = _parse_bounds(bounds, p0)    
+    popt, pcov = _curve_fit_chunk(coords, f, da_reduced, p0, bounds, map_over, unsafe_numpy_f, kwargs)
+    return _datagroup_outputs(da_reduced, p0, map_over, popt, pcov)
 
 
 __all__ = ['curve_fit']

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from functools import partial

import numpy as np
import pytest

import scipp as sc
from scipp import curve_fit
from scipp.compat.xarray_compat import from_xarray, to_xarray


def func(x, a, b):
    return a * sc.exp(-b * x)


def func2d(x, t, a, b):
    return a * sc.exp(-b * t / (1 + x))


def func3d(x, t, y, a, b):
    return a * y * sc.exp(-b * t / (1 + x))


def func_np(x, a, b):
    return a * np.exp(-b * x)


def func2d_np(x, t, a, b):
    return a * np.exp(-b * t / (1 + x))


def func3d_np(x, t, y, a, b):
    return a * y * np.exp(-b * t / (1 + x))


def array(coords, f, params, noise_scale, seed=1234):
    rng = np.random.default_rng(seed)
    da_coords = {
        c: sc.linspace(
            dim=kw.pop('dim', c),
            **kw,
        )
        for c, kw in coords.items()
    }
    data = f(**da_coords, **params)
    # Noise is random but avoiding unbounded values to avoid flaky tests
    data.values += noise_scale * np.clip(rng.normal(size=data.values.shape), -1.5, 1.5)
    return sc.DataArray(data, coords=da_coords)


def array1d(*, a=1.2, b=1.3, noise_scale=0.1, size=50):
    return array(
        dict(x=dict(dim='xx', start=-0.1, stop=4.0, num=size)),
        func,
        dict(a=a, b=b),
        noise_scale=noise_scale,
    )


def array2d(*, a=1.2, b=1.3, noise_scale=0.1, size=20):
    return array(
        dict(
            x=dict(dim='xx', start=-0.1, stop=4.0, num=size),
            t=dict(dim='tt', start=0.0, stop=1.0, num=size // 2),
        ),
        func2d,
        dict(a=a, b=b),
        noise_scale=noise_scale,
    )


def array3d(*, a=1.2, b=1.3, noise_scale=0.1, size=10):
    return array(
        dict(
            x=dict(dim='xx', start=-0.1, stop=4.0, num=size),
            t=dict(dim='tt', start=0.0, stop=1.0, num=size // 2),
            y=dict(dim='yy', start=2.0, stop=4.0, num=size),
        ),
        func3d,
        dict(a=a, b=b),
        noise_scale=noise_scale,
    )


def array1d_from_vars(*, a, b, noise_scale=0.1, size=50):
    return array(
        dict(x=dict(dim='xx', start=0.1, stop=4.0, num=size, unit='m')),
        func,
        dict(a=a, b=b),
        noise_scale=noise_scale,
    )


def test_should_not_raise_given_function_with_dimensionless_params_and_1d_array():
    curve_fit(['x'], func, array1d())


@pytest.mark.parametrize(
    "p0, params, bounds",
    (
        (None, dict(a=1.2, b=1.3), None),
        (dict(a=3, b=2), dict(a=1.2, b=1.3), None),
        (dict(a=0.2, b=-1), dict(a=1.2, b=1.3), {'a': (-3, 3), 'b': (-2, 2)}),
        (dict(a=0.2, b=-1), dict(a=1.2, b=1.3), {'a': (-3, 1.1), 'b': (-2, 1.1)}),
    ),
)
@pytest.mark.parametrize(
    "dims",
    (
        dict(x=10, t=10, y=10),
        dict(x=5, t=8, y=7),
    ),
)
@pytest.mark.parametrize(
    "coords, reduce_dims",
    (
        (['x'], []),
        (['x'], ['y']),
        (['x'], ['t', 'y']),
        (['x', 't'], []),
        (['x', 't'], ['y']),
        (['x', 't', 'y'], []),
    ),
)
def test_compare_to_curve_fit_xarray(dims, coords, reduce_dims, p0, params, bounds):
    f, fxarray = {
        1: (func, func_np),
        2: (func2d, lambda x, a, b: func2d_np(*x, a, b)),
        3: (func3d, lambda x, a, b: func3d_np(*x, a, b)),
    }[len(coords)]
    da = array(
        {c: dict(start=1, stop=3, num=n) for c, n in dims.items()},
        lambda **x: (
            sc.broadcast(
                f(**{c: x[c] for c in x if c in coords or c in params}),
                sizes=dims,
            ).copy()
        ),
        params,
        noise_scale=0.0,
    )

    result, _ = curve_fit(
        coords,
        f,
        da,
        p0=p0,
        bounds=bounds,
        reduce_dims=reduce_dims,
    )
    xresult = from_xarray(
        to_xarray(da).curvefit(
            coords,
            fxarray,
            p0=p0,
            bounds=bounds,
            reduce_dims=reduce_dims,
        )['curvefit_coefficients']
    )
    for param_name in result:
        assert sc.allclose(
            result[param_name].data,
            xresult['param', sc.scalar(param_name)].data,
        )
        if (
            bounds is None
            or bounds[param_name][0] <= params[param_name] <= bounds[param_name][1]
        ):
            assert sc.allclose(result[param_name].data, sc.scalar(params[param_name]))


def test_dimensions_present_in_reduce_dims_argument_are_not_present_in_output():
    popt, _ = curve_fit(['x'], func3d, array3d())
    assert 'tt' in popt['a'].dims
    assert 'yy' in popt['a'].dims

    popt, _ = curve_fit(['x'], func3d, array3d(), reduce_dims=['tt'])
    assert 'tt' not in popt['a'].dims
    assert 'yy' in popt['a'].dims

    popt, _ = curve_fit(['x'], func3d, array3d(), reduce_dims=['tt', 'yy'])
    assert 'tt' not in popt['a'].dims
    assert 'yy' not in popt['a'].dims


def test_should_not_raise_given_function_with_dimensionful_params_and_1d_array():
    curve_fit(
        ['x'],
        func,
        array1d_from_vars(a=sc.scalar(1.2, unit='s'), b=sc.scalar(100.0, unit='1/m')),
        p0={'a': sc.scalar(1.0, unit='s'), 'b': sc.scalar(1.0, unit='1/m')},
    )


def test_should_raise_TypeError_when_xdata_given_as_param():
    with pytest.raises(TypeError):
        curve_fit(['x'], func, array1d(), xdata=np.arange(4))


def test_should_raise_TypeError_when_ydata_given_as_param():
    with pytest.raises(TypeError):
        curve_fit(['x'], func, array1d(), ydata=np.arange(4))


def test_should_raise_TypeError_when_sigma_given_as_param():
    with pytest.raises(TypeError):
        curve_fit(['x'], func, array1d(), sigma=np.arange(4))


def test_should_raise_ValueError_when_sigma_contains_zeros():
    da = array1d(size=50)
    da.variances = np.random.default_rng().normal(0.0, 0.1, size=50) ** 2
    da['xx', 21].variance = 0.0
    with pytest.raises(ValueError):
        curve_fit(['x'], func, da)


def test_does_not_raise_when_sigma_contains_zeros_that_is_masked():
    da = array1d(size=50)
    da.variances = np.random.default_rng().normal(0.0, 0.1, size=50) ** 2
    da.masks['m'] = sc.full(value=False, sizes=da.sizes)
    da['xx', 21].variance = 0.0
    da.masks['m']['xx', 21] = True
    curve_fit(['x'], func, da)


def test_should_raise_KeyError_when_data_array_has_no_coord():
    da = array1d()
    for c in tuple(da.coords):
        del da.coords[c]
    with pytest.raises(KeyError):
        curve_fit(['x'], func, da)


def test_should_raise_BinEdgeError_when_data_array_is_histogram():
    da = array1d()
    hist = da[1:].copy()
    hist.coords['x'] = da.coords['x']
    with pytest.raises(sc.BinEdgeError):
        curve_fit(['x'], func, hist)


def test_masks_are_not_ignored():
    da = array1d(size=20)
    unmasked, _ = curve_fit(['x'], func, da)
    da.masks['mask'] = sc.zeros(sizes=da.sizes, dtype=bool)
    da.masks['mask'][-5:] = sc.scalar(True)
    masked, _ = curve_fit(['x'], func, da)
    assert not sc.identical(masked['a'], unmasked['a'])
    assert not sc.identical(masked['b'], unmasked['b'])


@pytest.mark.parametrize(
    "f,array,coords",
    (
        (func, array1d, ['x']),
        (func2d, array2d, ['x', 't']),
        (func3d, array3d, ['x', 't', 'y']),
    ),
)
@pytest.mark.parametrize(
    "noise_scale",
    [1e-1, 1e-2, 1e-3, 1e-6, 1e-9],
)
def test_optimized_params_approach_real_params_as_data_noise_decreases(
    noise_scale, f, array, coords
):
    popt, _ = curve_fit(coords, f, array(a=1.7, b=1.5, noise_scale=noise_scale))
    assert sc.allclose(
        popt['a'].data, sc.scalar(1.7), rtol=sc.scalar(2.0 * noise_scale)
    )
    assert sc.allclose(
        popt['b'].data, sc.scalar(1.5), rtol=sc.scalar(2.0 * noise_scale)
    )


@pytest.mark.parametrize(
    "f,array,coords",
    (
        (func3d, array3d, ['x', 't', 'y']),
        (func3d, array3d, ['y', 'x', 't']),
        (func3d, array3d, ['t', 'y', 'x']),
    ),
)
@pytest.mark.parametrize(
    "noise_scale",
    [1e-1, 1e-2, 1e-3, 1e-6, 1e-9],
)
def test_order_of_coords_does_not_matter(noise_scale, f, array, coords):
    popt, _ = curve_fit(coords, f, array(a=1.7, b=1.5, noise_scale=noise_scale))
    assert sc.allclose(
        popt['a'].data, sc.scalar(1.7), rtol=sc.scalar(2.0 * noise_scale)
    )
    assert sc.allclose(
        popt['b'].data, sc.scalar(1.5), rtol=sc.scalar(2.0 * noise_scale)
    )


@pytest.mark.parametrize(
    "f,fnp,array,coords",
    (
        (func, func_np, array1d, ['x']),
        (func2d, func2d_np, array2d, ['x', 't']),
        (func3d, func3d_np, array3d, ['x', 't', 'y']),
    ),
)
def test_scipp_fun_and_numpy_fun_finds_same_optimized_params(f, fnp, array, coords):
    data = array(a=1.7, b=1.5, noise_scale=1e-2)
    popt, _ = curve_fit(coords, f, data)
    popt_np, _ = curve_fit(coords, fnp, data, unsafe_numpy_f=True)

    for p, p_np in zip(popt.values(), popt_np.values()):
        assert sc.allclose(p.data, p_np.data, rtol=sc.scalar(2.0 * 1e-2))


def test_optimized_params_variances_are_diag_of_covariance_matrix():
    popt, pcov = curve_fit(['x'], func, array1d(a=1.7, b=1.5))
    assert popt['a'].variances == pcov['a']['a'].data.values
    assert popt['b'].variances == pcov['b']['b'].data.values


@pytest.mark.parametrize("mask_pos", [0, 1, -3])
@pytest.mark.parametrize("mask_size", [1, 2])
def test_masked_points_are_treated_as_if_they_were_removed(mask_pos, mask_size):
    da = array1d(size=10)
    da.masks['mask'] = sc.zeros(sizes=da.sizes, dtype=bool)
    da.masks['mask'][mask_pos : mask_pos + mask_size] = sc.scalar(True)
    masked, _ = curve_fit(['x'], func, da)
    removed, _ = curve_fit(
        ['x'], func, sc.concat([da[:mask_pos], da[mask_pos + mask_size :]], da.dim)
    )
    assert sc.identical(masked['a'], removed['a'])
    assert sc.identical(masked['b'], removed['b'])


@pytest.mark.parametrize(
    "variance,expected",
    [(1e9, 1.0), (1, 2.0), (1 / 3, 3.0), (1e-9, 5.0)],
    ids=['disabled', 'equal', 'high', 'dominant'],
)
def test_variances_determine_weights(variance, expected):
    x = sc.array(dims=['x'], values=[1, 2, 3, 4])
    y = sc.array(
        dims=['x'], values=[1.0, 5.0, 1.0, 1.0], variances=[1.0, 1.0, 1.0, 1.0]
    )
    da = sc.DataArray(data=y, coords={'x': x})
    da.variances[1] = variance
    # Fit a constant to highlight influence of weights
    popt, _ = curve_fit(['x'], lambda x, *, a: sc.scalar(a), da)
    assert popt['a'].value == pytest.approx(expected)


def test_fit_function_with_dimensionful_params_raises_UnitError_when_no_p0_given():
    def f(x, *, a, b):
        return a * sc.exp(-b * x)

    with pytest.raises(sc.UnitError):
        curve_fit(
            ['x'],
            f,
            array1d_from_vars(
                a=sc.scalar(1.2, unit='s'), b=sc.scalar(100.0, unit='1/m')
            ),
        )


def test_fit_function_with_dimensionful_params_yields_outputs_with_units():
    def f(x, *, a, b):
        return a * sc.exp(-b * x)

    x = sc.linspace(dim='x', start=0.5, stop=2.0, num=10, unit='m')
    da = sc.DataArray(f(x, a=1.2, b=1.3 / sc.Unit('m')), coords={'x': x})
    popt, pcov = curve_fit(['x'], f, da, p0={'a': 1.1, 'b': 1.2 / sc.Unit('m')})
    assert popt['a'].unit == sc.units.one
    assert popt['b'].unit == sc.Unit('1/m')
    assert not isinstance(pcov['a']['a'], sc.Variable)
    assert pcov['a']['b'].unit == sc.Unit('1/m')
    assert pcov['b']['a'].unit == sc.Unit('1/m')
    assert pcov['b']['b'].unit == sc.Unit('1/m**2')


def test_default_params_with_initial_guess_are_used_for_fit():
    noise_scale = 1e-3
    popt, _ = curve_fit(
        ['x'],
        partial(func, b=1.5),
        array1d(a=1.7, b=1.5, noise_scale=noise_scale),
        p0={'b': 1.1},
    )
    assert sc.allclose(
        popt['a'].data, sc.scalar(1.7), rtol=sc.scalar(2.0 * noise_scale)
    )
    assert sc.allclose(
        popt['b'].data, sc.scalar(1.5), rtol=sc.scalar(2.0 * noise_scale)
    )


def test_bounds_limit_param_range_without_units():
    data = array1d(a=40.0, b=30.0)
    unconstrained, _ = curve_fit(['x'], func, data, p0={'a': 1.0, 'b': 1.0})
    # Fit approaches correct value more closely than with the bound below.
    assert sc.abs(unconstrained['a']).value > 3.0
    assert sc.abs(unconstrained['b']).value > 2.0

    constrained, _ = curve_fit(
        ['x'],
        func,
        data,
        p0={'a': 1.0, 'b': 1.0},
        bounds={'a': (-3, 3), 'b': (sc.scalar(-2), sc.scalar(2))},
    )
    assert sc.abs(constrained['a']).value < 3.0
    assert sc.abs(constrained['b']).value < 2.0


def test_bounds_limit_param_range_with_units():
    data = array1d_from_vars(a=sc.scalar(20.0, unit='s'), b=sc.scalar(10.0, unit='1/m'))
    unconstrained, _ = curve_fit(
        ['x'],
        func,
        data,
        p0={'a': sc.scalar(1.0, unit='s'), 'b': sc.scalar(1.0, unit='1/m')},
    )
    # Fit approaches correct value more closely than with the bound below.
    assert (abs(unconstrained['a']) > sc.scalar(3.0, unit='s')).value
    assert (abs(unconstrained['b']) > sc.scalar(2.0, unit='1/m')).value

    constrained, _ = curve_fit(
        ['x'],
        func,
        data,
        p0={'a': sc.scalar(1.0, unit='s'), 'b': sc.scalar(1.0, unit='1/m')},
        bounds={
            'a': (sc.scalar(-3.0, unit='s'), sc.scalar(3.0, unit='s')),
            'b': (sc.scalar(-2, unit='1/m'), sc.scalar(2, unit='1/m')),
        },
    )

    assert (abs(constrained['a']) < sc.scalar(3.0, unit='s')).value
    assert (abs(constrained['b']) < sc.scalar(2.0, unit='1/m')).value


def test_bounds_limit_only_given_parameters_param_range():
    data = array1d_from_vars(a=sc.scalar(20.0, unit='s'), b=sc.scalar(10.0, unit='1/m'))
    unconstrained, _ = curve_fit(
        ['x'],
        func,
        data,
        p0={'a': sc.scalar(1.0, unit='s'), 'b': sc.scalar(1.0, unit='1/m')},
    )

    # Fit approaches correct value more closely than with the bound below.
    assert (abs(unconstrained['a']) > sc.scalar(5.0, unit='s')).value
    assert (abs(unconstrained['b']) > sc.scalar(2.0, unit='1/m')).value

    constrained, _ = curve_fit(
        ['x'],
        func,
        data,
        p0={'a': sc.scalar(1.0, unit='s'), 'b': sc.scalar(1.0, unit='1/m')},
        bounds={'b': (sc.scalar(-2, unit='1/m'), sc.scalar(2, unit='1/m'))},
    )

    # assert (abs(constrained['a']) > sc.scalar(5.0, unit='s')).value
    assert (abs(constrained['b']) <= sc.scalar(2.0, unit='1/m')).value


def test_bounds_must_have_unit_convertable_to_param_unit():
    data = array1d_from_vars(a=sc.scalar(1.2, unit='s'), b=sc.scalar(10.0, unit='1/m'))
    with pytest.raises(sc.UnitError):
        curve_fit(
            ['x'],
            func,
            data,
            p0={'a': sc.scalar(1.0, unit='s'), 'b': sc.scalar(1.0, unit='1/m')},
            bounds={'a': (sc.scalar(-10.0, unit='s'), sc.scalar(10.0, unit='kg'))},
        )


def test_jac_is_not_implemented():
    # replace this with an actual test once jac is implemented
    with pytest.raises(NotImplementedError):
        curve_fit(['x'], func, array1d(), jac=np.array([[1, 2], [3, 4]]))


def test_can_pass_extra_kwargs():
    data = array1d()

    # Does not raise
    curve_fit(['x'], func, data, method='lm')

    with pytest.raises(ValueError):
        curve_fit(['x'], func, data, method='bad_method')


def test_can_rename_coords():
    def func(apple, *, a, b):
        return a * sc.exp(-b * apple)

    curve_fit(dict(apple='x'), func, array1d())


def test_can_use_non_coord_in_fit():
    data = array1d()
    z = data.coords['x'].copy()
    curve_fit(dict(x=z), func, data)

import numpy as np
import pytest

import scipp as sc
from scipp.testing import assert_allclose


def test_parallel_curve_fit_basic():
    """Test basic parallel curve fitting with a simple exponential function"""
    def func(x, a, b):
        return a * sc.exp(-b * x)

    rng = np.random.default_rng(1234)
    x = sc.linspace(dim='x', start=0.0, stop=0.4, num=50, unit='m')
    y = func(x, a=5, b=17/sc.Unit('m'))
    y.values += 0.01 * rng.normal(size=50)
    da = sc.DataArray(y, coords={'x': x})

    popt, pcov = sc.curve_fit_parallel(['x'], func, da, p0={'b': 1.0 / sc.Unit('m')})
    
    # Check parameter values are close to true values
    assert_allclose(popt['a'], sc.scalar(5.0), rtol=1e-2)
    assert_allclose(popt['b'], sc.scalar(17.0, unit='1/m'), rtol=1e-2)

    # Check covariance matrix structure
    assert set(pcov.keys()) == {'a', 'b'}
    assert set(pcov['a'].keys()) == {'a', 'b'}
    assert set(pcov['b'].keys()) == {'a', 'b'}


def test_parallel_curve_fit_2d():
    """Test parallel fitting over multiple dimensions"""
    def func(x, z, a, b):
        return a * z * sc.exp(-b * x)

    rng = np.random.default_rng(1234)
    x = sc.linspace(dim='x', start=0.0, stop=0.4, num=50, unit='m')
    z = sc.linspace(dim='z', start=0.0, stop=1, num=10)
    y = func(x, z, a=5, b=17/sc.Unit('m'))
    y.values += 0.01 * rng.normal(size=500).reshape(10, 50)
    da = sc.DataArray(y, coords={'x': x, 'z': z})

    popt, pcov = sc.curve_fit_parallel(
        ['x', 'z'], func, da, p0={'b': 1.0 / sc.Unit('m')}, n_workers=2
    )
    
    assert_allclose(popt['a'], sc.scalar(5.0), rtol=1e-2)
    assert_allclose(popt['b'], sc.scalar(17.0, unit='1/m'), rtol=1e-2)


def test_parallel_curve_fit_with_map_over():
    """Test parallel fitting with mapping over dimensions"""
    def func(x, a, b):
        return a * sc.exp(-b * x)

    rng = np.random.default_rng(1234)
    x = sc.linspace(dim='x', start=0.0, stop=0.4, num=50, unit='m')
    z = sc.linspace(dim='z', start=0.0, stop=1, num=10)
    y = func(x, a=z, b=17/sc.Unit('m'))
    y.values += 0.01 * rng.normal(size=500).reshape(10, 50)
    da = sc.DataArray(y, coords={'x': x, 'z': z})

    popt, _ = sc.curve_fit_parallel(
        ['x'], func, da, p0={'b': 1.0 / sc.Unit('m')}, chunks=25
    )
    
    # Check that fitted 'a' parameter follows z coordinate values
    assert_allclose(popt['a'], da.coords['z'], rtol=1e-1)


def test_parallel_curve_fit_with_bounds():
    """Test parallel fitting with parameter bounds"""
    def func(x, a, b):
        return a * sc.exp(-b * x)

    rng = np.random.default_rng(1234)
    x = sc.linspace(dim='x', start=0.0, stop=0.4, num=50, unit='m')
    y = func(x, a=5, b=17/sc.Unit('m'))
    y.values += 0.01 * rng.normal(size=50)
    da = sc.DataArray(y, coords={'x': x})

    bounds = {
        'a': (4.0, 6.0),
        'b': (10.0/sc.Unit('m'), 20.0/sc.Unit('m'))
    }
    
    popt, _ = sc.curve_fit_parallel(
        ['x'], func, da, p0={'b': 1.0 / sc.Unit('m')}, bounds=bounds
    )
    
    assert_allclose(popt['a'], sc.scalar(5.0), rtol=1e-2)
    assert_allclose(popt['b'], sc.scalar(17.0, unit='1/m'), rtol=1e-2)


def test_parallel_curve_fit_with_mask():
    """Test parallel fitting with masked data"""
    def func(x, a, b):
        return a * sc.exp(-b * x)

    rng = np.random.default_rng(1234)
    x = sc.linspace(dim='x', start=0.0, stop=0.4, num=50, unit='m')
    y = func(x, a=5, b=17/sc.Unit('m'))
    y.values += 0.01 * rng.normal(size=50)
    da = sc.DataArray(y, coords={'x': x})
    
    # Mask some values
    mask = sc.zeros(dims=['x'], shape=[50], dtype='bool')
    mask.values[::5] = True  # Mask every 5th point
    da.masks['test_mask'] = mask

    popt, _ = sc.curve_fit_parallel(
        ['x'], func, da, p0={'b': 1.0 / sc.Unit('m')}
    )
    
    assert_allclose(popt['a'], sc.scalar(5.0), rtol=1e-2)
    assert_allclose(popt['b'], sc.scalar(17.0, unit='1/m'), rtol=1e-2)

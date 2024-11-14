import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose

def test_curve_fit_parallel_thread():
    def func(x, a, b):
        return a * sc.exp(-b * x.to(unit=None, dtype='float64'))

    x = sc.linspace(dim='x', start=0.0, stop=0.4, num=50, unit='m')
    z = sc.linspace(dim='z', start=0.0, stop=1, num=10)
    true_a = 5.0
    true_b = 17.0/sc.Unit('m')
        
    # Create a 2D array by broadcasting x and z
    x_2d = sc.broadcast(x, dims=['x', 'z'], shape=[x.sizes['x'], z.sizes['z']])
    z_2d = sc.broadcast(z, dims=['x', 'z'], shape=[x.sizes['x'], z.sizes['z']])
    y = func(x_2d, a=true_a, b=true_b)
    
    # Add some noise
    rng = np.random.default_rng(1234)
    y.values += 0.01 * rng.normal(size=y.values.shape)
    
    da = sc.DataArray(y, coords={'x': x_2d, 'z': z})
    
    popt, pcov = sc.curve_fit(
        ['x'], func, da,
        p0={'b': 1.0/sc.Unit('m')},
        parallel='thread',
        n_workers=2
    )
    
    assert_allclose(popt['a'].value, true_a, rtol=0.1)
    assert_allclose(popt['b'].value, true_b, rtol=0.1)

@pytest.mark.skipif(not sc.HAS_DASK, reason="Dask not available")
def test_curve_fit_parallel_dask():
    def func(x, a, b):
        return a * sc.exp(-b * x.to(unit=None, dtype='float64'))
    
    x = sc.linspace(dim='x', start=0.0, stop=0.4, num=50, unit='m')
    z = sc.linspace(dim='z', start=0.0, stop=1, num=10)
    true_a = 5.0
    true_b = 17.0/sc.Unit('m')
        
    # Create a 2D array by broadcasting x and z
    x_2d = sc.broadcast(x, sizes={'z': z.sizes['z']})
    z_2d = sc.broadcast(z, sizes={'x': x.sizes['x']})
    y = func(x_2d, a=true_a, b=true_b)
    
    # Add some noise
    rng = np.random.default_rng(1234)
    y.values += 0.01 * rng.normal(size=y.values.shape)
    
    da = sc.DataArray(y, coords={'x': x_2d, 'z': z})
    
    popt, pcov = sc.curve_fit(
        ['x'], func, da,
        p0={'b': 1.0/sc.Unit('m')},
        parallel='dask',
        n_workers=2
    )
    
    assert_allclose(popt['a'].value, true_a, rtol=0.1)
    assert_allclose(popt['b'].value, true_b, rtol=0.1)

def test_curve_fit_invalid_parallel():
    def func(x, a, b):
        return a * sc.exp(-b * x)

    x = sc.linspace(dim='x', start=0.0, stop=0.4, num=50, unit='m')
    y = func(x, a=5.0, b=17.0/sc.Unit('m'))
    da = sc.DataArray(y, coords={'x': x})
    
    with pytest.raises(ValueError, match="parallel must be one of"):
        sc.curve_fit(['x'], func, da, parallel='invalid')

def test_curve_fit_dask_not_available(monkeypatch):
    monkeypatch.setattr(sc.curve_fit, 'HAS_DASK', False)
    
    def func(x, a, b):
        return a * sc.exp(-b * x.to(unit=None, dtype='float64'))

    x = sc.linspace(dim='x', start=0.0, stop=0.4, num=50, unit='m')
    y = func(x, a=5.0, b=17.0/sc.Unit('m'))
    da = sc.DataArray(y, coords={'x': x})
    
    with pytest.raises(ImportError, match="Dask is required"):
        sc.curve_fit(['x'], func, da, parallel='dask')

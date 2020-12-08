import numpy as np
import scipp as sc


class ResamplingModel():
    def __init__(self, data, resolution={}, bounds={}):
        self._data = data
        self._resolution = resolution
        self._bounds = bounds
        self._resampled = None
        self._resampled_params = None

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, res):
        self._resolution = res

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bnds):
        self._bounds = bnds

    @property
    def data(self):
        self._call_resample()
        return self._resampled

    def _call_resample(self):
        out = self._data
        edges = []
        params = {}
        for dim, s in self.bounds.items():
            if isinstance(s, int):
                out = out[dim, s]
                params[dim] = s
            else:
                low, high = s
                params[dim] = (low.value, high.value, self.resolution[dim])
                out = out[dim, low:high]
                edges.append(
                    sc.Variable(dims=[dim],
                                values=np.linspace(low.value,
                                                   high.value,
                                                   num=self.resolution[dim] +
                                                   1)))
        if self._resampled is None or params != self._resampled_params:
            self._resampled_params = params
            self._resampled = self._resample(out, edges)


class ResamplingBinnedModel(ResamplingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _resample(self, data, edges):
        return sc.bin(data.bins, edges=edges).bins.sum()


class ResamplingDenseModel(ResamplingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _resample(self, data, edges):
        out = data
        for edge in edges:
            try:
                out = sc.rebin(out, edge.dims[-1], edge)
            except RuntimeError:  # Limitation of rebin for slice of inner dim
                out = sc.rebin(out.copy(), edge.dims[-1], edge)

        return out


def resampling_model(data, **kwargs):
    if data.bins is None:
        return ResamplingDenseModel(data, **kwargs)
    else:
        return ResamplingBinnedModel(data, **kwargs)

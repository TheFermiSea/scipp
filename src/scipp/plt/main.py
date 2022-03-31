# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from .. import Variable, DataArray, Dataset
from ..compat.dict import from_dict
from .dispatch import dispatch
from .plot import PlotDict, Plot
from ..typing import VariableLike, has_numeric_type

import itertools
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Union


def _ndarray_to_variable(ndarray: ArrayLike) -> Variable:
    """
    Convert a numpy ndarray to a Variable.
    Fake dimension labels begin at 'x' and cycle through the alphabet.
    """
    dims = [f"dim_{i}" for i in range(len(ndarray.shape))]
    return Variable(dims=dims, values=ndarray)


def _make_plot_key(plt_key: str, all_keys: list) -> str:
    if plt_key in all_keys:
        key_gen = (f'{plt_key}_{i}' for i in itertools.count(1))
        plt_key = next(x for x in key_gen if x not in all_keys)
    return plt_key


def _brief_str(obj: VariableLike) -> str:
    sizes = ', '.join([f'{dim}: {size}' for dim, size in obj.sizes.items()])
    return f'scipp.{type(obj).__name__}({sizes})'


def _input_to_data_array(item: Union[VariableLike, ArrayLike],
                         all_keys: list,
                         key: str = None) -> DataArray:
    """
    Convert an input for the plot function to a DataArray or a dict of
    DataArrays.
    """
    to_plot = {}
    if isinstance(item, Dataset):
        for name in sorted(item.keys()):
            if has_numeric_type(item[name]):
                proto_plt_key = f'{key}_{name}' if key else name
                to_plot[_make_plot_key(proto_plt_key, all_keys)] = item[name]
    elif isinstance(item, Variable):
        if has_numeric_type(item):
            if key is None:
                key = _brief_str(item)
            to_plot[_make_plot_key(key, all_keys)] = DataArray(data=item)
    elif isinstance(item, DataArray):
        if has_numeric_type(item):
            if key is None:
                key = item.name
            to_plot[_make_plot_key(key, all_keys)] = item
    elif isinstance(item, np.ndarray):
        if key is None:
            key = str(type(item))
        to_plot[_make_plot_key(key,
                               all_keys)] = DataArray(data=_ndarray_to_variable(item))
    else:
        raise RuntimeError("plot: Unknown input type: {}. Allowed inputs are "
                           "a Dataset, a DataArray, a Variable (and their "
                           "respective views), a numpy ndarray, and a dict of "
                           "Variables, DataArrays or ndarrays".format(type(item)))
    return to_plot


def plot(scipp_obj: Union[VariableLike, ArrayLike, Dict[str, Union[VariableLike,
                                                                   ArrayLike]]],
         projection: str = None,
         filters: list = None,
         **kwargs) -> Union[Plot, PlotDict]:
    """
    Wrapper function to plot a scipp object.

    Possible inputs are:
      - Variable
      - Dataset
      - DataArray
      - numpy ndarray
      - dict of Variables
      - dict of DataArrays
      - dict of numpy ndarrays
      - dict that can be converted to a Scipp object via `from_dict`

    1D Variables are grouped onto the same axes if they have the same dimension
    and the same unit.

    Any other Variables are displayed in their own figure.

    Returns a Plot object which can be displayed in a Jupyter notebook.
    """

    # Decompose the input and return a dict of DataArrays.
    inventory = {}
    if isinstance(scipp_obj, dict):
        try:
            inventory.update(
                _input_to_data_array(from_dict(scipp_obj), all_keys=inventory.keys()))
        except:  # noqa: E722
            for key, item in scipp_obj.items():
                inventory.update(
                    _input_to_data_array(item, all_keys=inventory.keys(), key=key))
    else:
        inventory.update(_input_to_data_array(scipp_obj, all_keys=inventory.keys()))

    # # Counter for 1d/event data
    # line_count = -1

    # # Create a list of variables which will then be dispatched to correct
    # # plotting function.
    # # Search through the variables and group the 1D datasets that have
    # # the same coordinates and units.
    # # tobeplotted is a dict that holds four items:
    # # {number_of_dimensions, Dataset, axes, line_parameters}.
    # tobeplotted = dict()
    # for name, item in sorted(inventory.items()):
    #     ndims = len(item.dims)
    #     if (ndims > 0) and (np.sum(item.shape) > 0):
    #         if ndims == 1 or projection == "1d" or projection == "1D":
    #             key = f"{item.dims}.{item.unit}"
    #             line_count += 1
    #         else:
    #             key = name

    #         if key not in tobeplotted.keys():
    #             tobeplotted[key] = dict(ndims=ndims, scipp_obj_dict=dict())
    #         tobeplotted[key]["scipp_obj_dict"][name] = item

    return Plot(models=inventory, filters=filters, **kwargs)

    # # Plot all the subsets
    # output = PlotDict()
    # for key, val in tobeplotted.items():
    #     output._items[key] = dispatch(scipp_obj_dict=val["scipp_obj_dict"],
    #                                   ndim=val["ndims"],
    #                                   projection=projection,
    #                                   **kwargs)

    # if len(output) > 1:
    #     return output
    # elif len(output) > 0:
    #     return output._items[list(output.keys())[0]]
    # else:
    #     raise ValueError("Input contains nothing that can be plotted. "
    #                      "Input may be of dtype vector or string, "
    #                      f"or may have zero dimensions:\n{scipp_obj}")

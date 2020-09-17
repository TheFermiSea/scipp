# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

# Scipp imports
from .. import config
from .model import PlotModel
# from .lineplot import LinePlot
from .render import render_plot
from .slicer import Slicer
from .tools import to_bin_centers, vars_to_err, mask_to_float
from .._utils import name_with_unit
from .._scipp import core as sc

# Other imports
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import ipywidgets as widgets
import warnings





class PlotModel1d(PlotModel):

    def __init__(self,
                 controller=None,
                 scipp_obj_dict=None):

        super().__init__(controller=controller,
            scipp_obj_dict=scipp_obj_dict)


        self.axparams = {"x": {}}

        return




    # def update_buttons(self, owner, event, dummy):
    #     for dim, button in self.parent.widgets.buttons.items():
    #         if dim == owner.dim:
    #             self.parent.widgets.slider[dim].disabled = True
    #             button.disabled = True
    #             self.parent.widgets.button_axis_to_dim["x"] = dim
    #         else:
    #             self.parent.widgets.slider[dim].disabled = False
    #             button.value = None
    #             button.disabled = False
    #     self.update_axes(owner.dim)
    #     self.parent.clear_keep_buttons()
    #     # self.keep_buttons = dict()
    #     self.parent.make_keep_button()
    #     # self.update_button_box_widget()
    #     return

    # # def update_button_box_widget(self):
    # #     self.mbox = self.vbox.copy()
    # #     for k, b in self.keep_buttons.items():
    # #         self.mbox.append(widgets.HBox(list(b.values())))
    # #     self.box.children = tuple(self.mbox)

    def update_axes(self, axparams):
        # # if not self.mpl_axes:
        # #     self.ax.lines = []
        # #     self.ax.collections = []
        # # self.members.update({
        # #     "lines": {},
        # #     "error_x": {},
        # #     "error_y": {},
        # #     "error_xy": {},
        # #     "masks": {}
        # # })

        # dim = list(limits.keys())[0]
        # self.axparams["x"]["dim"] = dim

        # self.axparams["x"]["labels"] = name_with_unit(
        #         self.data_arrays[self.name].coords[dim])




        # # # xmin = np.Inf
        # # # xmax = np.NINF
        # # to_line_plot = {}
        # # is_bin_edge = {}

        # xmin = np.Inf
        # xmax = np.NINF

        # self.axparams["x"]["hist"] = {}

        # for name, array in self.data_arrays.items():

        #     xmin = min(sc.min(array.coords[dim]).value, xmin)
        #     xmax = max(sc.max(array.coords[dim]).value, xmax)

        #     self.axparams["x"]["hist"][name] = limits[dim]["hist"][name]


        #     # # # new_x = self.slider_coord[name][dim].values
        #     # # # new_x = array.coords[dim].values
        #     # # xmin = min(sc.min(array.coords[dim]).value, xmin)
        #     # # xmax = max(sc.max(array.coords[dim]).value, xmax)

        #     # vslice = self.slice_data(array, name)
        #     # to_line_plot[name] = vslice
        #     # # dim = vslice.dims[0]
        #     # is_bin_edge[name] = self.histograms[name][dim][dim]

        # # if self.parent.figure is None:
        # #     self.parent.figure = LinePlot(to_line_plot, is_bin_edge=is_bin_edge,
        # #         mpl_line_params=self.parent.mpl_line_params)
        # # else:
        # # self.parent.figure.plot_data(to_line_plot, is_bin_edge=is_bin_edge, clear=True)

        # self.axparams["x"]["lims"] = [xmin, xmax]
        # return self.axparams
        return



    def slice_data(self, array, slices):
        data_slice = array
        # # Slice along dimensions with active sliders
        # for dim, val in self.parent.widgets.slider.items():
        #     if not val.disabled:

        for dim in slices:
            # deltax = self.controller.widgets.thickness_slider[dim].value
            deltax = slices[dim]["thickness"]
            loc = slices[dim]["location"]


                # deltax = self.parent.widgets.thickness_slider[dim].value
            data_slice = self.resample_image(data_slice,
                    rebin_edges={dim: sc.Variable([dim], values=[loc - 0.5 * deltax,
                                                                 loc + 0.5 * deltax],
                                                        unit=data_slice.coords[dim].unit)})[dim, 0]
            data_slice *= (deltax * sc.units.one)



        # if vslice.unaligned is not None:
        #     vslice = sc.histogram(vslice)
        #     # self.ylim = get_ylim(var=vslice,
        #     #                           ymin=self.ylim[0],
        #     #                           ymax=self.ylim[1],
        #     #                           errorbars=self.errorbars[name],
        #     #                           logy=self.logy)
        return data_slice

    # def slice_masks(self):
    #     mslice = self.masks
    #     for dim, val in self.slider.items():
    #         if not val.disabled and (dim in mslice.dims):
    #             mslice = mslice[dim, val.value]
    #     return mslice

    def update_data(self, slices, mask_info):
        # Define function to update slices.
        # Special key in the change dict: if "vslice" is found, it means we are
        # calling from a profile viewer, and the slice has hence already been
        # generate.

        # dim = list(slices.keys())[0]
        dim = self.axparams["x"]["dim"]
        new_values = {}

        # xmin = np.Inf
        # xmax = np.NINF

        for name, array in self.data_arrays.items():
            # if "vslice" in change:
            #     vslice = change["vslice"][name]
            # else:
            new_values[name] = {"values": {}, "variances": {}, "masks": {}}

            data_slice = self.slice_data(array, slices)

            # xmin = min(sc.min(array.coords[dim]).value, xmin)
            # xmax = max(sc.max(array.coords[dim]).value, xmax)

            ydata = data_slice.values
            xcenters = to_bin_centers(data_slice.coords[dim], dim).values

            # if self.histograms[name][dim][dim]:
            if self.axparams["x"]["hist"][name]:
                new_values[name]["values"]["x"] = data_slice.coords[dim].values
                new_values[name]["values"]["y"] = np.concatenate((ydata[0:1], ydata))
                # new_values[name]["data"]["hist"] = True
            else:
                new_values[name]["values"]["x"] = xcenters
                new_values[name]["values"]["y"] = ydata
                # new_values[name]["data"]["hist"] = False
            if data_slice.variances is not None:
                new_values[name]["variances"]["x"] = xcenters
                new_values[name]["variances"]["y"] = ydata
                new_values[name]["variances"]["e"] = vars_to_err(data_slice.variances)



        # self.parent.figure.update_data(new_slices)

        #     vals = vslice.values
        #     dim = self.button_axis_to_dim["x"]
        #     xcoord = vslice.coords[dim]
        #     hist = self.histograms[name][dim][dim]
        #     if hist:
        #         vals = np.concatenate((vals[0:1], vals))
        #     else:
        #         xcoord = to_bin_centers(xcoord, dim)
        #     # self.members["lines"][name].set_ydata(vals)
        #     self.members["lines"][name].set_data(xcoord.values, vals)

            # if len(self.masks[name]) > 0:
            if len(mask_info[name]) > 0:
                base_mask = sc.Variable(dims=data_slice.dims,
                                        values=np.ones(data_slice.shape,
                                                       dtype=np.int32))
                for m in mask_info[name]:
                    # Use automatic broadcast to broadcast 0D masks
                    msk = (base_mask * sc.Variable(
                        dims=data_slice.masks[m].dims,
                        values=data_slice.masks[m].values.astype(np.int32))).values
                    if self.axparams["x"]["hist"][name]:
                        msk = np.concatenate((msk[0:1], msk))

                    new_values[name]["masks"][m] = mask_to_float(msk, new_values[name]["values"]["y"])

                    # self.members["masks"][name][m].set_data(
                    #     xcoord.values,
                    #     self.mask_to_float(msk, vals))
                    # # self.members["masks"][name][m].set_ydata(
                    # #     self.mask_to_float(msk, vals))

        #     if self.errorbars[name]:
        #         coll = self.members["error_y"][name].get_children()[0]
        #         if hist:
        #             xcoord = to_bin_centers(xcoord, dim)
        #         coll.set_segments(
        #             self.change_segments_y(xcoord.values,
        #                                    vslice.values, vslice.variances))

        # # if self.input_contains_unaligned_data and (not self.mpl_axes):
        # #     with warnings.catch_warnings():
        # #         warnings.filterwarnings("ignore", category=UserWarning)
        # #         self.ax.set_ylim(self.ylim)

        return new_values

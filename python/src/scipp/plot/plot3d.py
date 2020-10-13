# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2020 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

# Scipp imports
from .controller3d import PlotController3d
from .model3d import PlotModel3d
from .panel3d import PlotPanel3d
from .sciplot import SciPlot
from .view3d import PlotView3d


# def plot3d(scipp_obj_dict=None,
#            positions=None,
#            axes=None,
#            masks=None,
#            filename=None,
#            figsize=None,
#            aspect=None,
#            cmap=None,
#            log=False,
#            vmin=None,
#            vmax=None,
#            color=None,
#            background="#f0f0f0",
#            nan_color="#d3d3d3",
#            pixel_size=1.0,
#            tick_size=None,
#            show_outline=True):
def plot3d(*args,
           filename=None,
           **kwargs):
    """
    Plot a 3D point cloud through a N dimensional dataset.
    For every dimension above 3, a slider is created to adjust the position of
    the slice in that particular dimension.
    It is possible to add cut surfaces as cartesian, cylindrical or spherical
    planes.
    """

    sp = SciPlot3d(*args, **kwargs)

    if filename is not None:
        sp.savefig(filename)

    return sp


class SciPlot3d(SciPlot):
    def __init__(self,
                 scipp_obj_dict=None,
                 positions=None,
                 axes=None,
                 masks=None,
                 cmap=None,
                 log=None,
                 vmin=None,
                 vmax=None,
                 color=None,
                 aspect=None,
                 background="#f0f0f0",
                 nan_color="#d3d3d3",
                 pixel_size=None,
                 tick_size=None,
                 show_outline=True):

        super().__init__(scipp_obj_dict=scipp_obj_dict,
                 axes=axes,
                 cmap=cmap,
                 norm=norm,
                 vmin=vmin,
                 vmax=vmax,
                 masks=masks,
                 positions=positions,
                 view_ndims=3)

        self.widgets = PlotWidgets(axes=self.axes,
                                   ndim=self.ndim,
                                   name=self.name,
                                   dim_to_shape=self.dim_to_shape,
                                   masks=self.masks,
                                   button_options=['X', 'Y', 'Z'])


        # The model which takes care of all heavy calculations
        # self.model = PlotModel3d(controller=self.controller,
        #                          scipp_obj_dict=scipp_obj_dict,
        #                          positions=positions,
        #                          cut_options=self.panel.cut_options)
        self.model = PlotModel3d(scipp_obj_dict=scipp_obj_dict,
                                 axes=self.axes,
                                 name=self.name,
                                 dim_to_shape=self.dim_to_shape,
                                 dim_label_map=self.dim_label_map,
                                 positions=positions,
                                 cut_options=self.panel.cut_options)

        # The view which will display the 3d scene and send pick events back to
        # the controller
        self.view = PlotView3d(
            cmap=self.params["values"][
                self.name]["cmap"],
            norm=self.params["values"][
                self.name]["norm"],
            unit=self.params["values"][
                self.name]["unit"],
            masks=self.masks[self.name],
            nan_color=nan_color,
            pixel_size=pixel_size,
            tick_size=tick_size,
            background=background,
            show_outline=show_outline)

        # # Connect controller to model, view, panel and profile
        # self._connect_controller_members()

        # An additional panel view with widgets to control the cut surface
        # Note that the panel needs to be created before the model.
        self.panel = PlotPanel3d(data_names=list(scipp_obj_dict.keys()),
                                 pixel_size=pixel_size)

        # The main controller module which connects all the parts
        self.controller = PlotController3d(
          # scipp_obj_dict=scipp_obj_dict,
                                           axes=self.axes,
                                           name=self.name,
                                           dim_to_shape=self.dim_to_shape,
                                           coord_shapes=self.coord_shapes,
                                           # masks=self.masks,
                                           # cmap=cmap,
                                           norm=norm,
                                           vmin=self.params["values"][self.name]["vmin"],
                                           vmax=self.params["values"][self.name]["vmax"],
                                           # color=color,
                                           # positions=positions,
                                           pixel_size=pixel_size)


        # Call update_slice once to make the initial image
        self.controller.update_axes()

        return

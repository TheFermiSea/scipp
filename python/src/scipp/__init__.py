# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2019 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock
from ._scipp import *
from ._scipp import __version__
from .show import show
from .table import table
from .plot import plot, config as plot_config
from .compat.mantid import load as __load

# from ._scipp import neutron as neutron
setattr(neutron, "load", __load)

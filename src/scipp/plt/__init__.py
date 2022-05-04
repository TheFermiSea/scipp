# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

# flake8: noqa

# from .main import plot
import matplotlib.pyplot as plt

plt.ioff()

from .plot import Plot
from .model import Node, Model
from .figure import Figure
from . import widgets

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from ..view import View

from functools import partial
import ipywidgets as ipw


class WidgetView(View):

    def __init__(self, widgets):
        super().__init__()
        self._base_func = None
        self._widgets = widgets
        for widget in self._widgets.values():
            widget.observe(self._update_and_notify, names="value")

    @property
    def values(self):
        return {key: widget.value for key, widget in self._widgets.items()}

    def _update_node_func(self, node):
        node.func = partial(self._base_func, **self.values)

    def _update_and_notify(self, _):
        nodes = next(iter(self._graph_nodes.values()))
        for node in nodes.values():
            self._update_node_func(node)
        self.notify_graphs()

    def add_graph_node(self, node):
        super().add_graph_node(node)
        self._base_func = node.func
        self._update_node_func(node)

    def _to_widget(self):
        return ipw.VBox([
            widget._to_widget() if hasattr(widget, "_to_widget") else widget
            for widget in self._widgets.values()
        ])

    def notify_view(self, _):
        return

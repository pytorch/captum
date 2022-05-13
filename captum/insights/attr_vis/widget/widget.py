#!/usr/bin/env python3
import ipywidgets as widgets
from captum.insights import AttributionVisualizer
from captum.insights.attr_vis.server import namedtuple_to_dict
from traitlets import Dict, Instance, List, observe, Unicode


@widgets.register
class CaptumInsights(widgets.DOMWidget):
    """A widget for interacting with Captum Insights."""

    _view_name = Unicode("CaptumInsightsView").tag(sync=True)
    _model_name = Unicode("CaptumInsightsModel").tag(sync=True)
    _view_module = Unicode("jupyter-captum-insights").tag(sync=True)
    _model_module = Unicode("jupyter-captum-insights").tag(sync=True)
    _view_module_version = Unicode("^0.1.0").tag(sync=True)
    _model_module_version = Unicode("^0.1.0").tag(sync=True)

    visualizer = Instance(klass=AttributionVisualizer)

    insights_config = Dict().tag(sync=True)
    label_details = Dict().tag(sync=True)
    attribution = Dict().tag(sync=True)
    config = Dict().tag(sync=True)
    output = List().tag(sync=True)

    def __init__(self, **kwargs) -> None:
        super(CaptumInsights, self).__init__(**kwargs)
        self.insights_config = self.visualizer.get_insights_config()
        self.out = widgets.Output()
        with self.out:
            print("Captum Insights widget created.")

    @observe("config")
    def _fetch_data(self, change):
        if not self.config:
            return
        with self.out:
            self.visualizer._update_config(self.config)
            self.output = namedtuple_to_dict(self.visualizer.visualize())
            self.config = dict()

    @observe("label_details")
    def _fetch_attribution(self, change):
        if not self.label_details:
            return
        with self.out:
            self.attribution = namedtuple_to_dict(
                self.visualizer._calculate_attribution_from_cache(
                    self.label_details["inputIndex"],
                    self.label_details["modelIndex"],
                    self.label_details["labelIndex"],
                )
            )
            self.label_details = dict()

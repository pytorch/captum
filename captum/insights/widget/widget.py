import ipywidgets as widgets
from captum.insights import AttributionVisualizer
from captum.insights.server import namedtuple_to_dict
from traitlets import Dict, Instance, List, Unicode, observe


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
    classes = List(trait=Unicode).tag(sync=True)

    label_details = Dict().tag(sync=True)
    attribution = Dict().tag(sync=True)

    config = Dict().tag(sync=True)
    output = List().tag(sync=True)

    def __init__(self, **kwargs):
        super(CaptumInsights, self).__init__(**kwargs)
        self.classes = self.visualizer.classes

    @observe("config")
    def _fetch_data(self, change):
        if self.config:
            self.visualizer._update_config(self.config)
            self.output = namedtuple_to_dict(self.visualizer.visualize())
            self.config = dict()

    @observe("label_details")
    def _fetch_attribution(self, change):
        if self.label_details:
            self.attribution = namedtuple_to_dict(
                self.visualizer._calculate_attribution_from_cache(
                    self.label_details["instance"], self.label_details["labelIndex"]
                )
            )
            self.label_details = dict()

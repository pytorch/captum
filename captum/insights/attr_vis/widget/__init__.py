from captum.insights.attr_vis.widget._version import __version__, version_info  # noqa
from captum.insights.attr_vis.widget.widget import *  # noqa


def _jupyter_nbextension_paths():
    return [
        {
            "section": "notebook",
            "src": "static",
            "dest": "jupyter-captum-insights",
            "require": "jupyter-captum-insights/extension",
        }
    ]

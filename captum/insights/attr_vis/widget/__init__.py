# pyre-strict
from typing import Dict, List

from captum.insights.attr_vis.widget._version import __version__, version_info
from captum.insights.attr_vis.widget.widget import CaptumInsights


def _jupyter_nbextension_paths() -> List[Dict[str, str]]:
    return [
        {
            "section": "notebook",
            "src": "static",
            "dest": "jupyter-captum-insights",
            "require": "jupyter-captum-insights/extension",
        }
    ]


__all__ = ["__version__", "version_info", "CaptumInsights"]

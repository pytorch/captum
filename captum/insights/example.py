# for legacy purposes

# pyre-strict
import warnings

# pyre-fixme[21]: Could not find name `Net` in `captum.insights.attr_vis.example`.
from captum.insights.attr_vis.example import *  # noqa

warnings.warn(
    "Deprecated. Please import from captum.insights.attr_vis.example instead."
)


main()  # noqa

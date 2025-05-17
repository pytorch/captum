# for legacy purposes

# pyre-strict
import warnings

from captum.insights.attr_vis.example import *  # noqa

warnings.warn(
    "Deprecated. Please import from captum.insights.attr_vis.example instead.",
    stacklevel=1,
)


main()  # noqa

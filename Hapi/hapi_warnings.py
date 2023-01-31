import warnings
# import numpy as np


class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""

    pass


warnings.simplefilter("always", InstabilityWarning)
warnings.simplefilter("always", UserWarning)



def SilenceShapelyWarning():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

"""Custom warning and silencing warnings."""
import warnings


class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""

    pass


warnings.simplefilter("always", InstabilityWarning)
warnings.simplefilter("always", UserWarning)


def SilencePandasWarning():
    """Silence pandas future warning."""
    warnings.simplefilter(action="ignore", category=FutureWarning)


def SilenceShapelyWarning():
    """Silence Shapely deprecation warning."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)

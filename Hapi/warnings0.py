import warnings


class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""

    pass


warnings.simplefilter("always", InstabilityWarning)
warnings.simplefilter("always", UserWarning)

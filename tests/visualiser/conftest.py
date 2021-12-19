import pytest

# xs in segment 3
fromxs = 230
toxs = 270

@pytest.fixture(scope="module")
def plot_xs_seg3_fromxs() -> int:
    return fromxs

@pytest.fixture(scope="module")
def plot_xs_seg3_toxs() -> int:
    return toxs
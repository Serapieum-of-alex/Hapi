import pytest

# xs in segment 3
fromxs = 230
toxs = 235
start = "1955-02-10"
end = "1955-02-11"


@pytest.fixture(scope="module")
def plot_xs_seg3_fromxs() -> int:
    return fromxs


@pytest.fixture(scope="module")
def plot_xs_seg3_toxs() -> int:
    return toxs


@pytest.fixture(scope="module")
def animate_start() -> str:
    return start


@pytest.fixture(scope="module")
def animate_end() -> str:
    return end

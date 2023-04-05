import pytest

WarmUpPeriod = 0
saveplots = True
novalue = -9

StatisticalPr_columns = [
    "mean",
    "std",
    "min",
    "5%",
    "25%",
    "median",
    "75%",
    "95%",
    "max",
    "t_beg",
    "t_end",
    "nyr",
    "t_end",
    "t_beg",
    "q1.5",
    "q2",
    "q5",
    "q10",
    "q25",
    "q50",
    "q100",
    "q200",
    "q500",
    "q1000",
]
gev_columns = ["c", "loc", "scale", "D-static", "P-Value"]
gum_columns = ["loc", "scale", "D-static", "P-Value"]


@pytest.fixture(scope="module")
def Discharge_WarmUpPeriod() -> int:
    return WarmUpPeriod


@pytest.fixture(scope="module")
def Discharge_gauge_long_ts() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/gauges/discharge_long_ts/"


@pytest.fixture(scope="module")
def Statistical_analisis_path() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/gauges/discharge_long_ts/statistical-analysis-results"


@pytest.fixture(scope="module")
def SavePlots() -> bool:
    return saveplots


@pytest.fixture(scope="module")
def SavePlots() -> bool:
    return saveplots


@pytest.fixture(scope="module")
def NoValue() -> int:
    return novalue


@pytest.fixture(scope="module")
def statisticalpr_columns() -> list:
    return StatisticalPr_columns


@pytest.fixture(scope="module")
def distributionpr_gev_columns() -> list:
    return gev_columns


@pytest.fixture(scope="module")
def distributionpr_gum_columns() -> list:
    return gum_columns

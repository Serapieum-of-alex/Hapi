import pytest


@pytest.fixture(scope="module")
def interface_Laterals_table_path() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/1d/topo/laterals.txt"


@pytest.fixture(scope="module")
def interface_Laterals_folder() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/1d/hydro/"


@pytest.fixture(scope="module")
def interface_Laterals_date_format() -> str:
    return "%d_%m_%Y"


@pytest.fixture(scope="module")
def interface_bc_path() -> str:
    return (
        "examples/hydrodynamic-models/test_case/inputs/1d/topo/boundaryconditions.txt"
    )


@pytest.fixture(scope="module")
def interface_bc_folder() -> str:
    return "examples/hydrodynamic-models/test_case/inputs/1d/hydro/"


@pytest.fixture(scope="module")
def interface_bc_date_format() -> str:
    return "%d_%m_%Y"


@pytest.fixture(scope="module")
def rrm_resutls_hm_location() -> str:
    return r"examples\Hydrodynamic-models\test_case\inputs\1d\hydro"


@pytest.fixture(scope="module")
def laterals_number_ts() -> int:
    return 80


@pytest.fixture(scope="module")
def no_laterals() -> int:
    return 9

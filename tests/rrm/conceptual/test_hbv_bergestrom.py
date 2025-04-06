import numpy as np
import pandas as pd
import pytest

from Hapi.rrm.hbv_bergestrom92 import HBVBergestrom92


@pytest.fixture()
def coello_lumped_meteo_data() -> np.ndarray:
    path = "tests/rrm/data/coello/meteo-lumped-data-MSWEP.csv"
    data = pd.read_csv(path, header=0, delimiter=",", index_col=0).to_numpy()
    return data


@pytest.fixture()
def coello_lumped_parameters() -> list:
    path = "tests/rrm/data/coello/coello-lumpedparameter-muskingum.txt"
    parameters = pd.read_csv(path, index_col=0, header=None)[1].tolist()
    return parameters


def test_hbv_lumped_model(
    coello_lumped_meteo_data: np.ndarray, coello_lumped_parameters: list
):
    p = coello_lumped_meteo_data[:, 0]
    et = coello_lumped_meteo_data[:, 1]
    t = coello_lumped_meteo_data[:, 2]
    tm = coello_lumped_meteo_data[:, 3]
    initial_conditions = [0, 10, 10, 10, 0]
    snow = False
    q_init = None
    conceptual_model = HBVBergestrom92()

    quz, qlz, state_variables = conceptual_model.simulate(
        p,
        t,
        et,
        tm,
        coello_lumped_parameters,
        init_st=initial_conditions,
        q_init=q_init,
        snow=snow,
    )
    assert len(quz) == len(qlz) == len(state_variables) == len(p) + 1
    assert state_variables.shape == (len(p) + 1, 5)

    expected_quz = np.array([3.5651002, 1.6847236, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert all(np.isclose(quz, expected_quz))

    expected_qlz = np.array(
        [
            0.32230002,
            0.49229392,
            0.57979316,
            0.56222373,
            0.54827434,
            0.58935225,
            0.6560198,
            0.7031159,
            0.71430916,
            0.70471734,
            0,
        ]
    )
    assert all(np.isclose(qlz, expected_qlz))

    expected_state_variables = np.array(
        [
            [0.0, 10.0, 10.0, 10.0, 0.0],
            [0.0, 9.767188, 3.0408764, 14.782106, 0.0],
            [0.0, 10.355642, 0.0, 17.409445, 0.0],
            [0.0, 10.257657, 0.0, 16.881887, 0.0],
            [0.0, 10.617212, 0.0, 16.46303, 0.0],
            [0.0, 19.594128, 0.0, 17.696476, 0.0],
            [0.0, 24.455137, 0.0, 19.698301, 0.0],
            [0.0, 26.748932, 0.0, 21.112457, 0.0],
            [0.0, 27.195393, 0.0, 21.448555, 0.0],
            [0.0, 26.771988, 0.0, 21.160543, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(state_variables, expected_state_variables)

import pytest
import pandas as pd
import numpy as np
from Hapi.rrm.hbv import HBV


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


def test_hbv_lumped_model(coello_lumped_meteo_data: np.ndarray, coello_lumped_parameters: list):
    p = coello_lumped_meteo_data[:, 0]
    et = coello_lumped_meteo_data[:, 1]
    t = coello_lumped_meteo_data[:, 2]
    tm = coello_lumped_meteo_data[:, 3]
    initial_conditions = [0, 10, 10, 10, 0]
    snow = False
    q_init = None
    conceptual_model = HBV()


    quz, qlz, state_variables = conceptual_model.simulate(
        p,
        t,
        et,
        coello_lumped_parameters,
        init_st=initial_conditions,
        ll_temp=tm,
        q_init=q_init,
        snow=snow,
    )
    assert len(quz) == len(qlz) == len(state_variables) == len(p) + 1
    assert state_variables.shape == (len(p) + 1, 5)

    expected_quz = np.array([np.inf, 4.115642, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert all(np.isclose(quz, expected_quz))

    expected_qlz = np.array([0.3223    , 0.49229392, 0.4764273 , 0.46107203, 0.44621167,
                             0.4318303 , 0.46364802, 0.5265117 , 0.5704197 , 0.57703793,
                             0.5617763 ])
    assert all(np.isclose(qlz, expected_qlz))

    expected_state_variables = np.array([[ 0.      , 10.      , 10.      , 10.      ,  0.      ],
                                         [ 0.      , 10.94771 ,  0.      , 14.782106,  0.      ],
                                         [ 0.      , 10.692835,  0.      , 14.305678,  0.      ],
                                         [ 0.      , 11.413589,  0.      , 13.844606,  0.      ],
                                         [ 0.      , 11.321631,  0.      , 13.398396,  0.      ],
                                         [ 0.      , 11.776372,  0.      , 12.966565,  0.      ],
                                         [ 0.      , 21.125572,  0.      , 13.921956,  0.      ],
                                         [ 0.      , 26.192625,  0.      , 15.809565,  0.      ],
                                         [ 0.      , 28.677124,  0.      , 17.12799 ,  0.      ],
                                         [ 0.      , 29.34668 ,  0.      , 17.326714,  0.      ],
                                         [ 0.      , 29.167826,  0.      , 16.868452,  0.      ]],
                                        dtype=np.float32)
    assert  np.array_equal(state_variables, expected_state_variables)

from typing import List

import numpy as np

from Hapi.sm.distributions import GEV, Gumbel


def test_create_gumbel_instance(
        time_series1: list,
):
    Gdist = Gumbel(time_series1)
    assert isinstance(Gdist.data, np.ndarray)
    assert isinstance(Gdist.data_sorted, np.ndarray)


def test_gumbel_estimate_parameter(
        time_series2: list,
        dist_estimation_parameters: List[str],
):
    Gdist = Gumbel(time_series2)
    for i in range(len(dist_estimation_parameters)):
        param = Gdist.EstimateParameter(method=dist_estimation_parameters[i], Test=False)
        assert isinstance(param, list)
        assert Gdist.loc
        assert Gdist.scale

def test_gumbel_ks(
        time_series2: list,
        dist_estimation_parameters_ks: str,
):
    Gdist = Gumbel(time_series2)
    Gdist.EstimateParameter(method=dist_estimation_parameters_ks, Test=False)
    Gdist.ks()
    assert Gdist.Dstatic
    assert Gdist.KS_Pvalue



def test_gumbel_chisquare(
        time_series2: list,
        dist_estimation_parameters_ks: str,
):
    Gdist = Gumbel(time_series2)
    Gdist.EstimateParameter(method=dist_estimation_parameters_ks, Test=False)
    Gdist.chisquare()
    assert Gdist.chistatic
    assert Gdist.chi_Pvalue





def test_create_gev_instance(
        time_series1: list,
):
    Gdist = GEV(time_series1)
    assert isinstance(Gdist.data, np.ndarray)
    assert isinstance(Gdist.data_sorted, np.ndarray)


def test_gev_estimate_parameter(
        time_series1: list,
        dist_estimation_parameters: List[str],
):
    Gdist = GEV(time_series1)
    for i in range(len(dist_estimation_parameters)):
        param = Gdist.EstimateParameter(method=dist_estimation_parameters[i], Test=False)
        assert isinstance(param, list)
        assert Gdist.loc
        assert Gdist.scale
        assert Gdist.shape


def test_gev_ks(
        time_series1: list,
        dist_estimation_parameters_ks: str,
):
    Gdist = GEV(time_series1)
    Gdist.EstimateParameter(method=dist_estimation_parameters_ks, Test=False)
    Gdist.ks()
    assert Gdist.Dstatic
    assert Gdist.KS_Pvalue

def test_gev_chisquare(
        time_series1: list,
        dist_estimation_parameters_ks: str,
):
    Gdist = GEV(time_series1)
    Gdist.EstimateParameter(method=dist_estimation_parameters_ks, Test=False)
    Gdist.chisquare()
    assert Gdist.chistatic
    assert Gdist.chi_Pvalue

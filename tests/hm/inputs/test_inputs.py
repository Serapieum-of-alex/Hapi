import os

import Hapi.hm.calibration as RC
import Hapi.hm.inputs as IN


def test_StatisticalProperties(
        gauges_table_path: str,
        ReadObservedQ_Path: str,
        dates: list,
        nodatavalu: int,
        gauge_date_format: str,
        version: int,
        Discharge_WarmUpPeriod: int,
        Discharge_gauge_long_ts: str,
        SavePlots: bool,
        Statistical_analisis_path: str,
        NoValue: int,
        statisticalpr_columns: list,
        distributionpr_gev_columns: list,
        distributionpr_gum_columns: list,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(gauges_table_path)

    Inputs35 = IN.Inputs("Observed_Q")
    computationalnodes = Calib.GaugesTable['oid'].tolist()

    Inputs35.StatisticalProperties(
        computationalnodes,
        Discharge_gauge_long_ts,
        dates[0],
        Discharge_WarmUpPeriod,
        SavePlots,
        Statistical_analisis_path,
        SeparateFiles=True,
        Filter=NoValue,
        Distibution='GUM',
        method='lmoments'
    )
    assert os.path.exists(os.path.join(Statistical_analisis_path, "Figures"))
    assert all(elem in Inputs35.StatisticalPr.columns.tolist() for elem in statisticalpr_columns)
    assert all(elem in Calib.GaugesTable['oid'].to_list() for elem in Inputs35.StatisticalPr.index.tolist())
    assert all(elem in Inputs35.DistributionPr.columns.tolist() for elem in distributionpr_gum_columns)
    assert all(elem in Calib.GaugesTable['oid'].to_list() for elem in Inputs35.DistributionPr.index.tolist())

    Inputs35.StatisticalProperties(
        computationalnodes,
        Discharge_gauge_long_ts,
        dates[0],
        Discharge_WarmUpPeriod,
        SavePlots,
        Statistical_analisis_path,
        SeparateFiles=True,
        Filter=NoValue,
        method='lmoments'
    )



    assert os.path.exists(os.path.join(Statistical_analisis_path, "Figures"))
    assert all(elem in Inputs35.StatisticalPr.columns.tolist() for elem in statisticalpr_columns)
    assert all(elem in Calib.GaugesTable['oid'].to_list() for elem in Inputs35.StatisticalPr.index.tolist())
    assert all(elem in Inputs35.DistributionPr.columns.tolist() for elem in distributionpr_gev_columns)
    assert all(elem in Calib.GaugesTable['oid'].to_list() for elem in Inputs35.DistributionPr.index.tolist())

from typing import List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import Hapi.hm.calibration as RC
import Hapi.hm.river as R
from Hapi.hm.interface import Interface


def test_create_river_instance(
        dates:list,
        rrm_start: str,
        version: int
):
    assert R.River('HM', version=version, start=dates[0], rrmstart=rrm_start)

def test_read_slope_method(
        version: int,
        slope_path: str
):
    River = R.River('HM', version=version)
    River.Slope(slope_path)
    assert len(River.slope) == 2 and len(River.slope.columns) == 2


def test_read_crosssections_method(
        version: int,
        river_cross_section_path: str,
        xs_total_no: int,
        xs_col_no: int
):
    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    assert len(River.crosssections) == xs_total_no and len(River.crosssections.columns) == xs_col_no

def test_read_rivernetwork_method(
        version: int,
        river_network_path: str
):
    River = R.River('HM', version=version)
    River.RiverNetwork(river_network_path)
    assert len(River.rivernetwork) == 3 and len(River.rivernetwork.columns) == 3


def test_create_sub_instance(
        segment1: int,
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        slope_path: str,
        create_sub_instance_firstxs: int,
        create_sub_instance_lastxs: int
):
    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.Slope(slope_path)
    Sub = R.Sub(segment1, River)
    assert Sub.firstxs == create_sub_instance_firstxs and Sub.lastxs == create_sub_instance_lastxs
    assert Sub.slope


def test_sub_GetFlow(
        segment1: int,
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        slope_path: str,
        dates: list,
        interface_bc_path: str,
        interface_bc_folder: str,
        interface_bc_date_format: str,
        interface_Laterals_table_path: str,
        interface_Laterals_folder: str,
        interface_Laterals_date_format: str,
        sub_GetFlow_lateralTable: List[int],
        test_time_series_length: int,
):
    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.Slope(slope_path)
    Sub = R.Sub(segment1, River)

    IF = Interface('Rhine', start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)
    IF.ReadBoundaryConditions(path=interface_bc_folder, date_format=interface_bc_date_format)

    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)
    IF.ReadLaterals(path=interface_Laterals_folder, date_format=interface_Laterals_date_format)

    Sub.GetFlow(IF)

    assert len(Sub.BC) == len(Sub.Laterals) == test_time_series_length and len(Sub.BC.columns) == 1 and len(Sub.Laterals.columns) == 4
    assert all(elem in Sub.LateralsTable for elem in sub_GetFlow_lateralTable)


def test_Read1DResult(
        version: int,
        river_cross_section_path: str,
        segment1: int,
        Read1DResult_path: str,
        Read1DResult_xsid: int,
        test_time_series_length: int,
):

    River = R.River('HM', version=version)
    River.onedresultpath = Read1DResult_path
    River.ReadCrossSections(river_cross_section_path)
    Sub = R.Sub(segment1, River)
    Sub.Read1DResult()
    assert len(Sub.Result1D) == test_time_series_length * 24 * (len(Sub.crosssections) + 1) and len(Sub.Result1D.columns) == 6
    assert len(Sub.XSHydrographs) == test_time_series_length * 24 and len(Sub.XSHydrographs.columns) == 2
    assert len(Sub.XSWaterLevel) == test_time_series_length * 24 and len(Sub.XSWaterLevel.columns) == 2
    assert len(Sub.XSWaterDepth) == test_time_series_length * 24 and len(Sub.XSWaterDepth.columns) == 2
    Sub.Read1DResult(xsid=Read1DResult_xsid)

    assert Read1DResult_xsid in Sub.XSHydrographs.columns.tolist()
    assert Read1DResult_xsid in Sub.XSWaterLevel.columns.tolist()
    assert Read1DResult_xsid in Sub.XSWaterDepth.columns.tolist()


def test_Sub_GetLaterals(
        segment1: int,
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        slope_path: str,
        dates: list,
        interface_bc_path: str,
        interface_bc_folder: str,
        interface_bc_date_format: str,

        interface_Laterals_table_path: str,
        interface_Laterals_folder: str,
        interface_Laterals_date_format: str,
        sub_GetFlow_lateralTable: List[int],
        create_sub_instance_lastxs: int,
        test_time_series_length: int,
):
    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.Slope(slope_path)
    Sub = R.Sub(segment1, River)

    IF = Interface('Rhine', start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)
    IF.ReadBoundaryConditions(path=interface_bc_folder, date_format=interface_bc_date_format)

    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)
    IF.ReadLaterals(path=interface_Laterals_folder, date_format=interface_Laterals_date_format)

    Sub.GetFlow(IF)
    Sub.GetLaterals(create_sub_instance_lastxs)

    assert len(Sub.BC) == len(Sub.Laterals) == test_time_series_length and len(Sub.BC.columns) == 1 \
           and len(Sub.Laterals.columns) == len(sub_GetFlow_lateralTable) + 1
    assert all(elem in Sub.LateralsTable for elem in sub_GetFlow_lateralTable)


def test_ReadRRMHydrograph_one_location(
        version: int,
        river_cross_section_path: str,
        ReadRRMHydrograph_path: str,
        segment1: int,
        ReadRRMHydrograph_station_id: int,
        ReadRRMHydrograph_date_format: str,
        test_time_series_length: int,
):
    River = R.River('HM', version=version)
    River.rrmpath = ReadRRMHydrograph_path
    River.ReadCrossSections(river_cross_section_path)
    Sub = R.Sub(segment1, River)
    Sub.ReadRRMHydrograph(ReadRRMHydrograph_station_id,
                          date_format=ReadRRMHydrograph_date_format,
                          )

    assert len(Sub.RRM) == test_time_series_length and len(Sub.RRM.columns) == 1
    assert Sub.RRM.columns.to_list()[0] == ReadRRMHydrograph_station_id


def test_ReadRRMHydrograph_two_location(
        version: int,
        river_cross_section_path: str,
        ReadRRMHydrograph_path: str,
        segment1: int,
        ReadRRMHydrograph_station_id: int,
        ReadRRMHydrograph_date_format: str,
        ReadRRMHydrograph_location_2: int,
        ReadRRMHydrograph_path2: str,
        test_time_series_length: int,
):
    River = R.River('HM', version=version)
    River.rrmpath = ReadRRMHydrograph_path
    River.ReadCrossSections(river_cross_section_path)
    Sub = R.Sub(segment1, River)
    Sub.ReadRRMHydrograph(ReadRRMHydrograph_station_id,
                          date_format=ReadRRMHydrograph_date_format,
                          location=ReadRRMHydrograph_location_2,
                          path2=ReadRRMHydrograph_path2
                          )

    assert len(Sub.RRM) == test_time_series_length and len(Sub.RRM.columns) == 1
    assert Sub.RRM.columns.to_list()[0] == ReadRRMHydrograph_station_id
    assert len(Sub.RRM2) == test_time_series_length and len(Sub.RRM.columns) == 1
    assert Sub.RRM2.columns.to_list()[0] == ReadRRMHydrograph_station_id


def test_ReadUSHydrograph(
        version: int,
        CustomizedRunspath: str,
        river_cross_section_path: str,
        river_network_path: str,
        segment3: int,
        test_time_series_length: int,
        segment3_us_subs: List[int]
):
    River = R.River('HM', version=version)
    River.CustomizedRunspath = CustomizedRunspath
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    Sub = R.Sub(segment3, River)
    Sub.ReadUSHydrograph()
    assert len(Sub.USHydrographs) == test_time_series_length \
           and len(Sub.USHydrographs.columns) == len(segment3_us_subs) +1
    assert all(elem in Sub.USHydrographs.columns.tolist() for elem in segment3_us_subs)

def test_GetTotalFlow(
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        CustomizedRunspath: str,
        segment3: int,
        create_sub_instance_lastxs: int,
        dates:list,
        interface_bc_path: str,
        interface_bc_folder: str,
        interface_bc_date_format: str,
        interface_Laterals_table_path: str,
        interface_Laterals_folder: str,
        interface_Laterals_date_format: str,
        test_time_series_length: int,
):
    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.CustomizedRunspath = CustomizedRunspath
    IF = Interface('Rhine', start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)
    IF.ReadBoundaryConditions(path=interface_bc_folder, date_format=interface_bc_date_format)

    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)
    IF.ReadLaterals(path=interface_Laterals_folder, date_format=interface_Laterals_date_format)

    Sub = R.Sub(segment3, River)
    Sub.GetFlow(IF)
    Sub.ReadUSHydrograph()

    Sub.GetTotalFlow(create_sub_instance_lastxs)
    assert len(Sub.TotalFlow) == test_time_series_length
    assert 'total' in Sub.TotalFlow.columns.to_list()


def test_PlotQ(
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        CustomizedRunspath: str,
        Read1DResult_path: str,
        segment3: int,
        create_sub_instance_lastxs: int,
        dates:list,
        interface_bc_path: str,
        interface_bc_folder: str,
        interface_bc_date_format: str,
        interface_Laterals_table_path: str,
        interface_Laterals_folder: str,
        interface_Laterals_date_format: str,
        gauges_table_path: str,
        ReadObservedQ_Path: str,
        nodatavalu: int,
        gauges_file_extension: str,
        gauge_date_format: str,
        segment3_specificxs_plot: int,

):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadObservedQ(ReadObservedQ_Path, dates[0], dates[1],
                        nodatavalu, file_extension=gauges_file_extension,
                        gauge_date_format=gauge_date_format)

    gaugei = 0
    gauges = Calib.GaugesTable.loc[Calib.GaugesTable['id'] == segment3, :]
    gauges.index = range(len(gauges))
    stationname = gauges.loc[gaugei, 'oid']
    gaugename = str(gauges.loc[gaugei, 'name'])
    gaugexs = gauges.loc[gaugei, 'xsid']
    segment_xs = str(segment3) + "_" + str(gaugexs)


    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.CustomizedRunspath = CustomizedRunspath
    River.onedresultpath = Read1DResult_path

    IF = Interface('Rhine', start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)
    IF.ReadBoundaryConditions(path=interface_bc_folder, date_format=interface_bc_date_format)

    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)
    IF.ReadLaterals(path=interface_Laterals_folder, date_format=interface_Laterals_date_format)

    Sub = R.Sub(segment3, River)
    Sub.GetFlow(IF)
    Sub.ReadUSHydrograph()
    Sub.Read1DResult()
    fig, ax = Sub.PlotQ(Calib, gaugexs, dates[0], dates[1], stationname, gaugename, segment_xs,
                        specificxs=segment3_specificxs_plot, xlabels=5, ylabels=5)
    plt.close()
    assert isinstance(fig, Figure)

def test_CalculateQMetrics(
        version: int,
        dates:list,
        river_cross_section_path: str,
        river_network_path: str,
        Read1DResult_path: str,
        gauges_table_path: str,
        ReadObservedQ_Path: str,
        nodatavalu: int,
        gauges_file_extension: str,
        gauge_date_format: str,
        segment3: int,

):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadObservedQ(ReadObservedQ_Path, dates[0], dates[1],
                        nodatavalu, file_extension=gauges_file_extension,
                        gauge_date_format=gauge_date_format)

    gaugei = 0
    gauges = Calib.GaugesTable.loc[Calib.GaugesTable['id'] == segment3, :]
    gauges.index = range(len(gauges))
    stationname = gauges.loc[gaugei, 'oid']

    River = R.River('HM', version=version, start=dates[0])
    River.onedresultpath = Read1DResult_path
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)

    Sub = R.Sub(segment3, River)
    Sub.Read1DResult()

    # without filter
    Sub.CalculateQMetrics(Calib, stationname, Sub.lastxs)

    Sub.CalculateQMetrics(Calib, stationname, Sub.lastxs, Filter=True,
                          start=dates[0], end=dates[1])

def test_PlotHydrographProgression(
        version: int,
        dates: list,
        river_cross_section_path: str,
        river_network_path: str,
        Read1DResult_path: str,
        segment3: int,
):
    River = R.River('HM', version=version, start=dates[0])
    River.onedresultpath = Read1DResult_path
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)

    Sub = R.Sub(segment3, River)
    Sub.Read1DResult()

    xss = []
    start = dates[0]
    end = dates[1]
    fromxs = ''
    toxs = ''
    fig, ax = Sub.PlotHydrographProgression(xss, start, end, fromxs=fromxs,
                                            toxs=toxs, linewidth=2, spacing=20,
                                            figsize=(6, 4), xlabels=5)
    plt.close()
    assert isinstance(fig, Figure)

def test_PlotWL(
        version: int,
        dates: list,
        gauges_table_path: str,
        ReadObservedWL_Path: str,
        nodatavalu: int,
        gauges_file_extension: str,
        gauge_date_format: str,
        segment3: int,
        river_cross_section_path: str,
        river_network_path: str,
        Read1DResult_path: str,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadObservedWL(ReadObservedWL_Path, dates[0], dates[1],
                        nodatavalu, file_extension=gauges_file_extension,
                        gauge_date_format=gauge_date_format)

    gaugei = 0
    gauges = Calib.GaugesTable.loc[Calib.GaugesTable['id'] == segment3, :]
    gauges.index = range(len(gauges))
    stationname = gauges.loc[gaugei, 'oid']
    gaugename = str(gauges.loc[gaugei, 'name'])
    gaugexs = gauges.loc[gaugei, 'xsid']

    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.onedresultpath = Read1DResult_path

    Sub = R.Sub(segment3, River)
    Sub.Read1DResult()

    Sub.PlotWL(Calib, dates[0], dates[1], gaugexs, stationname, gaugename,
               plotgauge=True)
    plt.close()


def test_CalculateWLMetrics(
        version: int,
        dates:list,
        river_cross_section_path: str,
        river_network_path: str,
        Read1DResult_path: str,
        gauges_table_path: str,
        ReadObservedWL_Path: str,
        nodatavalu: int,
        gauges_file_extension: str,
        gauge_date_format: str,
        segment3: int,

):
    Calib = RC.Calibration("HM", version=version)
    Calib.ReadGaugesTable(gauges_table_path)
    Calib.ReadObservedWL(ReadObservedWL_Path, dates[0], dates[1],
                         nodatavalu, file_extension=gauges_file_extension,
                         gauge_date_format=gauge_date_format)

    gaugei = 0
    gauges = Calib.GaugesTable.loc[Calib.GaugesTable['id'] == segment3, :]
    gauges.index = range(len(gauges))
    stationname = gauges.loc[gaugei, 'oid']

    River = R.River('HM', version=version, start=dates[0])
    River.onedresultpath = Read1DResult_path
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)

    Sub = R.Sub(segment3, River)
    Sub.Read1DResult()

    # without filter
    Sub.CalculateWLMetrics(Calib, stationname, Sub.lastxs)

    Sub.CalculateWLMetrics(Calib, stationname, Sub.lastxs,
                          Filter=True, start=dates[0], end=dates[1])

def test_SaveHydrograph(
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        CustomizedRunspath: str,
        Read1DResult_path: str,
        segment3: int,

):
    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.CustomizedRunspath = CustomizedRunspath
    River.onedresultpath = Read1DResult_path
    Sub = R.Sub(segment3, River)
    Sub.Read1DResult()
    Sub.SaveHydrograph(Sub.lastxs)
    # option 2
    Sub.SaveHydrograph(Sub.lastxs, Option=2)


def test_ReadBoundaryConditions(
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        dates:list,
        Read1DResult_path: str,
        usbc_path: str,
        segment3: int,
        test_time_series_length: int,
        test_hours: list
):
    River = R.River('HM', version=version, start=dates[0])
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.onedresultpath = Read1DResult_path
    River.usbcpath = usbc_path
    Sub = R.Sub(segment3, River)
    Sub.Read1DResult()
    Sub.ReadBoundaryConditions(start=dates[0], end=dates[1])

    assert len(Sub.QBC) == test_time_series_length and all(elem in test_hours for elem in Sub.QBC.columns.tolist())
    assert len(Sub.HBC) == test_time_series_length and all(elem in test_hours for elem in Sub.HBC.columns.tolist())


def test_ReadSubDailyResults(
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        dates:list,
        segment3: int,
        lastsegment: bool,
        subdailyresults_path: str,
        usbc_path: str,
        segment3_xs_ids_list: list,
        subdaily_no_timesteps: int,
        onemin_results_dates: list,
        onemin_results_len: int,
):
    River = R.River('HM', version=version, start=dates[0])
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.usbcpath = usbc_path
    River.oneminresultpath = subdailyresults_path
    Sub = R.Sub(segment3, River)
    Sub.ReadSubDailyResults(onemin_results_dates[0],
                            onemin_results_dates[1],
                            Lastsegment=lastsegment)
    assert len(Sub.h) == onemin_results_len*subdaily_no_timesteps
    assert all(elem in Sub.h.columns.tolist() for elem in segment3_xs_ids_list)
    assert len(Sub.q) == onemin_results_len*subdaily_no_timesteps
    assert all(elem in Sub.q.columns.tolist() for elem in segment3_xs_ids_list)
    assert len(Sub.QBCmin.columns) == subdaily_no_timesteps and len(Sub.QBCmin) == onemin_results_len
    assert len(Sub.HBCmin.columns) == subdaily_no_timesteps and len(Sub.HBCmin) == onemin_results_len

def test_PlotBC(
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        dates:list,
        segment3: int,
        lastsegment: bool,
        subdailyresults_path: str,
        usbc_path: str,
        onemin_results_dates: list,
):
    River = R.River('HM', version=version, start=dates[0])
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.usbcpath = usbc_path
    River.oneminresultpath = subdailyresults_path
    Sub = R.Sub(segment3, River)
    Sub.ReadSubDailyResults(onemin_results_dates[0],
                            onemin_results_dates[1],
                            Lastsegment=lastsegment)
    Sub.PlotBC(dates[0])


def test_StatisticalProperties(
        version: int,
        distribution_properties_fpath: str,
        statistical_properties_columns: list,
        distributionpr_gev_columns: list,
        distributionpr_gum_columns:list,
):
    River = R.River('HM', version=version)
    River.StatisticalProperties(distribution_properties_fpath, Distibution="GEV")
    assert all(elem in River.SP.columns.to_list() for elem in statistical_properties_columns)
    assert all(elem in River.SP.columns.to_list() for elem in distributionpr_gev_columns[:3])

    River.StatisticalProperties(distribution_properties_fpath, Distibution="Gumbel")
    assert all(elem in River.SP.columns.to_list() for elem in statistical_properties_columns)
    assert all(elem in River.SP.columns.to_list() for elem in distributionpr_gum_columns[:2])


# def test_GetCapacity(
#         version: int,
#         river_cross_section_path: str,
#         distribution_properties_hm_results_fpath: str,
# ):
#     River = R.River('HM', version=version)
#     River.ReadCrossSections(river_cross_section_path)
#     River.StatisticalProperties(distribution_properties_hm_results_fpath)
#     River.GetCapacity('Qbkf')
#     River.GetCapacity('Qc2', Option=2)
#     cols = River.crosssections.columns.tolist()
#     assert "Slope" in cols
#     assert "Qbkf" in cols
#     assert "QbkfRP" in cols
#     assert "Qc2" in cols
#     assert "Qc2RP" in cols

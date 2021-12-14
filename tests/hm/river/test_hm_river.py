from typing import List
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
        river_cross_section_path: str
):
    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    assert len(River.crosssections) == 300 and len(River.crosssections.columns) == 16

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
    Laterals = Sub.GetLaterals(create_sub_instance_lastxs)

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

# def test_GetTotalFlow(
#         version: int,
#         river_cross_section_path: str,
#         river_network_path: str,
# ):
#     River = R.River('HM', version=version)
#     River.ReadCrossSections(river_cross_section_path)
#     River.RiverNetwork(river_network_path)
#     Sub = R.Sub(segment3, River)
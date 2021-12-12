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
        create_sub_instance_subid: int,
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
    Sub = R.Sub(create_sub_instance_subid, River)
    assert Sub.firstxs == create_sub_instance_firstxs and Sub.lastxs == create_sub_instance_lastxs
    assert Sub.slope

def test_sub_GetFlow(
        create_sub_instance_subid: int,
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
):
    River = R.River('HM', version=version)
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.Slope(slope_path)
    Sub = R.Sub(create_sub_instance_subid, River)

    IF = Interface('Rhine', start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)
    IF.ReadBoundaryConditions(path=interface_bc_folder, date_format=interface_bc_date_format)

    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)
    IF.ReadLaterals(path=interface_Laterals_folder, date_format=interface_Laterals_date_format)

    Sub.GetFlow(IF)

    assert len(Sub.BC) == len(Sub.Laterals) == 80 and len(Sub.BC.columns) == 1 and len(Sub.Laterals.columns) == 4


def test_Read1DResult(
        version: int,
        river_cross_section_path: str,
        create_sub_instance_subid: int,
        test_Read1DResult_path: str,
        test_Read1DResult_xsid: int,
):

    River = R.River('HM', version=version)
    River.onedresultpath = test_Read1DResult_path
    River.ReadCrossSections(river_cross_section_path)
    Sub = R.Sub(create_sub_instance_subid, River)
    Sub.Read1DResult()
    assert len(Sub.Result1D) == 80 * 24 * (len(Sub.crosssections) + 1) and len(Sub.Result1D.columns) == 6
    assert len(Sub.XSHydrographs) == 80 * 24 and len(Sub.XSHydrographs.columns) == 2
    assert len(Sub.XSWaterLevel) == 80 * 24 and len(Sub.XSWaterLevel.columns) == 2
    assert len(Sub.XSWaterDepth) == 80 * 24 and len(Sub.XSWaterDepth.columns) == 2
    Sub.Read1DResult(xsid=test_Read1DResult_xsid)

    assert test_Read1DResult_xsid in Sub.XSHydrographs.columns.tolist()
    assert test_Read1DResult_xsid in Sub.XSWaterLevel.columns.tolist()
    assert test_Read1DResult_xsid in Sub.XSWaterDepth.columns.tolist()

    Sub.firstday == dt.datetime.strptime("1955-01-01", "%Y-%m-%d")

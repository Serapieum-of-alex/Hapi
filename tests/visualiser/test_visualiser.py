from collections import OrderedDict

from Hapi.visualizer import Visualize as V
import Hapi.hm.river as R
from Hapi.hm.interface import Interface

def test_create_visualize_instance(

):
    Vis = V(resolution="Hourly")
    assert isinstance(Vis.MarkerStyleList, list)
    assert isinstance(Vis.FigureDefaultOptions, dict)
    assert isinstance(Vis.linestyles, OrderedDict)

def test_GroundSurface(
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        dates:list,
        segment3: int,
        interface_bc_path: str,
        interface_bc_folder: str,
        interface_bc_date_format: str,
        interface_Laterals_table_path: str,
        interface_Laterals_folder: str,
        interface_Laterals_date_format: str,
        Read1DResult_path: str,

):
    River = R.River('HM', version=version, start=dates[0])
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.onedresultpath = Read1DResult_path
    Sub = R.Sub(segment3, River)

    IF = Interface('Rhine', start=dates[0])
    IF.ReadBoundaryConditionsTable(interface_bc_path)
    IF.ReadBoundaryConditions(path=interface_bc_folder, date_format=interface_bc_date_format)

    IF.ReadCrossSections(river_cross_section_path)
    IF.ReadLateralsTable(interface_Laterals_table_path)
    IF.ReadLaterals(path=interface_Laterals_folder, date_format=interface_Laterals_date_format)

    Sub.GetFlow(IF)
    Sub.Read1DResult()

    Vis = V(resolution="Hourly")
    Vis.GroundSurface(Sub, floodplain=True, plotlateral=True, nxlabels=20, option=2)
    Vis.GroundSurface(Sub, floodplain=True, plotlateral=True, nxlabels=20, option=1)


def test_CrossSections(
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        dates:list,
        segment3: int,
        interface_bc_path: str,
        interface_bc_folder: str,
        interface_bc_date_format: str,
        interface_Laterals_table_path: str,
        interface_Laterals_folder: str,
        interface_Laterals_date_format: str,
        Read1DResult_path: str,
        plot_xs_seg3_fromxs: int,
        plot_xs_seg3_toxs: int
):
    River = R.River('HM', version=version, start=dates[0])
    River.ReadCrossSections(river_cross_section_path)
    River.RiverNetwork(river_network_path)
    River.onedresultpath = Read1DResult_path
    Sub = R.Sub(segment3, River)
    Vis = V(resolution="Hourly")

    fig, ax = Vis.CrossSections(Sub, bedlevel=True, fromxs=plot_xs_seg3_fromxs,
                                toxs=plot_xs_seg3_toxs, samescale=True,
                                textspacing=[(1, 1), (1, 4)],
                                plottingoption=3)


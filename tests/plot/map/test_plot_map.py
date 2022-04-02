import pytest
from matplotlib.figure import Figure
from osgeo.gdal import Dataset

from Hapi.plot.map import Map


class TestPlotArray:

    def test_plot_array(
            self,
            src: Dataset
    ):
        fig, ax = Map.PlotArray(src, Title="Flow Accumulation")
        assert isinstance(fig, Figure)

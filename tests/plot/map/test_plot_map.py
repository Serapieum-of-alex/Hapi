import numpy as np
import pytest
from matplotlib.figure import Figure
from osgeo.gdal import Dataset

from Hapi.plot.map import Map


class TestPlotArray:

    def test_plot_gdal_object(
            self,
            src: Dataset
    ):
        fig, ax = Map.PlotArray(src, Title="Flow Accumulation")
        assert isinstance(fig, Figure)

    def test_plot_numpy_array(
            self,
            src_arr: np.ndarray,
            src_no_data_value: float,
    ):
        fig, ax = Map.PlotArray(src_arr, nodataval=src_no_data_value, Title="Flow Accumulation")
        assert isinstance(fig, Figure)

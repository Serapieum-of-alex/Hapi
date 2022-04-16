from typing import List

import numpy as np
import pandas as pd
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

    def test_plot_array_color_scale_1(
            self,
            src: Dataset,
            cmap: str,
            ColorScale: List[int],
            TicksSpacing: int
    ):
        fig, ax = Map.PlotArray(src, ColorScale=ColorScale[0], cmap=cmap, TicksSpacing=TicksSpacing)
        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_2(
            self,
            src: Dataset,
            cmap: str,
            color_scale_2_gamma: float,
            ColorScale: List[int],
            TicksSpacing: int
    ):
        fig, ax = Map.PlotArray(src, ColorScale=ColorScale[1], cmap=cmap, gamma=color_scale_2_gamma,
                                TicksSpacing=TicksSpacing)
        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_3(
            self,
            src: Dataset,
            cmap: str,
            ColorScale: List[int],
            TicksSpacing: int,
            color_scale_3_linscale: float,
            color_scale_3_linthresh: float,
    ):
        fig, ax = Map.PlotArray(
            src,
            ColorScale=ColorScale[2],
            linscale=color_scale_3_linscale,
            linthresh=color_scale_3_linthresh,
            cmap=cmap,
            TicksSpacing=TicksSpacing
        )

        assert isinstance(fig, Figure)


    def test_plot_array_color_scale_4(
            self,
            src: Dataset,
            cmap: str,
            ColorScale: List[int],
            TicksSpacing: int
    ):
        fig, ax = Map.PlotArray(
            src,
            ColorScale=ColorScale[3],
            cmap=cmap,
            TicksSpacing=TicksSpacing
        )

        assert isinstance(fig, Figure)

    def test_plot_array_color_scale_5(
            self,
            src: Dataset,
            cmap: str,
            ColorScale: List[int],
            TicksSpacing: int,
            midpoint: int
    ):
        fig, ax = Map.PlotArray(
            src,
            ColorScale=ColorScale[4],
            midpoint=midpoint,
            cmap=cmap,
            TicksSpacing=TicksSpacing
        )

        assert isinstance(fig, Figure)


    def test_plot_array_display_cell_values(
            self,
            src: Dataset,
            TicksSpacing: int,
            display_cellvalue: bool,
            NumSize: int,
            Backgroundcolorthreshold,
    ):



        fig, ax = Map.PlotArray(
            src,
            display_cellvalue=display_cellvalue,
            NumSize=NumSize,
            Backgroundcolorthreshold=Backgroundcolorthreshold,
            TicksSpacing=TicksSpacing,
        )

        assert isinstance(fig, Figure)


    def test_plot_array_with_points(
            self,
            src: Dataset,
            display_cellvalue: bool,
            points: pd.DataFrame,
            NumSize: int,
            Backgroundcolorthreshold,
            TicksSpacing: int,
            IDsize: int,
            IDcolor: str,
            Gaugesize: int,
            Gaugecolor: str,
    ):
        fig, ax = Map.PlotArray(
            src,
            Gaugecolor=Gaugecolor,
            Gaugesize=Gaugesize,
            IDcolor=IDcolor,
            IDsize=IDsize,
            points=points,
            display_cellvalue=display_cellvalue,
            NumSize=NumSize,
            Backgroundcolorthreshold=Backgroundcolorthreshold,
            TicksSpacing=TicksSpacing,
        )

        assert isinstance(fig, Figure)

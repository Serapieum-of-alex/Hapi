from typing import Any, List, Tuple, Union

import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame
from loguru import logger
from matplotlib import gridspec  # animation,
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import LogFormatter
from osgeo import gdal

from Hapi.gis.giscatchment import GISCatchment as GC
from Hapi.gis.raster import Raster
from Hapi.plot.visualizer import MidpointNormalize, Visualize
from Hapi.sm.statisticaltools import StatisticalTools as ST


class Map:
    """
    Map
    """


    def __init__(self):
        pass


    @staticmethod
    def PlotCatchment(
            Metrics: GeoDataFrame,
            ColumnName: Any,
            Basin: GeoDataFrame,
            River: GeoDataFrame,
            scheme: Any = None,
            scale_func: Any = '',
            cmap: str = "viridis",
            legend_values: List = [],
            legend_labels: List = [],
            figsize: Tuple = (8, 8),
            Title: Any = 'Title',
            TitleSize: int = 500,
            Save: Union[bool, str] = False,
    ):
        """PlotCatchment.

        Inputs:
        ------
            Metrics:[GeoDataFrame]
                geodataframe contains values to plot in one of its columns.
            ColumnName: [str]
                name of the column you want to plot its values.
            Basin: [GeoDataFrame]
                geodataframe contains polygon geometries.
            River: [GeoDataFrame]
                geodataframe contains linestring geometries.
            figsize: [Tuple]
                fize oif the figure.
            Title:[str]
                title of the figure.
            Save: [bool/str]
                if you want to save the plot provide the path with the extention,
                Default is False.
        """
        # unify the projection
        if not Basin.crs.is_geographic:
            logger.debug("The coordinate system of the Basin geodataframe is not geographic"
                         "SO, it will be reprojected to WGS-84")
            Basin.to_crs(4326, inplace=True)

        epsg = Basin.crs.to_json()
        River.to_crs(epsg, inplace=True)
        Metrics.to_crs(epsg, inplace=True)

        pointplot_kwargs = {'edgecolor': 'white', 'linewidth': 0.9}  # 'color': "crimson"

        # make sure that the plotted column is numeric
        Metrics[ColumnName] = Metrics[ColumnName].map(float)

        fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': gcrs.AlbersEqualArea()})
        if scheme:

            gplt.pointplot(Metrics, projection=gcrs.AlbersEqualArea(),
                           hue=ColumnName, cmap=cmap,
                           scale=ColumnName, limits=(4, 20),
                           scheme=scheme,
                           # scale_func = scale_func,
                           legend=True,
                           legend_var='scale',
                           legend_kwargs={  # 'loc': 'upper right',
                               'bbox_to_anchor': (1, 0.35)},
                           ax=ax, **pointplot_kwargs  # ,
                           )
        else:
            gplt.pointplot(Metrics, projection=gcrs.AlbersEqualArea(),
                           hue=ColumnName, cmap=cmap,
                           scale=ColumnName, limits=(4, 20),
                           scale_func=scale_func,
                           legend=True,
                           legend_var='scale',
                           legend_values=legend_values,
                           legend_labels=legend_labels,
                           legend_kwargs={  # 'loc': 'upper right',
                               'bbox_to_anchor': (1, 0.35)},
                           ax=ax, **pointplot_kwargs  # ,
                           )

        gplt.polyplot(Basin, ax=ax, edgecolor='grey', facecolor='grey',  # 'lightgray',
                      linewidth=0.5, extent=Basin.total_bounds)  # # , zorder=0

        gplt.polyplot(River, ax=ax, linewidth=10)

        plt.title(Title, fontsize=TitleSize)
        # plt.subplots_adjust(top=0.99999, right=0.9999, left=0.000005, bottom=0.000005)
        if Save:
            plt.savefig(Save, bbox_inches='tight', transparent=True)

        return fig, ax


    @staticmethod
    def AnimateArray(
            Arr,
            Time,
            NoElem,
            TicksSpacing=2,
            Figsize=(8, 8),
            PlotNumbers=True,
            NumSize=8,
            Title="Total Discharge",
            titlesize=15,
            Backgroundcolorthreshold=None,
            cbarlabel="Discharge m3/s",
            cbarlabelsize=12,
            textcolors=("white", "black"),
            Cbarlength=0.75,
            interval=200,
            cmap="coolwarm_r",
            Textloc=[0.1, 0.2],
            Gaugecolor="red",
            Gaugesize=100,
            ColorScale=1,
            gamma=0.5,
            linthresh=0.0001,
            linscale=0.001,
            midpoint=0,
            orientation="vertical",
            rotation=-90,
            IDcolor="blue",
            IDsize=10,
            **kwargs
    ):
        """AnimateArray.

        plot an animation for 3d arrays

        Parameters
        ----------
        Arr : [array]
            the array you want to animate.
        Time : [dataframe]
            dataframe contains the date of values.
        NoElem : [integer]
            Number of the cells that has values.
        TicksSpacing : [integer], optional
            Spacing in the colorbar ticks. The default is 2.
        Figsize : [tuple], optional
            figure size. The default is (8,8).
        PlotNumbers : [bool], optional
            True to plot the values intop of each cell. The default is True.
        NumSize : integer, optional
            size of the numbers plotted intop of each cells. The default is 8.
        Title : [str], optional
            title of the plot. The default is 'Total Discharge'.
        titlesize : [integer], optional
            title size. The default is 15.
        Backgroundcolorthreshold : [float/integer], optional
            threshold value if the value of the cell is greater, the plotted
            numbers will be black and if smaller the plotted number will be white
            if None given the maxvalue/2 will be considered. The default is None.
        textcolors : TYPE, optional
            Two colors to be used to plot the values i top of each cell. The default is ("white","black").
        cbarlabel : str, optional
            label of the color bar. The default is 'Discharge m3/s'.
        cbarlabelsize : integer, optional
            size of the color bar label. The default is 12.
        Cbarlength : [float], optional
            ratio to control the height of the colorbar. The default is 0.75.
        interval : [integer], optional
            number to controlthe speed of the animation. The default is 200.
        cmap : [str], optional
            color style. The default is 'coolwarm_r'.
        Textloc : [list], optional
            location of the date text. The default is [0.1,0.2].
        Gaugecolor : [str], optional
            color of the points. The default is 'red'.
        Gaugesize : [integer], optional
            size of the points. The default is 100.
        IDcolor : [str]
            the ID of the Point.The default is "blue".
        IDsize : [integer]
            size of the ID text. The default is 10.
        ColorScale : integer, optional
            there are 5 options to change the scale of the colors. The default is 1.
            1- ColorScale 1 is the normal scale
            2- ColorScale 2 is the power scale
            3- ColorScale 3 is the SymLogNorm scale
            4- ColorScale 4 is the PowerNorm scale
            5- ColorScale 5 is the BoundaryNorm scale
            ------------------------------------------------------------------
            gamma : [float], optional
                value needed for option 2 . The default is 1./2..
            linthresh : [float], optional
                value needed for option 3. The default is 0.0001.
            linscale : [float], optional
                value needed for option 3. The default is 0.001.
            midpoint : [float], optional
                value needed for option 5. The default is 0.
            ------------------------------------------------------------------
        orientation : [string], optional
            orintation of the colorbar horizontal/vertical. The default is 'vertical'.
        rotation : [number], optional
            rotation of the colorbar label. The default is -90.
        **kwargs : [dict]
            keys:
                Points : [dataframe].
                    dataframe contains two columns 'cell_row', and cell_col to
                    plot the point at this location

        Returns
        -------
        animation.FuncAnimation.

        """
        fig = plt.figure(60, figsize=Figsize)
        gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
        ax = fig.add_subplot(gs[:, :])
        ticks = np.arange(np.nanmin(Arr), np.nanmax(Arr), TicksSpacing)

        if ColorScale == 1:
            im = ax.matshow(
                Arr[:, :, 0], cmap=cmap, vmin=np.nanmin(Arr), vmax=np.nanmax(Arr)
            )
            cbar_kw = dict(ticks=ticks)
        elif ColorScale == 2:
            im = ax.matshow(
                Arr[:, :, 0],
                cmap=cmap,
                norm=colors.PowerNorm(
                    gamma=gamma, vmin=np.nanmin(Arr), vmax=np.nanmax(Arr)
                ),
            )
            cbar_kw = dict(ticks=ticks)
        elif ColorScale == 3:
            im = ax.matshow(
                Arr[:, :, 0],
                cmap=cmap,
                norm=colors.SymLogNorm(
                    linthresh=linthresh,
                    linscale=linscale,
                    base=np.e,
                    vmin=np.nanmin(Arr),
                    vmax=np.nanmax(Arr),
                ),
            )
            formatter = LogFormatter(10, labelOnlyBase=False)
            cbar_kw = dict(ticks=ticks, format=formatter)
        elif ColorScale == 4:
            bounds = np.arange(np.nanmin(Arr), np.nanmax(Arr), TicksSpacing)
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.matshow(Arr[:, :, 0], cmap=cmap, norm=norm)
            cbar_kw = dict(ticks=ticks)
        else:
            im = ax.matshow(
                Arr[:, :, 0], cmap=cmap, norm=MidpointNormalize(midpoint=midpoint)
            )
            cbar_kw = dict(ticks=ticks)

        # Create colorbar
        cbar = ax.figure.colorbar(
            im, ax=ax, shrink=Cbarlength, orientation=orientation, **cbar_kw
        )
        cbar.ax.set_ylabel(cbarlabel, rotation=rotation, va="bottom")
        cbar.ax.tick_params(labelsize=10)

        day_text = ax.text(Textloc[0], Textloc[1], " ", fontsize=cbarlabelsize)
        ax.set_title(Title, fontsize=titlesize)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_yticks([])
        Indexlist = list()

        for x in range(Arr.shape[0]):
            for y in range(Arr.shape[1]):
                if not np.isnan(Arr[x, y, 0]):
                    Indexlist.append([x, y])

        Textlist = list()
        for x in range(NoElem):
            Textlist.append(
                ax.text(
                    Indexlist[x][1],
                    Indexlist[x][0],
                    round(Arr[Indexlist[x][0], Indexlist[x][1], 0], 2),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=NumSize,
                )
            )
        # Points = list()
        PoitsID = list()
        if "Points" in kwargs.keys():
            row = kwargs["Points"].loc[:, "cell_row"].tolist()
            col = kwargs["Points"].loc[:, "cell_col"].tolist()
            IDs = kwargs["Points"].loc[:, "id"].tolist()
            Points = ax.scatter(col, row, color=Gaugecolor, s=Gaugesize)

            for i in range(len(row)):
                PoitsID.append(
                    ax.text(
                        col[i],
                        row[i],
                        IDs[i],
                        ha="center",
                        va="center",
                        color=IDcolor,
                        fontsize=IDsize,
                    )
                )

        # Normalize the threshold to the images color range.
        if Backgroundcolorthreshold is not None:
            Backgroundcolorthreshold = im.norm(Backgroundcolorthreshold)
        else:
            Backgroundcolorthreshold = im.norm(np.nanmax(Arr)) / 2.0


        def init():
            im.set_data(Arr[:, :, 0])
            day_text.set_text("")

            output = [im, day_text]

            if "Points" in kwargs.keys():
                # plot gauges
                # for j in range(len(kwargs['Points'])):
                row = kwargs["Points"].loc[:, "cell_row"].tolist()
                col = kwargs["Points"].loc[:, "cell_col"].tolist()
                # Points[j].set_offsets(col, row)
                Points.set_offsets(np.c_[col, row])
                output.append(Points)

                for x in range(len(col)):
                    PoitsID[x].set_text(IDs[x])

                output = output + PoitsID

            if PlotNumbers:
                for x in range(NoElem):
                    val = round(Arr[Indexlist[x][0], Indexlist[x][1], 0], 2)
                    Textlist[x].set_text(val)

                output = output + Textlist

            return output


        def animate(i):
            im.set_data(Arr[:, :, i])
            day_text.set_text("Date = " + str(Time[i])[0:10])

            output = [im, day_text]

            if "Points" in kwargs.keys():
                # plot gauges
                # for j in range(len(kwargs['Points'])):
                row = kwargs["Points"].loc[:, "cell_row"].tolist()
                col = kwargs["Points"].loc[:, "cell_col"].tolist()
                # Points[j].set_offsets(col, row)
                Points.set_offsets(np.c_[col, row])
                output.append(Points)

                for x in range(len(col)):
                    PoitsID[x].set_text(IDs[x])

                output = output + PoitsID

            if PlotNumbers:
                for x in range(NoElem):
                    val = round(Arr[Indexlist[x][0], Indexlist[x][1], i], 2)
                    kw = dict(
                        color=textcolors[
                            int(
                                im.norm(Arr[Indexlist[x][0], Indexlist[x][1], i])
                                > Backgroundcolorthreshold
                            )
                        ]
                    )
                    Textlist[x].update(kw)
                    Textlist[x].set_text(val)

                output = output + Textlist

            return output


        plt.tight_layout()
        # global anim
        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=np.shape(Arr)[2],
            interval=interval,
            blit=True,
        )

        return anim


    @staticmethod
    def Plot_Type1(
            Y1,
            Y2,
            Points,
            PointsY,
            PointMaxSize=200,
            PointMinSize=1,
            X_axis_label="X Axis",
            LegendNum=5,
            LegendLoc=(1.3, 1),
            PointLegendTitle="Output 2",
            Ylim=[0, 180],
            Y2lim=[-2, 14],
            color1="#27408B",
            color2="#DC143C",
            color3="grey",
            linewidth=4,
            **kwargs
    ):
        """Plot_Type1.

        !TODO Needs docs

        Parameters
        ----------
        Y1 : TYPE
            DESCRIPTION.
        Y2 : TYPE
            DESCRIPTION.
        Points : TYPE
            DESCRIPTION.
        PointsY : TYPE
            DESCRIPTION.
        PointMaxSize : TYPE, optional
            DESCRIPTION. The default is 200.
        PointMinSize : TYPE, optional
            DESCRIPTION. The default is 1.
        X_axis_label : TYPE, optional
            DESCRIPTION. The default is 'X Axis'.
        LegendNum : TYPE, optional
            DESCRIPTION. The default is 5.
        LegendLoc : TYPE, optional
            DESCRIPTION. The default is (1.3, 1).
        PointLegendTitle : TYPE, optional
            DESCRIPTION. The default is "Output 2".
        Ylim : TYPE, optional
            DESCRIPTION. The default is [0,180].
        Y2lim : TYPE, optional
            DESCRIPTION. The default is [-2,14].
        color1 : TYPE, optional
            DESCRIPTION. The default is '#27408B'.
        color2 : TYPE, optional
            DESCRIPTION. The default is '#DC143C'.
        color3 : TYPE, optional
            DESCRIPTION. The default is "grey".
        linewidth : TYPE, optional
            DESCRIPTION. The default is 4.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        ax1 : TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        fig : TYPE
            DESCRIPTION.

        """
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        ax2 = ax1.twinx()

        ax1.plot(
            Y1[:, 0],
            Y1[:, 1],
            zorder=1,
            color=color1,
            linestyle=Visualize.LineStyle(0),
            linewidth=linewidth,
            label="Model 1 Output1",
        )

        if "Y1_2" in kwargs.keys():
            Y1_2 = kwargs["Y1_2"]

            rows_axis1, cols_axis1 = np.shape(Y1_2)

            if "Y1_2_label" in kwargs.keys():
                label = kwargs["Y2_2_label"]
            else:
                label = ["label"] * (cols_axis1 - 1)
            # first column is the x axis
            for i in range(1, cols_axis1):
                ax1.plot(
                    Y1_2[:, 0],
                    Y1_2[:, i],
                    zorder=1,
                    color=color2,
                    linestyle=Visualize.LineStyle(i),
                    linewidth=linewidth,
                    label=label[i - 1],
                )

        ax2.plot(
            Y2[:, 0],
            Y2[:, 1],
            zorder=1,
            color=color3,
            linestyle=Visualize.LineStyle(6),
            linewidth=2,
            label="Output1-Diff",
        )

        if "Y2_2" in kwargs.keys():
            Y2_2 = kwargs["Y2_2"]
            rows_axis2, cols_axis2 = np.shape(Y2_2)

            if "Y2_2_label" in kwargs.keys():
                label = kwargs["Y2_2_label"]
            else:
                label = ["label"] * (cols_axis2 - 1)

            for i in range(1, cols_axis2):
                ax1.plot(
                    Y2_2[:, 0],
                    Y2_2[:, i],
                    zorder=1,
                    color=color2,
                    linestyle=Visualize.LineStyle(i),
                    linewidth=linewidth,
                    label=label[i - 1],
                )

        if "Points1" in kwargs.keys():
            # first axis in the x axis
            Points1 = kwargs["Points1"]

            vmax = np.max(Points1[:, 1:])
            vmin = np.min(Points1[:, 1:])

            vmax = max(Points[:, 1].max(), vmax)
            vmin = min(Points[:, 1].min(), vmin)

        else:
            vmax = max(Points)
            vmin = min(Points)

        vmaxnew = PointMaxSize
        vminnew = PointMinSize

        Points_scaled = [
            ST.Rescale(x, vmin, vmax, vminnew, vmaxnew) for x in Points[:, 1]
        ]
        f1 = np.ones(shape=(len(Points))) * PointsY
        scatter = ax2.scatter(
            Points[:, 0],
            f1,
            zorder=1,
            c=color1,
            s=Points_scaled,
            label="Model 1 Output 2",
        )

        if "Points1" in kwargs.keys():
            row_points, col_points = np.shape(Points1)
            PointsY1 = kwargs["PointsY1"]
            f2 = np.ones_like(Points1[:, 1:])

            for i in range(col_points - 1):
                Points1_scaled = [
                    ST.Rescale(x, vmin, vmax, vminnew, vmaxnew) for x in Points1[:, i]
                ]
                f2[:, i] = PointsY1[i]

                ax2.scatter(
                    Points1[:, 0],
                    f2[:, i],
                    zorder=1,
                    c=color2,
                    s=Points1_scaled,
                    label="Model 2 Output 2",
                )

        # produce a legend with the unique colors from the scatter
        legend1 = ax2.legend(
            *scatter.legend_elements(), bbox_to_anchor=(1.1, 0.2)
        )  # loc="lower right", title="RIM"

        ax2.add_artist(legend1)

        # produce a legend with a cross section of sizes from the scatter
        handles, labels = scatter.legend_elements(
            prop="sizes", alpha=0.6, num=LegendNum
        )
        # L = [vminnew] + [float(i[14:-2]) for i in labels] + [vmaxnew]
        L = [float(i[14:-2]) for i in labels]
        labels1 = [round(ST.Rescale(x, vminnew, vmaxnew, vmin, vmax) / 1000) for x in L]

        legend2 = ax2.legend(
            handles, labels1, bbox_to_anchor=LegendLoc, title=PointLegendTitle
        )
        ax2.add_artist(legend2)

        ax1.set_ylim(Ylim)
        ax2.set_ylim(Y2lim)
        #
        ax1.set_ylabel("Output 1 (m)", fontsize=12)
        ax2.set_ylabel("Output 1 - Diff (m)", fontsize=12)
        ax1.set_xlabel(X_axis_label, fontsize=12)
        ax1.xaxis.set_minor_locator(plt.MaxNLocator(10))
        ax1.tick_params(which="minor", length=5)
        fig.legend(
            loc="lower center",
            bbox_to_anchor=(1.3, 0.3),
            bbox_transform=ax1.transAxes,
            fontsize=10,
        )
        plt.rcParams.update({"ytick.major.size": 3.5})
        plt.rcParams.update({"font.size": 12})
        plt.title("Model Output Comparison", fontsize=15)

        plt.subplots_adjust(right=0.7)
        # plt.tight_layout()

        return (ax1, ax2), fig


    @staticmethod
    def PlotArray(
            src,
            nodataval=np.nan,
            Figsize=(8, 8),
            Title="Total Discharge",
            titlesize=15,
            Cbarlength=0.75,
            orientation="vertical",
            cbarlabelsize=12,
            cbarlabel="Color bar label",
            rotation=-90,
            TicksSpacing=5,
            NumSize=8,
            ColorScale=1,
            cmap="coolwarm_r",
            gamma=0.5,
            linscale=0.001,
            linthresh=0.0001,
            midpoint=0,
            display_cellvalue=False,
            Backgroundcolorthreshold=None,
            Gaugecolor="red",
            Gaugesize=100,
            IDcolor="blue",
            IDsize=10,
            **kwargs
    ):
        """PlotArray.

        plot an image for 2d arrays

        Parameters
        ----------
            src : [array/gdal.Dataset]
                the array/gdal raster you want to plot.
            nodataval : [numeric]
                value used to fill cells out of the domain. Optional, Default is np.nan
                needed only in case of plotting array
            Figsize : [tuple], optional
                figure size. The default is (8,8).
            Title : [str], optional
                title of the plot. The default is 'Total Discharge'.
            titlesize : [integer], optional
                title size. The default is 15.
            Cbarlength : [float], optional
                ratio to control the height of the colorbar. The default is 0.75.
            orientation : [string], optional
                orintation of the colorbar horizontal/vertical. The default is 'vertical'.
            cbarlabelsize : integer, optional
                size of the color bar label. The default is 12.
            cbarlabel : str, optional
                label of the color bar. The default is 'Discharge m3/s'.
            rotation : [number], optional
                rotation of the colorbar label. The default is -90.
            TicksSpacing : [integer], optional
                Spacing in the colorbar ticks. The default is 2.
            ColorScale : integer, optional
                there are 5 options to change the scale of the colors. The default is 1.
                1- ColorScale 1 is the normal scale
                2- ColorScale 2 is the power scale
                3- ColorScale 3 is the SymLogNorm scale
                4- ColorScale 4 is the PowerNorm scale
                5- ColorScale 5 is the BoundaryNorm scale
            gamma : [float], optional
                value needed for option 2 . The default is 1./2..
            linthresh : [float], optional
                value needed for option 3. The default is 0.0001.
            linscale : [float], optional
                value needed for option 3. The default is 0.001.
            midpoint : [float], optional
                value needed for option 5. The default is 0.
            cmap : [str], optional
                color style. The default is 'coolwarm_r'.
            display_cellvalue : [bool]
                True if you want to display the values of the cells as a text
            NumSize : integer, optional
                size of the numbers plotted intop of each cells. The default is 8.
            Backgroundcolorthreshold : [float/integer], optional
                threshold value if the value of the cell is greater, the plotted
                numbers will be black and if smaller the plotted number will be white
                if None given the maxvalue/2 will be considered. The default is None.
            Gaugecolor : [str], optional
                color of the points. The default is 'red'.
            Gaugesize : [integer], optional
                size of the points. The default is 100.
            IDcolor : [str]
                the ID of the Point.The default is "blue".
            IDsize : [integer]
                size of the ID text. The default is 10.
            IDcolor : []

            rotation : []

            midpoint : []

            **kwargs : [dict]
                keys:
                    Points : [dataframe].
                        dataframe contains two columns 'row', and col to
                        plot the point at this location

        Returns
        -------
            1- axes: [figure axes].
                the axes of the matplotlib figure
            2. fig: [matplotlib figure object]
                the figure object

        """
        if isinstance(src, gdal.Dataset):
            Arr, nodataval = Raster.GetRasterData(src)
            Arr = Arr.astype(np.float32)
            Arr[np.isclose(Arr, nodataval, rtol=0.001)] = np.nan

            no_elem = np.size(Arr[:, :]) - np.count_nonzero((Arr[np.isnan(Arr)]))

            if "points" in kwargs.keys():
                points = kwargs["points"]
                points["row"] = np.nan
                points["col"] = np.nan
                # to locte the points in the array
                points.loc[:, ["row", "col"]] = GC.NearestCell(
                    src, points[["x", "y"]][:]
                ).values
        elif isinstance(src, np.ndarray):
            Arr = src
            Arr[np.isclose(Arr, nodataval, rtol=0.001)] = np.nan
            no_elem = np.size(Arr[:, :]) - np.count_nonzero((Arr[np.isnan(Arr)]))

        fig = plt.figure(figsize=Figsize)  # 60,
        ax = fig.add_subplot()

        if np.mod(np.nanmax(Arr), TicksSpacing) == 0:
            ticks = np.arange(
                np.nanmin(Arr), np.nanmax(Arr) + TicksSpacing, TicksSpacing
            )
        else:
            ticks = np.arange(np.nanmin(Arr), np.nanmax(Arr), TicksSpacing)
            ticks = np.append(
                ticks,
                [int(np.nanmax(Arr) / TicksSpacing) * TicksSpacing + TicksSpacing],
            )

        if ColorScale == 1:
            im = ax.matshow(
                Arr[:, :], cmap=cmap, vmin=np.nanmin(Arr), vmax=np.nanmax(Arr)
            )
            cbar_kw = dict(ticks=ticks)
        elif ColorScale == 2:
            im = ax.matshow(
                Arr[:, :],
                cmap=cmap,
                norm=colors.PowerNorm(
                    gamma=gamma, vmin=np.nanmin(Arr), vmax=np.nanmax(Arr)
                ),
            )
            cbar_kw = dict(ticks=ticks)
        elif ColorScale == 3:
            im = ax.matshow(
                Arr[:, :],
                cmap=cmap,
                norm=colors.SymLogNorm(
                    linthresh=linthresh,
                    linscale=linscale,
                    base=np.e,
                    vmin=np.nanmin(Arr),
                    vmax=np.nanmax(Arr),
                ),
            )

            formatter = LogFormatter(10, labelOnlyBase=False)
            cbar_kw = dict(ticks=ticks, format=formatter)
        elif ColorScale == 4:
            bounds = ticks
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            im = ax.matshow(Arr[:, :], cmap=cmap, norm=norm)
            cbar_kw = dict(ticks=ticks)
        else:
            im = ax.matshow(
                Arr[:, :], cmap=cmap, norm=MidpointNormalize(midpoint=midpoint)
            )
            cbar_kw = dict(ticks=ticks)

        # Create colorbar
        cbar = ax.figure.colorbar(
            im, ax=ax, shrink=Cbarlength, orientation=orientation, **cbar_kw
        )
        cbar.ax.set_ylabel(
            cbarlabel, rotation=rotation, va="bottom", fontsize=cbarlabelsize
        )
        cbar.ax.tick_params(labelsize=10)

        ax.set_title(Title, fontsize=titlesize)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_yticks([])
        Indexlist = list()

        if display_cellvalue:
            for x in range(Arr.shape[0]):
                for y in range(Arr.shape[1]):
                    if not np.isnan(Arr[x, y]):
                        Indexlist.append([x, y])
            # add text for the cell values
            Textlist = list()
            for x in range(no_elem):
                Textlist.append(
                    ax.text(
                        Indexlist[x][1],
                        Indexlist[x][0],
                        round(Arr[Indexlist[x][0], Indexlist[x][1]], 2),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=NumSize,
                    )
                )
        #
        PoitsID = list()
        if "points" in kwargs.keys():
            row = points.loc[:, "row"].tolist()
            col = points.loc[:, "col"].tolist()
            IDs = points.loc[:, "id"].tolist()
            Points = ax.scatter(col, row, color=Gaugecolor, s=Gaugesize)

            for i in range(len(row)):
                PoitsID.append(
                    ax.text(
                        col[i],
                        row[i],
                        IDs[i],
                        ha="center",
                        va="center",
                        color=IDcolor,
                        fontsize=IDsize,
                    )
                )
        # Normalize the threshold to the images color range.
        if Backgroundcolorthreshold is not None:
            Backgroundcolorthreshold = im.norm(Backgroundcolorthreshold)
        else:
            Backgroundcolorthreshold = im.norm(np.nanmax(Arr)) / 2.0

        return fig, ax


class Scale:


    def __init__(self):
        pass


    def log_scale(minval, maxval):
        def scalar(val):
            val = val + abs(minval) + 1
            return np.log10(val)


        return scalar


    def power_scale(minval, maxval):
        def scalar(val):
            val = val + abs(minval) + 1
            return (val / 1000) ** 2


        return scalar


    def identity_scale(minval, maxval):
        def scalar(val):
            return 2


        return scalar

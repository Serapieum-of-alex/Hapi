"""Created on Wed Mar 31 02:10:49 2021.

@author: mofarrag
"""

__name__ = "catchment"

import datetime as dt
import math
import os
from types import ModuleType
from typing import Optional, Union

import geopandas as gpd
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statista.metrics as PC
from digitalearth.map import Map
from loguru import logger
from osgeo import gdal
from pyramids.catchment import Catchment as GC
from pyramids.raster import Raster

from Hapi.plot.visualizer import Visualize as Vis


class Catchment:
    """Catchment. The class include methods to read the meteorological and Spatial inputs of the distributed hydrological model. Catchment class also reads the data of the gauges, it is a super class that has the run subclass, so you need to build the catchment object and hand it as an inpit to the Run class to run the model.

    methods:
        1-ReadRainfall
        2-ReadTemperature
        3-ReadET
        4-ReadFlowAcc
        5-ReadFlowDir
        6-ReadFlowPathLength
        7-ReadParameters
        8-ReadLumpedModel
        9-ReadLumpedInputs
        10-ReadGaugeTable
        11-ReadDischargeGauges
        12-ReadParametersBounds
        13-ExtractDischarge
        14-PlotHydrograph
        15-PlotDistributedQ
        16-SaveResults
    """

    def __init__(
        self,
        name: str,
        startdata: str,
        end: str,
        fmt: str = "%Y-%m-%d",
        SpatialResolution: Optional[str] = "Lumped",
        TemporalResolution: Optional[str] = "Daily",
        RouteRiver: Optional[str] = "Muskingum",
    ):
        """Catchment.

        Parameters
        ----------
        name : [str]
            Name of the Catchment.
        startdata : [str]
            starting date.
        end : [str]
            end date.
        fmt : [str], optional
            format of the given date. The default is "%Y-%m-%d".
        SpatialResolution : TYPE, optional
            Lumped or 'Distributed' . The default is 'Lumped'.
        TemporalResolution : TYPE, optional
            "Hourly" or "Daily". The default is "Daily".

        Returns
        -------
        None.
        """
        self.name = name
        self.start = dt.datetime.strptime(startdata, fmt)
        self.end = dt.datetime.strptime(end, fmt)

        if not SpatialResolution.lower() in ["lumped", "distributed"]:
            raise ValueError(
                "available spatial resolutions are 'lumped' and 'distributed'"
            )
        self.SpatialResolution = SpatialResolution.lower()

        if not TemporalResolution.lower() in ["daily", "hourly"]:
            raise ValueError("available temporal resolutions are 'daily' and 'hourly'")
        self.TemporalResolution = TemporalResolution.lower()
        # assuming the default dt is 1 day
        conversionfactor = (1000 * 24 * 60 * 60) / (1000**2)
        if TemporalResolution.lower() == "daily":
            self.dt = 1  # 24
            self.conversionfactor = conversionfactor * 1
            self.Index = pd.date_range(self.start, self.end, freq="D")
        elif TemporalResolution.lower() == "hourly":
            self.dt = 1  # 24
            self.conversionfactor = conversionfactor * 1 / 24
            self.Index = pd.date_range(self.start, self.end, freq="H")
        else:
            # TODO calculate the temporal resolution factor
            # q mm , area sq km  (1000**2)/1000/f/24/60/60 = 1/(3.6*f)
            # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
            self.conversionfactor = 24

        self.RouteRiver = RouteRiver
        self.Parameters = None
        self.data = None
        self.Prec = None
        self.TS = None
        self.Temp = None
        self.ET = None
        self.ll_temp = None
        self.QGauges = None
        self.Snow = None
        self.Maxbas = None
        self.LumpedModel = None
        self.CatArea = None
        self.InitialCond = None
        self.q_init = None
        self.GaugesTable = None
        self.UB = None
        self.LB = None
        self.cols = None
        self.rows = None
        self.NoDataValue = None
        self.FlowAccArr = None
        self.no_elem = None
        self.acc_val = None
        self.Outlet = None
        self.CellSize = None
        self.px_area = None
        self.px_tot_area = None
        self.FlowDirArr = None
        self.FDT = None
        self.FPLArr = None
        (
            self.DEM,
            self.BankfullDepth,
            self.RiverWidth,
            self.RiverRoughness,
            self.FloodPlainRoughness,
        ) = (None, None, None, None, None)
        self.qout, self.Qtot = None, None
        self.quz_routed, self.qlz_translated, self.statevariables = None, None, None
        self.anim = None
        self.quz, self.qlz = None, None
        self.Qsim = None
        self.Metrics = None

    def ReadRainfall(self, Path: str, start: str = "", end: str = "", fmt: str = ""):
        """ReadRainfall.

        Parameters
        ----------
        Path : [String]
            path to the Folder contains precipitation rasters.
        fmt: [str]
            format of the given date
        start: [str]
            start date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        end: [str]
            end date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.


        Returns
        -------
        Prec : [array attribute]
            array containing the spatial rainfall values
        """
        if self.Prec is None:
            # data type
            assert isinstance(Path, str), "PrecPath input should be string type"
            # check wether the path exists or not
            assert os.path.exists(Path), f"{Path} you have provided does not exist"
            # check wether the folder has the rasters or not
            assert (
                len(os.listdir(Path)) > 0
            ), f"{Path} folder you have provided is empty"
            # read data
            self.Prec = Raster.readRastersFolder(Path, start=start, end=end, fmt=fmt)
            self.TS = (
                self.Prec.shape[2] + 1
            )  # no of time steps =length of time series +1
            assert isinstance(
                self.Prec, np.ndarray
            ), "array should be of type numpy array"

            logger.debug("Rainfall data are read successfully")

    def ReadTemperature(
        self,
        Path: str,
        ll_temp: Union[list, np.ndarray] = None,
        start: str = "",
        end: str = "",
        fmt: str = "",
    ):
        """ReadTemperature.

        Parameters
        ----------
        Path : [String]
            path to the Folder contains temperature rasters.
        fmt: [str]
            format of the given date
        start: [str]
            start date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        end: [str]
            end date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        ll_temp: [list/ ndarray]
            long term temperature

        Returns
        -------
        Temp : [array attribute]
            array containing the spatial temperature values
        """
        if self.Temp is None:
            # data type
            assert type(Path) == str, "PrecPath input should be string type"
            # check wether the path exists or not
            assert os.path.exists(Path), Path + " you have provided does not exist"
            # check wether the folder has the rasters or not
            assert len(os.listdir(Path)) > 0, (
                Path + " folder you have provided is empty"
            )
            # read data
            self.Temp = Raster.readRastersFolder(Path, start=start, end=end, fmt=fmt)
            assert type(self.Temp) == np.ndarray, "array should be of type numpy array"

            if ll_temp is None:
                self.ll_temp = np.zeros_like(self.Temp, dtype=np.float32)
                avg = self.Temp.mean(axis=2)
                for i in range(self.Temp.shape[0]):
                    for j in range(self.Temp.shape[1]):
                        self.ll_temp[i, j, :] = avg[i, j]

            logger.debug("Temperature data are read successfully")

    def ReadET(self, Path: str, start: str = "", end: str = "", fmt: str = ""):
        """ReadET.

        Parameters
        ----------
        Path : [String]
            path to the Folder contains Evapotranspiration rasters.
        fmt: [str]
            format of the given date
        start: [str]
            start date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.
        end: [str]
            end date if you want to read the input temperature for a specific period only,
            if not given all rasters in the given path will be read.

        Returns
        -------
        ET : [array attribute]
            array containing the spatial Evapotranspiration values
        """
        if self.ET is None:
            # data type
            assert type(Path) == str, "PrecPath input should be string type"
            # check wether the path exists or not
            assert os.path.exists(Path), Path + " you have provided does not exist"
            # check wether the folder has the rasters or not
            assert len(os.listdir(Path)) > 0, (
                Path + " folder you have provided is empty"
            )
            # read data
            self.ET = Raster.readRastersFolder(Path, start=start, end=end, fmt=fmt)
            assert type(self.ET) == np.ndarray, "array should be of type numpy array"
            logger.debug("Potential Evapotranspiration data are read successfully")

    def ReadFlowAcc(self, Path: str):
        """ReadFlowAcc.

        Parameters
        ----------
        Path : [String]
            path to the Flow Accumulation raster of the catchment
            (it should include the raster name and extension).

        Returns
        -------
        FlowAcc : [array attribute]
            array containing the spatial Evapotranspiration values
        rows: [integer]
            number of rows in the flow acc array
        cols:[integer]
            number of columns in the flow acc array
        NoDataValue:[numeric]
            the NoDataValue
        no_elem : [integer]
            number of cells in the domain
        """
        # data type
        assert isinstance(Path, str), "PrecPath input should be string type"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        # check the extension of the accumulation file
        assert (
            Path[-4:] == ".tif"
        ), "please add the extension at the end of the Flow accumulation raster path input"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"

        FlowAcc = gdal.Open(Path)
        [self.rows, self.cols] = FlowAcc.ReadAsArray().shape
        # check flow accumulation input raster
        self.NoDataValue = FlowAcc.GetRasterBand(1).GetNoDataValue()
        self.FlowAccArr = FlowAcc.ReadAsArray()

        # check if the flow acc array is integer convert it to float
        if self.FlowAccArr.dtype == "int":
            self.FlowAccArr = self.FlowAccArr.astype(float)

        for i in range(self.rows):
            for j in range(self.cols):
                if math.isclose(self.FlowAccArr[i, j], self.NoDataValue, rel_tol=0.001):
                    self.FlowAccArr[i, j] = np.nan

        self.no_elem = np.size(self.FlowAccArr[:, :]) - np.count_nonzero(
            (self.FlowAccArr[np.isnan(self.FlowAccArr)])
        )
        self.acc_val = [
            int(self.FlowAccArr[i, j])
            for i in range(self.rows)
            for j in range(self.cols)
            if not np.isnan(self.FlowAccArr[i, j])
        ]
        self.acc_val = list(set(self.acc_val))
        self.acc_val.sort()
        acc_val_mx = max(self.acc_val)

        if not (acc_val_mx == self.no_elem or acc_val_mx == self.no_elem - 1):
            message = (
                "flow accumulation raster values are not correct max "
                "value should equal number of cells or number of cells -1 "
                f"Max Value in the Flow Acc raster is {acc_val_mx}"
                f" while No of cells are {self.no_elem}"
            )
            logger.debug(message)

        # assert acc_val_mx == self.no_elem or acc_val_mx == self.no_elem -1,

        # location of the outlet
        # outlet is the cell that has the max flow_acc
        self.Outlet = np.where(self.FlowAccArr == np.nanmax(self.FlowAccArr))

        # calculate area covered by cells
        geo_trans = (
            FlowAcc.GetGeoTransform()
        )  # get the coordinates of the top left corner and cell size [x,dx,y,dy]
        dx = np.abs(geo_trans[1]) / 1000.0  # dx in Km
        dy = np.abs(geo_trans[-1]) / 1000.0  # dy in Km
        self.CellSize = dx * 1000

        # area of the cell
        self.px_area = dx * dy
        # no_cells=np.size(raster[:,:])-np.count_nonzero(raster[raster==no_val])
        self.px_tot_area = self.no_elem * self.px_area  # total area of pixels

        logger.debug("Flow Accmulation input is read successfully")

    def ReadFlowDir(self, Path: str):
        """ReadFlowDir method reads the flow direction raster.

        Parameters
        ----------
        Path : [str]
            Path to the flow direction raster.

        Returns
        -------
        FlowDirArr : [array].
            array of the flow direction raster
        FDT : [dictionary]
            flow direction table
        """
        # data type
        assert isinstance(Path, str), "PrecPath input should be string type"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        # check the extension of the accumulation file
        assert (
            Path[-4:] == ".tif"
        ), "please add the extension at the end of the Flow accumulation raster path input"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        FlowDir = gdal.Open(Path)

        [rows, cols] = FlowDir.ReadAsArray().shape
        self.FlowDirArr = FlowDir.ReadAsArray().astype(float)
        # check flow direction input raster
        fd_noval = FlowDir.GetRasterBand(1).GetNoDataValue()

        for i in range(rows):
            for j in range(cols):
                if math.isclose(self.FlowDirArr[i, j], fd_noval, rel_tol=0.001):
                    self.FlowDirArr[i, j] = np.nan

        fd_val = [
            int(self.FlowDirArr[i, j])
            for i in range(rows)
            for j in range(cols)
            if not np.isnan(self.FlowDirArr[i, j])
        ]
        fd_val = list(set(fd_val))
        fd_should = [1, 2, 4, 8, 16, 32, 64, 128]
        assert all(
            fd_val[i] in fd_should for i in range(len(fd_val))
        ), "flow direction raster should contain values 1,2,4,8,16,32,64,128 only "

        # create the flow direction table
        self.FDT = GC.flowDirectionTable(FlowDir)
        logger.debug("Flow Direction input is read successfully")

    def ReadFlowPathLength(self, Path: str):
        """ReadFlowPathLength method reads the flow path length.

        Parameters
        ----------
        Path : [str]
            Path to the file.

        Returns
        -------
        FPLArr : [array]
            flpw path length array
        rows : [integer]
            number of rows in the flow acc array
        cols : [integer]
            number of columns in the flow acc array
        NoDataValue : [numeric]
            the NoDataValue
        no_elem : [integer]
            number of cells in the domain
        """
        # data type
        assert isinstance(Path, str), "PrecPath input should be string type"
        # input values
        FPL_ext = Path[-4:]
        assert (
            FPL_ext == ".tif"
        ), "please add the extension at the end of the Flow accumulation raster path input"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"

        FPL = gdal.Open(Path)
        [self.rows, self.cols] = FPL.ReadAsArray().shape
        self.FPLArr = FPL.ReadAsArray()
        self.NoDataValue = FPL.GetRasterBand(1).GetNoDataValue()

        for i in range(self.rows):
            for j in range(self.cols):
                if math.isclose(self.FPLArr[i, j], self.NoDataValue, rel_tol=0.001):
                    self.FPLArr[i, j] = np.nan
        # check flow accumulation input raster
        self.no_elem = np.size(self.FPLArr[:, :]) - np.count_nonzero(
            (self.FPLArr[np.isnan(self.FPLArr)])
        )

        logger.debug("Flow Path length input is read successfully")

    def ReadRiverGeometry(
        self,
        DEMF: str,
        BankfulldepthF: str,
        RiverWidthF: str,
        RiverRoughnessF: str,
        FloodPlainRoughnessF: str,
    ):
        """ReadRiverGeometry.

        Parameters
        ----------
        DEMF
        BankfulldepthF
        RiverWidthF
        RiverRoughnessF
        FloodPlainRoughnessF

        Returns
        -------
        None
        """
        DEM = gdal.Open(DEMF)
        self.DEM = DEM.ReadAsArray()

        BankfullDepth = gdal.Open(BankfulldepthF)
        self.BankfullDepth = BankfullDepth.ReadAsArray()

        RiverWidth = gdal.Open(RiverWidthF)
        self.RiverWidth = RiverWidth.ReadAsArray()

        RiverRoughness = gdal.Open(RiverRoughnessF)
        self.RiverRoughness = RiverRoughness.ReadAsArray()

        FloodPlainRoughness = gdal.Open(FloodPlainRoughnessF)
        self.FloodPlainRoughness = FloodPlainRoughness.ReadAsArray()

    def ReadParameters(self, Path: str, Snow: int = False, Maxbas: bool = False):
        """ReadParameters.

        ReadParameters method reads the parameters' raster

        Parameters
        ----------
        Path : [str]
            path to the folder where the raster exist.
        Snow : [integer]
            False if you dont want to run the snow related processes and 1 if there is snow.
            in case of 1 (simulate snow processes) parameters related to snow simulation
            has to be provided. The default is 0.
        Maxbas : [bool], optional
            True if the routing is Maxbas. The default is False.

        Returns
        -------
        Parameters : [array].
            3d array containing the parameters
        Snow : [integer]
            0/1
        Maxbas : [bool]
            True/False
        """
        if self.SpatialResolution.lower() == "distributed":
            # data type
            assert type(Path) == str, "PrecPath input should be string type"
            # check wether the path exists or not
            assert os.path.exists(Path), Path + " you have provided does not exist"
            # check wether the folder has the rasters or not
            assert len(os.listdir(Path)) > 0, (
                Path + " folder you have provided is empty"
            )
            # parameters
            self.Parameters = Raster.readRastersFolder(Path)
        else:
            if not os.path.exists(Path):
                raise FileNotFoundError ("The parameters file you have entered does not exist")

            self.Parameters = pd.read_csv(Path, index_col=0, header=None)[1].tolist()

        assert (
            not Snow or Snow
        ), " snow input defines whether to consider snow subroutine or not it has to be True or False"

        self.Snow = Snow
        self.Maxbas = Maxbas

        if self.SpatialResolution == "distributed":
            if Snow and Maxbas:
                assert self.Parameters.shape[2] == 16, (
                    "current version of HBV (with snow) takes 16 parameter you have entered "
                    + str(self.Parameters.shape[2])
                )
            elif not Snow and Maxbas:
                assert self.Parameters.shape[2] == 11, (
                    "current version of HBV (with snow) takes 11 parameter you have entered "
                    + str(self.Parameters.shape[2])
                )
            elif Snow and not Maxbas:
                assert self.Parameters.shape[2] == 17, (
                    "current version of HBV (with snow) takes 17 parameter you have entered "
                    + str(self.Parameters.shape[2])
                )
            elif not Snow and not Maxbas:
                assert self.Parameters.shape[2] == 12, (
                    "current version of HBV (with snow) takes 12 parameter you have entered "
                    + str(self.Parameters.shape[2])
                )
        else:
            if Snow and Maxbas:
                assert len(self.Parameters) == 16, (
                    "current version of HBV (with snow) takes 16 parameter you have entered "
                    + str(len(self.Parameters))
                )
            elif not Snow and Maxbas:
                assert len(self.Parameters) == 11, (
                    "current version of HBV (with snow) takes 11 parameter you have entered "
                    + str(len(self.Parameters))
                )
            elif Snow and not Maxbas:
                assert len(self.Parameters) == 17, (
                    "current version of HBV (with snow) takes 17 parameter you have entered "
                    + str(len(self.Parameters))
                )
            elif not Snow and not Maxbas:
                assert len(self.Parameters) == 12, (
                    "current version of HBV (with snow) takes 12 parameter you have entered "
                    + str(len(self.Parameters))
                )

        logger.debug("Parameters are read successfully")

    def ReadLumpedModel(
        self, LumpedModel, CatArea: Union[float, int], InitialCond: list, q_init=None
    ):
        """ReadLumpedModel.

        Parameters
        ----------
        LumpedModel : [module]
            HBV module.
        CatArea : [numeric]
            Catchment area (km2).
        InitialCond : [list]
            list of the inial condition [SnowPack,SoilMoisture,Upper Zone,
                                         Lower Zone, Water Content].
        q_init : [numeric], optional
            initial discharge. The default is None.

        Returns
        -------
        LumpedModel : [module].
            the lumped conceptual model.
        q_init : [numeric]
            initial discharge.
        InitialCond : [list]
            initial conditions.
        """

        assert isinstance(
            LumpedModel, ModuleType
        ), "ConceptualModel should be a module or a python file contains functions "

        self.LumpedModel = LumpedModel
        self.CatArea = CatArea

        assert (
            len(InitialCond) == 5
        ), f"state variables are 5 and the given initial values are {len(InitialCond)}"

        self.InitialCond = InitialCond

        if q_init is not None:
            assert type(q_init) == float, "q_init should be of type float"
        self.q_init = q_init

        if self.InitialCond is not None:
            assert isinstance(self.InitialCond, list), "init_st should be of type list"

        logger.debug("Lumped model is read successfully")

    def ReadLumpedInputs(self, Path: str, ll_temp: Union[list, np.ndarray] = None):
        """ReadLumpedInputs. ReadLumpedInputs method read the meteorological data of lumped mode.

        [precipitation, Evapotranspiration, temperature, long term average temperature]

        Parameters
        ----------
        Path : [string]
            Path to the input file, data has to be in the order of
            [date, precipitation, ET, Temp].
        ll_temp : [bool], optional
            average long term temperature, if None it is calculated inside the
            code. The default is None.

        Returns
        -------
        data : [array].
            meteorological data.
        ll_temp : [array]
            average long term temperature.
        """
        self.data = pd.read_csv(
            Path, header=0, delimiter=",", index_col=0  # "\t", #skiprows=11,
        )
        self.data = self.data.values

        if ll_temp is None:
            self.ll_temp = np.zeros(shape=(len(self.data)), dtype=np.float32)
            self.ll_temp = self.data[:, 2].mean()

        assert (
            np.shape(self.data)[1] == 3 or np.shape(self.data)[1] == 4
        ), " meteorological data should be of length at least 3 (prec, ET, temp) or 4(prec, ET, temp, tm) "

        logger.debug("Lumped Model inputs are read successfully")

    def ReadGaugeTable(self, Path: str, FlowaccPath: str = "", fmt: str = "%Y-%m-%d"):
        """ReadGaugeTable. ReadGaugeTable reads the table where the data about the gauges are listed.

        [x coordinate, y coordinate, 'area ratio', 'weight'], the coordinates are
        mandatory to enter , to locate the gauges and be able to extract the
        discharge at the coresponding cells.

        Parameters
        ----------
        Path : [str]
            Path to the gauge file.
        FlowaccPath : [str], optional
            Path to the Flow acc raster. The default is ''.
        fmt: [str]
            Default is "%Y-%m-%d"

        Returns
        -------
        hm_gauges : [dataframe]
            the table of the gauges.
        """
        # read the gauge table
        if Path.endswith(".geojson"):
            self.GaugesTable = gpd.read_file(Path, driver="GeoJSON")
        else:
            self.GaugesTable = pd.read_csv(Path)
        col_list = self.GaugesTable.columns.tolist()

        if "start" in col_list:
            for i in range(len(self.GaugesTable)):
                self.GaugesTable.loc[i, "start"] = dt.datetime.strptime(
                    self.GaugesTable.loc[i, "start"], fmt
                )
                self.GaugesTable.loc[i, "end"] = dt.datetime.strptime(
                    self.GaugesTable.loc[i, "end"], fmt
                )
        if FlowaccPath != "" and "cell_row" not in col_list:
            # if hasattr(self, 'FlowAcc'):
            FlowAcc = gdal.Open(FlowaccPath)
            # calculate the nearest cell to each station
            self.GaugesTable.loc[:, ["cell_row", "cell_col"]] = GC.nearestCell(
                FlowAcc, self.GaugesTable[["id", "x", "y"]][:]
            )  # ,'weight'

        logger.debug("Gauge Table is read successfully")

    def ReadDischargeGauges(
        self,
        Path: str,
        delimiter: str = ",",
        column: str = "id",
        fmt: str = "%Y-%m-%d",
        Split: bool = False,
        Date1: str = "",
        Date2: str = "",
        readfrom: str = "",
    ):
        """ReadDischargeGauges. ReadDischargeGauges method read the gauge discharge data, discharge of each gauge has to be stored separetly in a file, and the name of the file has to be stored in the Gauges table you entered ubing the method "ReadGaugeTable" under the column "id", the file should contains the date at the first column.

        Parameters
        ----------
        Path : [str]
            Path the the gauge discharge data.
        delimiter : [str], optional
            the delimiter between the date and the discharge column. The default is ",".
        column : [str], optional
            the name of the column where you stored the files names. The default is 'id'.
        fmt : [str], optional
            date format. The default is "%Y-%m-%d".
        Split : bool, optional
            True if you want to split the data between two dates. The default is False.
        Date1 : [str], optional
            start date. The default is ''.
        Date2 : [str], optional
            end date. The default is ''.
        readfrom: [str]
            Default is "".

        Returns
        -------
        GaugesTable : [dataframe].
            dataframe containing the discharge data
        """
        if self.TemporalResolution.lower() == "daily":
            ind = pd.date_range(self.start, self.end, freq="D")
        else:
            ind = pd.date_range(self.start, self.end, freq="H")

        if self.SpatialResolution.lower() == "distributed":
            assert hasattr(self, "GaugesTable"), "please read the gauges table first"

            self.QGauges = pd.DataFrame(
                index=ind, columns=self.GaugesTable[column].tolist()
            )

            for i in range(len(self.GaugesTable)):
                name = self.GaugesTable.loc[i, "id"]
                if readfrom != "":
                    f = pd.read_csv(
                        Path + str(name) + ".csv",
                        index_col=0,
                        delimiter=delimiter,
                        skiprows=readfrom,
                    )  # ,#delimiter="\t"
                else:
                    f = pd.read_csv(
                        Path + str(name) + ".csv",
                        header=0,
                        index_col=0,
                        delimiter=delimiter,
                    )

                f.index = [dt.datetime.strptime(i, fmt) for i in f.index.tolist()]

                self.QGauges[int(name)] = f.loc[self.start : self.end, f.columns[-1]]
        else:
            if not os.path.exists(Path):
                raise FileNotFoundError(f"The file you have entered{Path} does not exist")

            self.QGauges = pd.DataFrame(index=ind)
            f = pd.read_csv(
                Path, header=0, index_col=0, delimiter=delimiter
            )  # ,#delimiter="\t", skiprows=11,
            f.index = [dt.datetime.strptime(i, fmt) for i in f.index.tolist()]
            self.QGauges[f.columns[0]] = f.loc[self.start : self.end, f.columns[0]]

        if Split:
            Date1 = dt.datetime.strptime(Date1, fmt)
            Date2 = dt.datetime.strptime(Date2, fmt)
            self.QGauges = self.QGauges.loc[Date1:Date2]

        logger.debug("Gauges data are read successfully")

    def ReadParametersBounds(
        self,
        UB: Union[list, np.ndarray],
        LB: Union[list, np.ndarray],
        Snow: bool = False,
        Maxbas: bool = False,
    ):
        """ReadParametersBounds. ReadParametersBounds method reads the lower and upper boundarys for each parameter.

        Parameters
        ----------
        UB : [list]
            upper bound.
        LB : [list]
            lower bound.
        Snow : [integer]
            0 if you dont want to run the snow related processes and 1 if there is snow.
            in case of 1 (simulate snow processes) parameters related to snow simulation
            has to be provided. The default is 0.
        Maxbas: [bool]
            True if the parameters has maxbas.

        Returns
        -------
        UB : [list]
            upper bound.
        LB : [list]
            lower bound.
        Snow : [integer]
            Snow
        """
        assert len(UB) == len(LB), "length of UB should be the same like LB"
        self.UB = np.array(UB)
        self.LB = np.array(LB)

        if not isinstance(Snow, bool):
            raise ValueError(
                " snow input defines whether to consider snow subroutine or not it has to be "
                "True or False"
            )
        self.Snow = Snow
        self.Maxbas = Maxbas

        logger.debug("Parameters bounds are read successfully")

    def ExtractDischarge(
        self, CalculateMetrics=True, FW1=False, Factor=None, OnlyOutlet=False
    ):
        """ExtractDischarge. ExtractDischarge method extracts and sums the discharge from the Quz_routed and Qlz_translated arrays at the location of the gauges.

        Parameters
        ----------
        CalculateMetrics : bool, optional
            True if you want to calculate the performance metrics. The default is True.
        FW1 : [bool], optional
            True if the routing function is Maxbas. The default is False.
        Factor : [list/None]
            list of factor if you want to multiply the simulated discharge by
            a factor you have to provide a list of the factor (as many factors
            as the number of gauges). The default is False.
        OnlyOutlet: [bool]
            True to extract the discharge only at the outlet.

        Returns
        -------
        Qsim : [dataframe]
            dataframe containing the discharge time series of the cells where
            the gauges are located.
        Metrics : [dataframe]
            data frame containing the followinf metrics ['RMSE', 'NSE',
            'NSEhf', 'KGE', 'WB','Pearson-CC','R2'] calculated between the simulated
            hydrographs and the gauge data
        """

        if not FW1:
            self.Qsim = pd.DataFrame(index=self.Index, columns=self.QGauges.columns)
            if CalculateMetrics:
                index = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]
                self.Metrics = pd.DataFrame(index=index, columns=self.QGauges.columns)
            # sum the lower zone and the upper zone discharge
            outletx = self.Outlet[0][0]
            outlety = self.Outlet[1][0]

            # self.qout = self.qlz_translated[outletx,outlety,:] + self.quz_routed[outletx,outlety,:]
            # self.Qtot = self.qlz_translated + self.quz_routed
            self.qout = self.Qtot[outletx, outlety, :]

            for i in range(len(self.GaugesTable)):
                Xind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_row"])
                Yind = int(self.GaugesTable.loc[self.GaugesTable.index[i], "cell_col"])
                gaugeid = self.GaugesTable.loc[self.GaugesTable.index[i], "id"]

                # Quz = np.reshape(self.quz_routed[Xind,Yind,:-1],self.TS-1)
                # Qlz = np.reshape(self.qlz_translated[Xind,Yind,:-1],self.TS-1)
                # Qsim = Quz + Qlz

                Qsim = np.reshape(self.Qtot[Xind, Yind, :-1], self.TS - 1)
                if Factor is not None:
                    self.Qsim.loc[:, gaugeid] = Qsim * Factor[i]
                else:
                    self.Qsim.loc[:, gaugeid] = Qsim

                if CalculateMetrics:
                    Qobs = self.QGauges.loc[:, gaugeid]
                    self.Metrics.loc["RMSE", gaugeid] = round(PC.RMSE(Qobs, Qsim), 3)
                    self.Metrics.loc["NSE", gaugeid] = round(PC.NSE(Qobs, Qsim), 3)
                    self.Metrics.loc["NSEhf", gaugeid] = round(PC.NSEHF(Qobs, Qsim), 3)
                    self.Metrics.loc["KGE", gaugeid] = round(PC.KGE(Qobs, Qsim), 3)
                    self.Metrics.loc["WB", gaugeid] = round(PC.WB(Qobs, Qsim), 3)
                    self.Metrics.loc["Pearson-CC", gaugeid] = round(
                        PC.PearsonCorre(Qobs, Qsim), 3
                    )
                    self.Metrics.loc["R2", gaugeid] = round(PC.R2(Qobs, Qsim), 3)
        elif FW1 or OnlyOutlet:
            self.Qsim = pd.DataFrame(index=self.Index)
            gaugeid = self.GaugesTable.loc[self.GaugesTable.index[-1], "id"]
            Qsim = np.reshape(self.qout, self.TS - 1)
            self.Qsim.loc[:, gaugeid] = Qsim

            if CalculateMetrics:
                index = ["RMSE", "NSE", "NSEhf", "KGE", "WB", "Pearson-CC", "R2"]
                self.Metrics = pd.DataFrame(index=index)

                # if CalculateMetrics:
                Qobs = self.QGauges.loc[:, gaugeid]
                self.Metrics.loc["RMSE", gaugeid] = round(PC.RMSE(Qobs, Qsim), 3)
                self.Metrics.loc["NSE", gaugeid] = round(PC.NSE(Qobs, Qsim), 3)
                self.Metrics.loc["NSEhf", gaugeid] = round(PC.NSEHF(Qobs, Qsim), 3)
                self.Metrics.loc["KGE", gaugeid] = round(PC.KGE(Qobs, Qsim), 3)
                self.Metrics.loc["WB", gaugeid] = round(PC.WB(Qobs, Qsim), 3)
                self.Metrics.loc["Pearson-CC", gaugeid] = round(
                    PC.PearsonCorre(Qobs, Qsim), 3
                )
                self.Metrics.loc["R2", gaugeid] = round(PC.R2(Qobs, Qsim), 3)

    def PlotHydrograph(
        self,
        plotstart: str,
        plotend: str,
        gaugei: int,
        Hapicolor: Union[tuple, str] = "#004c99",
        gaugecolor: Union[tuple, str] = "#DC143C",
        linewidth: int = 3,
        Hapiorder: int = 1,
        Gaugeorder: int = 0,
        labelfontsize: int = 10,
        XMajorfmt: str = "%Y-%m-%d",
        Noxticks: int = 5,
        Title: str = "",
        Xaxis_fmt: str = "%d\n%m",
        label: str = "",
        fmt: str = "%Y-%m-%d",
    ):
        """PlotHydrograph. plot the simulated and gauge hydrograph.

        Parameters
        ----------
        plotstart : [str]
            starting date.
        plotend : [str]
            end date.
        gaugei : [integer]
            order if the gauge in the GaugeTable.
        Hapicolor : [str], optional
            color of the Simulated hydrograph. The default is "#004c99".
        gaugecolor : [str], optional
            color of the gauge. The default is "#DC143C".
        linewidth : [numeric], optional
            line width. The default is 3.
        Hapiorder : [integer], optional
            the order of the simulated hydrograph to control which hydrograph
            is in the front . The default is 1.
        Gaugeorder : TYPE, optional
            the order of the simulated hydrograph to control which hydrograph
            is in the front . The default is 0.
        labelfontsize : numeric, optional
            label size. The default is 10.
        XMajorfmt : [str], optional
            format of the x-axis labels. The default is '%Y-%m-%d'.
        Noxticks : [integer], optional
            number of x-axis ticks. The default is 5.
        Title: [str]
            Default is "".
        Xaxis_fmt: [str]
            Default is "%d\n%m".
        label: [str]
            Default is "".
        fmt:[str]
            Default is "%Y-%m-%d".

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        ax : [matplotlib axes]
            you can control the figure from the axes.
        """
        plotstart = dt.datetime.strptime(plotstart, fmt)
        plotend = dt.datetime.strptime(plotend, fmt)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))

        if self.SpatialResolution == "distributed":
            gaugeid = self.GaugesTable.loc[gaugei, "id"]

            if Title == "":
                Title = "Gauge - " + str(self.GaugesTable.loc[gaugei, "name"])

            if label == "":
                label = str(self.GaugesTable.loc[gaugei, "name"])

            ax.plot(
                self.Qsim.loc[plotstart:plotend, gaugeid],
                "-.",
                label=label,
                linewidth=linewidth,
                color=Hapicolor,
                zorder=Hapiorder,
            )
            ax.set_title(Title, fontsize=20)
        else:
            gaugeid = self.QGauges.columns[0]
            if Title == "":
                Title = "Gauge - " + str(gaugeid)
            if label == "":
                label = str(gaugeid)

            ax.plot(
                self.Qsim.loc[plotstart:plotend, gaugeid],
                "-.",
                label=Title,
                linewidth=linewidth,
                color=Hapicolor,
                zorder=Hapiorder,
            )
            ax.set_title(Title, fontsize=20)

        ax.plot(
            self.QGauges.loc[plotstart:plotend, gaugeid],
            label="Gauge",
            linewidth=linewidth,
            color=gaugecolor,
            zorder=Gaugeorder,
        )

        ax.tick_params(axis="both", which="major", labelsize=labelfontsize)
        # ax.locator_params(axis="x", nbins=4)

        XMajorfmt = dates.DateFormatter(XMajorfmt)
        ax.xaxis.set_major_formatter(XMajorfmt)
        # ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(1),
        # interval=1))

        ax.xaxis.set_minor_formatter(dates.DateFormatter(Xaxis_fmt))

        ax.xaxis.set_major_locator(plt.MaxNLocator(Noxticks))

        ax.legend(fontsize=12)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Discharge m3/s", fontsize=12)
        plt.tight_layout()

        if self.Metrics:
            logger.debug("----------------------------------")
            logger.debug("Gauge - " + str(gaugeid))
            logger.debug("RMSE= " + str(round(self.Metrics.loc["RMSE", gaugeid], 2)))
            logger.debug("NSE= " + str(round(self.Metrics.loc["NSE", gaugeid], 2)))
            logger.debug("NSEhf= " + str(round(self.Metrics.loc["NSEhf", gaugeid], 2)))
            logger.debug("KGE= " + str(round(self.Metrics.loc["KGE", gaugeid], 2)))
            logger.debug("WB= " + str(round(self.Metrics.loc["WB", gaugeid], 2)))
            logger.debug(
                "Pearson-CC= " + str(round(self.Metrics.loc["Pearson-CC", gaugeid], 2))
            )
            logger.debug("R2= " + str(round(self.Metrics.loc["R2", gaugeid], 2)))

        return fig, ax

    def PlotDistributedResults(
        self,
        start: str,
        end: str,
        fmt: str = "%Y-%m-%d",
        Option: int = 1,
        Gauges: bool = False,
        **kwargs,
    ):
        """PlotDistributedResults. PlotDistributedResults animate the time series of the meteorological inputs and the result calculated by the model  like the total discharge, upper zone, and lower zone discharge and the state variables.

        Parameters
        ----------
        start : [str]
            starting date
        end : [str]
            end date
        fmt : [str]
            format of the gicen date. The default is "%Y-%m-%d"
        Option : [str]
            1- Total discharge, 2-Upper zone discharge, 3-ground water,
            4-Snowpack state variable, 5-Soil moisture, 6-Upper zone,
            7-Lower zone, 8-Water content, 9-Precipitation input. 10-ET,
            11-Temperature. The default is 1
        Gauges : [str]
            . The default is False
        **kwargs :
            possible keyward args
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
            Interval : [integer], optional
                number to controlthe speed of the animation. The default is 200.
            cmap : [str], optional
                color style. The default is 'coolwarm_r'.
            Textloc : [list], optional
                location of the date text. The default is [0.1,0.2].
            Gaugecolor : [str], optional
                color of the points. The default is 'red'.
            Gaugesize : [integer], optional
                size of the points. The default is 100.
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
        start = dt.datetime.strptime(start, fmt)
        end = dt.datetime.strptime(end, fmt)

        starti = np.where(self.Index == start)[0][0]
        endi = np.where(self.Index == end)[0][0]

        if 1 > Option > 11:
            raise ValueError("Plotting options are from 1 to 11")

        if Option == 1:
            self.Qtot[self.FlowAccArr == self.NoDataValue, :] = np.nan
            Arr = self.Qtot[:, :, starti:endi]
            Title = "Total Discharge"
        elif Option == 2:
            self.quz_routed[self.FlowAccArr == self.NoDataValue, :] = np.nan
            Arr = self.quz_routed[:, :, starti:endi]
            Title = "Surface Flow"
        elif Option == 3:
            self.qlz_translated[self.FlowAccArr == self.NoDataValue, :] = np.nan
            Arr = self.qlz_translated[:, :, starti:endi]
            Title = "Ground Water Flow"
        elif Option == 4:
            self.statevariables[self.FlowAccArr == self.NoDataValue, :, 0] = np.nan
            Arr = self.statevariables[:, :, starti:endi, 0]
            Title = "Snow Pack"
        elif Option == 5:
            self.statevariables[self.FlowAccArr == self.NoDataValue, :, 1] = np.nan
            Arr = self.statevariables[:, :, starti:endi, 1]
            Title = "Soil Moisture"
        elif Option == 6:
            self.statevariables[self.FlowAccArr == self.NoDataValue, :, 2] = np.nan
            Arr = self.statevariables[:, :, starti:endi, 2]
            Title = "Upper Zone"
        elif Option == 7:
            self.statevariables[self.FlowAccArr == self.NoDataValue, :, 3] = np.nan
            Arr = self.statevariables[:, :, starti:endi, 3]
            Title = "Lower Zone"
        elif Option == 8:
            self.statevariables[self.FlowAccArr == self.NoDataValue, :, 4] = np.nan
            Arr = self.statevariables[:, :, starti:endi, 4]
            Title = "Water Content"
        elif Option == 9:
            self.Prec[self.FlowAccArr == self.NoDataValue, :] = np.nan
            Arr = self.Prec[:, :, starti:endi]
            Title = "Precipitation"
        elif Option == 10:
            self.ET[self.FlowAccArr == self.NoDataValue, :] = np.nan
            Arr = self.ET[:, :, starti:endi]
            Title = "ET"
        elif Option == 11:
            self.Temp[self.FlowAccArr == self.NoDataValue, :] = np.nan
            Arr = self.Temp[:, :, starti:endi]
            Title = "Temperature"

        Time = self.Index[starti:endi]

        if Gauges:
            kwargs["Points"] = self.GaugesTable

        anim = Map.AnimateArray(Arr, Time, self.no_elem, Title=Title, **kwargs)

        self.anim = anim

        return anim

    def SaveAnimation(self, VideoFormat="gif", Path="", SaveFrames=20):
        """SaveAnimation. SaveAnimation.

        Parameters
        ----------
        VideoFormat : [str], optional
            possible formats ['mp4','mov', 'avi', 'gif']. The default is "gif".
        Path : [str], optional
            path inclusinf the video format. The default is ''.
        SaveFrames : [integer], optional
            speed of the video. The default is 20.

        Returns
        -------
        None.
        """
        Vis.SaveAnimation(
            self.anim, VideoFormat=VideoFormat, Path=Path, SaveFrames=SaveFrames
        )

    def SaveResults(
        self,
        FlowAccPath: str = "",
        Result: int = 1,
        start: str = "",
        end: str = "",
        Path: str = "",
        Prefix: str = "",
        fmt: str = "%Y-%m-%d",
    ):
        """SaveResults. SaveResults save the results into rasters.

        Parameters
        ----------
        FlowAccPath : [path]
            Path to Flow acc raster.
        Result : [integer], optional
            1 for the total discharge, 2 for the upper zone discharge, 3 for
            the lower zone discharge, 4 for the snow pack, 5 for the soil
            moisture, 6 upper zone, 7 for the lower zone, 8 for the water content.
            The default is 1.
        start : [str], optional
            start date. The default is ''.
        end : [str], optional
            end date. The default is ''.
        Path : [str], optional
            PAth to the directory where you want to save the results. The default is ''.
        Prefix : [str], optional
            prefix to add to the name of the result files. The default is ''.
        fmt : [str], optional
            format of the date. The default is "%Y-%m-%d".

        Returns
        -------
        None.
        """
        if start == "":
            start = self.Index[0]
        else:
            start = dt.datetime.strptime(start, fmt)

        if end == "":
            end = self.Index[-1]
        else:
            end = dt.datetime.strptime(end, fmt)

        starti = np.where(self.Index == start)[0][0]
        endi = np.where(self.Index == end)[0][0] + 1

        if self.SpatialResolution == "distributed":
            assert (
                FlowAccPath != ""
            ), "Please enter the  FlowAccPath parameter to the SaveResults method"
            src = gdal.Open(FlowAccPath)

            if Prefix == "":
                Prefix = "Result_"

            # create list of names
            Path = Path + Prefix
            names = [Path + str(i)[:10] for i in self.Index[starti:endi]]
            names = [i.replace("-", "_") for i in names]
            names = [i.replace(" ", "_") for i in names]
            names = [i + ".tif" for i in names]
            if Result == 1:
                Raster.rastersLike(src, self.Qtot[:, :, starti:endi], names)
            elif Result == 2:
                Raster.rastersLike(src, self.quz_routed[:, :, starti:endi], names)
            elif Result == 3:
                Raster.rastersLike(src, self.qlz_translated[:, :, starti:endi], names)
            elif Result == 4:
                Raster.rastersLike(
                    src, self.statevariables[:, :, starti:endi, 0], names
                )
            elif Result == 5:
                Raster.rastersLike(
                    src, self.statevariables[:, :, starti:endi, 1], names
                )
            elif Result == 6:
                Raster.rastersLike(
                    src, self.statevariables[:, :, starti:endi, 2], names
                )
            elif Result == 7:
                Raster.rastersLike(
                    src, self.statevariables[:, :, starti:endi, 3], names
                )
            elif Result == 8:
                Raster.rastersLike(
                    src, self.statevariables[:, :, starti:endi, 4], names
                )
        else:
            ind = pd.date_range(start, end, freq="D")
            data = pd.DataFrame(index=ind)

            data["date"] = ["'" + str(i)[:10] + "'" for i in data.index]

            if Result == 1:
                data["Qsim"] = self.Qsim[starti:endi]
                data.to_csv(Path, index=False, float_format="%.3f")
            elif Result == 2:
                data["Quz"] = self.quz[starti:endi]
                data.to_csv(Path, index=False, float_format="%.3f")
            elif Result == 3:
                data["Qlz"] = self.qlz[starti:endi]
                data.to_csv(Path, index=False, float_format="%.3f")
            elif Result == 4:
                data[["SP", "SM", "UZ", "LZ", "WC"]] = self.statevariables[
                    starti:endi, :
                ]
                data.to_csv(Path, index=False, float_format="%.3f")
            elif Result == 5:
                data["Qsim"] = self.Qsim[starti:endi]
                data["Quz"] = self.quz[starti:endi]
                data["Qlz"] = self.qlz[starti:endi]
                data[["SP", "SM", "UZ", "LZ", "WC"]] = self.statevariables[
                    starti:endi, :
                ]
                data.to_csv(Path, index=False, float_format="%.3f")
            else:
                assert False, "the possible options are from 1 to 5"

        logger.debug("Data is saved successfully")

    def ListAttributes(self):
        """Print Attributes List."""

        logger.debug("\n")
        logger.debug(
            "Attributes List of: "
            + repr(self.__dict__["name"])
            + " - "
            + self.__class__.__name__
            + " Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                logger.debug(str(key) + " : " + repr(self.__dict__[key]))

        logger.debug("\n")


class Lake:
    """Lake. Lake class reads the meteorological inputs, and the module to simulate a lake as a lumped model, using a rating curve, the lake and the upstream sub-catchments are going to be considered as one lumped model than result in a discharge input to the lake, the discharge input is going to change the volume of the water in the lake, and from the volume-outflow curve the outflow can be obtained.

    methods:
        1- ReadMeteoData
        2- ReadParameters
        3- ReadLumpedModel
    """

    def __init__(
        self,
        start="",
        end="",
        fmt="%Y-%m-%d",
        TemporalResolution="Daily",
        Split=False,
    ):
        """Lake. Lake class for lake simulation.

        Parameters
        ----------
        start : [str], optional
            start date. The default is ''.
        end : [str], optional
            end date. The default is ''.
        fmt : [str], optional
            date format. The default is "%Y-%m-%d".
        TemporalResolution : [str], optional
            "Daily" ot "Hourly". The default is "Daily".
        Split : bool, optional
            true if you want to split the date between two dates. The default is False.

        Returns
        -------
        None.
        """

        self.OutflowCell = None
        self.Snow = None
        self.Split = Split
        self.start = dt.datetime.strptime(start, fmt)
        self.end = dt.datetime.strptime(end, fmt)

        if TemporalResolution.lower() == "daily":
            self.Index = pd.date_range(start, end, freq="D")
        elif TemporalResolution.lower() == "hourly":
            self.Index = pd.date_range(start, end, freq="H")
        else:
            assert False, "Error"
        self.MeteoData = None
        self.Parameters = None
        self.LumpedModel, self.CatArea, self.LakeArea, self.InitialCond = (
            None,
            None,
            None,
            None,
        )
        self.StageDischargeCurve = None

    def ReadMeteoData(self, Path: str, fmt: str):
        """ReadMeteoData. ReadMeteoData reads the meteorological data for the lake.

        [rainfall, ET, temperature]

        Parameters
        ----------
        Path : [str]
            path to the meteo data file, containing the data in the follwoing
            order [date, rainfall, ET, temperature].
        fmt : [str]
            date format.

        Returns
        -------
        MeteoData : [array].
            array containing the meteorological data
        """

        df = pd.read_csv(Path, index_col=0)
        df.index = [dt.datetime.strptime(date, fmt) for date in df.index]

        if self.Split:
            df = df.loc[self.start : self.end, :]

        self.MeteoData = df.values  # lakeCalibArray = lakeCalibArray[:,0:-1]

        logger.debug("Lake Meteo data are read successfully")

    def ReadParameters(self, Path):
        """ReadParameters. ReadParameters method reads the lake parameters.

        Parameters
        ----------
        Path : [str]
            Path to the parameter file.

        Returns
        -------
        Parameters : [array].
        """
        Parameters = np.loadtxt(Path).tolist()
        self.Parameters = Parameters
        logger.debug("Lake Parameters are read successfully")

    def ReadLumpedModel(
        self,
        LumpedModel,
        CatArea,
        LakeArea,
        InitialCond,
        OutflowCell,
        StageDischargeCurve,
        Snow,
    ):
        """ReadLumpedModel.

        ReadLumpedModel reads the lumped model module

        Parameters
        ----------
        LumpedModel : [module]
            lumped conceptual model.
        CatArea : [numeric]
            catchment area in mk2.
        LakeArea : [numeric]
            area of the lake in km2.
        InitialCond : [list]
            initial condition [Snow Pack, Soil Moisture, Upper Zone, Lower Zone,
                               Water Content, Lake volume].
        OutflowCell : [list]
            indeces of the cell where the lake hydrograph is going to be added.
        StageDischargeCurve : [array]
            volume-outflow curve.
        Snow : [integer]
            0 if you dont want to run the snow related processes and 1 if there is snow.
            in case of 1 (simulate snow processes) parameters related to snow simulation
            has to be provided. The default is 0.

        Returns
        -------
        LumpedModel : [module]
            HBV module.
        CatArea : [numeric]
            Catchment area (km2).
        InitialCond : [list]
            list of the inial condition [SnowPack,SoilMoisture,Upper Zone,
                                         Lower Zone, Water Content].
        Snow : [integer]
            0/1
        StageDischargeCurve : [array]
        """
        assert isinstance(
            LumpedModel, ModuleType
        ), "ConceptualModel should be a module or a python file contains functions "
        self.LumpedModel = LumpedModel

        self.CatArea = CatArea
        self.LakeArea = LakeArea
        self.InitialCond = InitialCond

        if self.InitialCond is not None:
            assert type(self.InitialCond) == list, "init_st should be of type list"

        self.Snow = Snow
        self.OutflowCell = OutflowCell
        self.StageDischargeCurve = StageDischargeCurve
        logger.debug("Lumped model is read successfully")


if __name__ == "__main__":
    logger.debug("Catchment module")

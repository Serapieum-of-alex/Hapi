# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 02:10:49 2021

@author: mofarrag
"""
import math
import numpy as np
import pandas as pd
import datetime as dt
import os
import gdal
from types import ModuleType
import matplotlib.pyplot as plt
import matplotlib.dates as dates


from Hapi.gis.raster import Raster
from Hapi.gis.giscatchment import GISCatchment as GC
import Hapi.statistics.performancecriteria as PC
from Hapi.visualizer import Visualize as Vis


class Catchment():
    """
    ================================
        Catchment
    ================================
    Catchment class include methods to read the meteorological and Spatial inputs
    of the distributed hydrological model. Catchment class also reads the data
    of the gauges, it is a super class that has the run subclass, so you
    need to build the catchment object and hand it as an inpit to the Run class
    to run the model

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

    def __init__(self, name, StartDate, EndDate, fmt="%Y-%m-%d", SpatialResolution = 'Lumped',
                 TemporalResolution = "Daily", RouteRiver="Muskingum"):
        """
        =============================================================================
            Catchment(name, StartDate, EndDate, fmt="%Y-%m-%d", SpatialResolution = 'Lumped',
                             TemporalResolution = "Daily")
        =============================================================================
        Parameters
        ----------
        name : [str]
            Name of the Catchment.
        StartDate : [str]
            starting date.
        EndDate : [str]
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
        self.StartDate = dt.datetime.strptime(StartDate,fmt)
        self.EndDate = dt.datetime.strptime(EndDate,fmt)
        self.SpatialResolution = SpatialResolution
        self.TemporalResolution = TemporalResolution

        if TemporalResolution == "Daily":
            self.Timef = 24
            self.Index = pd.date_range(self.StartDate, self.EndDate, freq = "D" )
        elif TemporalResolution == "Hourly":
            self.Timef = 1
            self.Index = pd.date_range(self.StartDate, self.EndDate, freq = "H" )
        else:
            #TODO calculate the teporal resolution factor
            # q mm , area sq km  (1000**2)/1000/f/60/60 = 1/(3.6*f)
            # if daily tfac=24 if hourly tfac=1 if 15 min tfac=0.25
            self.Tfactor = 24

        self.RouteRiver = RouteRiver
        pass

    def ReadRainfall(self,Path):
        """
        =========================================================
             ReadRainfall(Path)
        =========================================================

        Parameters
        ----------
        Path : [String]
            path to the Folder contains precipitation rasters.

        Returns
        -------
        Prec : [array attribute]
            array containing the spatial rainfall values
        """
        if not hasattr(self, "Prec"):
            # data type
            assert type(Path) == str, "PrecPath input should be string type"
            # check wether the path exists or not
            assert os.path.exists(Path), Path + " you have provided does not exist"
            # check wether the folder has the rasters or not
            assert len(os.listdir(Path)) > 0, Path+" folder you have provided is empty"
            # read data
            self.Prec = Raster.ReadRastersFolder(Path)
            self.TS = self.Prec.shape[2] + 1 # no of time steps =length of time series +1
            assert type(self.Prec) == np.ndarray, "array should be of type numpy array"
            print("Rainfall data are read successfully")


    def ReadTemperature(self,Path, ll_temp=None):
        """
        =========================================================
            ReadTemperature(Path)
        =========================================================

        Parameters
        ----------
        Path : [String]
            path to the Folder contains temperature rasters.

        Returns
        -------
        Temp : [array attribute]
            array containing the spatial temperature values

        """
        if not hasattr(self, 'Temp'):
            # data type
            assert type(Path) == str, "PrecPath input should be string type"
            # check wether the path exists or not
            assert os.path.exists(Path), Path + " you have provided does not exist"
            # check wether the folder has the rasters or not
            assert len(os.listdir(Path)) > 0, Path+" folder you have provided is empty"
            # read data
            self.Temp = Raster.ReadRastersFolder(Path)
            assert type(self.Temp) == np.ndarray, "array should be of type numpy array"

            if ll_temp is None:
                self.ll_temp = np.zeros_like(self.Temp,dtype=np.float32)
                avg = self.Temp.mean(axis=2)
                for i in range(self.Temp.shape[0]):
                    for j in range(self.Temp.shape[1]):
                        self.ll_temp[i,j,:] = avg[i,j]

            print("Temperature data are read successfully")


    def ReadET(self,Path):
        """
        =========================================================
            ReadET(Path)
        =========================================================

        Parameters
        ----------
        Path : [String]
            path to the Folder contains Evapotranspiration rasters.

        Returns
        -------
        ET : [array attribute]
            array containing the spatial Evapotranspiration values

        """
        if not hasattr(self, 'ET'):
            # data type
            assert type(Path) == str, "PrecPath input should be string type"
            # check wether the path exists or not
            assert os.path.exists(Path), Path + " you have provided does not exist"
            # check wether the folder has the rasters or not
            assert len(os.listdir(Path)) > 0, Path+" folder you have provided is empty"
            # read data
            self.ET = Raster.ReadRastersFolder(Path)
            assert type(self.ET) == np.ndarray, "array should be of type numpy array"
            print("Potential Evapotranspiration data are read successfully")


    def ReadFlowAcc(self, Path):
        """
        =========================================================
            ReadET(Path)
        =========================================================

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
        assert type(Path) == str, "PrecPath input should be string type"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        # check the extension of the accumulation file
        assert Path[-4:] == ".tif", "please add the extension at the end of the Flow accumulation raster path input"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"

        FlowAcc = gdal.Open(Path)
        [self.rows,self.cols] = FlowAcc.ReadAsArray().shape
        # check flow accumulation input raster
        self.NoDataValue = FlowAcc.GetRasterBand(1).GetNoDataValue()
        self.FlowAccArr = FlowAcc.ReadAsArray()

        for i in range(self.rows):
            for j in range(self.cols):
                if math.isclose(self.FlowAccArr[i,j], self.NoDataValue, rel_tol=0.001):
                    self.FlowAccArr[i,j] = np.nan

        self.no_elem = np.size(self.FlowAccArr[:,:]) - np.count_nonzero((self.FlowAccArr[np.isnan(self.FlowAccArr)]))
        self.acc_val = [int(self.FlowAccArr[i,j]) for i in range(self.rows) for j in range(self.cols) if not np.isnan(self.FlowAccArr[i,j])]
        self.acc_val = list(set(self.acc_val))
        self.acc_val.sort()
        acc_val_mx = max(self.acc_val)
        if not (acc_val_mx == self.no_elem or acc_val_mx == self.no_elem -1):
            message = ('flow accumulation raster values are not correct max '\
            'value should equal number of cells or number of cells -1 '\
            'Max Value in the Flow Acc raster is {x}'
            ' while No of cells are {y}').format(x=acc_val_mx, y=self.no_elem, sep='\n')
            print(message)

        # assert acc_val_mx == self.no_elem or acc_val_mx == self.no_elem -1,

        # location of the outlet
        # outlet is the cell that has the max flow_acc
        self.Outlet = np.where(self.FlowAccArr == np.nanmax(self.FlowAccArr))

        # calculate area covered by cells
        geo_trans = FlowAcc.GetGeoTransform() # get the coordinates of the top left corner and cell size [x,dx,y,dy]
        dx = np.abs(geo_trans[1])/1000.0  # dx in Km
        dy = np.abs(geo_trans[-1])/1000.0  # dy in Km
        # area of the cell
        self.px_area = dx*dy
        # no_cells=np.size(raster[:,:])-np.count_nonzero(raster[raster==no_val])
        self.px_tot_area = self.no_elem*self.px_area # total area of pixels

        print("Flow Accmulation input is read successfully")


    def ReadFlowDir(self, Path):
        """
        ================================================================
            ReadFlowDir(Path)
        ================================================================
        ReadFlowDir method reads the flow direction raster

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
        assert type(Path) == str, "PrecPath input should be string type"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        # check the extension of the accumulation file
        assert Path[-4:] == ".tif", "please add the extension at the end of the Flow accumulation raster path input"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        FlowDir = gdal.Open(Path)

        [rows,cols] = FlowDir.ReadAsArray().shape

        # check flow direction input raster
        fd_noval = np.float32(FlowDir.GetRasterBand(1).GetNoDataValue())
        self.FlowDirArr = FlowDir.ReadAsArray()
        fd_val = [int(self.FlowDirArr[i,j]) for i in range(rows) for j in range(cols) if self.FlowDirArr[i,j] != fd_noval]
        fd_val = list(set(fd_val))
        fd_should = [1,2,4,8,16,32,64,128]
        assert all(fd_val[i] in fd_should for i in range(len(fd_val))), "flow direction raster should contain values 1,2,4,8,16,32,64,128 only "

        # create the flow direction table
        self.FDT = GC.FlowDirecTable(FlowDir)
        print("Flow Direction input is read successfully")


    def ReadFlowPathLength(self, Path):
        """
        ==============================================================
            ReadFlowPathLength(Path)
        ==============================================================
        ReadFlowPathLength method reads the flow path length

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
        assert type(Path) == str, "PrecPath input should be string type"
        # input values
        FPL_ext = Path[-4:]
        assert FPL_ext == ".tif", "please add the extension at the end of the Flow accumulation raster path input"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"

        FPL = gdal.Open(Path)
        [self.rows,self.cols] = FPL.ReadAsArray().shape
        # check flow accumulation input raster
        self.NoDataValue = np.float32(FPL.GetRasterBand(1).GetNoDataValue())
        self.FPLArr = FPL.ReadAsArray()
        self.no_elem = np.size(self.FPLArr[:,:])-np.count_nonzero((self.FPLArr[self.FPLArr==self.NoDataValue]))

        print("Flow Path length input is read successfully")
    def ReadRiverGeometry(self, BankfulldepthF, RiverWidthF, RiverRoughnessF, FloodPlainRoughnessF):

        Bankfulldepth = gdal.Open(BankfulldepthF)
        self.Bankfulldepth = Bankfulldepth.ReadAsArray()

        RiverWidth = gdal.Open(RiverWidthF)
        self.RiverWidth = RiverWidth.ReadAsArray()

        RiverRoughness = gdal.Open(RiverRoughnessF)
        self.RiverRoughness = RiverRoughness.ReadAsArray()

        FloodPlainRoughness = gdal.Open(FloodPlainRoughnessF)
        self.FloodPlainRoughness = FloodPlainRoughness.ReadAsArray()

    def ReadParameters(self, Path, Snow=0, Maxbas=False):
        """
        ============================================================================
             ReadParameters(Path, Snow, Maxbas=False)
        ============================================================================
        ReadParameters method reads the parameters' raster

        Parameters
        ----------
        Path : [str]
            path to the folder where the raster exist.
        Snow : [integer]
            0 if you dont want to run the snow related processes and 1 if there is snow.
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
        if self.SpatialResolution == 'Distributed':
            # data type
            assert type(Path) == str, "PrecPath input should be string type"
            # check wether the path exists or not
            assert os.path.exists(Path), Path + " you have provided does not exist"
            # check wether the folder has the rasters or not
            assert len(os.listdir(Path)) > 0, Path+" folder you have provided is empty"
            # parameters
            self.Parameters = Raster.ReadRastersFolder(Path)
        else:
            self.Parameters = pd.read_csv(Path, index_col = 0, header = None)[1].tolist()

        assert Snow == 0 or Snow == 1, " snow input defines whether to consider snow subroutine or not it has to be 0 or 1"

        self.Snow = Snow
        self.Maxbas = Maxbas

        if self.SpatialResolution == 'Distributed':
            if Snow == 1 and Maxbas:
                assert self.Parameters.shape[2] == 16, "current version of HBV (with snow) takes 16 parameter you have entered "+str(self.Parameters.shape[2])
            elif Snow == 0 and Maxbas:
                assert self.Parameters.shape[2] == 11, "current version of HBV (with snow) takes 11 parameter you have entered "+str(self.Parameters.shape[2])
            elif Snow == 1 and not Maxbas:
                assert self.Parameters.shape[2] == 17, "current version of HBV (with snow) takes 17 parameter you have entered "+str(self.Parameters.shape[2])
            elif Snow == 0 and not Maxbas:
                assert self.Parameters.shape[2] == 12, "current version of HBV (with snow) takes 12 parameter you have entered "+str(self.Parameters.shape[2])
        else:
            if Snow == 1 and Maxbas:
                assert len(self.Parameters) == 16, "current version of HBV (with snow) takes 16 parameter you have entered "+str(len(self.Parameters))
            elif Snow == 0 and Maxbas:
                assert len(self.Parameters) == 11, "current version of HBV (with snow) takes 11 parameter you have entered "+str(len(self.Parameters))
            elif Snow == 1 and not Maxbas:
                assert len(self.Parameters) == 17, "current version of HBV (with snow) takes 17 parameter you have entered "+str(len(self.Parameters))
            elif Snow == 0 and not Maxbas:
                assert len(self.Parameters) == 12, "current version of HBV (with snow) takes 12 parameter you have entered "+str(len(self.Parameters))

        print("Parameters are read successfully")


    def ReadLumpedModel(self, LumpedModel, CatArea, InitialCond, q_init=None):
        """
        =============================================================================
              ReadLumpedModel(LumpedModel, CatArea, InitialCond, q_init=None)
        =============================================================================

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

        assert isinstance(LumpedModel,ModuleType) , "ConceptualModel should be a module or a python file contains functions "
        self.LumpedModel = LumpedModel
        self.CatArea = CatArea

        assert len(InitialCond) == 5, "state variables are 5 and the given initial values are "+str(len(InitialCond))

        self.InitialCond = InitialCond

        if q_init != None:
            assert type(q_init) == float, "q_init should be of type float"
        self.q_init = q_init

        if self.InitialCond != None:
            assert type(self.InitialCond)==list, "init_st should be of type list"

        print("Lumped model is read successfully")


    def ReadLumpedInputs(self, Path, ll_temp=None):
        """
        ================================================================
              ReadLumpedInputs(Path, ll_temp=None)
        ================================================================
        ReadLumpedInputs method read the meteorological data of lumped mode
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
        self.data = pd.read_csv(Path,header=0 ,delimiter=',',#"\t", #skiprows=11,
                   index_col=0)
        self.data = self.data.values

        if ll_temp is None :
            self.ll_temp = np.zeros(shape=(len(self.data)),dtype=np.float32)
            self.ll_temp = self.data[:,2].mean()

        assert np.shape(self.data)[1] == 3 or np.shape(self.data)[1] == 4," meteorological data should be of length at least 3 (prec, ET, temp) or 4(prec, ET, temp, tm) "
        print("Lumped Model inputs are read successfully")

    def ReadGaugeTable(self, Path, FlowaccPath=''):
        """
        ==========================================================================
               ReadGaugeTable(self, Path, FlowaccPath='')
        ==========================================================================
        ReadGaugeTable reads the table where the data about the gauges are listed
        [x coordinate, y coordinate, 'area ratio', 'weight'], the coordinates are
        mandatory to enter , to locate the gauges and be able to extract the
        discharge at the coresponding cells.

        Parameters
        ----------
        Path : [str]
            Path to the gauge file.
        FlowaccPath : [str], optional
            Path to the Flow acc raster. The default is ''.

        Returns
        -------
        GaugesTable : [dataframe]
            the table of the gauges.

        """
        # read the gauge table
        self.GaugesTable = pd.read_csv(Path)
        col_list = self.GaugesTable.columns.tolist()
        if FlowaccPath != '' and 'cell_row' not in col_list:
            # if hasattr(self, 'FlowAcc'):
            FlowAcc = gdal.Open(FlowaccPath)
            # calculate the nearest cell to each station
            self.GaugesTable.loc[:,["cell_row","cell_col"]] = GC.NearestCell(FlowAcc,self.GaugesTable[['id','x','y','weight']][:])

        print("Gauge Table is read successfully")


    def ReadDischargeGauges(self, Path, delimiter=",", column='id',fmt="%Y-%m-%d",
                            Split=False, Date1='', Date2=''):
        """
        =========================================================================
            ReadDischargeGauges(Path, delimiter=",", column='id',fmt="%Y-%m-%d",
                                 Split=False, Date1='', Date2='')
        ==========================================================================
        ReadDischargeGauges method read the gauge discharge data, discharge of
        each gauge has to be stored separetly in a file, and the name of the file
        has to be stored in the Gauges table you entered ubing the method "ReadGaugeTable"
        under the column "id", the file should contains the date at the first column

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

        Returns
        -------
        QGauges : [dataframe].
            dataframe containing the discharge data

        """

        if self.TemporalResolution == "Daily":
            ind = pd.date_range(self.StartDate, self.EndDate, freq="D")
        else:
            ind = pd.date_range(self.StartDate, self.EndDate, freq="H")

        if self.SpatialResolution == "Distributed":
            assert hasattr(self, 'GaugesTable'), 'please read the gauges table first'

            self.QGauges = pd.DataFrame(index=ind, columns = self.GaugesTable[column].tolist())

            for i in range(len(self.GaugesTable)):
                name = self.GaugesTable.loc[i,'id']
                f = pd.read_csv(Path + str(name) + '.csv', header=0, index_col=0, delimiter=delimiter)# ,#delimiter="\t", skiprows=11,

                f.index = [ dt.datetime.strptime(i,fmt) for i in f.index.tolist()]

                self.QGauges[int(name)] = f.loc[self.StartDate:self.EndDate,f.columns[0]]
        else:
            self.QGauges = pd.DataFrame(index=ind)
            f = pd.read_csv(Path, header=0, index_col=0, delimiter=delimiter)# ,#delimiter="\t", skiprows=11,
            f.index = [ dt.datetime.strptime(i,fmt) for i in f.index.tolist()]
            self.QGauges[f.columns[0]] = f.loc[self.StartDate:self.EndDate,f.columns[0]]


        if Split:
            Date1 = dt.datetime.strptime(Date1,fmt)
            Date2 = dt.datetime.strptime(Date2,fmt)
            self.QGauges = self.QGauges.loc[Date1:Date2]

        print("Gauges data are read successfully")


    def ReadParametersBounds(self, UB, LB, Snow, Maxbas=False):
        """
        =======================================================================
            ReadParametersBounds(UB, LB, Snow)
        =======================================================================
        ReadParametersBounds method reads the lower and upper boundarys for each
        parameter

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

        Returns
        -------
        UB : [list]
            upper bound.
        LB : [list]
            lower bound.
        Snow : [integer]
            Snow
        """
        assert len(UB)==len(LB), "length of UB should be the same like LB"
        self.UB = np.array(UB)
        self.LB = np.array(LB)

        assert Snow == 0 or Snow == 1, " snow input defines whether to consider snow subroutine or not it has to be 0 or 1"
        self.Snow = Snow
        self.Maxbas = Maxbas

        print("Parameters bounds are read successfully")


    def ExtractDischarge(self, CalculateMetrics=True, FW1=False, Factor=None,
                         OnlyOutlet=False):
        """
        =============================================================================
              ExtractDischarge(CalculateMetrics=True, FW1=False)
        =============================================================================
        ExtractDischarge method extracts and sums the discharge from the
        Quz_routed and Qlz_translated arrays at the location of the gauges

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
            self.Qsim = pd.DataFrame(index = self.Index, columns = self.QGauges.columns)
            if CalculateMetrics:
                index = ['RMSE', 'NSE', 'NSEhf', 'KGE', 'WB','Pearson-CC','R2']
                self.Metrics = pd.DataFrame(index = index, columns = self.QGauges.columns)

            for i in range(len(self.GaugesTable)):
                Xind = int(self.GaugesTable.loc[self.GaugesTable.index[i],"cell_row"])
                Yind = int(self.GaugesTable.loc[self.GaugesTable.index[i],"cell_col"])
                gaugeid = self.GaugesTable.loc[self.GaugesTable.index[i],"id"]

                # Quz = np.reshape(self.quz_routed[Xind,Yind,:-1],self.TS-1)
                # Qlz = np.reshape(self.qlz_translated[Xind,Yind,:-1],self.TS-1)
                # Qsim = Quz + Qlz

                Qsim = np.reshape(self.Qtot[Xind,Yind,:-1],self.TS-1)
                if Factor != None:
                    self.Qsim.loc[:,gaugeid] = Qsim * Factor[i]
                else:
                    self.Qsim.loc[:,gaugeid] = Qsim

                if CalculateMetrics:
                    Qobs = self.QGauges.loc[:,gaugeid]
                    self.Metrics.loc['RMSE',gaugeid] = round(PC.RMSE(Qobs, Qsim),3)
                    self.Metrics.loc['NSE',gaugeid] = round(PC.NSE(Qobs, Qsim),3)
                    self.Metrics.loc['NSEhf',gaugeid] = round(PC.NSEHF(Qobs, Qsim),3)
                    self.Metrics.loc['KGE',gaugeid] = round(PC.KGE(Qobs, Qsim),3)
                    self.Metrics.loc['WB',gaugeid] = round(PC.WB(Qobs, Qsim),3)
                    self.Metrics.loc['Pearson-CC',gaugeid] = round(PC.PearsonCorre(Qobs, Qsim),3)
                    self.Metrics.loc['R2',gaugeid] = round(PC.R2(Qobs, Qsim),3)
        elif FW1 or OnlyOutlet:
            self.Qsim = pd.DataFrame(index = self.Index)
            gaugeid = self.GaugesTable.loc[self.GaugesTable.index[-1],"id"]
            Qsim = np.reshape(self.qout,self.TS-1)
            self.Qsim.loc[:,gaugeid] = Qsim

            if CalculateMetrics:
                index = ['RMSE', 'NSE', 'NSEhf', 'KGE', 'WB','Pearson-CC', 'R2']
                self.Metrics = pd.DataFrame(index = index)
            if CalculateMetrics:
                    Qobs = self.QGauges.loc[:,gaugeid]
                    self.Metrics.loc['RMSE',gaugeid] = round(PC.RMSE(Qobs, Qsim),3)
                    self.Metrics.loc['NSE',gaugeid] = round(PC.NSE(Qobs, Qsim),3)
                    self.Metrics.loc['NSEhf',gaugeid] = round(PC.NSEHF(Qobs, Qsim),3)
                    self.Metrics.loc['KGE',gaugeid] = round(PC.KGE(Qobs, Qsim),3)
                    self.Metrics.loc['WB',gaugeid] = round(PC.WB(Qobs, Qsim),3)
                    self.Metrics.loc['Pearson-CC',gaugeid] = round(PC.PearsonCorre(Qobs, Qsim),3)
                    self.Metrics.loc['R2',gaugeid] = round(PC.R2(Qobs, Qsim),3)


    def PlotHydrograph(self, plotstart, plotend, gaugei, Hapicolor="#004c99",
                       gaugecolor="#DC143C", linewidth = 3, Hapiorder = 1,
                       Gaugeorder = 0, labelfontsize=10, XMajorfmt='%Y-%m-%d',
                       Noxticks=5, Title='', Xaxis_fmt = '%d\n%m', label=''):
        """
        =========================================================================
        PlotHydrograph(plotstart, plotend, gaugei, Hapicolor="#004c99",
                       gaugecolor="#DC143C", linewidth = 3, Hapiorder = 1,
                       Gaugeorder = 0, labelfontsize=10, XMajorfmt='%Y-%m-%d',
                       Noxticks=5)
        =========================================================================
        PlotHydrograph plot the simulated and gauge hydrograph

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

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        ax : [matplotlib axes]
            you can control the figure from the axes.

        """

        plotstart = dt.datetime.strptime(plotstart,"%Y-%m-%d")
        plotend = dt.datetime.strptime(plotend,"%Y-%m-%d")

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))

        if self.SpatialResolution == "Distributed":
            gaugeid = self.GaugesTable.loc[gaugei,'id']

            if Title == '':
                Title = "Gauge - "+str(self.GaugesTable.loc[gaugei,'name'])

            if label == '':
                label = str(self.GaugesTable.loc[gaugei,'name'])

            ax.plot(self.Qsim.loc[plotstart:plotend, gaugeid], '-.',
                label = label,
                linewidth=linewidth, color=Hapicolor, zorder = Hapiorder)
            ax.set_title(Title, fontsize = 20)
        else:
            gaugeid = self.QGauges.columns[0]
            if Title == '':
                Title = "Gauge - "+str(gaugeid)
            if label == '':
                label = str(gaugeid)

            ax.plot(self.Qsim.loc[plotstart:plotend, gaugeid], '-.',
                label = Title,
                linewidth=linewidth, color=Hapicolor, zorder = Hapiorder)
            ax.set_title(Title, fontsize = 20)

        ax.plot(self.QGauges.loc[plotstart:plotend,gaugeid],label = 'Gauge',
                      linewidth=linewidth, color = gaugecolor, zorder = Gaugeorder)

        ax.tick_params(axis='both', which='major', labelsize=labelfontsize)
        # ax.locator_params(axis="x", nbins=4)

        XMajorfmt = dates.DateFormatter(XMajorfmt)
        ax.xaxis.set_major_formatter(XMajorfmt)
        # ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(1),
                                            # interval=1))

        ax.xaxis.set_minor_formatter(dates.DateFormatter(Xaxis_fmt))

        ax.xaxis.set_major_locator(plt.MaxNLocator(Noxticks))


        ax.legend(fontsize = 12)
        ax.set_xlabel("Time", fontsize = 12)
        ax.set_ylabel("Discharge m3/s", fontsize = 12)
        plt.tight_layout()

        if hasattr(self, "Metrics") :
            #print the metrics
            print("----------------------------------")
            print("Gauge - " +str(gaugeid))
            print("RMSE= " + str(round(self.Metrics.loc['RMSE',gaugeid],2)))
            print("NSE= " + str(round(self.Metrics.loc['NSE',gaugeid],2)))
            print("NSEhf= " + str(round(self.Metrics.loc['NSEhf',gaugeid],2)))
            print("KGE= " + str(round(self.Metrics.loc['KGE',gaugeid],2)))
            print("WB= " + str(round(self.Metrics.loc['WB',gaugeid],2)))
            print("Pearson-CC= " + str(round(self.Metrics.loc['Pearson-CC',gaugeid],2)))
            print("R2= " + str(round(self.Metrics.loc['R2',gaugeid],2)))

        return fig, ax


    def PlotDistributedResults(self, StartDate, EndDate, fmt="%Y-%m-%d", Option = 1,
                         Gauges=False,**kwargs):
        """
         =============================================================================
           PlotDistributedResults(StartDate, EndDate, fmt="%Y-%m-%d", Option = 1, Gauges=False,
                            TicksSpacing = 2, Figsize=(8,8), PlotNumbers=True,
                            NumSize= 8, Title = 'Total Discharge',titlesize = 15, Backgroundcolorthreshold=None,
                            cbarlabel = 'Discharge m3/s', cbarlabelsize = 12, textcolors=("white","black"),
                            Cbarlength = 0.75, Interval = 200,cmap='coolwarm_r', Textloc=[0.1,0.2],
                            Gaugecolor='red',Gaugesize=100, ColorScale = 1,gamma=1./2.,linthresh=0.0001,
                            linscale=0.001, midpoint=0, orientation='vertical', rotation=-90,
                            **kwargs):
        =============================================================================
        PlotDistributedResults animate the time series of the meteorological inputs and
        the result calculated by the model  like the total discharge, upper zone,
        and lower zone discharge and the state variables

        Parameters
        ----------
        StartDate : [str]
            starting date
        EndDate : [str]
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
        StartDate = dt.datetime.strptime(StartDate,fmt)
        EndDate = dt.datetime.strptime(EndDate,fmt)

        starti = np.where(self.Index == StartDate)[0][0]
        endi = np.where(self.Index == EndDate)[0][0]

        if Option == 1:
            self.Qtot[self.FlowAccArr == self.NoDataValue,:] = np.nan
            Arr = self.Qtot[:,:,starti:endi]
            Title = 'Total Discharge'
        elif Option == 2:
            self.quz_routed[self.FlowAccArr == self.NoDataValue,:] = np.nan
            Arr = self.quz_routed[:,:,starti:endi]
            Title = 'Surface Flow'
        elif Option == 3:
            self.qlz_translated[self.FlowAccArr == self.NoDataValue,:] = np.nan
            Arr = self.qlz_translated[:,:,starti:endi]
            Title = 'Ground Water Flow'
        elif Option == 4:
            self.statevariables[self.FlowAccArr == self.NoDataValue,:,0] = np.nan
            Arr = self.statevariables[:,:,starti:endi,0]
            Title = 'Snow Pack'
        elif Option == 5:
            self.statevariables[self.FlowAccArr == self.NoDataValue,:,1] = np.nan
            Arr = self.statevariables[:,:,starti:endi,1]
            Title = 'Soil Moisture'
        elif Option == 6:
            self.statevariables[self.FlowAccArr == self.NoDataValue,:,2] = np.nan
            Arr = self.statevariables[:,:,starti:endi,2]
            Title = 'Upper Zone'
        elif Option == 7:
            self.statevariables[self.FlowAccArr == self.NoDataValue,:,3] = np.nan
            Arr = self.statevariables[:,:,starti:endi,3]
            Title = 'Lower Zone'
        elif Option == 8:
            self.statevariables[self.FlowAccArr == self.NoDataValue,:,4] = np.nan
            Arr = self.statevariables[:,:,starti:endi,4]
            Title = 'Water Content'
        elif Option == 9:
            self.Prec[self.FlowAccArr == self.NoDataValue,:] = np.nan
            Arr = self.Prec[:,:,starti:endi]
            Title = 'Precipitation'
        elif Option == 10:
            self.ET[self.FlowAccArr == self.NoDataValue,:] = np.nan
            Arr = self.ET[:,:,starti:endi]
            Title = 'ET'
        elif Option == 11:
            self.Temp[self.FlowAccArr == self.NoDataValue,:] = np.nan
            Arr = self.Temp[:,:,starti:endi]
            Title = 'Temperature'

        Time = self.Index[starti:endi]

        if Gauges:
            kwargs['Points'] = self.GaugesTable

        anim = Vis.AnimateArray(Arr, Time, self.no_elem, Title = Title, **kwargs)

        self.anim = anim

        return anim


    def SaveAnimation(self, VideoFormat="gif",Path='',SaveFrames=20):
        """
        =====================================================================
            SaveAnimation(VideoFormat="gif",Path='',SaveFrames=20)
        =====================================================================

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
        Vis.SaveAnimation(self.anim, VideoFormat=VideoFormat,Path=Path,SaveFrames=SaveFrames)


    def SaveResults(self, FlowAccPath='', Result=1, StartDate='', EndDate='',
                    Path='', Prefix='', fmt="%Y-%m-%d"):
        """
        =========================================================================
        SaveResults(FlowAccPath, Result=1, StartDate='', EndDate='',
                    Path='', Prefix='', fmt="%Y-%m-%d")
        =========================================================================
        SaveResults save the results into rasters

        Parameters
        ----------
        FlowAccPath : [path]
            Path to Flow acc raster.
        Result : [integer], optional
            1 for the total discharge, 2 for the upper zone discharge, 3 for
            the lower zone discharge, 4 for the snow pack, 5 for the soil
            moisture, 6 upper zone, 7 for the lower zone, 8 for the water content.
            The default is 1.
        StartDate : [str], optional
            start date. The default is ''.
        EndDate : [str], optional
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
        if StartDate == '' :
            StartDate = self.Index[0]
        else:
            StartDate = dt.datetime.strptime(StartDate,fmt)

        if EndDate == '' :
            EndDate = self.Index[-1]
        else:
            EndDate = dt.datetime.strptime(EndDate,fmt)

        starti = np.where(self.Index == StartDate)[0][0]
        endi = np.where(self.Index == EndDate)[0][0]+1

        if self.SpatialResolution == "Distributed":
            assert FlowAccPath != '', "Please enter the  FlowAccPath parameter to the SaveResults method"
            src = gdal.Open(FlowAccPath)

            if Prefix == '' :
                Prefix = 'Result_'

            # create list of names
            Path = Path + Prefix
            names = [Path + str(i)[:10] for i in self.Index[starti:endi]]
            names = [i.replace("-","_") for i in names]
            names = [i.replace(" ","_") for i in names]
            names = [i+".tif" for i in names]
            if Result ==1:
                Raster.RastersLike(src,self.Qtot[:,:,starti:endi],names)
            elif Result ==2:
                Raster.RastersLike(src,self.quz_routed[:,:,starti:endi],names)
            elif Result ==3:
                Raster.RastersLike(src,self.qlz_translated[:,:,starti:endi],names)
            elif Result ==4:
                Raster.RastersLike(src,self.statevariables[:,:,starti:endi,0],names)
            elif Result ==5:
                Raster.RastersLike(src,self.statevariables[:,:,starti:endi,1],names)
            elif Result ==6:
                Raster.RastersLike(src,self.statevariables[:,:,starti:endi,2],names)
            elif Result ==7:
                Raster.RastersLike(src,self.statevariables[:,:,starti:endi,3],names)
            elif Result ==8:
                Raster.RastersLike(src,self.statevariables[:,:,starti:endi,4],names)
        else:
            ind = pd.date_range(StartDate,EndDate,freq='D')
            data = pd.DataFrame(index = ind)

            data['date'] = ["'" + str(i)[:10] + "'" for i in data.index]

            if Result ==1:
                data['Qsim'] = self.Qsim[starti:endi]
                data.to_csv(Path, index = False, float_format="%.3f")
            elif Result ==2:
                data['Quz'] = self.quz[starti:endi]
                data.to_csv(Path, index = False, float_format="%.3f")
            elif Result ==3:
                data['Qlz'] = self.qlz[starti:endi]
                data.to_csv(Path, index = False, float_format="%.3f")
            elif Result ==4:
                data[['SP','SM','UZ','LZ','WC']] = self.statevariables[starti:endi,:]
                data.to_csv(Path, index = False, float_format="%.3f")
            elif Result ==5:
                data['Qsim'] = self.Qsim[starti:endi]
                data['Quz'] = self.quz[starti:endi]
                data['Qlz'] = self.qlz[starti:endi]
                data[['SP','SM','UZ','LZ','WC']] = self.statevariables[starti:endi,:]
                data.to_csv(Path, index = False, float_format="%.3f")

        print("Data is saved successfully")


    def ListAttributes(self):
        """
        Print Attributes List
        """

        print('\n')
        print('Attributes List of: ' + repr(self.__dict__['name']) + ' - ' + self.__class__.__name__ + ' Instance\n')
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != 'name':
                print(str(key) + ' : ' + repr(self.__dict__[key]))

        print('\n')


class Lake():
    """
    ================================
        Lake class
    ================================
    Lake class reads the meteorological inputs, and the module to simulate a lake
    as a lumped model, using a rating curve, the lake and the upstream sub-catchments
    are going to be considered as one lumped model than result in a discharge input
    to the lake, the discharge input is going to change the volume of the water
    in the lake, and from the volume-outflow curve the outflow can be obtained.

    methods:
        1- ReadMeteoData
        2- ReadParameters
        3- ReadLumpedModel
    """

    def __init__(self, StartDate='', EndDate='', fmt="%Y-%m-%d",
                 TemporalResolution="Daily", Split=False):
        """
        =========================================================================
            Lake(StartDate='', EndDate='', fmt="%Y-%m-%d",
                         TemporalResolution="Daily", Split=False)
        =========================================================================
        Lake class for lake simulation

        Parameters
        ----------
        StartDate : [str], optional
            start date. The default is ''.
        EndDate : [str], optional
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

        self.Split = Split
        self.StartDate = dt.datetime.strptime(StartDate,fmt)
        self.EndDate = dt.datetime.strptime(EndDate,fmt)

        if TemporalResolution == "Daily":
            self.Index = pd.date_range(StartDate, EndDate, freq = "D" )
        elif TemporalResolution == "Hourly":
            self.Index = pd.date_range(StartDate, EndDate, freq = "H" )
        else:
            assert False , "Error"
        pass

    def ReadMeteoData(self, Path, fmt):
        """
        =================================================================
              ReadMeteoData(Path, fmt)
        =================================================================
        ReadMeteoData reads the meteorological data for the lake
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

        df = pd.read_csv(Path, index_col = 0)
        df.index = [dt.datetime.strptime(date,fmt) for date in df.index]

        if self.Split:
             df = df.loc[self.StartDate:self.EndDate,:]

        self.MeteoData = df.values # lakeCalibArray = lakeCalibArray[:,0:-1]

        print("Lake Meteo data are read successfully")


    def ReadParameters(self, Path):
        """
        ====================================================================
             ReadParameters(Path)
        ====================================================================
        ReadParameters method reads the lake parameters

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
        print("Lake Parameters are read successfully")


    def ReadLumpedModel(self, LumpedModel, CatArea, LakeArea, InitialCond,
                        OutflowCell, StageDischargeCurve, Snow):
        """
        ==========================================================================
            ReadLumpedModel(self, LumpedModel, CatArea, LakeArea, InitialCond,
                             OutflowCell, StageDischargeCurve, Snow)
        ==========================================================================
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
        assert isinstance(LumpedModel,ModuleType) , "ConceptualModel should be a module or a python file contains functions "
        self.LumpedModel = LumpedModel

        self.CatArea = CatArea
        self.LakeArea = LakeArea
        self.InitialCond = InitialCond

        if self.InitialCond != None:
            assert type(self.InitialCond)==list, "init_st should be of type list"

        self.Snow = Snow
        self.OutflowCell = OutflowCell
        self.StageDischargeCurve = StageDischargeCurve
        print("Lumped model is read successfully")
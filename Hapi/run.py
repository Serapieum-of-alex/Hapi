# -*- coding: utf-8 -*-
"""
RUN

RUN contains functions to to connect the parameter spatial distribution
function with the with both component of the spatial representation of the hydrological
process (conceptual model & spatial routing) to calculate the predicted
runoff at known locations based on given performance function

Created on Sun Jun 24 21:02:34 2018

@author: Mostafa
"""
import os
import gdal
import numpy as np
import pandas as pd
import datetime as dt
from types import ModuleType


from Hapi.wrapper import Wrapper
from Hapi.raster import Raster as raster
from Hapi.giscatchment import GISCatchment as GC
import Hapi.performancecriteria as PC

class Model():

    def __init__(self, name, StartDate, EndDate, fmt="%Y-%m-%d", SpatialResolution = 'Lumped',
                 TemporalResolution = "Daily"):
        """
        SpatialR : TYPE, optional
            Spatial Resolution "Distributed" or "Lumped". The default is 'Lumped'.

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
        # data type
        assert type(Path) == str, "PrecPath input should be string type"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        # check wether the folder has the rasters or not
        assert len(os.listdir(Path)) > 0, Path+" folder you have provided is empty"
        # read data
        self.Prec = raster.ReadRastersFolder(Path)
        self.TS = self.Prec.shape[2] + 1 # no of time steps =length of time series +1
        assert type(self.Prec) == np.ndarray, "array should be of type numpy array"
        print("Rainfall data are read successfully")


    def ReadTemperature(self,Path):
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
        # data type
        assert type(Path) == str, "PrecPath input should be string type"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        # check wether the folder has the rasters or not
        assert len(os.listdir(Path)) > 0, Path+" folder you have provided is empty"
        # read data
        self.Temp = raster.ReadRastersFolder(Path)
        assert type(self.Temp) == np.ndarray, "array should be of type numpy array"
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
        # data type
        assert type(Path) == str, "PrecPath input should be string type"
        # check wether the path exists or not
        assert os.path.exists(Path), Path + " you have provided does not exist"
        # check wether the folder has the rasters or not
        assert len(os.listdir(Path)) > 0, Path+" folder you have provided is empty"
        # read data
        self.ET = raster.ReadRastersFolder(Path)
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
        rows:
        cols:
        NoDataValue:
        no_elem:

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
        self.NoDataValue = np.float32(FlowAcc.GetRasterBand(1).GetNoDataValue())
        self.FlowAccArr = FlowAcc.ReadAsArray()
        self.no_elem = np.size(self.FlowAccArr[:,:])-np.count_nonzero((self.FlowAccArr[self.FlowAccArr==self.NoDataValue]))
        self.acc_val = [int(self.FlowAccArr[i,j]) for i in range(self.rows) for j in range(self.cols) if self.FlowAccArr[i,j] != self.NoDataValue]
        self.acc_val = list(set(self.acc_val))
        self.acc_val.sort()
        acc_val_mx = max(self.acc_val)
        if not (acc_val_mx == self.no_elem or acc_val_mx == self.no_elem -1):
            message = """ flow accumulation raster values are not correct max value should equal number of cells or number of cells -1 """
            message = message + " Max Value in the Flow Acc raster is " + str(acc_val_mx)
            message = message + " while No of cells are " + str(self.no_elem)
            print(message)
        # assert acc_val_mx == self.no_elem or acc_val_mx == self.no_elem -1,

        # location of the outlet
        # outlet is the cell that has the max flow_acc
        self.Outlet = np.where(self.FlowAccArr == np.nanmax(self.FlowAccArr[self.FlowAccArr != self.NoDataValue ]))

        # calculate area covered by cells
        geo_trans = FlowAcc.GetGeoTransform() # get the coordinates of the top left corner and cell size [x,dx,y,dy]
        dx = np.abs(geo_trans[1])/1000.0  # dx in Km
        dy = np.abs(geo_trans[-1])/1000.0  # dy in Km
        # area of the cell
        self.px_area = dx*dy
        # no_cells=np.size(raster[:,:])-np.count_nonzero(raster[raster==no_val])
        self.px_tot_area = self.no_elem*self.px_area # total area of pixels

        print("Flow Accmulation inputs is read successfully")


    def ReadFlowDir(self, Path):
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
        print("Flow Direction inputs is read successfully")

    def ReadParameters(self,Path):
        if self.SpatialResolution == 'Distributed':
            # data type
            assert type(Path) == str, "PrecPath input should be string type"
            # check wether the path exists or not
            assert os.path.exists(Path), Path + " you have provided does not exist"
            # check wether the folder has the rasters or not
            assert len(os.listdir(Path)) > 0, Path+" folder you have provided is empty"
            # parameters
            self.Parameters = raster.ReadRastersFolder(Path)
        else:
            self.Parameters = pd.read_csv(Path, index_col = 0, header = None)[1].tolist()

        print("Parameters are read successfully")


    def ReadLumpedModel(self, LumpedModel, CatArea, InitialCond, Snow):
        assert isinstance(LumpedModel,ModuleType) , "ConceptualModel should be a module or a python file contains functions "
        self.LumpedModel = LumpedModel
        self.CatArea = CatArea
        self.InitialCond = InitialCond
        if self.InitialCond != None:
            assert type(self.InitialCond)==list, "init_st should be of type list"

        self.Snow = Snow
        print("Lumped model is read successfully")


    def ReadLumpedInputs(self,Path):
        self.data = pd.read_csv(Path,header=0 ,delimiter=',',#"\t", #skiprows=11,
                   index_col=0)
        self.data = self.data.values
        assert np.shape(self.data)[1] == 3 or np.shape(self.data)[1] == 4," meteorological data should be of length at least 3 (prec, ET, temp) or 4(prec, ET, temp, tm) "
        print("Lumped Model inputs are read successfully")

    def ReadGaugeTable(self, Path, FlowaccPath=''):
        # read the gauge table
        self.GaugesTable = pd.read_csv(Path)
        
        if FlowaccPath != '':
            # if hasattr(self, 'FlowAcc'):
            FlowAcc = gdal.Open(FlowaccPath)
            # calculate the nearest cell to each station
            self.GaugesTable.loc[:,["cell_row","cell_col"]] = GC.NearestCell(FlowAcc,self.GaugesTable[['id','x','y','weight']][:])
        
        print("Gauge Table is read successfully")


    def ReadDischargeGauges(self, Path, delimiter=",", column='id',fmt="%Y-%m-%d",
                            Split=False, Date1='', Date2=''):

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


    def ReadParametersBounds(self, UB, LB):
        assert len(UB)==len(LB), "length of UB should be the same like LB"
        self.UB = np.array(UB)
        self.LB = np.array(LB)
        print("Parameters bounds are read successfully")

    
    def ExtractDischarge(self, CalculateMetrics=True):
        self.Qsim = pd.DataFrame(index = self.Index, columns = self.QGauges.columns)
        
        if CalculateMetrics:
            index = ['RMSE', 'NSE', 'NSEhf', 'KGE', 'WB']
            self.Metrics = pd.DataFrame(index = index, columns = self.QGauges.columns)
        
        for i in range(len(self.GaugesTable)):
            Xind = int(self.GaugesTable.loc[self.GaugesTable.index[i],"cell_row"])
            Yind = int(self.GaugesTable.loc[self.GaugesTable.index[i],"cell_col"])
            gaugeid = self.GaugesTable.loc[self.GaugesTable.index[i],"id"]
            
            Quz = np.reshape(self.quz_routed[Xind,Yind,:-1],self.TS-1)
            Qlz = np.reshape(self.qlz_translated[Xind,Yind,:-1],self.TS-1)
            Qsim = Quz + Qlz
            self.QSim.loc[:,gaugeid] = Qsim
            if CalculateMetrics:
                Qobs = self.QGauges.loc[:,gaugeid]
                self.Metrics.loc['RMSE',gaugeid] = round(PC.RMSE(Qobs, Qsim),3)
                self.Metrics.loc['NSE',gaugeid] = round(PC.NSE(Qobs, Qsim),3)
                self.Metrics.loc['NSEhf',gaugeid] = round(PC.NSEHF(Qobs, Qsim),3)
                self.Metrics.loc['KGE',gaugeid] = round(PC.KGE(Qobs, Qsim),3)
                self.Metrics.loc['WB',gaugeid] = round(PC.WB(Qobs, Qsim),3)

class Lake():

    def __init__(self, StartDate='', EndDate='', fmt="%Y-%m-%d",
                 TemporalResolution="Daily", Split=False):

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

        df = pd.read_csv(Path, index_col = 0)
        df.index = [dt.datetime.strptime(date,fmt) for date in df.index]

        if self.Split:
             df = df.loc[self.StartDate:self.EndDate,:]

        self.MeteoData = df.values # lakeCalibArray = lakeCalibArray[:,0:-1]

        print("Lake Meteo data are read successfully")

    def ReadParameters(self, Path):
        Parameters = np.loadtxt(Path).tolist()
        self.Parameters = Parameters
        print("Lake Parameters are read successfully")

    def ReadLumpedModel(self, LumpedModel, CatArea, LakeArea, InitialCond,
                        OutflowCell, StageDischargeCurve, Snow):
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


class Run(Model):

    def __init__(self):
        pass


    def RunHapi(self):
        """
        =======================================================================
            RunModel(PrecPath, Evap_Path, TempPath, DemPath, FlowAccPath, FlowDPath, ParPath, p2)
        =======================================================================
        this function runs the conceptual distributed hydrological model

        Inputs:
        ----------
            1-Paths:

                4-FlowAccPath:

                5-FlowDPath:
                    [String] path to the Flow Direction raster of the catchment (it should
                    include the raster name and extension)
            7-ParPath:
                [String] path to the Folder contains parameters rasters of the catchment
            8-p2:
                [List] list of unoptimized parameters
                p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                p2[1] = catchment area in km2

        Outputs:
        ----------
            1-statevariables: [numpy attribute] 
                4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
            2-qlz: [numpy attribute] 
                3D array of the lower zone discharge
            3-quz: [numpy attribute] 
                3D array of the upper zone discharge
            4-qout: [numpy attribute] 
                1D timeseries of discharge at the outlet of the catchment
                of unit m3/sec
            5-quz_routed: [numpy attribute] 
                3D array of the upper zone discharge  accumulated and
                routed at each time step
            6-qlz_translated: [numpy attribute] 
                3D array of the lower zone discharge translated at each time step

        Example:
        ----------
            PrecPath = prec_path="meteodata/4000/calib/prec"
            Evap_Path = evap_path="meteodata/4000/calib/evap"
            TempPath = temp_path="meteodata/4000/calib/temp"
            DemPath = "GIS/4000/dem4000.tif"
            FlowAccPath = "GIS/4000/acc4000.tif"
            FlowDPath = "GIS/4000/fd4000.tif"
            ParPath = "meteodata/4000/parameters"
            p2=[1, 227.31]
            st, q_out, q_uz_routed = RunModel(PrecPath,Evap_Path,TempPath,DemPath,
                                              FlowAccPath,FlowDPath,ParPath,p2)
        """
        ### input data validation
        # data type
        # assert type(self.FlowAcc)==gdal.Dataset, "flow_acc should be read using gdal (gdal dataset please read it using gdal library) "
        # assert type(self.FlowDir)==gdal.Dataset, "flow_direct should be read using gdal (gdal dataset please read it using gdal library) "

        # input dimensions
        [fd_rows,fd_cols] = self.FlowDirArr.shape
        assert fd_rows == self.rows and fd_cols == self.cols, "all input data should have the same number of rows"

        # input dimensions
        assert np.shape(self.Prec)[0] == self.rows and np.shape(self.ET)[0] == self.rows and np.shape(self.Temp)[0] == self.rows and np.shape(self.Parameters)[0] == self.rows, "all input data should have the same number of rows"
        assert np.shape(self.Prec)[1] == self.cols and np.shape(self.ET)[1] == self.cols and np.shape(self.Temp)[1] == self.cols and np.shape(self.Parameters)[1] == self.cols, "all input data should have the same number of columns"
        assert np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2], "all meteorological input data should have the same length"

        #run the model
        Wrapper.HapiModel(self)
        # extract gischarge at the gauges
        


    def RunHAPIwithLake(self, Lake):
        """
        =======================================================================
            RunDistwithLake(PrecPath, Evap_Path, TempPath, DemPath, FlowAccPath, FlowDPath, ParPath, p2)
        =======================================================================
        this function runs the conceptual distributed hydrological model

        Inputs:
        ----------
            1-Paths:
                1-PrecPath:
                    [String] path to the Folder contains precipitation rasters
                2-Evap_Path:
                    [String] path to the Folder contains Evapotranspiration rasters
                3-TempPath:
                    [String] path to the Folder contains Temperature rasters
                4-FlowAccPath:
                    [String] path to the Flow Accumulation raster of the catchment (it should
                    include the raster name and extension)
                5-FlowDPath:
                    [String] path to the Flow Direction raster of the catchment (it should
                    include the raster name and extension)
            7-ParPath:
                [String] path to the Folder contains parameters rasters of the catchment
            8-p2:
                [List] list of unoptimized parameters
                p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                p2[1] = catchment area in km2

        Outputs:
        ----------
            1- st:
                [4D array] state variables
            2- q_out:
                [1D array] calculated Discharge at the outlet of the catchment
            3- q_uz:
                [3D array] Distributed discharge for each cell

        Example:
        ----------
            PrecPath = prec_path="meteodata/4000/calib/prec"
            Evap_Path = evap_path="meteodata/4000/calib/evap"
            TempPath = temp_path="meteodata/4000/calib/temp"
            DemPath = "GIS/4000/dem4000.tif"
            FlowAccPath = "GIS/4000/acc4000.tif"
            FlowDPath = "GIS/4000/fd4000.tif"
            ParPath = "meteodata/4000/parameters"
            p2=[1, 227.31]
            st, q_out, q_uz_routed = RunModel(PrecPath,Evap_Path,TempPath,DemPath,
                                              FlowAccPath,FlowDPath,ParPath,p2)
        """
        ### input data validation
        # data type
        assert type(self.FlowAcc)==gdal.Dataset, "flow_acc should be read using gdal (gdal dataset please read it using gdal library) "
        assert type(self.FlowDir)==gdal.Dataset, "flow_direct should be read using gdal (gdal dataset please read it using gdal library) "

        # input dimensions
        [rows,cols] = self.FlowAcc.ReadAsArray().shape
        [fd_rows,fd_cols] = self.FlowDir.ReadAsArray().shape
        assert fd_rows == rows and fd_cols == cols, "all input data should have the same number of rows"

        # input dimensions
        assert np.shape(self.Prec)[0] == rows and np.shape(self.ET)[0] == rows and np.shape(self.Temp)[0] == rows and np.shape(self.Parameters)[0] == rows, "all input data should have the same number of rows"
        assert np.shape(self.Prec)[1] == cols and np.shape(self.ET)[1] == cols and np.shape(self.Temp)[1] == cols and np.shape(self.Parameters)[1] == cols, "all input data should have the same number of columns"
        assert np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2], "all meteorological input data should have the same length"

        assert np.shape(Lake.MeteoData)[0] == np.shape(self.Prec)[2], "Lake meteorological data has to have the same length as the distributed raster data"
        assert np.shape(Lake.MeteoData)[1] >= 3, "Lake Meteo data has to have at least three columns rain, ET, and Temp"

        #run the model
        q_out, q_uz, q_lz = Wrapper.HapiWithlake(self, Lake)

        return q_out, q_uz, q_lz


    @staticmethod
    def RunFW1withLake(ConceptualModel, Paths, ParPath, p2, init_st, snow,
                        lakeCalibArray, StageDischargeCurve, LakeParameters ,
                        lakecell,Lake_init_st,LumpedPar = True):
        """
        =======================================================================
            RunDistwithLake(PrecPath, Evap_Path, TempPath, DemPath, FlowAccPath, FlowDPath, ParPath, p2)
        =======================================================================
        this function runs the conceptual distributed hydrological model

        Inputs:
        ----------
            1-Paths:
                1-PrecPath:
                    [String] path to the Folder contains precipitation rasters
                2-Evap_Path:
                    [String] path to the Folder contains Evapotranspiration rasters
                3-TempPath:
                    [String] path to the Folder contains Temperature rasters
                4-FlowAccPath:
                    [String] path to the Flow Accumulation raster of the catchment (it should
                    include the raster name and extension)
                5-FlowDPath:
                    [String] path to the Flow Direction raster of the catchment (it should
                    include the raster name and extension)
            7-ParPath:
                [String] path to the Folder contains parameters rasters of the catchment
            8-p2:
                [List] list of unoptimized parameters
                p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                p2[1] = catchment area in km2

        Outputs:
        ----------
            1- st:
                [4D array] state variables
            2- q_out:
                [1D array] calculated Discharge at the outlet of the catchment
            3- q_uz:
                [3D array] Distributed discharge for each cell

        Example:
        ----------
            PrecPath = prec_path="meteodata/4000/calib/prec"
            Evap_Path = evap_path="meteodata/4000/calib/evap"
            TempPath = temp_path="meteodata/4000/calib/temp"
            DemPath = "GIS/4000/dem4000.tif"
            FlowAccPath = "GIS/4000/acc4000.tif"
            FlowDPath = "GIS/4000/fd4000.tif"
            ParPath = "meteodata/4000/parameters"
            p2=[1, 227.31]
            st, q_out, q_uz_routed = RunModel(PrecPath,Evap_Path,TempPath,DemPath,
                                              FlowAccPath,FlowDPath,ParPath,p2)
        """
        # input data validation
        assert len(Paths) == 4, "Paths should include 5 folder pathes " +str(len(Paths))+" paths are only provided"

        PrecPath = Paths[0]
        Evap_Path = Paths[1]
        TempPath = Paths[2]
        FlowPathLengthPath = Paths[3]

        # data type
        assert type(PrecPath) == str, "PrecPath input should be string type"
        assert type(Evap_Path) == str, "Evap_Path input should be string type"
        assert type(TempPath) == str, "TempPath input should be string type"
        assert type(FlowPathLengthPath) == str, "Flow Path Length Path input should be string type"
        assert type(ParPath) == str, "ParPath input should be string type"


        # input values
        FPL_ext = FlowPathLengthPath[-4:]
        assert FPL_ext == ".tif", "please add the extension at the end of the Flow accumulation raster path input"
        # check wether the path exists or not
        assert os.path.exists(PrecPath), PrecPath + " you have provided does not exist"
        assert os.path.exists(Evap_Path), Evap_Path+" path you have provided does not exist"
        assert os.path.exists(TempPath), TempPath+" path you have provided does not exist"
        assert os.path.exists(FlowPathLengthPath), FlowPathLengthPath + " you have provided does not exist"
        # check wether the folder has the rasters or not
        assert len(os.listdir(PrecPath)) > 0, PrecPath+" folder you have provided is empty"
        assert len(os.listdir(Evap_Path)) > 0, Evap_Path+" folder you have provided is empty"
        assert len(os.listdir(TempPath)) > 0, TempPath+" folder you have provided is empty"

        # read data
        ### meteorological data
        prec = raster.ReadRastersFolder(PrecPath)
        evap = raster.ReadRastersFolder(Evap_Path)
        temp = raster.ReadRastersFolder(TempPath)
        print("meteorological data are read successfully")

        #### GIS data
    #    dem= gdal.Open(DemPath)
        FPL = gdal.Open(FlowPathLengthPath)
        print("GIS data are read successfully")

        # parameters
    #    if LumpedPar == True:
    #        parameters = np.loadtxt(ParPath)#.tolist()
    #    else:
        parameters = raster.ReadRastersFolder(ParPath)

        print("Parameters are read successfully")


        #run the model
        st, q_out, q_uz, q_lz = Wrapper.FW1Withlake(ConceptualModel, FPL, prec, evap,
                                                   temp, parameters, p2, snow, init_st,
                                                   lakeCalibArray, StageDischargeCurve,
                                                   LakeParameters, lakecell,Lake_init_st)

        return st, q_out, q_uz, q_lz


    def RunLumped(self, Route=0, RoutingFn=[]):
        """
        =============================================================
            RunLumped(ConceptualModel,data,parameters,p2,init_st,snow,Routing=0, RoutingFn=[])
        =============================================================
        this function runs lumped conceptual model

        Inputs:
        ----------
            1-ConceptualModel:
                [function] conceptual model and it should contain a function called simulate
            2-data:
                [numpy array] meteorological data as array with the first column as precipitation
                second as evapotranspiration, third as temperature and forth column as
                long term average temperature
            3- parameters:
                [numpy array] conceptual model parameters as array
            4-p2:
                [List] list of unoptimized parameters
                p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                p2[1] = catchment area in km2
            5-init_st:
                [list] initial state variables values [sp, sm, uz, lz, wc].
            6-Routing:
                [0 or 1] to decide wether t route the generated discharge hydrograph or not
            7-RoutingFn:
                [function] function to route the dischrge hydrograph.

        Outputs:
        ----------
            1- st:
                [numpy array] 3d array of the 5 state variable data for each cell
            2- q_lz:
                [numpy array] 1d array of the calculated discharge.

        Examples:
        ----------
            p2=[24, 1530]
            #[sp,sm,uz,lz,wc]
            init_st=[0,5,5,5,0]
            snow=0
        """
        if self.TemporalResolution == "Daily":
            ind = pd.date_range(self.StartDate, self.EndDate, freq="D")
        else:
            ind = pd.date_range(self.StartDate, self.EndDate, freq="H")

        Qsim = pd.DataFrame(index = ind)

        Wrapper.Lumped(self, Route, RoutingFn)
        Qsim['q'] = self.Qsim
        self.Qsim = Qsim[:]


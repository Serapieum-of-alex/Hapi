# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 02:10:49 2021

@author: mofarrag
"""
import numpy as np
import pandas as pd
import datetime as dt
import os
import gdal
from types import ModuleType
import matplotlib.pyplot as plt
import matplotlib.dates as dates


from Hapi.raster import Raster
from Hapi.giscatchment import GISCatchment as GC
import Hapi.performancecriteria as PC
from Hapi.visualizer import Visualize as Vis

    
class Catchment():

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

        print("Flow Accmulation input is read successfully")


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
        print("Flow Direction input is read successfully")


    def ReadFlowPathLength(self, Path):
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
        
        
    def ReadParameters(self, Path, Snow, Maxbas=False):
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
            Path to the input file.
        ll_temp : [bool], optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

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


    def ReadParametersBounds(self, UB, LB, Snow):
        assert len(UB)==len(LB), "length of UB should be the same like LB"
        self.UB = np.array(UB)
        self.LB = np.array(LB)
        
        assert Snow == 0 or Snow == 1, " snow input defines whether to consider snow subroutine or not it has to be 0 or 1"
        self.Snow = Snow
        
        print("Parameters bounds are read successfully")

    
    def ExtractDischarge(self, CalculateMetrics=True, FW1=False):
        """
        =============================================================================
              ExtractDischarge(CalculateMetrics=True, FW1=False)
        =============================================================================
        ExtractDischarge method extracts and sums the discharge from the 
        Quz_routed and Qlz_translated arrays at the location of the gauges
        
        Parameters
        ----------
        CalculateMetrics : TYPE, optional
            DESCRIPTION. The default is True.
        FW1 : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

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
        else:
            
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
                       Noxticks=5):
        
        plotstart = dt.datetime.strptime(plotstart,"%Y-%m-%d")
        plotend = dt.datetime.strptime(plotend,"%Y-%m-%d")
        
        gaugeid = self.GaugesTable.loc[gaugei,'id']
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))
        
        ax.plot(self.Qsim.loc[plotstart:plotend, gaugeid], '-.',
                label = str(self.GaugesTable.loc[gaugei,'name']), 
                linewidth=linewidth, color=Hapicolor, zorder = Hapiorder)
        ax.plot(self.QGauges.loc[plotstart:plotend,gaugeid],label = 'Gauge',
                      linewidth=linewidth, color = gaugecolor, zorder = Gaugeorder)
        
        ax.tick_params(axis='both', which='major', labelsize=labelfontsize)
        # ax.locator_params(axis="x", nbins=4)
        
        XMajorfmt = dates.DateFormatter(XMajorfmt)
        ax.xaxis.set_major_formatter(XMajorfmt)
        # ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(1),
                                            # interval=1))
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%d\n%m'))
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(Noxticks))

        ax.set_title("Gauge - "+str(self.GaugesTable.loc[gaugei,'name']), fontsize = 20)
        ax.legend(fontsize = 12)
        ax.set_xlabel("Time", fontsize = 12)
        ax.set_ylabel("Discharge m3/s", fontsize = 12)
        plt.tight_layout()
        
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
        

    def PlotDistributedQ(self, StartDate, EndDate, fmt="%Y-%m-%d", Option = 1,
                         TicksSpacing = 2, Figsize=(8,8), PlotNumbers=True,
                         NumSize=8, Title = 'Total Discharge',titlesize = 15, 
                         threshold=None, cbarlabel = 'Discharge m3/s', cbarlabelsize = 12, 
                         Cbarlength = 0.75, Interval = 200, Gauges=False): #
        
        
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
            kwargs = dict(Points = self.GaugesTable)
        else:
            kwargs = dict()
            
        anim = Vis.AnimateArray(Arr, Time, self.no_elem, TicksSpacing = TicksSpacing, 
                                Figsize=Figsize, PlotNumbers=PlotNumbers, NumSize= NumSize,
                                Title = Title,titlesize = titlesize, threshold=threshold, cbarlabel = cbarlabel, 
                                cbarlabelsize = cbarlabelsize, Cbarlength = Cbarlength, 
                                Interval = Interval,**kwargs) #
        
        return anim
    
    
    def SaveResults(self, FlowAccPath, Result=1, StartDate='', EndDate='', 
                    Path='', Prefix='', fmt="%Y-%m-%d"):
    
        src = gdal.Open(FlowAccPath)
        if StartDate == '' :
            StartDate = self.Index[0]
        else:
            StartDate = dt.datetime.strptime(StartDate,fmt)
            
        if EndDate == '' :
            EndDate = self.Index[-1]
        else:
            EndDate = dt.datetime.strptime(EndDate,fmt)    
        
        if Prefix == '' :
            Prefix = 'Result_'
            
        starti = np.where(self.Index == StartDate)[0][0]
        endi = np.where(self.Index == EndDate)[0][0]+1
        
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
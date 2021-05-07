import sys
import os
import pandas as pd
import numpy as np
import datetime as dt
from bisect import bisect
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import zipfile

from Hapi.gis.raster import Raster as raster
from Hapi.statistics import performancecriteria as Pf
import Hapi.visualizer as V
# for line styling
Vis = V.Visualize(1)

hours = list(range(1,25))


class River():
    """
    ========================
       River
    ========================
    River class reads all the data of the river, (cross sections, simulation results)
    and analyse the results and do visualisation

    """

    def __init__(self, name, Version = 3, start = "1950-1-1", days = 36890,
                 RRMstart = "1950-1-1", RRMdays =36890,
                 leftOvertopping_Suffix = "_left.txt",
                 RightOvertopping_Suffix = "_right.txt", DepthPrefix = "DepthMax",
                 DurationPrefix = "Duration", ReturnPeriodPrefix = "ReturnPeriod",
                 Compressed=True, OneDResultPath='', TwoDResultPath='', fmt="%Y-%m-%d"):
        """
        =============================================================================
            River(self, name, Version = 2, start = "1950-1-1", days = 36890,
                  RRMstart = "1950-1-1", RRMdays =36890,
                  leftOvertopping_Suffix = "_left.txt",
                  RightOvertopping_Suffix = "_right.txt", DepthPrefix = "DepthMax",
                  DurationPrefix = "Duration", ReturnPeriodPrefix = "ReturnPeriod",
                  Compressed=True, OneDResultPath='', TwoDResultPath=''):
        =============================================================================
        to instantiate the river class you need to provide the following parameters

        Parameters
        ----------
        name : [str]
            name of the river.
        Version : [integer], optional
            version of the model hydraulic model. The default is 3.
        start : [str], optional
            start date. The default is "1950-1-1".
        days : [integer], optional
            length of the simulation in days. The default is 36890.
        RRMstart : [str], optional
            the start date of the rainfall-runoff data. The default is "1950-1-1".
        RRMdays : [integer], optional
            the length of the data of the rainfall-runoff data in days. The default is 36890.
        leftOvertopping_Suffix : [str], optional
            the prefix you used to name the overtopping form the left bank files.
            The default is "_left.txt".
        RightOvertopping_Suffix : TYPE, optional
            the prefix you used to name the overtopping form the right bank files.
            The default is "_right.txt".
        DepthPrefix : [str], optional
            the prefix you used to name the Max depth raster result maps.
            The default is "DepthMax".
        DurationPrefix : [str], optional
            the prefix you used to name the inundation duration raster result maps.
            The default is "Duration".
        ReturnPeriodPrefix : [str], optional
            the prefix you used to name the Return Period raster result maps.
            The default is "ReturnPeriod".
        Compressed : [bool], optional
            True if the result raster/ascii files are compressed. The default is True.
        OneDResultPath : [str], optional
            path to the folder where the 1D river routing results exist. The default is ''.
        TwoDResultPath : [str], optional
            path to the folder where the 1D river routing results exist. The default is ''.

        Returns
        -------
        None.

        """

        self.name = name
        self.Version = Version
        self.start = dt.datetime.strptime(start,fmt)
        self.end = self.start + dt.timedelta(days = days)

        self.RRMstart = dt.datetime.strptime(RRMstart,fmt)
        self.RRMend = self.RRMstart  + dt.timedelta(days = RRMdays)

        self.leftOvertopping_Suffix = leftOvertopping_Suffix
        self.RightOvertopping_Suffix = RightOvertopping_Suffix
        self.OneDResultPath = OneDResultPath
        self.TwoDResultPath = TwoDResultPath
        self.DepthPrefix = DepthPrefix
        self.DurationPrefix = DurationPrefix
        self.ReturnPeriodPrefix = ReturnPeriodPrefix
        self.Compressed = Compressed

        Ref_ind = pd.date_range(self.start, self.end, freq='D')
        # the last day is not in the results day Ref_ind[-1]
        # write the number of days + 1 as python does not include the last number in the range
        # 19723 days so write 19724
        self.ReferenceIndex = pd.DataFrame(index = list(range(1,days+1)))
        self.ReferenceIndex['date'] = Ref_ind[:-1]

        Ref_ind = pd.date_range(self.RRMstart,self.RRMend, freq='D')
        self.RRMReferenceIndex = pd.DataFrame(index = list(range(1,RRMdays+1)))
        self.RRMReferenceIndex['date'] = Ref_ind[:-1]
        self.NoTimeSteps = len(self.RRMReferenceIndex)
        self.FigureOptions = Vis.FigureDefaultOptions


    def IndexToDate(self,Index):
        """
        =======================================================
           IndexToDate(Index)
        =======================================================
        IndexToDate takes an integer number and returns the date coresponding
        to this date based on a time series starting from the "start" attribute
        of  River object and for a length of the value of the "days" attribute

        Parameters
        ----------
        Index : [Integer]
            Integer number ranges from 1 and max value of the value of the attribute
            "days" of the River object.

        Returns
        -------
        [Date time ]
            date object.

        """
        # convert the index into date
        return self.ReferenceIndex.loc[Index,'date']

    def DateToIndex(self,Date):
        """
        ===================================================
             DateToIndex(Date)
        ===================================================
        DateToIndex takes a date and returns a the order of the days in the
        time series. The time series starts from the value of the "start" for
        a length of "days" value

        Parameters
        ----------
        Date : [string/date time object]
            string in the format of "%Y-%m-%d" or a date time object.

        Returns
        -------
        [Integer]
            the order oif the date in the time series.

        """
        if type(Date) == str:
            Date = dt.datetime.strptime(Date,"%Y-%m-%d")
        
        try:
            return np.where(self.ReferenceIndex['date'] == Date)[0][0]+1    
        except:
            print("The input date " + str(Date) + " is out of the range")
            print("Simulation is between " + str(self.ReferenceIndex.loc[1,'date']) + " and " + str(self.ReferenceIndex.loc[len(self.ReferenceIndex),'date']))
            sys.exit
        

    def IndexToDateRRM(self,Index):
        """
        =======================================================
           IndexToDateRRM(Index)
        =======================================================
        IndexToDate takes an integer number and returns the date coresponding
        to this date based on a time series starting from the "start" attribute
        of  River object and for a length of the value of the "days" attribute

        Parameters
        ----------
        Index : [Integer]
            Integer number ranges from 1 and max value of the value of the attribute
            "days" of the River object.

        Returns
        -------
        [Date time ]
            date object.

        """
        # convert the index into date
        return self.ReferenceIndex.loc[Index,'date']

    def DateToIndexRRM(self,Date):
        """
        ===================================================
             DateToIndexRRM(Date)
        ===================================================
        DateToIndex takes a date and returns a the order of the days in the
        time series. The time series starts from the value of the "start" for
        a length of "days" value

        Parameters
        ----------
        Date : [string/date time object]
            string in the format of "%Y-%m-%d" or a date time object.

        Returns
        -------
        [Integer]
            the order oif the date in the time series.

        """
        if type(Date) == str:
            Date = dt.datetime.strptime(Date,"%Y-%m-%d")
        return np.where(self.ReferenceIndex['date'] == Date)[0][0]+1

    def Read1DConfigFile(self, Path):
        File  = open(Path)
        Wholefile = File.readlines()
        File.close()
        # project path
        self.WorkDirectory = Wholefile[1][:-1]
        # cross sections file
        self.XSF  = Wholefile[4][:-1]
        self.ReadCrossSections(self.WorkDirectory + "/inputs/1d/topo/" + self.XSF)
        # Laterals file, BC file
        self.LateralsF, self.BCF= Wholefile[6][:-1].split(" ")
        # RiverNetwork file
        self.RiverNetworkF = Wholefile[8][:-1]
        self.RiverNetwork(self.WorkDirectory + "/inputs/1d/topo/" + self.RiverNetworkF)
        # Slope File
        self.SlopeF = Wholefile[10][:-1]
        self.Slope(self.WorkDirectory + "/inputs/1d/topo/" + self.SlopeF)
        self.NoSeg = len(self.slope)
        # Calibration file
        self.CalibrationF = Wholefile[12][:-1]
        # 1D-2D Coupling file
        self.Coupling1D2DF = Wholefile[14][:-1]
        # Run mode
        RunMode, SubID = Wholefile[16][:-1].split("#")[0].strip().split(" ")
        self.RunMode = int(RunMode)
        # Segment ID
        self.SubID = int(SubID)
        # Customized Run file
        self.Customized_BC_F = Wholefile[18][:-1]
        self.CustomizedRunsPath = Wholefile[19][:-1]

        # Results
        DeleteOld = Wholefile[22][:-1]
        DeleteOld = int(DeleteOld)
        SaveHourly, SaveOverToppingTot, SaveOverToppingDetailed = Wholefile[24][:-1].split(" ")
        SaveHourly = int(SaveHourly)
        SaveOverToppingTot = int(SaveOverToppingTot)
        SaveOverToppingDetailed = int(SaveOverToppingDetailed)
        SaveQ, SaveWL, SaveUSBC = Wholefile[26].split("#")[0][:-1].split(" ")
        SaveQ = int(SaveQ)
        SaveWL = int(SaveWL)
        SaveUSBC = int(SaveUSBC)

        self.ResultsDetails = dict(DeleteOld=DeleteOld, SaveHourly=SaveHourly,
                       SaveOverToppingTot=SaveOverToppingTot,
                       SaveOverToppingDetailed=SaveOverToppingDetailed,
                       SaveQ = SaveQ, SaveWL=SaveWL, SaveUSBC=SaveUSBC)

        # Temporal Resolution
        self.RRMTemporalResolution = Wholefile[29].split("#")[0][:-1]
        self.HMTemporalResolution = Wholefile[30].split("#")[0][:-1]
        self.HMTemporalResolution = int(self.HMTemporalResolution)

        if self.RRMTemporalResolution == "Daily":
            # number of time steps for the 1D model
            self.TS = 24*self.HMTemporalResolution

        Start, End = Wholefile[32][:-1].split(" ")
        self.SimStartIndex = int(Start)
        self.SimEndIndex = int(End)
        self.SimStart = self.IndexToDate(self.SimStartIndex)
        self.SimEnd = self.IndexToDate(self.SimEndIndex)
        self.OneDTempR = 60 # in seconds



        # 1D thresholds
        MinQ, MinDepth = Wholefile[34][:-1].split(" ")
        MinQ = float(MinQ)
        MinDepth = float(MinDepth)

        # 1D or 2D
        ModelMode, OvertoppingOnly = Wholefile[36][:-1].split(" ")
        ModelMode = int(ModelMode)
        OvertoppingOnly = int(OvertoppingOnly)

        self.D1 = dict(MinQ=MinQ, MinDepth=MinDepth, ModelMode=ModelMode,
                       OvertoppingOnly=OvertoppingOnly)

        DischargeThreshold = Wholefile[38][:-1]
        DischargeThreshold = int(DischargeThreshold)

        SimulationMode = Wholefile[40][:-1]
        SimulationMode = int(SimulationMode)
        ResultsFormat = Wholefile[42][:-1]
        ResultsFormat = int(ResultsFormat)
        CalcInundationDuration = Wholefile[44][:-1]
        CalcInundationDuration = int(CalcInundationDuration)
        CalcReturnPerion = Wholefile[46][:-1]
        CalcReturnPerion = int(CalcReturnPerion)
        self.D2 = dict(DischargeThreshold=DischargeThreshold, SimulationMode=SimulationMode,
                       ResultsFormat=ResultsFormat, CalcInundationDuration=CalcInundationDuration,
                       CalcReturnPerion=CalcReturnPerion)

    def ReadCrossSections(self,Path):
        """
        ===========================================
          CrossSections(self,Path)
        ===========================================
        CrossSections method reads the cross section data of the river and assign it
        to an attribute "Crosssections" of type dataframe
        """
        if self.Version == 1 or self.Version == 2:
            self.crosssections = pd.read_csv(Path, delimiter = ',', skiprows =1  )
        else:
            self.crosssections = pd.read_csv(Path, delimiter = ',')
            self.XSno = len(self.crosssections)

    def Slope(self,Path):
        """
        ====================================
            Slope(Path)
        ====================================

        Parameters
        ----------
            1-Path : [String]
                path to the Guide.csv file including the file name and extention
                "RIM1Files + "/Guide.csv".

        Returns
        -------
        None.

        """
        if self.Version == 1 or self.Version == 2:
            self.slope = pd.read_csv(Path, delimiter = ",",header = None)
            self.slope.columns = ['id','f1','slope','f2']
        else:
            self.slope = pd.read_csv(Path, delimiter = ",",header = None)
            self.slope.columns = ['id','slope']

    def ReturnPeriod(self,Path):
        """
        ==========================================
             ReturnPeriod(Path)
        ==========================================
        ReturnPeriod method reads the HQ file which contains all the computational nodes
        with HQ2, HQ10, HQ100
        Parameters
        ----------
            1-Path : [String]
                path to the HQ.csv file including the file name and extention
                "RIM1Files + "/HQRhine.csv".

        Returns
        -------
            1-RP:[data frame attribute]
                containing the river computational node and calculated return period
                for with columns ['node','HQ2','HQ10','HQ100']
        """
        self.RP = pd.read_csv(Path, delimiter = ",",header = None)
        self.RP.columns = ['node','HQ2','HQ10','HQ100']


    def RiverNetwork(self, Path):
        """
        =====================================================
              RiverNetwork(Path)
        =====================================================
        RiverNetwork method rad the table of each computational node followed by
        upstream and then downstream node (TraceAll file)

        ==============   ====================================================
        Keyword          Description
        ==============   ====================================================
        1-Path :         [String] path to the Trace.txt file including the file name and extention
                            "path/Trace.txt".

        Returns
        -------
            1-rivernetwork:[data frame attribute]
                containing the river network with columns ['SubID','US','DS']
        """
        if self.Version == 1 or self.Version == 2:
            self.rivernetwork = pd.read_csv(Path, delimiter = ',') #,header = None
            self.rivernetwork.columns = ['id','us','ds']
        else:
            File  = open(Path)
            Wholefile = File.readlines()
            File.close()
            rivernetwork = pd.DataFrame(columns=Wholefile[0][:-1].split(','))
            # all lines excpt the last line
            for i in range(1,len(Wholefile)-1):
                rivernetwork.loc[i-1,rivernetwork.columns[0:2].tolist()] = [int(j) for j in Wholefile[i][:-1].split(',')[0:2]]
                rivernetwork.loc[i-1,rivernetwork.columns[2]] = [int(j) for j in Wholefile[i][:-1].split(',')[2:]]
            # last line does not have the \n at the end
            i = len(Wholefile)-1
            rivernetwork.loc[i-1,rivernetwork.columns[0:2].tolist()] = [int(j) for j in Wholefile[i][:-1].split(',')[0:2]]
            rivernetwork.loc[i-1,rivernetwork.columns[2]] = [int(j) for j in Wholefile[i].split(',')[2:]]
            rivernetwork.columns = ["No","id","us"]
            self.rivernetwork = rivernetwork[:]
            self.Segments = self.rivernetwork['id'].tolist()


    def TraceSegment(self,ID):
        """
        ========================================
        	Trace(self,SubID)
        ========================================
        Trace method takes sub basin Id and trace it to get the upstream and
        down stream computational nodes


        Parameters
        ----------
            1- SubID : TYPE
                DESCRIPTION.

        Returns
        -------
            1-SWIMUS : [Integer]
                the Upstream computational node from SWIM Configuration file.
            2-SWIMDS : [Integer]
                the Downstream computational node from SWIM Configuration file.

        Example:
            SubID = 42
            SWIMUS, SWIMDS = River1.Trace(SubID)
            print("sub-basin = "+str(SubID))
            print("DS node= "+str(SWIMDS))
            print("US node= "+str(SWIMUS))
        """

        if self.Version == 1 or self.Version == 2:
            DS = int(self.rivernetwork['ds'][np.where(self.rivernetwork['id'] == ID)[0][0]])
            US = int(self.rivernetwork['us'][np.where(self.rivernetwork['id'] == ID)[0][0]])
        else:
            US = self.rivernetwork['us'][np.where(self.rivernetwork['id'] == ID)[0][0]]
            for i in range(len(self.rivernetwork)):
                if ID in self.rivernetwork.loc[i,'us']:
                    DS = self.rivernetwork.loc[i,'id']
                    break
                else:
                    DS = []

        return  US, DS


    def Trace2(self, ID, US):
        # trace the given segment
            US1, _ = self.TraceSegment(ID)
            if len(US1) > 0 :
                for i in range(len(US1)):
                    US.append(US1[i])
                    self.Trace2(US1[i], US)

    def Trace(self, ID):
        """
        ============================================================
             Trace(self, ID ,US=[])
        ============================================================
        Trace method takes the Id of the segment and trace it upstream

        Parameters
        ----------
        ID : TYPE
            DESCRIPTION.

        Returns
        -------
        US : [list attribute], optional
            the ID of all the upstream segments are going to be stored in a list
            attribute.
        """
        self.US = []
        self.Trace2(ID, self.US)

    def StatisticalProperties(self, Path, Filter = True):
        """
        ====================================================
           StatisticalProperties(Path)
        ====================================================
        StatisticalProperties method reads the parameters of the distribution and
        calculates the the 2, 5, 10, 15, 20, 50, 100 return period discharge for each
        sub-basin to create the parameters file use the code StatisticalProperties
        in the 07ReturnPeriod folder

        Parameters
        ----------
            1-Path : [String]
                path to the "Statistical Properties.txt" file including the
                file name and extention "path/Statistical Properties.txt".
            2-Filter:[Boolen]
                true to filter the nodes to the nodes used in RIM (upstream nodes only)
        Returns
        -------
            1-SP:[data frame attribute]
                containing the river computational nodes US of the Sub basins
                and estimated gumbel distribution parameters that fit the time
                series
                ['node','HQ2','HQ10','HQ100']

        """
        self.SP = pd.read_csv(Path, delimiter = ",") #,header = None,skiprows = 0

        if Filter:
            assert hasattr(self, "slope"), "please read the Guide file with the Slope Method as this method uses the the SubID in the guide file as ID"
            assert hasattr(self, "rivernetwork"), "Please read the Tracefile.txt with the RiverNetwork method data "
            # filter all the computational nodes in the file to those only
            # exist in the slope attribute (the nodes in the guide file)
            NewSP = pd.DataFrame(columns = ['id','loc','scale'])
            NewSP['id'] = self.slope['id']
            for i in range(len(self.slope)):
                # get the location of the USnode in the rivernetwork attribute
                loc = np.where(self.rivernetwork['id'] == self.slope.loc[i,'id'])[0][0]
                #  get the location of USnode in the SP attribute
                try:
                    loc = np.where(self.SP['id'] == self.rivernetwork.loc[loc,'US'])[0][0]
                    NewSP.loc[i,['loc','scale']] = self.SP.loc[loc,['loc','scale']]
                except:
                    NewSP.loc[i,['loc','scale']] = [-1,-1]
                    continue

            self.SP = NewSP
        # calculate the 2, 5, 10, 15, 20 return period doscharge
        T = np.array([2, 5, 10, 15, 20, 50, 100, 200, 500, 1000, 5000])
        self.SP = self.SP.assign(RP2 = 0, RP5 = 0, RP10 = 0, RP15 = 0, RP20 = 0, RP50 = 0,
                                 RP100 = 0, RP200 = 0, RP500 = 0, RP1000 = 0,
                                 RP5000 = 0)
        F = 1 - (1/T)
        for i in range(len(self.SP)):
            if self.SP.loc[i,'loc'] != -1:
                self.SP.loc[i,self.SP.keys()[3:].tolist()] = gumbel_r.ppf(F,loc=self.SP.loc[i,'loc'],
                                                                          scale=self.SP.loc[i,'scale']).tolist()

    def GetReturnPeriod(self, SubID, Q):
        """
        =================================================================
             GetReturnPeriod(self, SubID, Q)
        =================================================================
        GetReturnPeriod method takes given discharge and using the distribution
        properties of a particular sub basin it it calculates the return period

        Parameters
        ----------
            1-SubID : [Integer]
                Sub-basin id.
            2-Q : [Float]
                Discharge value.

        Returns
        -------
            1-return Period :[Float]
                return periodcal culated for the given discharge using the parameters
            of the distribution for the catchment.

        """
        assert hasattr(self, "SP"), "Please read the statistical properties file for the catchment first"

        try:
            loc = np.where(self.SP['id'] == SubID)[0][0]
            F = gumbel_r.cdf(Q, loc = self.SP.loc[loc,'loc'],
                         scale = self.SP.loc[loc,'scale'])
            return 1/(1-F)
        except:
            return -1

    def GetQForReturnPeriod(self, SubID, T):

        assert hasattr(self, "SP"), "Please read the statistical properties file for the catchment first"
        F = 1 - (1/T)
        try:
            loc = np.where(self.SP['id'] == SubID)[0][0]
            Q = gumbel_r.ppf(F, loc=self.SP.loc[loc,"loc"],
                             scale=self.SP.loc[loc,"scale"])
            return Q
        except:
            return -1


    def GetBankfullDepth(self,function,ColumnName):
        """
        =========================================================
            GetBankfullDepth(function,ColumnName)
        =========================================================
        GetBankfullDepth method takes a function that calculates the bankful depth
        as a function of bankful width and calculate the depth

        Parameters
        ----------
            1-function : [function]
                function that takes one input and calculates the depth.
            2-ColumnName : [String]
                A name for the column to store the calculated depth at the
                cross section dataframe.

        Returns
        -------
            dataframe column in the cross section attribute with the calculated
            depth.

        """
        self.crosssections[ColumnName] = self.crosssections['b'].to_frame().applymap(function)


    def GetCapacity(self,ColumnName, Option = 1):
        """
        ======================================================
              GetCapacity(ColumnName)
        ======================================================
        GetCapacity method calculates the discharge that is enough to fill the
        cross section using kinematic wave approximation (using bed slope with manning)

        Parameters
        ----------
            1-ColumnName : [String]
                A name for the column to store the calculated depth at the
                cross section dataframe.
            2-Option : [Integer]
                1 if you want to calculate the capacity of the bankfull area,
                2 if you want to calculate the capacity of the whole cross section
                to the lelvel of the lowest dike

        Returns
        -------
            1- Discharge: [dataframe column]
                the calculated discharge will be stored in the crosssections
                attribute in the River object in a columns with the given ColumnName
            2- ColumnName+"RP":[dataframe column]
                if you already rad the statistical properties another column containing
                the coresponding return period to the discharge,
                the calculated return period will be stored in a column with a name
                the given ColumnName+"RP", if the ColumnName was QC then the discharge
                will be in a Qc columns and the return period will be in QcRP column
        """
        for i in range(len(self.crosssections)-1):
            # get the slope
            if self.crosssections.loc[i,'swimid'] == self.crosssections.loc[i+1,'swimid']:
                slope = (self.crosssections.loc[i,'gl'] - self.crosssections.loc[i+1,'gl'])/500
            else:
                slope = abs(self.crosssections.loc[i,'gl'] - self.crosssections.loc[i-1,'gl'])/500
            self.crosssections.loc[i,'Slope'] = slope

            if Option == 1:
                # bankfull area
                self.crosssections.loc[i,ColumnName] = (1/self.crosssections.loc[i,'m']) * self.crosssections.loc[i,'b'] * (self.crosssections.loc[i,'dbf']) ** (5/3)
                self.crosssections.loc[i,ColumnName]  = self.crosssections.loc[i,ColumnName] * (slope)**(1/2)

            else:
                # lowest dike
                # get the vortices of the cross sections
                H = self.crosssections.loc[i,['zl','zr']].min()
                Hl, Hr, Bl, Br, B, Dbf = self.crosssections.loc[i,['hl','hr','bl','br','b','dbf']].tolist()
                BedLevel = self.crosssections.loc[i,'gl']
                Coords = self.GetVortices(H - BedLevel, Hl, Hr, Bl, Br, B, Dbf)
                # get the area and perimeters
                Area, Perimeter = self.PolygonGeometry(Coords)
                # self.crosssections.loc[i,'Area'] = Area
                # self.crosssections.loc[i,'Perimeter'] = Perimeter
                self.crosssections.loc[i,ColumnName] = (1/self.crosssections.loc[i,'m']) * Area * ((Area/Perimeter) ** (2/3))
                self.crosssections.loc[i,ColumnName]  = self.crosssections.loc[i,ColumnName] * (slope)**(1/2)

            if hasattr(self, "SP"):
                # print(self.GetReturnPeriod(self.crosssections.loc[i,'swimid'],self.crosssections.loc[i,ColumnName]))
                RP = self.GetReturnPeriod(self.crosssections.loc[i,'swimid'],self.crosssections.loc[i,ColumnName])
                if np.isnan(RP):
                    RP = -1
                self.crosssections.loc[i,ColumnName + "RP"] = round(RP,2)


    def CalibrateDike(self, ObjectiveRP, CurrentRP):
        """
        ========================================================
            CalibrateDike(ObjectiveRP, CurrentRP)
        ========================================================
        CalibrateDike method takes cross section and based on a

        Parameters
        ----------
        ObjectiveRP : [string]
            Column name in the cross section dataframe you have to enter it.
        CurrentRP : [string]
            Column name in the cross section dataframe created by the GetCapacity
            method.

        Returns
        -------
        New RP: [data frame column]
            column in the cross section dataframe containing the new return period.
        New Capacity: [data frame column]
            column in the cross section dataframe containing the the discharge enough
            to fill the cross section after raising the dikes to the objective
            return period.
        """

        assert hasattr(self, 'SP'), "Please read the statistical properties file first using StatisticalProperties method"
        assert hasattr(self, 'crosssections'), "please read the cross section data first with the method CrossSections"
        assert CurrentRP in self.crosssections.columns, CurrentRP + " in not in the cross section data please use GetCapacity method to calculate the current return perion "
        assert ObjectiveRP in self.crosssections.columns, ObjectiveRP + " in not in the cross section data please create a column in the cross section data containing the objective return period"

        # self.crosssections['zlnew'] = -1
        # self.crosssections['zrnew'] = -1
        self.crosssections.loc[:,'zlnew'] = self.crosssections.loc[:,'zl']
        self.crosssections.loc[:,'zrnew'] = self.crosssections.loc[:,'zr']

        for i in range(len(self.crosssections)-2):

            if self.crosssections.loc[i,'swimid'] == self.crosssections.loc[i+1,'swimid']:
                slope = (self.crosssections.loc[i,'gl'] - self.crosssections.loc[i+1,'gl'])/500
            else:
                slope = abs(self.crosssections.loc[i,'gl'] - self.crosssections.loc[i-1,'gl'])/500
            # self.crosssections.loc[i,'Slope'] = slope
            Hl, Hr, Bl, Br, B, Dbf = self.crosssections.loc[i,['hl','hr','bl','br','b','dbf']].tolist()
            BedLevel = self.crosssections.loc[i,'gl']



            if self.crosssections.loc[i,CurrentRP] < self.crosssections.loc[i,ObjectiveRP] and self.crosssections.loc[i, CurrentRP] != -1:
                print("XS-"+str(self.crosssections.loc[i,'xsid']))
                print('Old RP = ' + str(self.crosssections.loc[i, CurrentRP]))
                print('Old H = ' + str(self.crosssections.loc[i,['zl','zr']].min()))

                self.crosssections.loc[i,"New RP"] = self.crosssections.loc[i, CurrentRP]
                # self.crosssections.loc[i,'zlnew'] = self.crosssections.loc[i,'zl']
                # self.crosssections.loc[i,'zrnew'] = self.crosssections.loc[i,'zr']

                while self.crosssections.loc[i,"New RP"] < self.crosssections.loc[i, ObjectiveRP]:
                    # get the vortices of the cross sections
                    if self.crosssections.loc[i,'zlnew'] < self.crosssections.loc[i,'zrnew'] :
                        self.crosssections.loc[i,'zlnew'] = self.crosssections.loc[i,'zlnew'] + 0.1
                    else:
                        self.crosssections.loc[i,'zrnew'] = self.crosssections.loc[i,'zrnew'] + 0.1

                    H = self.crosssections.loc[i,['zlnew','zrnew']].min()
                    # print("H=" + str(H))

                    Coords = self.GetVortices(H - BedLevel, Hl, Hr, Bl, Br, B, Dbf)
                    # get the area and perimeters
                    Area, Perimeter = self.PolygonGeometry(Coords)
                    # self.crosssections.loc[i,'Area'] = Area
                    # self.crosssections.loc[i,'Perimeter'] = Perimeter
                    self.crosssections.loc[i,"New Capacity"] = (1/self.crosssections.loc[i,'m']) * Area * ((Area/Perimeter) ** (2/3))
                    self.crosssections.loc[i,"New Capacity"]  = self.crosssections.loc[i, "New Capacity"] * (slope)**(1/2)
                    # print("New QC " + str(self.crosssections.loc[i,ColumnName+"calib"]))
                    # print(self.GetReturnPeriod(self.crosssections.loc[i,'swimid'],self.crosssections.loc[i,ColumnName]))
                    RP = self.GetReturnPeriod(self.crosssections.loc[i,'swimid'],self.crosssections.loc[i,"New Capacity"])

                    self.crosssections.loc[i,"New RP"] = round(RP,2)
                    # print("New RP = "+str(RP))
                print("New RP = "+str(round(RP,2)))
                print("New H = " + str(round(H,2)))
                print("---------------------------")


    def Overtopping(self,OvertoppingResultPath=''):

        """
        =====================================================
          Overtopping(self, OvertoppingResultF)
        =====================================================

        Overtopping method reads the overtopping files and for each cross section
        in each sub-basin it will strore the days where overtopping happens
        in this cross section.

        you do not need to delete empty files or anything just give the code
        the sufix you used for the left overtopping file and the sufix you used
        for the right overtopping file

        Inputs:
        -------
            1-OvertoppingResultF:
                [String] a path to the folder includng 2D results.

        Returns
        -------
            1-OverToppingSubsLeft : [dictionary attribute]
                dictionary having sub-basin IDs as a key and for each sub-basins
                it contains dictionary for each cross section having the days of
                overtopping.
            1-OverToppingSubsRight : [dictionary attribute]
                dictionary having sub-basin IDs as a key and for each sub-basins
                it contains dictionary for each cross section having the days of
                overtopping.
        """

        #sort files
        leftOverTop = list()
        RightOverTop = list()
        # get names of files that has _left or _right at its end
        if OvertoppingResultPath == '':
            OvertoppingResultPath = self.OneDResultPath

        All1DFiles = os.listdir(OvertoppingResultPath)
        for i in range(len(All1DFiles)) :
            if All1DFiles[i].endswith(self.leftOvertopping_Suffix):
                leftOverTop.append(All1DFiles[i])
            if All1DFiles[i].endswith(self.RightOvertopping_Suffix):
                RightOverTop.append(All1DFiles[i])

        # two dictionaries for overtopping left and right
        OverToppingSubsLeft = dict()
        OverToppingSubsRight = dict()
        # the _left and _right files has all the overtopping discharge
        # but sometimes the sum of all the overtopping is less than a threshold specified
        # and then the 2D  algorithm does not run so these cross sections you will not find
        # any inundation beside it in the maps but you will find it in the _left or _right maps

        # for each sub-basin that has overtopping from the left dike
        for i in range(len(leftOverTop)):

            try:
                # open the file (if there is no column sthe file is empty)
                data = pd.read_csv(self.OneDResultPath + leftOverTop[i],header =None,delimiter = r'\s+')
                # add the sub basin to the overtopping dictionary of sub-basins
                OverToppingSubsLeft[leftOverTop[i][:-len(self.leftOvertopping_Suffix)]] = dict()
            except:
                continue
            # get the XS that overtopping happened from
            XSs = list(set(data.loc[:,2]))
            # for each XS get the days
            for j in range(len(XSs)):
                OverToppingSubsLeft[leftOverTop[i][:-len(self.leftOvertopping_Suffix)]][XSs[j]] = list(set(data[0][data[2] == XSs[j]].tolist()))

        for i in range(len(RightOverTop)):

            try:
                # open the file
                data = pd.read_csv(self.OneDResultPath + RightOverTop[i],header =None,delimiter = r'\s+')
                # add the sub basin to the overtopping dictionary of sub-basins
                OverToppingSubsRight[RightOverTop[i][:-len(self.RightOvertopping_Suffix)]] = dict()
            except :
                continue
            # get the XS that overtopping happened from
            XSs = list(set(data.loc[:,2]))
            # for each XS get the days
            for j in range(len(XSs)):
                OverToppingSubsRight[RightOverTop[i][:-len(self.RightOvertopping_Suffix)]][XSs[j]] = list(set(data[0][data[2] == XSs[j]].tolist()))

        self.OverToppingSubsLeft = OverToppingSubsLeft
        self.OverToppingSubsRight = OverToppingSubsRight


    def GetOvertoppedXS(self, day, allEventdays = True):
        """
        ==========================================================
             GetOvertoppedXS( day, allEventdays = True)
        ==========================================================
        GetOvertoppedXS method get the cross sections that was overtopped in
        a given date(you have to read the overtopping data first with the method
        Overtopping), since inudation maps gets the max depth for the whole event
        the method can also trace the event back to the beginning and get all
        the overtopped XS from the beginning of the Event till the given day
        (you have to give the River object the EventIndex attribute from the
        Event Object)

        Parameters
        ----------
            1-day : [Integer]
                the day you want to get the overtopped cross section for.
            2-allEventdays : [Bool], optional
                if you want to get the overtopped cross section for this day only
                or for the whole event. The default is True.

        Returns
        -------
            1-XSLeft : [list]
                list of cross section IDs that has overtopping from the left bank.
            2-XSRight : [list]
                list of cross section IDs that has overtopping from the right bank.
        Example:
            1- for a given day
                RIM2River = RV.River('RIM2.0')
                RIM2River.Overtopping(wpath2 + "/results/1d/")
                day = 1122
                XSleft, XSright = RIM2River.GetOvertoppedXS(day,False)

            2- from the beginning of the event till the given day
                RIM2River = RV.River('RIM2.0')
                RIM2River.Overtopping(wpath2 + "/results/1d/")
                # read precreated EventIndex table
                RIM2Event.ReadEventIndex(wpath2 + "/" + "EventIndex.txt")
                # give the EventIndex table to the River Object
                RIM2River.EventIndex = RIM1.EventIndex
                day = 1122
                XSleft, XSright = RIM2River.GetOvertoppedXS(day,False)

        """
        if allEventdays:
            loc = np.where(self.EventIndex['id'] == day)[0][0]
            # get all the days in the same event before that day as the inundation in the maps may
            # happen due to any of the days before not in this day
            Eventdays = self.EventIndex.loc[loc - self.EventIndex.loc[loc,'IndDiff'] : loc,'id'].tolist()
        else:
            Eventdays = [day,]

        XSLeft = list()
        XSRight = list()

        for k in range(len(Eventdays)):
            dayi = Eventdays[k]
            # for each sub-basin in the overtopping left dict
            for i in range(len(self.OverToppingSubsLeft.keys())):
                SubID = list(self.OverToppingSubsLeft.keys())[i]
                # get all cross section that overtopped before
                XSs = list(self.OverToppingSubsLeft[SubID].keys())
                # for each xross section check if the day is sored inside
                for j in range(len(XSs)):
                    if dayi in self.OverToppingSubsLeft[SubID][XSs[j]]:
                        XSLeft.append(XSs[j])

            for i in range(len(self.OverToppingSubsRight.keys())):
                SubID = list(self.OverToppingSubsRight.keys())[i]
                XSs = list(self.OverToppingSubsRight[SubID].keys())
                for j in range(len(XSs)):
                    if dayi in self.OverToppingSubsRight[SubID][XSs[j]]:
                        XSRight.append(XSs[j])

        XSLeft = list(set(XSLeft))
        XSRight = list(set(XSRight))

        return XSLeft,  XSRight

    def GetSubBasin(self,xsid):
        """
        ===========================================
             GetSubBasin(xsid)
        ===========================================
        GetSubBasin method returned the sub-basin that the Cross section belong
        Parameters
        ----------
            1-xsid : [Integer]
                cross section id.

        Returns
        -------
            [Integer]
                sub-basin ID.
        """
        loc = np.where(self.crosssections['xsid'] == xsid)[0][0]
        return self.crosssections.loc[loc,'swimid']


    def GetFloodedSubs(self, OvertoppedXS = [], day = [1,], allEventdays = True):
        """
        =============================================================================
            GetFloodedSubs(OvertoppedXS = [], day = [1,], allEventdays = True)
        =============================================================================
        GetFloodedSubs gets the inundeated sub-basins

        Parameters
        ----------
            1-OvertoppedXS : [list], optional
                list of cross sections overtopped (if you already used the GetOvertoppedXS
                method to get the overtopped XSs for a specific day).The default is [].
                If entered the algorithm is not going to look at the over arguments
                of the method.
            2-day : [list], optional
                if you want to get the flooded subs for a specific list of days. The default is 1.
            3-allEventdays : [Bool], optional in case user entered OvertoppedXS
                if the user entered day the allEventdays is a must. The default is True.

        Returns
        -------
            1-Subs : TYPE
                DESCRIPTION.

        Examples:
            1- get the flooded subs for a specific days
                floodedSubs = RIM1River.GetFloodedSubs(day = [1122,1123], allEventdays=False)

            2- get the flooded subs from already obtained overtopped XSs
                day = 1122
                XSleft, XSright = RIM1River.GetOvertoppedXS(day,False)
                floodedSubs = RIM1River.GetFloodedSubs(OvertoppedXS = XSleft + XSright, allEventdays=False)
        """
        Subs = list()
        # if you already used the GetOvertoppedXS and have a list of xs overtopped
        # at specific day
        if len(OvertoppedXS) > 0:
            OvertoppedXS = list(set(OvertoppedXS))
            for i in range(len(OvertoppedXS)):
                Subs.append(self.GetSubBasin(OvertoppedXS[i]))
        else:
            for j in range(len(day)):
                XSLeft, XSRight = self.GetOvertoppedXS(day[j], allEventdays)
                OvertoppedXS = XSLeft + XSRight
                OvertoppedXS = list(set(OvertoppedXS))

                for i in range(len(OvertoppedXS)):
                    Subs.append(self.GetSubBasin(OvertoppedXS[i]))

        # to remove duplicate subs
        Subs = list(set(Subs))
        return Subs

    def DetailedOvertopping(self,floodedSubs,eventdays):
        """
        ========================================================
           DetailedOvertopping(floodedSubs,eventdays)
        ========================================================
        DetailedOvertopping method takes list of days and the flooded subs-basins
        in those days and get the left and right overtopping for each sub-basin for
        each day

        Parameters
        ----------
            1-floodedSubs : [list]
                list of sub-basins that are flooded during the given days.
            2-eventdays : [list]
                list od daysof an event.

        Returns
        -------
            1-DetailedOvertoppingLeft : [dataframe attribute]
                dataframe having for each day of the event the left overtopping
                to each sub-basin.

            2-DetailedOvertoppingRight : [dataframe attribute]
                dataframe having for each day of the event the right overtopping
                to each sub-basin.
        """
        columns = floodedSubs + ['sum']
        self.DetailedOvertoppingLeft  = pd.DataFrame(index = eventdays + ['sum'], columns = columns)
        self.DetailedOvertoppingRight = pd.DataFrame(index = eventdays + ['sum'], columns = columns)

        # Left Bank
        for i in range(len(floodedSubs)):
            try :
                # try to open and read the overtopping file
                data = pd.read_csv(self.OneDResultPath + str(floodedSubs[i]) + self.leftOvertopping_Suffix,
                                   header =None,delimiter = r'\s+')
                # get the days in the sub
                days = list(set(data.loc[:,0]))

                for j in range(len(eventdays)):
                    # check whether this sub basin has flooded in this particular day
                    if eventdays[j] in days:
                        # filter the dataframe to the discharge column (3) and the days
                        self.DetailedOvertoppingLeft.loc[eventdays[j],floodedSubs[i]] = data.loc[data[0] == eventdays[j] ,3].sum()
                    else:
                        self.DetailedOvertoppingLeft.loc[eventdays[j],floodedSubs[i]] = 0
            except:
                self.DetailedOvertoppingLeft.loc[:,floodedSubs[i]] = 0
                continue

        # right Bank
        for i in range(len(floodedSubs)):
            try :
                # try to open and read the overtopping file
                data = pd.read_csv(self.OneDResultPath + str(floodedSubs[i]) + self.RightOvertopping_Suffix,
                                   header =None,delimiter = r'\s+')
                # get the days in the sub
                days = list(set(data.loc[:,0]))

                for j in range(len(eventdays)):
                    # check whether this sub basin has flooded in this particular day
                    if eventdays[j] in days:
                        # filter the dataframe to the discharge column (3) and the days
                        self.DetailedOvertoppingRight.loc[eventdays[j],floodedSubs[i]] = data.loc[data[0] == eventdays[j] ,3].sum()
                    else:
                        self.DetailedOvertoppingRight.loc[eventdays[j],floodedSubs[i]] = 0
            except:
                self.DetailedOvertoppingRight.loc[eventdays[j],floodedSubs[i]] = 0
                continue

        # sum overtopping for each day
        for j in range(len(eventdays)):
            self.DetailedOvertoppingLeft.loc[eventdays[j],'sum'] = self.DetailedOvertoppingLeft.loc[eventdays[j],:].sum()
            self.DetailedOvertoppingRight.loc[eventdays[j],'sum'] = self.DetailedOvertoppingRight.loc[eventdays[j],:].sum()
        # sum overtopping for each sub basin
        for j in range(len(floodedSubs)):
            self.DetailedOvertoppingLeft.loc['sum',floodedSubs[j]] = self.DetailedOvertoppingLeft.loc[:,floodedSubs[j]].sum()
            self.DetailedOvertoppingRight.loc['sum',floodedSubs[j]] = self.DetailedOvertoppingRight.loc[:,floodedSubs[j]].sum()

        # self.DetailedOvertoppingLeft.loc['sum','sum'] = self.DetailedOvertoppingLeft.loc[:,'sum'].sum()
        # self.DetailedOvertoppingRight.loc['sum','sum'] = self.DetailedOvertoppingRight.loc[:,'sum'].sum()

    def Coordinates(self, Bankful = False):
        """
        ==================================================
            Coordinates(Bankful = False)
        ==================================================
        Coordinates method calculate the real coordinates for all the vortixes
        of the cross section

        Parameters
        ----------
            Bankful : [Bool], optional
                if the cross section data has a bankful depth or not. The default is False.

        Returns
        -------
            1-coordenates will be added to the "crosssection" attribute.

        """
        if Bankful == True:
            self.crosssections = self.crosssections.assign(x1 =0,y1=0,z1=0,
                                                           x2=0,y2=0,z2=0,
                                                           x3=0,y3=0,z3=0,
                                                           x4=0,y4=0,z4=0,
                                                           x5=0,y5=0,z5=0,
                                                           x6=0,y6=0,z6=0,
                                                           x7=0,y7=0,z7=0,
                                                           x8=0,y8=0,z8=0)

            for i in range(len(self.crosssections)):
                inputs = self.crosssections.loc[i,list(self.crosssections.columns)[3:15]].tolist()
                Dbf = self.crosssections.loc[i,list(self.crosssections.columns)[16]]

                outputs = self.GetCoordinates(inputs,Dbf)

                self.crosssections.loc[i,['x1','x2','x3','x4','x5','x6','x7','x8']] = outputs[0]

                self.crosssections.loc[i,['y1','y2','y3','y4','y5','y6','y7','y8']] = outputs[1]

                self.crosssections.loc[i,['z1','z2','z3','z4','z5','z6','z7','z8']] = outputs[2]
        else:
            self.crosssections = self.crosssections.assign(x1 =0,y1=0,z1=0,
                                                           x2=0,y2=0,z2=0,
                                                           x3=0,y3=0,z3=0,
                                                           x4=0,y4=0,z4=0,
                                                           x5=0,y5=0,z5=0,
                                                           x6=0,y6=0,z6=0)
            Dbf = False
            for i in range(len(self.crosssections)):
                inputs = self.crosssections.loc[i,list(self.crosssections.columns)[3:15]].tolist()

                outputs = self.GetCoordinates(inputs, Dbf)

                self.crosssections.loc[i,['x1','x2','x3','x4','x5','x6']] = outputs[0]

                self.crosssections.loc[i,['y1','y2','y3','y4','y5','y6']] = outputs[1]

                self.crosssections.loc[i,['z1','z2','z3','z4','z5','z6']] = outputs[2]

        # TODO create a method to take the created coordinates and convert each cross section
        # into  a polygon
        # TODO another method to take the cross section coordinates of a whole sub basins
        # and convert them into one polygon
    # def CreatePolygons(self):

    @staticmethod
    def PolygonGeometry(Coords):
        """
        ======================================
            PolygonGeometry(Coords)
        ======================================
        PolygonGeometry method calculates the area and perimeter of some coordinates

            Parameters
            ----------
                1-Coords : [array]
                    numpy array in the shape of (n*2) where n is the number of
                    points

            Returns
            -------
                1-area : [float]
                    area between the coordinates.
                2-peri : [float]
                    perimeter between the coordinates.

            Example:
            -------
                coords = np.array([[0,1],[0,0],[5,0],[5,1]])
                RV.River.PolygonGeometry(coords)

        """
        area = 0.0
        peri = 0.0
        for i in range(np.shape(Coords)[0]-1):
            area = area + Coords[i,0]*Coords[i+1,1] - Coords[i+1,0]*Coords[i,1]
            peri = peri + ( (Coords[i+1,0] - Coords[i,0])**2 + (Coords[i+1,1]-Coords[i,1])**2 )**0.5
        area = area + Coords[np.shape(Coords)[0]-1,0] * Coords[0,1] - Coords[0,0] * Coords[np.shape(Coords)[0]-1,1]
        area = area*0.5

        return area, peri

    @staticmethod
    def PolyArea(Coords):
        """
        =================================
            PolyArea(Coords)
        =================================
        PolyArea method calculates the the area between given coordinates

            Parameters
            ----------
                1-Coords : [array]
                    numpy array in the shape of (n*2) where n is the number of
                    points

            Returns
            -------
                1-area : [float]
                    area between the coordinates.

            Example:
            -------
                coords = np.array([[0,1],[0,0],[5,0],[5,1]])
                River.PolyArea(coords)
        """

        area = 0.0
        for i in range(np.shape(Coords)[0]-1):
            # cros multiplication
            area = area + Coords[i,0]*Coords[i+1,1] - Coords[i+1,0]*Coords[i,1]
        area = area + Coords[np.shape(Coords)[0]-1,0] * Coords[0,1] - Coords[0,0] * Coords[np.shape(Coords)[0]-1,1]
        area = area*0.5

        return area

    @staticmethod
    def PolyPerimeter(Coords):
        """
        ====================================
            PolyPerimeter
        ====================================
        PolyPerimeter method calculates the the perimeter between given coordinates

            Parameters
            ----------
                1-Coords : [array]
                    numpy array in the shape of (n*2) where n is the number of
                    points

            Returns
            -------
                2-peri : [float]
                    perimeter between the coordinates.

            Example:
            -------
                coords = np.array([[0,1],[0,0],[5,0],[5,1]])
                RV.River.PolyPerimeter(coords)

        """
        peri = 0.0
        for i in range(np.shape(Coords)[0]-1):
            # next point coord - current point coord
            peri = peri + ( (Coords[i+1,0] - Coords[i,0])**2 + (Coords[i+1,1] - Coords[i,1])**2 )**0.5

        return peri

    @staticmethod
    def GetCoordinates(XSGeometry, Dbf):
        """
        GetCoordinates calculates the coordinates of all the points (vortices)
        of the cross section

        Parameters
        ----------
            1- BedLevel : [Float]
                DESCRIPTION.
            2- BankLeftLevel : [Float]
                DESCRIPTION.
            3- BankRightLevel : [Float]
                DESCRIPTION.
            4- InterPLHeight : [Float]
                Intermediate point left.
            5- InterPRHeight : [Float]
                Intermediate point right.
            6- Bl : [Float]
                DESCRIPTION.
            7- Br : [Float]
                DESCRIPTION.
            8- xl : [Float]
                DESCRIPTION.
            9- yl : [Float]
                DESCRIPTION.
            10- xr : [Float]
                DESCRIPTION.
            11- yr : [Float]
                DESCRIPTION.
            12- B : [Float]
                DESCRIPTION.
            13- Dbf : [Float/Bool]
                DESCRIPTION.

        Returns
        -------
        Xcoords : [List]
            DESCRIPTION.
        Xcoords : [List]
            DESCRIPTION.
        Zcoords : [List]
            DESCRIPTION.

        """
        BedLevel = XSGeometry[0]
        BankLeftLevel = XSGeometry[1]
        BankRightLevel = XSGeometry[2]
        InterPLHeight = XSGeometry[3]
        InterPRHeight = XSGeometry[4]
        Bl = XSGeometry[5]
        Br = XSGeometry[6]
        xl = XSGeometry[7]
        yl = XSGeometry[8]
        xr = XSGeometry[9]
        yr = XSGeometry[10]
        B = XSGeometry[11]

        Xcoords = list()
        Ycoords = list()
        Zcoords = list()
        # point 1
        Xcoords.append(xl)
        Ycoords.append(yl)
        Zcoords.append(BankLeftLevel)
        # 8 points cross sections
        if Dbf != False:
            # point 2
            Xcoords.append(xl)
            Ycoords.append(yl)
            Zcoords.append(BedLevel + Dbf + InterPLHeight)
            # point 3
            Xcoords.append(xl + (Bl / (Bl + Br + B)) * (xr - xl))
            Ycoords.append(yl + (Bl / (Bl + Br + B)) * (yr - yl))
            Zcoords.append(BedLevel + Dbf)
            # point 4
            Xcoords.append(xl + (Bl / (Bl + Br + B)) * (xr - xl))
            Ycoords.append(yl + (Bl / (Bl + Br + B)) * (yr - yl))
            Zcoords.append(BedLevel)
            # point 5
            Xcoords.append(xl + ((Bl+B) / (Bl + Br + B)) * (xr - xl))
            Ycoords.append(yl + ((Bl+B) / (Bl + Br + B)) * (yr - yl))
            Zcoords.append(BedLevel)
            # point 6
            Xcoords.append(xl + ((Bl+B) / (Bl + Br + B)) * (xr - xl))
            Ycoords.append(yl + ((Bl+B) / (Bl + Br + B)) * (yr - yl))
            Zcoords.append(BedLevel + Dbf)
            # point 7
            Xcoords.append(xr)
            Ycoords.append(yr)
            Zcoords.append(BedLevel + Dbf + InterPRHeight)
        else:
            # point 2
            Xcoords.append(xl)
            Ycoords.append(yl)
            Zcoords.append(BedLevel + InterPLHeight)
            # point 3
            Xcoords.append(xl + (Bl / (Bl + Br + B)) * (xr - xl))
            Ycoords.append(yl + (Bl / (Bl + Br + B)) * (yr - yl))
            Zcoords.append(BedLevel)
            # point 4
            Xcoords.append(xl + ((Bl+B) / (Bl + Br + B)) * (xr - xl))
            Ycoords.append(yl + ((Bl+B) / (Bl + Br + B)) * (yr - yl))
            Zcoords.append(BedLevel)
            # point 5
            Xcoords.append(xr)
            Ycoords.append(yr)
            Zcoords.append(BedLevel + InterPRHeight)

        # point 8
        Xcoords.append(xr)
        Ycoords.append(yr)
        Zcoords.append(BankRightLevel)

        return Xcoords, Ycoords, Zcoords

    @staticmethod
    def GetVortices(H, Hl, Hr, Bl, Br, B, Dbf):
        """
        GetCoordinates calculates the coordinates of all the points (vortices)
        of the cross section

        Parameters
        ----------
            1- H:[Float]
                water depth
            4- InterPLHeight : [Float]
                Intermediate point right height above the bankfull level.
            5- InterPRHeight : [Float]
                Intermediate point right height above the bankfull level.
            6- Bl : [Float]
                DESCRIPTION.
            7- Br : [Float]
                DESCRIPTION.
            12- B : [Float]
                DESCRIPTION.
            13- Dbf : [Float/Bool]
                DESCRIPTION.

        Returns
        -------
        Xcoords : [List]
            DESCRIPTION.
        Xcoords : [List]
            DESCRIPTION.
        Zcoords : [List]
            DESCRIPTION.

        """
        # left side slope  Horizontal: Vertical = 1:sl
        Sl = Hl/Bl
        # right side slope Horizontal: Vertical = 1:sr
        Sr = Hr/Br
        # put vertexes to the local coordination system
        Xcoords = list()
        Ycoords = list()

        if H <= Dbf:
            # Point 1
            Xcoords.append(0)
            Ycoords.append(H)
            # Point 2
            Xcoords.append(0)
            Ycoords.append(0)
            # Point 3
            Xcoords.append(B)
            Ycoords.append(0)
            # Point 4
            Xcoords.append(B)
            Ycoords.append(H)
            # PropXS8P = PolygonGeometry(mp)
        elif H - Dbf < min(Hl, Hr):
            Hnew = H - Dbf
            # Trapizoidal cross section with 4 points
            # case 1
            # Point 1
            Xcoords.append(0)
            Ycoords.append(H)
            # Point 2
            Xcoords.append(Hnew/Sl)
            Ycoords.append(Dbf)
            # Point 3
            Xcoords.append(Hnew/Sl)
            Ycoords.append(0)
            # Point 4
            Xcoords.append((Hnew/Sl) + B)
            Ycoords.append(0)
            # Point 5
            Xcoords.append((Hnew/Sl) + B)
            Ycoords.append(Dbf)
            # Point 6
            Xcoords.append(Hnew/Sl + B + Hnew/Sr)
            Ycoords.append(H)
        elif H - Dbf < max(Hl, Hr) and Hl < Hr:
            Hnew = H - Dbf
            # the height of one of the slopes is higher than the water depth.
            # so the shape of the cross section is 5 points
            # case 2
            # Point 1
            Xcoords.append(0)
            Ycoords.append(H)
            # Point 2
            Xcoords.append(0)
            Ycoords.append(Hl + Dbf)
            # Point 3
            Xcoords.append(Bl)
            Ycoords.append(Dbf)
            # Point 4
            Xcoords.append(Bl)
            Ycoords.append(0)
            # Point 5
            Xcoords.append(Bl + B)
            Ycoords.append(0)
            # Point 6
            Xcoords.append(Bl + B)
            Ycoords.append(Dbf)
            # Point 7
            Xcoords.append(Bl + B + Hnew/Sr)
            Ycoords.append(H)
        elif H - Dbf < max(Hl, Hr) and Hl > Hr:
            Hnew = H - Dbf
            # case 3
            # Point 1
            Xcoords.append(0)
            Ycoords.append(H)
            # Point 2
            Xcoords.append(Hnew/Sl)
            Ycoords.append(Dbf)
            # Point 3
            Xcoords.append(Hnew/Sl)
            Ycoords.append(0)
            # Point 4
            Xcoords.append(Hnew/Sl + B)
            Ycoords.append(0)
            # Point 5
            Xcoords.append(Hnew/Sl + B)
            Ycoords.append(Dbf)
            # Point 6
            Xcoords.append(Hnew/Sl + B + Br)
            Ycoords.append(Hr + Dbf)
            # Point 7
            Xcoords.append(Hnew/Sl + B + Br)
            Ycoords.append(H)
        else:
#       elif H - Dbf  > max(Hl,Hr):
            # Hnew = H - Dbf
            # the whole 6 points cross section
            # case 4
            # Point 1
            Xcoords.append(0)
            Ycoords.append(H)
            # Point 2
            Xcoords.append(0)
            Ycoords.append(Hl + Dbf)
            # Point 3
            Xcoords.append(Bl)
            Ycoords.append(Dbf)
            # Point 4
            Xcoords.append(Bl)
            Ycoords.append(0)
            # Point 5
            Xcoords.append(Bl + B)
            Ycoords.append(0)
            # Point 6
            Xcoords.append(Bl + B)
            Ycoords.append(Dbf)
            # Point 7
            Xcoords.append(Bl + B + Br)
            Ycoords.append(Hr + Dbf)
            # Point 8
            Xcoords.append(Bl + B + Br)
            Ycoords.append(H)
            # calculate the area & perimeter of the bankful area
            # area
            # PropXS8P(1) = PropXS8P(1) + (Dbf * mw)
            # perimeter
            # PropXS8P(2) = PropXS8P(2) + (2*Dbf)
        Coords = np.array([Xcoords, Ycoords]).transpose()

        return Coords


    def GetRatingCurve(self, MaxH=20, Interval=0.02, dx=500):

        So = self.slope/dx
        # Rating Curve
        geom = self.crosssections.loc[self.crosssections.index[0],:]
        # 0  1   2   3  4 5  6  7  8  9  10 11 12 13 14 15
        "id,xsid,gl,zl,zr,hl,hr,bl,br,xl,yl,xr,yr,b,m,dbf"

        Nint = int((max(geom['zl'] - geom['gl'], geom['zr'] - geom['gl']) + MaxH)/Interval)

        Table = np.zeros(shape=(Nint,10))
        HQ = np.zeros(shape=(Nint,2))

        for i in range(Nint):
            Table[i,0] = Interval*(i+1)
            # calculate area & perimeter
            Coords = self.GetVortices(Table[i,0], geom['hl'], geom['hr'], geom['bl'], geom['br'], geom['b'], geom['dbf'])
            Table[i,1:3] = self.PolygonGeometry(Coords)

            # area & perimeter of the Upper part only
            if Table[i,0] <= geom['dbf']:
            	Table[i,3] = 0
            	Table[i,4] = 0
            else :
            	# area of upper part only
            	Table[i,3] =  Table[i,1] - geom['dbf'] * geom['b']
            	# perimeter of upper part
            	Table[i,4] =  Table[i,2] -  2* geom['dbf'] - geom['b']

            # area & perimeter of the lower part only
            Table[i,5] = Table[i,1] - Table[i,3]
            Table[i,6] = Table[i,2] - Table[i,4]

            # Discharge of Upper & lower
            if Table[i,0] <= geom['dbf']:
            	Table[i,7] = 0
            else:
            	Table[i,7] = Table[i,3] * ((Table[i,3] / Table[i,4])** (2.0/3.0))

            Table[i,8] = Table[i,5] * ((Table[i,5]/Table[i,6])** (2.0/3.0))

            Table[i,7] = (1.0/geom['m']) * Table[i,7] * (abs(So)**(0.5))
            Table[i,8] = (1.0/geom['m']) * Table[i,8] * (abs(So)**(0.5))

            # total discharge
            Table[i,9] = Table[i,7] + Table[i,8]

        HQ[:,0] = Table[:,0]
        HQ[:,1] = Table[:,9]

        self.HQ = HQ[:,:]


    def GetDays(self,FromDay,ToDay):
        """
        ========================================================
            GetDays(FromDay,ToDay)
        ========================================================
        GetDays method check if input days exist in the 1D result data
        or not since RIM1.0 simulate only days where discharge is above
        a certain value (2 years return period), you have to enter the
        OneDResultPath attribute of the instance first to read the results of
        the given sub-basin

        Parameters
        ----------
            1-FromDay : [integer]
                the day you want to read the result from.
            2-ToDay : [integer]
                the day you want to read the result to.

        Returns
        -------
            1-Message stating whether the given days exist or not, and if not two
            alternatives are given instead (the earliest day before the given day
                                            and the earliest day after the given
                                            day).
        """

        data = pd.read_csv(self.OneDResultPath + str(self.ID) +'.txt',
                               header =None,delimiter = r'\s+')
        data.columns=["day" , "hour", "xs", "q", "h", "wl"]
        days = list(set(data['day']))
        days.sort()

        if FromDay not in days:
            Alt1 = FromDay

            stop = 0
            # search for the FromDay in the days column
            while stop == 0:
            # for i in range(0,10):
                try:
                    np.where(data['day'] == Alt1)[0][0] #loc =
                    stop = 1
                except:
                    Alt1 = Alt1 - 1
                    # print(Alt1)
                    if Alt1 <= 0 :
                        stop = 1
                    continue

            Alt2 = FromDay
            # FromDay =
            # search for closest later days
            stop = 0
            while stop == 0:
            # for i in range(0,10):
                try:
                    np.where(data['day'] == Alt2)[0][0] #loc =
                    stop = 1
                except:
                    Alt2 = Alt2 + 1
                    # print(Alt2)
                    if Alt2 >= data.loc[len(data)-1,'day']:
                        stop = 1
                    continue

            text = """"
            the FromDay you entered does not exist in the data, and the closest day earlier than your input day is
            """ + str(Alt1) + """  and the closest later day is """ + str(Alt2)
            print(text)

            if abs(Alt1 - FromDay) > abs(Alt2 - FromDay):
                Alt1 = Alt2
        else:
            print("FromDay you entered does exist in the data ")
            # Alt1 = False
            Alt1 = FromDay


        # if ToDay does not exist in the results
        if ToDay not in days:
            Alt3 = ToDay

            stop = 0
            # search for the ToDay in the days column
            while stop == 0:
            # for i in range(0,10):
                try:
                    np.where(data['day'] == Alt3)[0][0] # loc =
                    stop = 1
                except:
                    Alt3 = Alt3 - 1
                    # print(Alt1)
                    if Alt3 <= 0 :
                        stop = 1
                    continue

            Alt4 = ToDay
            # FromDay =
            # search for closest later days
            stop = 0
            while stop == 0:
            # for i in range(0,10):
                try:
                    np.where(data['day'] == Alt4)[0][0] #loc =
                    stop = 1
                except:
                    Alt4 = Alt4 + 1
                    # print(Alt2)
                    if Alt4 >= data.loc[len(data)-1,'day']:
                        Alt4 = data.loc[len(data)-1,'day']
                        stop = 1
                    continue
            # Alt3 = [Alt3, Alt4]
            text = """"
            the Today you entered does not exist in the data, and the closest day earlier than your input day is
            """ + str(Alt3) + """  and the closest later day is """ + str(Alt4)
            print(text)

            if abs(Alt3 - ToDay) > abs(Alt4 - ToDay):
                Alt3 = Alt4

        else:
            print("ToDay you entered does exist in the data ")
            # Alt3 = False
            Alt3 = ToDay


        return Alt1, Alt3


    def Read1DResult(self, SubID, FromDay ='' , ToDay = '', Path = '', FillMissing = False):
        """
        =============================================================================
          Read1DResult(SubID, FromDay = [], ToDay = [], Path = '', FillMissing = False)
        =============================================================================
        Read1DResult method reads the 1D results and fill the missing days in the middle

        Parameters
        ----------
            1-SubID : [integer]
                ID of the sub-basin you want to read its data.
            2-FromDay : [integer], optional
                the index of the day you want the data to start from. The default is empty.
                means read everything
            3-ToDay : [integer], optional
                the index of the day you want the data to end to. The default is empty.
                means read everything
            4-Path : [String], optional
                Path to read the results from. The default is ''.
            5-FillMissing : [Bool], optional
                Fill the missing days. The default is False.

        Returns
        -------
            6-Result1D : [attribute]
                the results read will be stored (as it is without any filter)
                in the attribute "Result1D"
        """

        # if the path is not given try to read from the object predefined OneDresultPath
        if Path == '':
            Path = self.OneDResultPath

        data = pd.read_csv( Path + str(SubID) +'.txt',
                               header =None,delimiter = r'\s+')

        data.columns=["day" , "hour", "xs", "q", "h", "wl"]
        days = list(set(data['day']))
        days.sort()

        if FromDay != '':
            assert  FromDay in days, "please use the GetDays method to select FromDay that exist in the data"
        if ToDay != '':
            assert ToDay in days, "please use the GetDays method to select FromDay that exist in the data"

        if FromDay != '':
            data = data.loc[data['day'] >= FromDay,:]

        if ToDay != '':
            data = data.loc[data['day'] <= ToDay]

        data.index = list(range(0,len(data)))

        # Cross section data
        XSname = data['xs'][data['day'] == data['day'][1]][data['hour'] == 1].tolist()

        if FillMissing == True:
            # check if there is missing days (Q was < threshold so the model didn't run)
            # fill these values with 0
            days = list(set(data['day']))
            days.sort()
            #hours = list(range(0,25))
            hours = list(range(1,25))

            missing_days = list()
            for i in range(days[0],days[-1]):
                if i not in days:
                    missing_days.append(i)
                    print("day = " + str(i) + " is missing")


            if len(missing_days) > 0:
                if len(missing_days) > 10000:
                    missing_1 = list()
                    missing_2 = list()
                    missing_3 = list()
                    for i in missing_days:
                        for j in hours:
                            for h in XSname:
                                missing_1.append(i)
                                missing_2.append(j)
                                missing_3.append(h)
                else :
                    missing = [(i,j,h) for i in missing_days for j in hours for h in XSname]
                    missing_1 = [i[0] for i in missing]
                    missing_2 = [i[1] for i in missing]
                    missing_3 = [i[2] for i in missing]

                print("done")
                missing = pd.DataFrame(index=range(len(missing_1)) ,dtype = np.float64)

            #    i=0
            #    h=100000
            #    if len(missing_1) > h:
            #        while (i+1)*h-1 < len(missing_1):
            #            missing.loc[i*h:(i+1)*h-1,'day'] = missing_1[i*h:(i+1)*h]
            #            missing.loc[i*h:(i+1)*h-1,'hour'] = missing_2[i*h:(i+1)*h]
            #            missing.loc[i*h:(i+1)*h-1,'xs']  = missing_3[i*h:(i+1)*h]
            #            print(i)
            #            i=i+1
            #
            #        missing.loc[i*h:len(missing_1),'day'] = missing_1[i*h:len(missing_1)]
            #        missing.loc[i*h:len(missing_1),'hour'] = missing_2[i*h:len(missing_1)]
            #        missing.loc[i*h:len(missing_1),'xs'] = missing_3[i*h:len(missing_1)]
            #    else:
                missing['day'] = missing_1
                missing['hour'] = missing_2
                missing['xs'] = missing_3

                missing['q'] = 0
                missing['h'] = 0
                missing['wl'] = 0
            #    missing['area'] = 0
            #    missing['perimeter'] = 0
                data  = data.append(missing)

                del missing, missing_1, missing_2, missing_3, missing_days
                data = data.sort_values(by=['day','hour','xs'], ascending=True)
                data.index = list(range(len(data)))

        self.Result1D = data


    @staticmethod
    def Collect1DResults(Path, FolderNames, Left, Right, SavePath, OneD,
                         fromf='', tof='', FilterbyName = False):
        """
        ======================================================================
            Collect1DResults(Path, FolderNames, Left, Right, SavePath, OneD,
                                fromf='', tof='', FilterbyName = False)
        ======================================================================
        Collect1DResults method reads the 1D separated result files and filter
        then between two number to remove any warmup period if exist then stack
        the result in one table then write it.

        Parameters
        ----------
            1-Path : [String]
                path to the folder containing the separated folder.
            2-FolderNames : [List]
                list containing folder names.
            3-Left : [Bool]
                True if you want to combine left overtopping files.
            4-Right : [Bool]
                True if you want to combine right overtopping files.
            5-SavePath : [String]
                path to the folder where data will be saved.
            6-OneD : [Bool]
                True if you want to combine 1D result files.
            7-fromf : [Integer], optional
                if the files are very big and the cache memory has a problem
                reading all the files you can specify here the order of the file
                the code will start from to combine. The default is ''.
            8-tof : [Integer], optional
                if the files are very big and the cache memory has a problem
                reading all the files you can specify here the order of the file
                the code will end to combine. The default is ''.
            9-FilterbyName : [Bool], optional
                if the results include a wanm up period at the beginning
                or has results for some days at the end you want to filter out
                you want to include the period you want to be combined only
                in the name of the folder between () and separated with -
                ex 1d(5000-80000). The default is False.

        Returns
        -------
            combined files will be written to the SavePath .

        """

        second = "=pd.DataFrame()"
        if fromf == '':
            fromf = 0

        for i in range(len(FolderNames)):
            print(str(i) + "-" + FolderNames[i])

            if tof == '':
                tof = len(os.listdir(Path +"/" + FolderNames[i]))

            FileList = os.listdir(Path +"/" + FolderNames[i])[fromf:tof]
            # tof is only renewed if it is equal to ''
            tof = ''
            if FilterbyName == True:
                filter1 = int(FolderNames[i].split('(')[1].split('-')[0])
                filter2 = int(FolderNames[i].split('(')[1].split('-')[1].split(')')[0])

            for j in range(len(FileList)):

                go = False

                if Left and FileList[j].split('.')[0].endswith("_left"):
                    print(str(i) + "-" + str(j) +"-" + FileList[j])
                    # create data frame for the sub-basin
                    first = "L" + FileList[j].split('.')[0]
                    go = True

                elif Right and FileList[j].split('.')[0].endswith("_right"):
                    print(str(i) + "-" + str(j) +"-" + FileList[j])
                    first = "R" + FileList[j].split('.')[0]
                    go = True

                ## try to get the integer of the file name to make sure that it is
                ## one of the 1D results file
                elif OneD and not FileList[j].split('.')[0].endswith("_right") and not FileList[j].split('.')[0].endswith("_left"):
                    print(str(i) + "-" + str(j) +"-" + FileList[j])
                    # create data frame for the sub-basin
                    first = "one" + FileList[j].split('.')[0]
                    go = True

                if go == True:
                    # get the updated list of variable names
                    variables = locals()

                    # read the file
                    try:
                        temp_df = pd.read_csv(Path + "/" + FolderNames[i] + "/" + FileList[j],header = None,
                                              delimiter = r'\s+')

                        if FilterbyName == True:
                            temp_df = temp_df[temp_df[0] >= filter1]
                            temp_df = temp_df[temp_df[0] <= filter2]
                        # check whether the variable exist or not
                        # if this is the first time this file exist
                        if not first in variables.keys():
                            # create a datafame with the name of the sub-basin
                            total = first+ second
                            exec(total)

                        # concatenate the
                        exec(first +  "= pd.concat([" + first+ ", temp_df])")
                    except:
                        continue

        # Save files
        variables = list(locals().keys())
        # get sub-basins variables (starts with "One")
        for i in range(len(variables)):
            var = variables[i]
            if var.endswith("_left"):
                # put the dataframe in order first
                exec(var + ".sort_values(by=[0,1,2],ascending = True, inplace = True)")
                path = SavePath + '/' + var[1:]+ '.txt'
                exec(var + ".to_csv(path ,index= None, sep = ' ', header = None)")
            elif var.endswith("_right"):
                # put the dataframe in order first
                exec(var + ".sort_values(by=[0,1,2],ascending = True, inplace = True)")
                path = SavePath + '/' + var[1:]+ '.txt'
                exec(var + ".to_csv(path ,index= None, sep = ' ', header = None)")
            elif var.startswith("one"):
                # put the dataframe in order first
                exec(var + ".sort_values(by=[0,1,2],ascending = True, inplace = True)")
                print("Saving " + var[3:]+ '.txt')
                path = SavePath + '/' + var[3:]+ '.txt'
                exec(var + ".to_csv(path ,index= None, sep = ' ', header = None)")

    @staticmethod
    def ReadRRMResults(Version, RRMReferenceIndex, Path, NodeID, FromDay, ToDay,
                       date_format="%d_%m_%Y"):
        """
        ===================================================================
             ReadRRMResults(Path, NodeID, FromDay, ToDay)
        ===================================================================
        ReadRRMResults is a static method to read the results of the rainfall-runoff
        model

        Parameters
        ----------
        Path : [String]
            Path to the result files.
        NodeID : [Integer]
            the ID given the the sub-basin .
        FromDay : [integer], optional
                the day you want to read the result from, the first day is 1 not zero.The default is ''.
        ToDay : [integer], optional
                the day you want to read the result to.

        Returns
        -------
        Q : [Dataframe]
            time series of the runoff .

        """

        if Version < 3:
            Q = pd.read_csv(Path + "/" + str(NodeID) + '.txt',header = None)
            Q = Q.rename(columns = {0:NodeID})
            Q.index = list(range(1,len(Q)+1))

            if FromDay == '':
                FromDay = 1
            if ToDay == '':
                ToDay = len(Q)

            Q = Q.loc[Q.index >= FromDay,:]
            Q = Q.loc[Q.index <= ToDay]
        else:
            Q = pd.read_csv(Path + "/" + str(NodeID) + '.txt',header = None, skiprows=1)
            Q.index = [dt.datetime.strptime(date,date_format) for date in Q[0]]
            del Q[0]
            
            
            # convert the date into integer index
            s = np.where(RRMReferenceIndex['date'] == Q.index[0])[0][0]+1
            e = np.where(RRMReferenceIndex['date'] == Q.index[-1])[0][0]+1
            Q.index = list(range(s,e+1))

            if FromDay == '':
                FromDay = s
            if ToDay == '':
                ToDay = e

            Q = Q.loc[Q.index >= FromDay,:]
            Q = Q.loc[Q.index <= ToDay, :]
            
            Q = Q[1].to_frame()
            Q = Q.rename(columns = {1:NodeID})
        return Q


    @staticmethod
    def CorrectMaps(DEMPath,Filelist, Resultpath, MapsPrefix, FilterValue, Saveto):
        """
        =========================================================================
             CorrectMaps(DEMPath,Filelist, Resultpath, MapsPrefix, Saveto)
        =========================================================================
        CorrectMaps method check every 2D result that starts with the given Mapsprefix
        and replace the Nan value with zeros and the values higher than 99 with zeros

        Parameters
        ----------
        DEMPath : [String]
            Path to the DEM ascii file including the name and extension
            (i.e., c\files\RhineDEM.asc) .
        Filelist : [String]
            - if you have a list of files to correct enter the Filelist as the path to the file
            containing the names
            ex,
                Filelist = "F:/RFM/RIM_all/RIM1.0/M35(done)/errorlist.txt"

            - if you want to check all the files in the resultpath enter the
            Filelist as '0'
            ex,
                Filelist = '0'
        Resultpath : [String]
            Path where the Maps exist.
        MapsPrefix : [String]
            the name prefix that distinguish the maps you want to correct from
            other maps in the same folder, like the first part of the name you
            use to name all files.
        Saveto : [String]
            Path to where you will save the corrected files.

        Returns
        -------
        Errors : [list]
            list of the files' names that has errors and are already corrected.
        """


        DEM, SpatialRef = raster.ReadASCII(DEMPath)
        NoDataValue = SpatialRef[-1]


        # filter and get the required maps
        if Filelist == '0' :
            # read list of file names
            AllResults = os.listdir(Resultpath)

            MapsNameList = list()
            for i in range(len(AllResults)):
                if AllResults[i].startswith(MapsPrefix):
                    MapsNameList.append(AllResults[i])
        elif type(Filelist) == str:
            MapsNameList = pd.read_csv(Filelist, header = None)[0].tolist()

        Errors = list()

        for k in range(len(MapsNameList)):
            try:
                # open the zip file
                Compressedfile = zipfile.ZipFile( Resultpath + "/" + MapsNameList[k])
            except:
                print("Error Opening the compressed file")
                Errors.append(MapsNameList[k][len(MapsPrefix):-4])
                continue

            # get the file name
            fname = Compressedfile.infolist()[0]
            # get the time step from the file name
            timestep = int(fname.filename[len(MapsPrefix):-4])
            print("File No = " + str(k))

            ASCIIF = Compressedfile.open(fname)
            SpatialRef = ASCIIF.readlines()[:6]
            ASCIIF = Compressedfile.open(fname)
            ASCIIRaw = ASCIIF.readlines()[6:]
            rows = len(ASCIIRaw)
            cols = len(ASCIIRaw[0].split())
            MapArray = np.ones((rows,cols), dtype = np.float32)*0
            # read the ascii file
            for i in range(rows):
                x = ASCIIRaw[i].split()
                MapArray[i,:] = list(map(float, x ))

            Save = 0
            # Clip all maps
            if MapArray[DEM == NoDataValue].max() > 0:
                MapArray[DEM == NoDataValue] = 0
                Save = 1
            # replace nan values with zero
            if len(MapArray[np.isnan(MapArray)]) > 0:
                MapArray[np.isnan(MapArray)] = 0
                Save = 1
            # replace FilterValue in the result raster with 0
            if len(MapArray[MapArray >= FilterValue]) > 0 :
                MapArray[MapArray >= FilterValue] = 0
                Save = 1

            if Save == 1:
                print("File= " + str(timestep))
                # write the new file
                fname = MapsPrefix + str(timestep) + ".asc"
                newfile = Saveto + "/" + fname

                with open(newfile,'w') as File:
                    # write the first lines
                    for i in range(len(SpatialRef)):
                        File.write(str(SpatialRef[i].decode()[:-2]) + "\n")


                    for i in range(rows):
                        File.writelines(list(map(raster.StringSpace,MapArray[i,:])))
                        File.write("\n")

                #zip the file
                with zipfile.ZipFile(Saveto + "/" + fname[:-4] + ".zip","w",zipfile.ZIP_DEFLATED) as newzip:
                    newzip.write( Saveto + "/" + fname, arcname = fname)
                # delete the file
                os.remove(Saveto + "/"  + fname)

        return Errors




    # @staticmethod
    def Histogram(self, v1, v2, NoAxis=2, filter1=0.2, Save = False, pdf=True, **kwargs):
        """
        ===========================================
              Histogram(v1, v2)
        ===========================================
        Histogram method plots the histogram of two given list of values

        Parameters
        ----------
            1-v1 : [List]
                first list of values.
            2-v2 : [List]
                second list of values.

        Returns
        -------
            - histogram plot.

        """
        # update the default options
        Fkeys = list(kwargs.keys())
        for key in Fkeys:
            if key in self.FigureOptions.keys():
                self.FigureOptions[key] = kwargs[key]



        v1 = np.array([j for j in v1 if j > filter1])
        v2 = np.array([j for j in v2 if j > filter1])

        if pdf :
            param_dist1 = gumbel_r.fit(np.array(v1))
            param_dist2 = gumbel_r.fit(np.array(v2))

            d1 = np.linspace(v1.min(), v1.max(), v1.size)
            d2 = np.linspace(v2.min(), v2.max(), v2.size)
            pdf_fitted1 = gumbel_r.pdf(d1, loc=param_dist1[0], scale=param_dist1[1])
            pdf_fitted2 = gumbel_r.pdf(d2, loc=param_dist2[0], scale=param_dist2[1])
        #
        # color1 = '#3D59AB'
        # color2 = "#DC143C"

        if NoAxis == 1:
            # if bins in kwargs.keys():

            plt.figure(60,figsize=(10,8))
            n, bins, patches  = plt.hist([v1,v2], color=['#3D59AB','#DC143C' ])
            # plt.xlabel("Depth Values (m)")
            # plt.ylabel("Frequency")

            # for key in kwargs.keys():
            #     if key == 'legend':
            #         plt.legend(kwargs['legend'])
            #     if key == 'legend size':
            #         plt.legend(kwargs['legend'],fontsize = int(kwargs['legend_size']))
            #     if key == 'xlabel':
            #         plt.xlabel(kwargs['xlabel'])
            #     if key == 'ylabel':
            #         plt.ylabel(kwargs['ylabel'])
            # #     # if key == 'xlabel':
            # #         # xlabel = kwargs['xlabel']
            # #     # if key == 'xlabel':
            # #         # xlabel = kwargs['xlabel']

        elif NoAxis == 2:
            fig, ax1 = plt.subplots(figsize = self.FigureOptions['figsize'])
            # n1= ax1.hist([v1,v2], bins=15, alpha = 0.7, color=[color1,color2])
            n1= ax1.hist([v1,v2], bins=15, alpha = 0.7, color = [self.FigureOptions['color1'],
                                                                 self.FigureOptions['color2']])
                         # label=['RIM1.0','RIM2.0']) #width = 0.2,

            ax1.set_ylabel("Frequency", fontsize = self.FigureOptions['fontsize'])
            # ax1.yaxis.label.set_color(color1)
            ax1.set_xlabel("Inundation Depth Ranges (m)", fontsize = self.FigureOptions['fontsize'])

            # ax1.tick_params(axis='y', color = color1)
            # ax1.spines['right'].set_color(color1)
            if pdf:
                ax2 = ax1.twinx()
                ax2.plot(d1, pdf_fitted1, '-.', color = self.FigureOptions['color1'],
                         linewidth = self.FigureOptions['linewidth'], label ="RIM1.0 pdf")
                ax2.plot(d2, pdf_fitted2, '-.', color = self.FigureOptions['color2'],
                         linewidth = self.FigureOptions['linewidth'], label ="RIM2.0 pdf")
                ax2.set_ylabel("Probability density function (pdf)", fontsize = self.FigureOptions['labelsize'])
            # else:
            #     ax2.yaxis.set_ticklabels([])
            #     # ax2.yaxis.set_major_formatter(plt.NullFormatter())
            #     # ax2.tick_params(right='off', labelright='off')
            #     ax2.set_xticks([])
            #     ax2.tick_params(axis='y', color = color2)

            #     # if key == 'xlabel':
            #         # xlabel = kwargs['xlabel']
            #     # if key == 'xlabel':
            #         # xlabel = kwargs['xlabel']

            # n2 = ax2.hist(v2,  bins=n1[1], alpha = 0.4, color=color2)#width=0.2,
            # ax2.set_ylabel("Frequency", fontsize = 15)
            # ax2.yaxis.label.set_color(color2)

            # ax2.tick_params(axis='y', color = color2)
            # plt.title("Sub-Basin = " + str(SubID), fontsize = 15)

            # minall = min(min(n1[1]), min(n2[1]))
            # if minall < 0:
            #     minall =0

            # maxall = max(max(n1[1]), max(n2[1]))
            # ax1.set_xlim(minall, maxall)
            #    ax1.set_yticklabels(ax1.get_yticklabels(), color = color1)
            #    ax2.set_yticklabels(ax2.get_yticklabels(), color = color2)

            # # options
            # for key in kwargs.keys():
                # if key == 'legend':
                    # ax1.legend(self.FigureOptions['legend'])
                # if key == 'legend_size':
            ax1.legend(kwargs['legend'],fontsize = int(self.FigureOptions['legend_size']))
                # if key == 'xlabel':
                    # ax1.set_xlabel(self.FigureOptions['xlabel'])
                # if key == 'ylabel':
                    # ax1.set_ylabel(self.FigureOptions['ylabel'])
                # if key == 'labelsize':
            ax1.set_xlabel(self.FigureOptions['xlabel'], fontsize = self.FigureOptions['labelsize'])
            ax1.set_ylabel(self.FigureOptions['ylabel'], fontsize = self.FigureOptions['labelsize'])
                # if key == 'fontsize':
            plt.rcParams.update({'font.size': int(self.FigureOptions['Axisfontsize'])})

            # fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes,fontsize = 15)

            plt.tight_layout()

        if Save == True:
            plt.savefig(self.FigureOptions['name'] +"-hist.tif", transparent=True)
            # plt.close()

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

class Sub(River):

    def __init__(self,ID, River, RunModel=False):
        self.ID = ID
        self.RIM = River.name
        self.Version = River.Version
        self.RightOvertopping_Suffix = River.RightOvertopping_Suffix
        self.leftOvertopping_Suffix = River.leftOvertopping_Suffix
        self.DepthPrefix = River.DepthPrefix
        self.DurationPrefix = River.DurationPrefix
        self.ReturnPeriodPrefix = River.ReturnPeriodPrefix
        self.Compressed = River.Compressed
        self.TwoDResultPath = River.TwoDResultPath

        if hasattr(River, 'CustomizedRunsPath'):
                self.CustomizedRunsPath = River.CustomizedRunsPath
        if hasattr(River,'USbndPath'):
            self.USbndPath = River.USbndPath
        if hasattr(River,"OneMinResultPath"):
            self.OneMinResultPath = River.OneMinResultPath
        if hasattr(River, "USbndPath"):
            self.USbndPath = River.USbndPath
        # filter the whole cross section file and get the cross section of the segment
        if self.Version == 1 or self.Version == 2:
            self.crosssections = River.crosssections[River.crosssections['swimid'] == ID]
        else:
            self.crosssections = River.crosssections[River.crosssections['id'] == ID]
            if RunModel:
                self.xsid = Sub.crosssections.loc[:,'xsid'].values
                self.Dbf = Sub.crosssections.loc[:,'dbf'].values
                self.bedlevel = Sub.crosssections.loc[:,'gl'].values
                self.hl = Sub.crosssections.loc[:,'hl'].values
                self.cl = Sub.crosssections.loc[:,'bl'].values
                self.zl = Sub.crosssections.loc[:,'zl'].values
                self.hr = Sub.crosssections.loc[:,'hr'].values
                self.cr = Sub.crosssections.loc[:,'br'].values
                self.zr = Sub.crosssections.loc[:,'zr'].values
                self.mw = Sub.crosssections.loc[:,'b'].values
                self.mn = Sub.crosssections.loc[:,'m'].values

        self.crosssections.index = list(range(len(self.crosssections)))
        self.LastXS = self.crosssections.loc[len(self.crosssections)-1,'xsid']
        self.FirstXS = self.crosssections.loc[0,'xsid']
        self.XSname = self.crosssections['xsid'].tolist()
        self.XSno = len(self.XSname)

        self.ReferenceIndex = River.ReferenceIndex
        self.RRMReferenceIndex = River.RRMReferenceIndex

        self.OneDResultPath = River.OneDResultPath

        if hasattr(River, 'slope'):
            if self.Version == 1 or self.Version == 2 :
                self.slope = River.slope[River.slope['id']==ID]['slope'].tolist()[0]
            else:
                self.slope = River.slope[River.slope['id']==ID]['slope'].tolist()[0]

        if hasattr(River, "rivernetwork"):
            self.USnode, self.DSnode = River.TraceSegment(ID)

        if hasattr(River, 'RP'):
            self.RP = River.RP.loc[River.RP['node'] == self.USnode,['HQ2','HQ10','HQ100']]
        if hasattr(River,"SP"):
            self.SP = River.SP.loc[River.SP['id'] == self.ID,:]
            self.SP.index = list(range(len(self.SP)))

        if hasattr(River,"RRMPath"):
            self.RRMPath = River.RRMPath
        # create dictionary to store any extracted values from maps
        self.ExtractedValues = dict()

        self.FigureOptions = River.FigureOptions

    def Read1DResult(self, FromDay = '', ToDay = '', FillMissing = True,
                     addHQ2 = False, Path = '', XSID = ''):
        """
        ===================================================================
           Read1DResult(FromDay = '', ToDay = '', FillMissing = False)
        ===================================================================
        Read1DResult method reads the 1D (1D-2D coupled) result of the sub-basin the object is
        created for and return the hydrograph of the first and last cross section
        the method will not read the 1D result file again if you tried to read results
        of the same sub-basin again, so you have to re-instantiate the object

        Parameters
        ----------
            1-FromDay : [integer], optional
                the index of the day you want the data to start from. The default is empty.
                means read everything
            2-ToDay : [integer], optional
                the index of the day you want the data to end to. The default is empty.
                means read everything
            3-FillMissing : [Bool], optional
                Fill the missing days. The default is False.
            4-addHQ2 : [Bool], optional
                to add the value of HQ2. The default is False.
            5-Path : [String], optional
                Path to read the results from. The default is ''.
            6-XSID : [Integer], optional
                ID of a specific cross section you want to get the results on it. The default is ''.
        Returns
        -------
            1-Result1D : [attribute]
                the results read will be stored (as it is without any filter)
                in the attribute "Result1D"
            2-XSHydrographs : [dataframe attribute]
                dataframe containing hydrographs at the position of the first and last cross section
            3-XSWaterLevel : [dataframe attribute]
                dataframe containing waterlevels at the position of the first and last cross section
            4-FirstDayResults:[attribute]
                the first day in the 1D result
            5-LastDay:[attribute]
                the last day in the 1D result
        """
        # if the results are not read yet read it
        if not hasattr(self, "Result1D") :
            River.Read1DResult(self,self.ID, FromDay, ToDay, Path = Path, FillMissing = FillMissing)

        if FromDay == '':
            FromDay = self.Result1D.loc[0,'day']
        if ToDay ==  '':
            ToDay = self.Result1D.loc[len(self.Result1D)-1,'day']

        start = self.IndexToDate(FromDay)
        end = self.IndexToDate(ToDay+1)

        if not hasattr(self, "XSHydrographs") :
            self.XSHydrographs = pd.DataFrame(index = pd.date_range(start,end,freq = 'H')[:-1])
            self.XSWaterLevel = pd.DataFrame(index = pd.date_range(start,end,freq = 'H')[:-1])
            self.XSWaterDepth = pd.DataFrame(index = pd.date_range(start,end,freq = 'H')[:-1])

        #check if the XSID is in the sub-basin
        if XSID != '':
            XSsub = list(set(self.Result1D['xs']))
            assert  XSID in XSsub, "The given cross-section " + str(XSID) + " does not exist inside the current Segment of the river, first XS is " + str(self.FirstXS) + ", and last XS is " + str(self.LastXS)

        # get the simulated hydrograph and add the cutted HQ2
        if addHQ2:
            self.XSHydrographs[self.LastXS] = self.Result1D['q'][self.Result1D['xs'] == self.LastXS ].values + self.RP['HQ2'].tolist()[0]
            self.XSHydrographs[self.FirstXS] = self.Result1D['q'][self.Result1D['xs'] == self.FirstXS ].values + self.RP['HQ2'].tolist()[0]

            if XSID != '':
                self.XSHydrographs[XSID] = self.Result1D['q'][self.Result1D['xs'] == XSID ].values + self.RP['HQ2'].tolist()[0]
        else:
            self.XSHydrographs[self.LastXS] = self.Result1D['q'][self.Result1D['xs'] == self.LastXS ].values
            self.XSHydrographs[self.FirstXS] = self.Result1D['q'][self.Result1D['xs'] == self.FirstXS ].values
            if XSID != '':
                self.XSHydrographs[XSID] = self.Result1D['q'][self.Result1D['xs'] == XSID ].values

        self.XSWaterLevel[self.LastXS]  = self.Result1D['wl'][self.Result1D['xs'] == self.LastXS ].values
        self.XSWaterLevel[self.FirstXS] = self.Result1D['wl'][self.Result1D['xs'] == self.FirstXS ].values
        
        self.XSWaterDepth[self.LastXS]  = self.Result1D['h'][self.Result1D['xs'] == self.LastXS ].values
        self.XSWaterDepth[self.FirstXS] = self.Result1D['h'][self.Result1D['xs'] == self.FirstXS ].values

        if XSID != '':
            self.XSWaterLevel[XSID]  = self.Result1D['wl'][self.Result1D['xs'] == XSID ].values
            self.XSWaterDepth[XSID]  = self.Result1D['h'][self.Result1D['xs'] == XSID ].values


        # check the first day in the results and get the date of the first day and last day
        ## create time series

        self.from_beginning  = self.Result1D['day'][0]

        # else:
        self.FirstDay =  self.ReferenceIndex.loc[self.from_beginning,'date']
        # if there are empty days at the beginning the filling missing days is not going to detect it
        # so ignore it here by starting from the first day in the data (data['day'][0]) dataframe
        # empty days at the beginning

        self.FirstDayResults = self.ReferenceIndex.loc[self.Result1D['day'][0],'date']
        self.LastDay = self.ReferenceIndex.loc[self.Result1D['day'][self.Result1D.index[-1]],'date']

        # last days+1 as range does not include the last element
        self.Daylist = list(range(self.Result1D['day'][0], self.Result1D['day'][self.Result1D.index[-1]]+1))
        self.ReferenceIndex_Results = pd.date_range(self.FirstDayResults, self.LastDay,freq = "D")


    def ExtractXS(self, XSID, addHQ2=False, WaterLevel=True):
        """
        ========================================================
            ExtractXS(XSID, addHQ2).
        ========================================================

        ExtractXS method extracts the hydrodraph and water levels of a specific
        XSID from the already been read 1D results

        Parameters
        ----------
            1-XSID : [Integer], optional
                ID of a specific cross section you want to get the results on it.
            2-addHQ2 : [Bool], optional
                to add the value of HQ2. The default is False.

        Returns
        -------
        None.
        """
        assert hasattr(self,"Result1D"), "please use the Read1DResult method to read the results first"
        # assert hasattr(self,"RP"), "please use the Read1DResult method to read the results first"
        if addHQ2:
            self.XSHydrographs[XSID] = self.Result1D['q'][self.Result1D['xs'] == XSID ].values + self.RP['HQ2'].tolist()[0]
        else:
            self.XSHydrographs[XSID] = self.Result1D['q'][self.Result1D['xs'] == XSID ].values

        if WaterLevel:
            self.XSWaterLevel[XSID]  = self.Result1D['wl'][self.Result1D['xs'] == XSID ].values


    def CheckNegativeQ(self, plot = False, TS = 'hourly'):
        """
        ==================================================
            CheckNegativeQ
        ==================================================
        CheckNegativeQ check whether there are any negative discharge values
        in the 'q' column in the 1D results or not, you need to read the result first


        Returns
        -------
            1-Negative.[attribute]
                dictionary with ['NegQ', 'NegXS', 'NegQind'] as keys
        """
        if TS == 'hourly':
            assert hasattr(self, "Result1D") , "please use the Result1D method to read the result of this sub-basin first"

            if self.Result1D['q'].min() < 0:
                print("NegativeDischarge")
                # extract -ve discharge data if exist
                self.Negative = dict()
                self.Negative['NegQ'] =  self.Result1D[self.Result1D['q'] < 0]
                self.Negative['NegXS'] = list(set(self.Negative['NegQ']['xs']))
                self.Negative['NegQind'] = self.Negative['NegQ'].index.tolist()

                self.Negative['QN'] = pd.DataFrame()
                for i in range(len(self.Negative['NegXS'])):
                    self.Negative['QN'][self.Negative['NegXS'][i]] = self.Result1D['q'][self.Result1D['xs'] == self.Negative['NegXS'][i] ]

                self.Negative['QN'].index = self.XSHydrographs.index

                if plot :
                    plt.figure(30, figsize = (15,8))
                    for i in range(len(self.Negative['NegXS'])):
                        plt.plot(self.Negative['QN'][self.Negative['NegXS'][i]])

                    plt.title("Discharge ", fontsize = 25)
                    plt.legend(self.Negative['NegXS'],fontsize = 15)
                    plt.xlabel("Time", fontsize = 15)
                    plt.ylabel("Discharge m3/s", fontsize = 15)

            else:
                print("There is no -ve Discharge")

        elif TS == "1min":
            assert hasattr(self, "Qmin") , "please use the Result1D method to read the result of this sub-basin first"
            NegQmin = self.Qmin[:]
            # NegQmin = pd.DataFrame()
            NegQmin.loc[:,'date'] = self.Qmin.index
            NegQmin.index=range(len(NegQmin.index))
            f = NegQmin[NegQmin[self.XSname[0]] < 0]

            for i in range(len(self.XSname[1:])):
                f = f.append( NegQmin[NegQmin[self.XSname[i+1]] < 0])

            self.NegQmin = f


    def ReadBoundaryConditions(self, FromDay = '', ToDay = '', Path = '', fmt="%Y-%m-%d"):
        """
        =============================================================================
            ReadBoundaryConditions(FromDay = '', ToDay = '', Path = '')
        =============================================================================

        ReadBoundaryConditions method reads the BC files and since these files are separated each day is
        written in a file so the code is reading a lot of files, therefor you can specify
        a specific day to to start read the BC H & Q from that day till the end of
        the simulated period

        Parameters
        ----------
        1-FromDay : [integer], optional
                the day you want to read the result from, the first day is 1 not zero.The default is ''.
        2-ToDay : [integer], optional
                the day you want to read the result to.
        4-Path : [String], optional
            Path to read the results from. The default is ''.

        Returns
        -------
            1-QBC: [dataframe attribute]
                dataframe contains the Discharge boundary conditions for each
                day as a row and for each column are the hours
            1-HBC: [dataframe attribute]
                dataframe contains the water depth boundary conditions for each
                day as a row and for each column are the hours
        """
        if FromDay == '':
            FromDay = 1
        if ToDay ==  '':
            ToDay = len(self.ReferenceIndex_Results) - 1
        
        if type(FromDay) == str:
            FromDay = dt.datetime.strptime(FromDay,fmt)
            FromDay = np.where(self.ReferenceIndex_Results == FromDay)[0][0] + 1
        if type(ToDay) == str:
            ToDay = dt.datetime.strptime(ToDay,fmt)
            ToDay = np.where(self.ReferenceIndex_Results == ToDay)[0][0] + 1
            
        if Path != '':
            self.USbndPath = Path
        
        QBC = pd.DataFrame(index = self.ReferenceIndex_Results[FromDay-1:ToDay] ,columns = hours)
        HBC = pd.DataFrame(index = self.ReferenceIndex_Results[FromDay-1:ToDay] ,columns = hours)

        for i in self.Daylist[FromDay-1:ToDay]:
            bc_q = np.loadtxt(self.USbndPath +str(self.ID) + "-" + str(i) +'.txt',dtype = np.float16)
            QBC.loc[self.ReferenceIndex.loc[i,'date'],:]= bc_q[:,0].tolist()[0:bc_q.shape[0]:60]
            HBC.loc[self.ReferenceIndex.loc[i,'date'],:]= bc_q[:,1].tolist()[0:bc_q.shape[0]:60]

        self.QBC = QBC
        self.HBC = HBC



    def Read1DResults1Min(self, Smin, Emin,fmt="%Y-%m-%d"):
        
        Smin = dt.datetime.strptime(Smin,fmt)
        Emin = dt.datetime.strptime(Emin,fmt)        
        indmin = pd.date_range(Smin,Emin,freq = "min")[:-1]
        
        XSname = [int(i) for i in self.XSname]
        Hmin = pd.DataFrame(index = indmin,columns = XSname)
        Qmin = pd.DataFrame(index = indmin,columns = XSname)

        # US boundary condition
        index_daily = pd.date_range(Smin,Emin,freq = "D")[:-1]
        BC_q = pd.DataFrame(index = index_daily ,columns = list(range(1,1441)))
        BC_h = pd.DataFrame(index = index_daily ,columns = list(range(1,1441)))
        
        ii = self.DateToIndex(Smin)
        # ii = Sub.ReferenceIndex.index[np.where(Sub.ReferenceIndex == Smin)[0][0]]  # old (Smin-s).days+1
        # ii2 = Sub.ReferenceIndex.index[np.where(self.ReferenceIndex == Emin)[0][0]]  # old (Emin-s).days+1
        ii2 = self.DateToIndex(Emin)
        
        list2 = list(range(ii,ii2))


        for i in list2:
            h = np.transpose(np.loadtxt(self.OneMinResultPath + "h/" +str(self.ID)+"-h-"+str(i) +'.txt',dtype = np.float16))[:,:-1]
            q = np.transpose(np.loadtxt(self.OneMinResultPath + "q/" +str(self.ID)+"-q-"+str(i) +'.txt'))[:,:-1]
            h = h + self.crosssections['gl'].tolist()
            # old is i-(Smin-s).days-1 new is i-list2[0]
            Hmin.loc[Hmin.index[(i-list2[0])*1440]:Hmin.index[(i-list2[0])*1440+1439]]= h
            Qmin.loc[Hmin.index[(i-list2[0])*1440]:Hmin.index[(i-list2[0])*1440+1439]]= q
            # BC
            bc_q = np.loadtxt(self.USbndPath +str(self.ID)+"-"+str(i) +'.txt',dtype = np.float16)
            BC_q.loc[BC_q.index[i-list2[0]]]= bc_q[:,0]
            BC_h.loc[BC_h.index[i-list2[0]]]= bc_q[:,1]

        self.Hmin = Hmin
        self.Qmin = Qmin
        self.QBCmin = BC_q
        self.HBCmin = BC_h


    def ReadRRMHydrograph(self, NodeID, FromDay = '', ToDay = '',Path = '',
                          date_format="%d_%m_%Y"):
        """
        =================================================================
            ReadHydrographs(NodeID, FromDay = [], ToDay = [])
        =================================================================
        ReadHydrographs method reads the results of the Rainfall-runoff model
        (SWIM) for the given node id for a specific period.

        Parameters
        ----------
            1-NodeID : [Integer]
                DESCRIPTION.
            2-FromDay : [Integer], optional
                start day of the period you wanrt to read its results.
                The default is [].
            3-ToDay : [Integer], optional
                end day of the period you wanrt to read its results.
                The default is [].

        Returns
        -------
            1-RRM:[data frame attribute]
                containing the computational node and rainfall-runoff results
                (hydrograph)with columns ['ID', NodeID ]

        """
        if not hasattr(self,"RRM"):
            self.RRM = pd.DataFrame()

        if Path == '':
            Path = self.RRMPath

        self.RRM[NodeID] = self.ReadRRMResults(self.Version, self.RRMReferenceIndex,
                                               Path, NodeID, FromDay, ToDay,
                                               date_format)[NodeID].tolist()

        if FromDay == '':
            FromDay = 1
        if ToDay == '':
            ToDay = len(self.RRM[NodeID])

        start = self.RRMReferenceIndex.loc[FromDay,'date']
        end = self.RRMReferenceIndex.loc[ToDay,'date']


        self.RRM.index = pd.date_range(start, end, freq = 'D')
        # get the simulated hydrograph and add the cutted HQ2


    def Resample(self, XSID, ColumnName, FromDay='', ToDay = '', Delete=False):
        """
        =============================================================================
            Resample(self, XSID, ColumnName, FromDay='', ToDay = '', Delete=False)
        =============================================================================
        Resample method extract the value at the last hour of the dat [hour == 24]
        from the 1D Result  file, for the discharge, water level, and water depth.

        Parameters
        ----------
        XSID : [Integer]
            cross-section id.
        ColumnName : [string]
            the column name you want to resample in the results1D. ColumnName
            could be 'q' for discharge, 'wl' for water level, and 'h' for water
            depth.
        FromDay : [integer], optional
            starting day. The default is ''.
        ToDay : [integer], optional
            end day. The default is ''.
        Delete : [boolen], optional
            to delete the previously resampled data frame to create another one.
            The default is False.

        Returns
        -------
        ResampledQ, ResampledWL, ResampledH: [dataframe attribute]
            depends on the given ColumnName the attribute will be created,
            if 'q' the attribute will be ResampledQ, and the same for "wl", and "H"
            and inside the ResampledQ a column will be created with the given XSID
            containing the resampled valeus
        """
        assert hasattr(self,"Result1D") , "please read the 1D results"

        if FromDay == '':
            FromDay = self.Result1D.loc[0,'day']
        if ToDay ==  '':
            ToDay = self.Result1D.loc[len(self.Result1D)-1,'day']

        # start = self.IndexToDate(FromDay)
        # end = self.IndexToDate(ToDay)

        # start = self.ReferenceIndex.loc[FromDay,'date']
        # end = self.ReferenceIndex.loc[ToDay,'date']

        ind = pd.date_range(self.IndexToDate(FromDay), self.IndexToDate(ToDay), freq = 'D')

        if ColumnName == 'q' and not hasattr(self,"ResampledQ"):
            self.ResampledQ = pd.DataFrame(index=ind)
        elif ColumnName == 'q':
            if Delete==True:
                del self.ResampledQ

        if ColumnName == 'wl' and not hasattr(self,"ResampledWL"):
            self.ResampledWL = pd.DataFrame(index=ind)
        elif ColumnName == 'wl':
            if Delete==True:
                del self.ResampledWL

        if ColumnName == 'h' and not hasattr(self,"ResampledH"):
            self.ResampledH = pd.DataFrame(index=ind)
        elif ColumnName == 'h':
            if Delete==True:
                del self.ResampledH

        Q = self.Result1D[self.Result1D['xs'] == XSID][self.Result1D['hour'] == 24]
        Q = Q[ColumnName][self.Result1D['day'] >= FromDay][self.Result1D['day'] <= ToDay]

        # self.Q = Q
        if ColumnName == 'q':
            self.ResampledQ.loc[:,XSID] = Q.tolist()
        elif ColumnName == "wl":
            self.ResampledWL.loc[:,XSID] = Q.tolist()
        elif ColumnName == "h":
            self.ResampledH.loc[:,XSID] = Q.tolist()


    def DetailedStatisticalCalculation(self, T):
        """
        ===============================================================
             DetailedStatisticalCalculation(T)
        ===============================================================
        DetailedStatisticalCalculation method calculates the discharge related
        to a specific given return period

        Parameters
        ----------
        T : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        assert hasattr(self,"SP"), "you "
        F = 1-(1/T)
        self.Qrp = pd.DataFrame()
        self.Qrp['RP'] = T
        self.Qrp['Q'] = gumbel_r.ppf(F,loc=self.SP.loc[0,"loc"], scale=self.SP.loc[0,"scale"])



    def DetailedOvertopping(self, eventdays):
        """
        =========================================================
                DetailedOvertopping(eventdays)
        =========================================================

        DetailedOvertopping method takes list of days and get the left and right
        overtopping for the sub-basin each day

        Parameters
        ----------
            1-eventdays : [list]
                list od daysof an event.

        Returns
        -------
            1- DetailedOvertoppingLeft:[data frame attribute]
                containing the computational node and rainfall-runoff results
                (hydrograph)with columns ['ID', NodeID ]
            2-DetailedOvertoppingRight:[data frame attribute]
                containing the computational node and rainfall-runoff results
                (hydrograph)with columns ['ID', NodeID ]
            3-AllOvertoppingVSXS:
            4-AllOvertoppingVSTime:

        """

        # River.DetailedOvertopping(self, [self.ID], eventdays)
        XSs = self.crosssections.loc[:,'xsid'].tolist()
        columns = [self.ID] + XSs + ['sum']
        self.DetailedOvertoppingLeft  = pd.DataFrame(index = eventdays + ['sum'], columns = columns)
        self.DetailedOvertoppingLeft.loc[:,columns] = 0
        self.DetailedOvertoppingRight = pd.DataFrame(index = eventdays + ['sum'], columns = columns)
        self.DetailedOvertoppingRight.loc[:,columns] = 0
        # Left Bank
        try :
            # try to open and read the overtopping file
            data = pd.read_csv(self.OneDResultPath + str(self.ID) + self.leftOvertopping_Suffix,
                               header =None,delimiter = r'\s+')

            data.columns = ['day','hour','xsid','q','wl']
            # get the days in the sub
            days = list(set(data.loc[:,'day']))

            for j in range(len(eventdays)):
                # check whether this sub basin has flooded in this particular day
                if eventdays[j] in days:
                    # filter the dataframe to the discharge column (3) and the days
                    self.DetailedOvertoppingLeft.loc[eventdays[j],self.ID] = data.loc[data['day'] == eventdays[j] ,'q'].sum()
                    # get the xss that was overtopped in that particular day
                    XSday = list(set(data.loc[data['day'] == eventdays[j] ,'xsid'].tolist()))

                    for i in range(len(XSday)):
                        # dataXS = data['q'].loc[data['day'] == eventdays[j]][data['xsid'] == XSday[i]].sum()
                        self.DetailedOvertoppingLeft.loc[eventdays[j], XSday[i]] = data['q'].loc[data['day'] == eventdays[j]][data['xsid'] == XSday[i]].sum()
                else:
                    self.DetailedOvertoppingLeft.loc[eventdays[j],self.ID] = 0
        except:
            self.DetailedOvertoppingLeft.loc[:,self.ID] = 0

        # right Bank
        try :
            # try to open and read the overtopping file
            data = pd.read_csv(self.OneDResultPath + str(self.ID) + self.RightOvertopping_Suffix,
                               header =None,delimiter = r'\s+')
            data.columns = ['day','hour','xsid','q','wl']
            # get the days in the sub
            days = list(set(data.loc[:,'day']))

            for j in range(len(eventdays)):
                # check whether this sub basin has flooded in this particular day
                if eventdays[j] in days:
                    # filter the dataframe to the discharge column (3) and the days
                    self.DetailedOvertoppingRight.loc[eventdays[j], self.ID] = data.loc[data['day'] == eventdays[j] ,'q'].sum()
                    # get the xss that was overtopped in that particular day
                    XSday = list(set(data.loc[data['day'] == eventdays[j] ,'xsid'].tolist()))

                    for i in range(len(XSday)):
                        # dataXS = data['q'].loc[data['day'] == eventdays[j]][data['xsid'] == XSday[i]].sum()
                        self.DetailedOvertoppingRight.loc[eventdays[j], XSday[i]] = data['q'].loc[data['day'] == eventdays[j]][data['xsid'] == XSday[i]].sum()

                else:
                    self.DetailedOvertoppingRight.loc[eventdays[j], self.ID] = 0
        except:
            # print("file did not open")
            self.DetailedOvertoppingRight.loc[:,self.ID] = 0

        # sum overtopping for each day
        for j in range(len(eventdays)):
            self.DetailedOvertoppingLeft.loc[eventdays[j],'sum'] = self.DetailedOvertoppingLeft.loc[eventdays[j],XSs].sum()
            self.DetailedOvertoppingRight.loc[eventdays[j],'sum'] = self.DetailedOvertoppingRight.loc[eventdays[j],XSs].sum()
        # sum overtopping for each sub basin
        for j in range(len(XSs)):
            self.DetailedOvertoppingLeft.loc['sum',XSs[j]] = self.DetailedOvertoppingLeft.loc[:,XSs[j]].sum()
            self.DetailedOvertoppingRight.loc['sum',XSs[j]] = self.DetailedOvertoppingRight.loc[:,XSs[j]].sum()

        self.DetailedOvertoppingLeft.loc['sum', self.ID] = self.DetailedOvertoppingLeft.loc[:, self.ID].sum()
        self.DetailedOvertoppingRight.loc['sum', self.ID] = self.DetailedOvertoppingRight.loc[:, self.ID].sum()

        self.AllOvertoppingVSXS = self.DetailedOvertoppingLeft.loc['sum', XSs] + self.DetailedOvertoppingRight.loc['sum', XSs]

        self.AllOvertoppingVSTime = pd.DataFrame()
        self.AllOvertoppingVSTime['id']  = eventdays
        self.AllOvertoppingVSTime.loc[:,'Overtopping'] = (self.DetailedOvertoppingLeft.loc[eventdays, 'sum'] + self.DetailedOvertoppingRight.loc[eventdays, 'sum']).tolist()
        self.AllOvertoppingVSTime.loc[:,'date'] = (self.ReferenceIndex.loc[eventdays[0]:eventdays[-1],'date']).tolist()
    # def Read1DResult1Donly(self,Path):
        # River.Read1DResult(self,self.ID, FromDay, ToDay, FillMissing)


    def SaveHydrograph(self, xsid, Path='', Option=1):
        """
        ==============================================================
            SaveHydrograph(xsid, Path='')
        ==============================================================
        SaveHydrograph method saves the hydrograph of any cross-section in
        the segment.
        Mainly the method is created to to be used to save the last cross-section
        hydrograph to use it as as a boundary condition for the downstream segment


        Parameters
        ----------
        xsid : [integer]
            the id of the cross section.
        Path : [string], optional
            Path to the directory where you want to save the file to. The default is ''.
        Option : [integer]
            1 to write water level results, 2 to write water depth results
        Returns
        -------
        None.

        """

        if Path == '' :
            assert hasattr(self, 'CustomizedRunsPath'), "please enter the value of the CustomizedRunsPath or use the Path argument to specify where to save the file"
            Path = self.CustomizedRunsPath

        saveDS = self.XSHydrographs[xsid].resample('D').mean()
        f = pd.DataFrame(index = saveDS.index)
        f['date'] = ["'" + str(i)[:10] + "'" for i in saveDS.index]
        f['discharge(m3/s)'] = saveDS
        if Option == 1 :
            f['water depth(m)'] = self.XSWaterDepth[xsid].resample('D').mean().values
        else:
            f['water level(m)'] = self.XSWaterLevel[xsid].resample('D').mean().values
            
        f.to_csv(Path+str(self.ID)+".txt", index = False, float_format="%.3f")  #header = None ,


    def PlotHydrographProgression(self, XSs, plotstart, plotend, linewidth = 4,
                                  Spacing=5, FigSize=(7,5)):
        """
        ============================================================================
             PlotHydrographProgression(XSs, plotstart, plotend, linewidth = 4, Spacing=5)
        =============================================================================

        Parameters
        ----------
        XSs : TYPE
            DESCRIPTION.
        plotstart : TYPE
            DESCRIPTION.
        plotend : TYPE
            DESCRIPTION.
        linewidth : [integer], optional
            width of the plots. The default is 4.
        Spacing : [integer]
            hydrographs are going to be plots every spacing. The default is 5.
        Returns
        -------
        None.

        """
        plotstart = dt.datetime.strptime(plotstart,"%Y-%m-%d")
        plotend = dt.datetime.strptime(plotend,"%Y-%m-%d")

        XSlist = self.XSname[Spacing:self.XSno:Spacing]

        XSlist = XSlist + XSs
        # to remove repeated XSs
        XSlist = list(set(XSlist))
        # extract the XS hydrographs
        for i in range(len(XSlist)):
            self.Read1DResult(XSID=XSlist[i])

        XSlist = [self.FirstXS] + XSlist + [self.LastXS]
        XSlist.sort()


        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))

        for i in range(len(XSlist)):
            ax.plot(self.XSHydrographs.loc[plotstart:plotend,XSlist[i]], label = "XS-" + str(XSlist[i]),
                        linewidth=linewidth),#color = XScolor,zorder=XSorder

        ax.legend(fontsize = 10, loc='best')
        ax.set_xlabel("Time", fontsize = 10)
        ax.set_ylabel("Discharge m3/s", fontsize = 10)
        plt.tight_layout()

        return fig, ax

    def ReadUSHydrograph(self, FromDay = '', ToDay = '', Path = '',
                          date_format="'%Y-%m-%d'"):
        """
        =======================================================================
            ReadUSHydrograph(FromDay = '', ToDay = '', Path = '',
                                date_format="'%Y-%m-%d'")
        =======================================================================

        Parameters
        ----------
        1-FromDay : [integer], optional
                the day you want to read the result from, the first day is 1 not zero.The default is ''.
        2-ToDay : [integer], optional
                the day you want to read the result to.
        3-Path : [String], optional
            Path to read the results from. The default is ''.
        4-date_format : "TYPE, optional
            DESCRIPTION. The default is "'%Y-%m-%d'".

        Returns
        -------
        1-USHydrographs : [dataframe attribute].
            dataframe contains the hydrograph of each of the upstream segments
            with segment id as a column name and a column 'total' contains the
            sum of all the hydrographs.
        """
        self.USHydrographs = pd.DataFrame()

        if Path == '':
            Path = self.CustomizedRunsPath

        if len(self.USnode) > 1 :
            # there is more than one upstream segment
            if type(self.USnode) == list:
                for i in range(len(self.USnode)):
                    NodeID = self.USnode[i]
                    self.USHydrographs[NodeID]  = self.ReadRRMResults(self.Version, self.RRMReferenceIndex,
                                                                    Path, NodeID, FromDay, ToDay,
                                                                    date_format)[NodeID]
                    
            #there is one upstream segment
        elif self.USnode != []:
            NodeID = self.USnode[0]
            self.USHydrographs[NodeID] = self.ReadRRMResults(self.Version, self.RRMReferenceIndex,
                                                                   Path, NodeID, FromDay, ToDay,
                                                                    date_format)[NodeID]
        else:
            print("the Segment Does not have any Upstream Segments")
            return
        
        self.USHydrographs['total'] = self.USHydrographs.sum(axis=1)
        if FromDay == '':
            # FromDay = 1
            FromDay = self.USHydrographs.index[0]
        if ToDay == '':
            # ToDay = len(self.USHydrographs[NodeID])
            ToDay = self.USHydrographs.index[-1]

        start = self.ReferenceIndex.loc[FromDay,'date']
        end = self.ReferenceIndex.loc[ToDay,'date']

        self.USHydrographs.index = pd.date_range(start, end, freq = 'D')



    def GetUSHydrograph(self, River):
        """
        ==================================================================
            GetUSHydrograph(River)
        ==================================================================
        GetUSHydrograph methods gets the sum of all the upstream hydrographs
        whither it is routed inside the model or a boundary condition.

        Parameters
        ----------
        River : [object]
            the object of the river.

        Returns
        -------
        USHydrographs : [array].
            array of the hydrograph
        """

        self.USHydrographs = np.zeros(shape=(River.NoTimeSteps))

        if len(self.USnode) > 1 :
            # there is more than one upstream segment
            if type(self.USnode) == list:
                for i in range(len(self.USnode)):
                    NodeID = self.USnode[i]
                    # get the order of the segment
                    River.Segments.index(NodeID)
                    self.USHydrographs = self.USHydrographs + River.RoutedQ[:,River.Segments.index(NodeID)]
            #there is one upstream segment
        elif self.USnode != []:
            NodeID = self.USnode[0]
            River.Segments.index(NodeID)
            self.USHydrographs = self.USHydrographs + River.RoutedQ[:,River.Segments.index(NodeID)]

        if type(self.BC) != bool:
            self.USHydrographs = self.USHydrographs + self.BC.values.reshape(len(self.USHydrographs))


    def GetXSGeometry(self):
        AreaPerLow = np.zeros(shape=(self.XSno,2))
        AreaPerHigh = np.zeros(shape=(self.XSno,2))
        for i in range(self.XSno):
            geom = self.crosssections.loc[i,:]
            H = min(geom['hl'], geom['hr']) + geom['dbf']
            Coords = self.GetVortices(H, geom['hl'], geom['hr'], geom['bl'], geom['br'], geom['b'], geom['dbf'])
            AreaPerLow[i,:] = self.PolygonGeometry(Coords)
            H = max(geom['hl'], geom['hr']) + geom['dbf']
            Coords = self.GetVortices(H, geom['hl'], geom['hr'], geom['bl'], geom['br'], geom['b'], geom['dbf'])
            AreaPerHigh[i,:] = self.PolygonGeometry(Coords)
        self.AreaPerHigh = AreaPerHigh[:,:]
        self.AreaPerLow = AreaPerLow[:,:]



    def GetFlow(self, IF, ID, FromDay='', ToDay='', date_format="%d_%m_%Y"):
        """
        =============================================================================
             GetFlow(IF, ID, FromDay='', ToDay='', date_format="%d_%m_%Y")
        =============================================================================

        Parameters
        ----------
        IF : [Interface object]
            You Have to create the interface object then read the laterals and the
            boundary conditions first.
        ID : [Integer]
            the id of the segment you want to extract the laterals for.
        FromDay : [string], optional
            the starting day. The default is ''.
        ToDay : [string], optional
            the ending day. The default is ''.
        date_format : [string], optional
            the format of the given dates. The default is "%d_%m_%Y".

        Returns
        -------
        BC : [dataframe attribute].
            dataframe containing the boundary condition under a column by the name
            of the segment id.
        Laterals : [dataframe attribute].
            dataframe containing a column for each cross section that has a lateral.
        """
        hasattr(IF, "BC"), "the boundary condition does not exist you have to read it first"
        hasattr(IF, "Laterals"), "the Laterals does not exist you have to read it first"

        if FromDay == '':
            FromDay = IF.BC.index[0]
        else:
            FromDay = dt.datetime.strptime(FromDay,date_format)

        if ToDay == '':
            ToDay = IF.BC.index[-1]
        else:
            ToDay = dt.datetime.strptime(ToDay,date_format)

        if ID in IF.BCTable['id'].tolist():
            self.BC = IF.BC.loc[FromDay:ToDay,ID].to_frame()
        else:
            self.BC = False

        self.LateralsTable = [value for value in self.XSname if value in IF.LateralsTable['xsid'].tolist()]

        if len(self.LateralsTable) > 0:
            self.Laterals = pd.DataFrame(index = pd.date_range(FromDay, ToDay, freq='D'),
                                         columns=self.LateralsTable)

            for i in self.LateralsTable:
                self.Laterals.loc[:,i] = IF.Laterals.loc[FromDay:ToDay,i]

            self.Laterals['total'] = self.Laterals.sum(axis=1)


    def GetLaterals(self, xsid):
        """
        =================================================================
               GetLaterals(xsid)
        =================================================================
        GetLaterals method gets the sum of the laterals of all the cross sections
        in the segment upstream of a given xsid

        Parameters
        ----------
        xsid : [integer]
            id of the cross section.

        Returns
        -------
        dataframe
            sum of the laterals of all the cross sections in the segment
            upstream of a given xsid.

        """
        USgauge = self.LateralsTable[:bisect(self.LateralsTable, xsid)]
        return self.Laterals[USgauge].sum(axis=1).to_frame()


    def H2Q(self, Q):

        H = np.zeros(shape=(len(Q)))

        for i in range(len(Q)):
        	# if Qbnd >  calculated Q for the highest depth in the table
        	# highest depth = (dike height + 15 m)
        	if Q[i] > self.HQ[-1,1] :
        		# depth = highest depth
        		H[i] = self.HQ[-1,0]
        	# if not calculate the Q for each discretized depth in the table
        	# and take the one with the smallest difference from the q given
        	else :
        		qq = Q[i]
        		diff = abs(qq - self.HQ[:,1])
        		H[i] = self.HQ[np.argmin(diff),0]

        return H

    def PlotQ(self, Calib, GagueXS,plotstart,plotend,stationname, Gaugename, Segment_XS,
             PlotLaterals=True, plotUS=True, SpecificXS=False, plotRRM=True,
             PlotGauge=True, RIMcolor="#004c99", gaugecolor="#DC143C",BCcolor = "grey",
             RRMcolor = "green",Latcolor = (0.3,0,0),XScolor = "grey",linewidth = 4,
             RIMorder = 6, Gaugeorder = 5, RRMorder = 4, BCorder = 7, USHorder = 2,
             XSorder = 1, fmt="%Y-%m-%d", NXlabels=False):

        plotstart = dt.datetime.strptime(plotstart,fmt)
        plotend = dt.datetime.strptime(plotend,fmt)
        #"#FF34B3"

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))

        if hasattr(self,"XSHydrographs"):
            # plot if you read the results using ther read1DResults
            ax.plot(self.XSHydrographs.loc[plotstart:plotend,GagueXS], label = "RIM",zorder=RIMorder,
                     linewidth=linewidth, linestyle = Vis.LineStyle(6), color = RIMcolor)

            if PlotLaterals:
                Laterals = self.GetLaterals(GagueXS)
                if type(self.BC) != bool:
                    ax.plot(self.BC.loc[plotstart:plotend,self.ID], label = "BC",zorder=BCorder,
                            linewidth=linewidth, linestyle = Vis.LineStyle(9), color = BCcolor)
                if len(self.LateralsTable) > 0:
                    ax.plot(Laterals.loc[plotstart:plotend,0], label = "Laterals",zorder=RRMorder,
                            linewidth=linewidth, linestyle = Vis.LineStyle(9), color = Latcolor)

            if self.USnode != [] and plotUS :
                ax.plot(self.USHydrographs.loc[plotstart:plotend,'total'], label = "US Hydrograph",zorder=USHorder,
                        linewidth=linewidth, linestyle = Vis.LineStyle(9), color = RRMcolor)

            if PlotGauge:
                # plot the gauge data
                ax.plot(Calib.QGauges.loc[plotstart:plotend,stationname],label = 'Gauge',
                        linewidth=linewidth, zorder=Gaugeorder, color = gaugecolor,
                        linestyle = Vis.LineStyle(9))

            # plot for specific XS
            if type(SpecificXS) != bool :
                self.Read1DResult(XSID=SpecificXS)
                ax.plot(self.XSHydrographs.loc[plotstart:plotend,SpecificXS], label = "XS-" + str(SpecificXS),
                        zorder=XSorder, linewidth=linewidth, color = XScolor,
                        linestyle = Vis.LineStyle(3))
            if plotRRM:
               ax.plot(self.RRM.loc[plotstart:plotend,stationname], label="mHM",
                        zorder=RRMorder, linewidth=linewidth,
                        linestyle = Vis.LineStyle(6), color = "green")

        elif hasattr(Calib,"CalibrationQ"):
            # plot if you read the data using ReadCalirationResult
            ax.plot(Calib.CalibrationQ[Segment_XS], label = "RIM",zorder=3, linewidth=linewidth,
                      linestyle = Vis.LineStyle(6), color = RIMcolor)
            # plot the gauge data
            ax.plot(Calib.QGauges.loc[Calib.CalibrationQ.index[0]:Calib.CalibrationQ.index[-1],stationname],
                    label = 'Gauge-' + str(self.ID), linewidth=linewidth, color = gaugecolor)
            if plotRRM:
                ax.plot(Sub.RRM.loc[Calib.CalibrationQ.index[0]:Calib.CalibrationQ.index[-1],stationname],
                        label="RRM")

        if type(NXlabels) != bool :
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.linspace(start,end,NXlabels))
        # xlabels = ["1990-01-01","1990-11-13","1991-11-15","1992-11-16"]
        # ax.set_xticklabels(xlabels)

        ax.set_title("Discharge - " + Gaugename, fontsize = 20)

        ax.legend(fontsize = 12)
        ax.set_xlabel("Time", fontsize = 12)
        ax.set_ylabel("Discharge m3/s", fontsize = 12)
        plt.tight_layout()

        return fig, ax 


    def CalculateQMetrics(self, Calib, stationname, startError, endError, GagueXS,
                          GaugeStart, GaugeEnd,Filter=False, fmt="%Y-%m-%d"):

        QRIM = pd.DataFrame()
        startError = dt.datetime.strptime(startError,fmt)
        endError = dt.datetime.strptime(endError,fmt)
        
        # GaugeStart = Calib.GaugesTable[Calib.GaugesTable['xsid'] == GagueXS]['Qstart'].values[0]
        # GaugeEnd = Calib.GaugesTable[Calib.GaugesTable['xsid'] == GagueXS]['Qend'].values[0]
        
        if Filter :
            # get the latest date of the filter date and the first date in the result
            # get the earliest date of the end and the last date in the result
            st2 = max(GaugeStart,startError, self.FirstDayResults)
            end2 = min(GaugeEnd,endError, self.EndDays)

            # starti = River.DateToIndex(st2)
            # endi = River.DateToIndex(end2)

            # get the observed discharge
            Qobs = Calib.QGauges.loc[st2:end2,stationname]

            # resample the times series to average daily
            ind = pd.date_range(self.FirstDayResults, self.EndDays + dt.timedelta(days=1), freq = 'h')[:-1]
            Q = self.Result1D[self.Result1D['xs'] == self.LastXS]#['q']#[Sub.Result1D['hour'] == 24]
            Q.index = ind
            QRIM['q'] = Q['q'].resample('D').mean()
            QRIM['q'] = QRIM.loc[st2:end2,'q']

            # try:
            #     Sub.Resample(GagueXS, 'q', starti, endi, Delete=True)
            # except:
            #     Sub.Resample(GagueXS, 'q', starti, endi, Delete=False)
            # QRIM['q']  = Sub.ResampledQ[GagueXS][:]
            # QRIM.index = pd.date_range(st2, end2)

        else:
            st2 = max(GaugeStart,self.FirstDayResults)
            end2 = min(GaugeEnd,self.LastDay)
            # get the observed discharge
            Qobs = Calib.QGauges.loc[st2:end2,stationname]

            # resample the times series to average daily
            ind = pd.date_range(self.FirstDayResults, self.LastDay + dt.timedelta(days=1), freq = 'h')[:-1]
            Q = self.Result1D[self.Result1D['xs'] == self.LastXS]#['q']#[Sub.Result1D['hour'] == 24]
            Q.index = ind
            QRIM['q'] = Q['q'].resample('D').mean()
            QRIM['q'] = QRIM.loc[st2:end2,'q']

            # old
            # QRIM['q'] = Sub.Result1D['q'][Sub.Result1D['xs'] == GagueXS][Sub.Result1D['hour'] == 24][:]
            # QRIM.index = pd.date_range(st2, end2)

        rmse = round(Pf.RMSE(Qobs,QRIM.loc[st2:end2,'q'].tolist()),0)
        kge = round(Pf.KGE(Qobs,QRIM.loc[st2:end2,'q'].tolist()),2)
        wb = round(Pf.WB(Qobs,QRIM.loc[st2:end2,'q'].tolist()),0)
        nsehf = round(Pf.NSEHF(Qobs,QRIM.loc[st2:end2,'q'].tolist()),2)
        nse = round(Pf.NSE(Qobs,QRIM.loc[st2:end2,'q'].tolist()),2)
        print("--------------------")
        print("RMSE = " + str(rmse))
        print("KGE = " + str(kge))
        print("WB = " + str(wb))
        print("NSEHF = " + str(nsehf))
        print("NSE = " + str(nse))

        return rmse, kge, wb, nsehf, nse
    
    
    def PlotWL(self, Calib, plotstart, plotend, GagueXS, stationname, Gaugename,
               Filter = False, gaugecolor = "#DC143C", RIMcolor = "#004c99",
               linewidth=2, RIMorder=1, Gaugeorder=0, PlotGauge=True, fmt="%Y-%m-%d"):
        
        plotstart = dt.datetime.strptime(plotstart,fmt)
        plotend = dt.datetime.strptime(plotend,fmt)
        
        if PlotGauge:
            
            GaugeStart = Calib.GaugesTable[Calib.GaugesTable['xsid'] == GagueXS]['WLstart']
            GaugeEnd = Calib.GaugesTable[Calib.GaugesTable['xsid'] == GagueXS]['WLend']
            
            if len(GaugeStart.values) != 0:
                print("The XS you provided does not exist in the GaugesTable")    
                GaugeStart = GaugeStart.values[0]
                GaugeEnd = GaugeEnd.values[0]
        
                if GaugeStart > plotstart and  GaugeStart > plotend:
                    print("Availabel data for the gauge starts from " + str(GaugeStart))
                    print("The period you provided is between " + str(plotstart) + " and " + str(plotend)) 
                    PlotGauge = False
                elif GaugeEnd < plotstart and  GaugeStart < plotend:
                    print("Availabel data for the gauge starts from" + str(GaugeEnd))
                    print("Out of Gauge dates") 
                    PlotGauge = False   
            
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))

        # if self.ID == SubID :
        """
        these lines were made to only plot the gauge if existed inside the processed sub-basin
        """
        # extract the water levels at the gauge cross section
        self.ExtractXS(GagueXS)

        ax.plot(self.XSWaterLevel.loc[plotstart:plotend,GagueXS],label="RIM" ,
                 zorder=RIMorder, linewidth=linewidth, color = RIMcolor, linestyle = Vis.LineStyle(6))
        if PlotGauge:
            ax.plot(Calib.WLGauges.loc[plotstart:plotend,stationname],label='Gauge',
                     zorder = Gaugeorder, linewidth=linewidth, color = gaugecolor)

        # ax1.plot(Sub2.XSWaterLevel.loc[plotstart:plotend,GagueXS],label="step-1" + str(GagueXS),
        #          zorder=3, linewidth=linewidth, color = 'grey')

        # else:
        #     """
        #     these lines are made to plot the closest downstreeam gauge
        #     """
        #     ax.plot(Sub.XSWaterLevel.loc[plotstart:plotend,Sub.LastXS], label = "RIM-XS " + str(Sub.LastXS),
        #              zorder=3, linewidth = linewidth, color = RIMcolor)
        #     ax.plot(Calib.WLGauges.loc[plotstart:plotend,stationname], label = 'Gauge' +str(stationname),
        #              linewidth = linewidth)

        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.linspace(start,end,4))
        # xlabels = ["1990-01-01","1990-11-13","1991-11-15","1992-11-16"]
        # ax1.set_xticklabels(xlabels)

        ax.set_title("Water Level - " + Gaugename, fontsize = 20)
        plt.legend(fontsize = 15)
        ax.set_xlabel("Time", fontsize = 15)
        ax.set_ylabel("Water Level m", fontsize = 15)
        plt.tight_layout()

        return ax

    def CalculateWLMetrics(self, Calib, stationname, startError,  
                           endError, GagueXS, Filter=False): #GaugeStart, GaugeEnd, 

        startError = dt.datetime.strptime(startError,"%Y-%m-%d")
        endError = dt.datetime.strptime(endError,"%Y-%m-%d")
        
        GaugeStart = Calib.GaugesTable[Calib.GaugesTable['xsid'] == GagueXS]['WLstart'].values[0]
        GaugeEnd = Calib.GaugesTable[Calib.GaugesTable['xsid'] == GagueXS]['WLend'].values[0]
        
        # if GaugeStart > plotstart and  GaugeStart > plotend:
        #    print("Out of Gauge dates") 
        #    PlotGauge = False
        # elif GaugeEnd < plotstart and  GaugeStart < plotend:
        #    print("Out of Gauge dates") 
        #    PlotGauge = False
           
        if Filter:
            st2 = max(GaugeStart,startError, self.FirstDayResults)
            end2 = min(GaugeEnd,endError,self.EndDays)
            # observed
            obs = np.array(Calib.WLGauges.loc[st2:end2,stationname])

            # RIM
            ind = pd.date_range(Sub.FirstDayResults, Sub.EndDays+dt.timedelta(days=1), freq = 'h')[:-1]
            mod = Sub.Result1D[Sub.Result1D['xs'] == Sub.LastXS]#['q']#[Sub.Result1D['hour'] == 24]
            mod.index = ind
            mod = mod['wl'].resample('D').mean()
            mod = mod.loc[st2:end2]


            # RIM
            # try:
            #     Sub.Resample(GagueXS, 'wl', River.DateToIndex(st2), River.DateToIndex(end2), Delete = True)
            # except:
            #     Sub.Resample(GagueXS, 'wl', River.DateToIndex(st2), River.DateToIndex(end2), Delete = False)
            # series1 = np.array(Sub.ResampledWL[GagueXS])

        else:
            st2 = max(GaugeStart,self.FirstDayResults)
            end2 = min(GaugeEnd,self.LastDay)
            # Observed
            obs = np.array(Calib.WLGauges.loc[st2:end2,stationname])


            # RIM
            ind = pd.date_range(self.FirstDayResults, self.LastDay + dt.timedelta(days=1), freq = 'h')[:-1]
            mod = self.Result1D[self.Result1D['xs'] == GagueXS]#['q']#[Sub.Result1D['hour'] == 24]
            mod.index = ind
            mod = mod['wl'].resample('D').mean()
            mod = mod.loc[st2:end2]

            # RIM
            # Sub.Resample(GagueXS, 'wl', River.DateToIndex(st2), River.DateToIndex(end2), Delete = False)
            # series1 = np.array(Sub.ResampledWL[GagueXS])
        
        if len(obs) != len(mod) or len(mod) ==0:
            print("Availabel data for the gauge starts from " + str(GaugeStart)+ " To " + str(GaugeEnd))
            return 

        MBE = round(Pf.MBE(obs, mod),2)
        MAE = round(Pf.MAE(obs, mod),2)
        RMSE = round(Pf.RMSE(obs, mod),2)
        KGE = round(Pf.KGE(obs, mod),2)
        NSEHF = round(Pf.NSEHF(obs, mod),2)
        NSE = round(Pf.NSE(obs, mod),2)


        print("RMSE= " + str(RMSE))
        print("KGE= " + str(KGE))
        print("NSEHF= " + str(NSEHF))
        print("NSE= " + str(NSE))
        print("MBE= " + str(MBE))
        print("MAE= " + str(MAE))

        return MBE, MAE, RMSE, KGE, NSEHF, NSE

    def Histogram(self, Day, BaseMapF, ExcludeValue, OccupiedCellsOnly, Map = 1,
                  filter1 = 0.2, filter2 = 15):

        # check if the object has the attribute ExtractedValues
        if hasattr(self,'ExtractedValues'):
            # depth map
            if Map == 1:
                Path = self.TwoDResultPath + self.DepthPrefix + str(Day) + ".zip"
            elif Map == 2:
                Path = self.TwoDResultPath + self.DurationPrefix + str(Day) + ".zip"
            else:
                Path = self.TwoDResultPath + self.ReturnPeriodPrefix + str(Day) + ".zip"


            ExtractedValues, NonZeroCells = raster.OverlayMap(Path, BaseMapF,
                                                ExcludeValue, self.Compressed,
                                                OccupiedCellsOnly)

            self.ExtractedValues = ExtractedValues[self.ID]
        # filter values
        ExtractedValues = [j for j in ExtractedValues if j > filter1]
        ExtractedValues = [j for j in ExtractedValues if j < filter2]
        #plot
        fig, ax1 = plt.subplots(figsize=(10,8))
        ax1.hist(self.ExtractedValues, bins=15, alpha = 0.4) #width = 0.2,
        ax1.set_ylabel("Frequency RIM1.0", fontsize = 15)
        ax1.yaxis.label.set_color('#27408B')
        ax1.set_xlabel("Depth Ranges (m)", fontsize = 15)
        #ax1.cla

        ax1.tick_params(axis='y', color = '#27408B')
        
        
    def PlotBC(self, date, fmt="%Y-%m-%d"):
        date = dt.datetime.strptime(date,fmt)
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.HBCmin.loc[date])
        ax1.set_xlabel('Date', fontsize = 15)
        ax1.set_ylabel('H', fontsize = 15)
        ax1.set_xlim(0,1440)
        ax2.plot(self.QBCmin.loc[date])
        ax2.set_ylabel('Q', fontsize = 15)
    
    # def CalculateHydraulics():
        # Sub.C

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
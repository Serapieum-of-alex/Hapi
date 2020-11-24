# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:27:32 2020

@author: mofarrag
"""
import pandas as pd
import datetime as dt
import numpy as np
import datetime as dt

datafn = lambda x: dt.datetime.strptime(x,"%Y-%m-%d")

class RIMCalibration():

    def __init__(self, name, Version, start, days):
        self.name = name
        self.Version = Version
        
        self.start = dt.datetime.strptime(start,"%Y-%m-%d")
        self.end = self.start + dt.timedelta(days = days)
        self.days = days
        
        Ref_ind = pd.date_range(self.start, self.end, freq='D')
        self.ReferenceIndex = pd.DataFrame(index = list(range(1,days+1)))
        self.ReferenceIndex['date'] = Ref_ind[:-1]
        

    def IndexToDate(self,Index):
        # convert the index into date
        return self.ReferenceIndex.loc[Index,'date']

    def DateToIndex(self,Date):
        # convert the index into date+
        # Date = dt.datetime(1950,1,1)
        if type(Date) == str:
            Date = dt.datetime.strptime(Date,"%Y-%m-%d")
        return np.where(self.ReferenceIndex['date'] == Date)[0][0]+1
    
    def ReadGaugesTable(self,Path):
        self.GaugesTable = pd.read_csv(Path)
        
        
    def ReadObservedWL(self, Path, StartDate, EndDate, NoValue, # GaugesTable,
                       column='oid'):
        """
        ============================================================================
            ReadObservedWL( WLGauges, Path, StartDate, EndDate, NoValue)
        ============================================================================

        Parameters
        ----------
            1-GaugesTable : [dataframe]
                Dataframe contains columns [Segment,xsid,datum(m)].
            2-Path : [String]
                    path to the folder containing the text files of the water level gauges
            3-StartDate : [datetime object]
                the starting date of the water level time series.
            4-EndDate : [datetime object]
                the end date of the water level time series.
            5-NoValue : [integer]
                value used to fill the missing values.

        Returns
        -------
            1-WLGauges: [dataframe attiribute].
                dataframe containing the data of the water level gauges and 
                the index as the time series from the StartDate till the EndDate
                and the gaps filled with the NoValue 
            2-WLGaugesTable:[dataframe attiribute].
                the input WLGaugesTable dataframe with the index replaced to 
                be the segment ID
        """

        ind = pd.date_range(StartDate, EndDate)
        columns = self.GaugesTable['oid'].tolist()
        
        WLGauges = pd.DataFrame(index = ind)
        # WLGaugesf.loc[:,:] = NoValue
        WLGauges.loc[:,0] = ind
        
        for i in range(len(columns)):
            if self.GaugesTable.loc[i,'waterlevel'] == 1:
                name = self.GaugesTable.loc[i,column]
                f = pd.read_csv(Path + str(name) + ".txt",
                               delimiter = ",", header = None)
                f[0] = f[0].map(datafn)
                # sort by date as some values are missed up
                f.sort_values(by = [0], ascending = True, inplace = True)
                # filter to the range we want
                f = f.loc[f[0] >= ind[0],:]
                f = f.loc[f[0] <= ind[-1],:]
                # reindex
                f.index = list(range(len(f)))
                # add datum and convert to meter
                f.loc[f[1]!= NoValue,1] = (f.loc[f[1]!= NoValue,1] / 100) + self.GaugesTable.loc[i,'datum(m)']
                f = f.rename(columns={1:columns[i]})
        
                # assign the values in the dateframe
                # WLGauges.loc[:,WLGauges.columns[i]].loc[f[0][0]:f[0][len(f)-1]] = f[1].tolist()
                # use merge as there are some gaps in the middle
                WLGauges = WLGauges.merge(f, on=0, how='left', sort=False)
        
        WLGauges.replace(to_replace = np.nan, value = NoValue, inplace=True)
        WLGauges.index = ind
        del WLGauges[0]
        self.WLGauges = WLGauges
        
        # GaugesTable.index = GaugesTable['segment'].tolist()
        self.GaugesTable['WLstart'] = 0
        self.GaugesTable['WLend'] = 0
        for i in range(len(columns)):
            if self.GaugesTable.loc[i,'waterlevel'] == 1:
                st1 = WLGauges[columns[i]][WLGauges[columns[i]] != NoValue].index[0]
                end1 = WLGauges[columns[i]][WLGauges[columns[i]] != NoValue].index[-1]
                self.GaugesTable.loc[i,'WLstart'] = st1
                self.GaugesTable.loc[i,'WLend'] = end1
        
        # self.GaugesTable = GaugesTable


    def ReadObservedQ(self, Path, StartDate, EndDate, NoValue, column='oid'): #Gauges,
        """
        ========================================================================
            ReadObservedQ(Gauges, Path, StartDate, EndDate, NoValue)
        ========================================================================
        ReadObservedQ method reads discharge data and store it in a dataframe
        attribute "QGauges"

        Parameters
        ----------
        Gauges : [DATAFRAME]
            Dataframe containing sub-id of the gauges under a column with a name 0.
        Path : [String]
            path to the folder where files for the gauges exist.
        StartDate : [datetime object]
            starting date of the time series.
        EndDate : [datetime object]
            ending date of the time series.
        NoValue : [numeric]
            value stored in gaps.

        Returns
        -------
        QGauges:[dataframe attribute]
            dataframe containing the hydrograph of each gauge under a column by
            the name of  gauge.
        QGaugesTable:[dataframe attribute]
            dataframe containing gauge dataframe entered toi the method.
        """

        # ind = pd.date_range(StartDate, EndDate)
        # GRDC = pd.DataFrame(index = ind)
        # # ID = Gauges.columns[0]
        # columns = Gauges['gid'].tolist()
        # for i in range(len(Gauges)):
        #     if Gauges.loc[Gauges['oid']==columns[i],'discharge'] ==1:
        #         name = Gauges.loc[Gauges['oid']==columns[i],column].values[0]
                
        #         GRDC.loc[:,int(Gauges['oid'][i])] = np.loadtxt(Path +
        #                   str(int(Gauges[ID][i])) + '.txt') #,skiprows = 0
        # self.QGauges = GRDC

        # GaugesTable = pd.DataFrame(index = Gauges['segment'])
        # GaugesTable['start'] = 0
        # GaugesTable['end'] = 0

        # for i in range(len(Gauges[ID])):
        #     st1 = GRDC[Gauges['segment'][i]][GRDC[Gauges['segment'][i]] != NoValue].index[0]
        #     end1 = GRDC[Gauges['segment'][i]][GRDC[Gauges['segment'][i]] != NoValue].index[-1]
        #     # GaugesTable.loc[GaugesTable.loc[:,'SubID'] == Gauges[0][i],'start'] = st1
        #     # GaugesTable.loc[GaugesTable.loc[:,'SubID'] == Gauges[0][i],'end'] = end1
        #     GaugesTable.loc[Gauges['segment'][i],'start'] = st1
        #     GaugesTable.loc[Gauges['segment'][i],'end'] = end1
        
        # self.QGaugesTable = GaugesTable
        
        ind = pd.date_range(StartDate, EndDate)
        QGauges = pd.DataFrame(index = ind)
        # ID = Gauges.columns[0]
        # columns = Gauges['gid'].tolist()
        for i in range(len(self.GaugesTable)):
            if self.GaugesTable.loc[i,'discharge'] == 1:
                name = self.GaugesTable.loc[i,column]
                QGauges.loc[:,int(name)] = np.loadtxt(Path +
                          str(int(name)) + '.txt') #,skiprows = 0
                
        self.QGauges = QGauges
        
        # Gauges = pd.DataFrame(index = Gauges['segment'])
        self.GaugesTable['Qstart'] = 0
        self.GaugesTable['Qend'] = 0
        
        for i in range(len(self.GaugesTable)):
            if self.GaugesTable.loc[i,'discharge'] == 1:
                ii = self.GaugesTable.loc[i,column]
                st1 = QGauges[ii][QGauges[ii] != NoValue].index[0]
                end1 = QGauges[ii][QGauges[ii] != NoValue].index[-1]
                self.GaugesTable.loc[i,'Qstart'] = st1
                self.GaugesTable.loc[i,'Qend'] = end1
        
        # self.QGaugesTable = GaugesTable


    def ReadRRM(self, Qgauges, Path, StartDate, EndDate):
        """
        ==============================================================
            ReadRRM(Qgauges, Path, StartDate, EndDate)
        ==============================================================
        ReadRRM method reads the discharge results of the rainfall runoff model
        and store it in a dataframe attribute "QRRM"

        Parameters
        ----------
        Qgauges : [DATAFRAME]
            Dataframe containing sub-id of the gauges under a column with a name 0.
        Path : [String]
            path to the folder where files for the gauges exist.
        StartDate : [datetime object]
            starting date of the time series.
        EndDate : [datetime object]
            ending date of the time series.

        Returns
        -------
        None.

        """

        ind = pd.date_range(StartDate,EndDate)
        QSWIM = pd.DataFrame(index = ind)


        for i in range(len(Qgauges[0])):
            # read SWIM data
            # only at the begining to get the length of the time series
            QSWIM.loc[:,int(Qgauges[0][i])] = np.loadtxt(Path +
                  str(int(Qgauges[0][i])) + '.txt')#,skiprows = 0
        self.QRRM = QSWIM


    def ReadRIMQ(self, Qgauges, Path, StartDate, days, NoValue, AddHQ2=False,
                 Shift=False, ShiftSteps=0):
        """
        ===============================================================
             ReadRIMQ(Qgauges, Path, StartDate, days, NoValue)
        ===============================================================

        Parameters
        ----------
        Qgauges : [DATAFRAME]
            Dataframe containing sub-id of the gauges under a column with a name 0.
        Path : [String]
            path to the folder where files for the gauges exist.
        StartDate : [datetime object]
            starting date of the time series.
        days : [integer]
            length of the simulation (how many days after the start date) .
        NoValue : [numeric value]
            the value used to fill the gaps in the time series or to fill the days
            that is not simulated (discharge is less than threshold).

        Returns
        -------
        QRIM : [dataframe attribute]
            dataframe containing the hydrograph of sub-basins in the Qgauge entered dataframe.

        """
        if AddHQ2 and self.Version == 1:
            assert hasattr(self,"rivernetwork"), "please read the traceall file using the RiverNetwork method"
            assert hasattr(self, "RP"), "please read the HQ file first using ReturnPeriod method"
        EndDate = StartDate + dt.timedelta(days = days-1)
        ind = pd.date_range(StartDate,EndDate)
        QRIM = pd.DataFrame(index = ind, columns = Qgauges[0].tolist())
        # for RIM1.0 don't fill with -9 as the empty days will be filled with 0 so to get
        # the event days we have to filter 0 and -9
        if self.Version == 1:
            QRIM.loc[:,:] = 0
        else:
            QRIM.loc[:,:] = NoValue

        # fill non modelled time steps with zeros
        for i in range(len(Qgauges)):
            f = np.loadtxt( Path + str(int(QRIM.columns[i])) + ".txt",
                       delimiter = ",")
            f1 = list(range(int(f[0,0]),int(f[-1,0])+1))
            f2 = list()

            if AddHQ2 and self.Version == 1:
                USnode = self.rivernetwork.loc[np.where(self.rivernetwork['SubID'] == Qgauges.loc[i,0])[0][0],'US']
                CutValue = self.RP.loc[np.where(self.RP['node'] == USnode)[0][0],'HQ2']


            for j in range(len(f1)):
                # if the index exist in the original list
                if f1[j] in f[:,0]:
                    # put the coresponding value in f2
                    f2.append(f[np.where(f[:,0] == f1[j])[0][0],1])
                else:
                    # if it does not exist put zero
                    if AddHQ2 and self.Version == 1:
                        f2.append(CutValue)
                    else:
                        f2.append(0)

            if Shift:
                f2[ShiftSteps:-1] = f2[0:-(ShiftSteps+1)]

            # QRIM.loc[:,QRIM.columns[i]].loc[ind[f1[0]-1]:ind[f1[-1]-1]] = f2
            QRIM.loc[ind[f1[0]-1]:ind[f1[-1]-1],QRIM.columns[i]] = f2

        self.QRIM = QRIM[:]


    def ReadRIMWL(self, WLGaugesTable, Path, StartDate, days, NoValue, Shift=False, ShiftSteps=0):
        """
        =============================================================================
            ReadRIMWL(WLGaugesTable, Path, StartDate, days, NoValue, Shift=False)
        =============================================================================

        Parameters
        ----------
        WLGaugesTable : TYPE
            DESCRIPTION.
        Path : TYPE
            DESCRIPTION.
        StartDate : TYPE
            DESCRIPTION.
        days : TYPE
            DESCRIPTION.
        NoValue : TYPE
            DESCRIPTION.
        Shift : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        EndDate = StartDate + dt.timedelta(days = days-1)
        ind = pd.date_range(StartDate,EndDate)

        WLRIM = pd.DataFrame(index = ind, columns = WLGaugesTable['Segment'].tolist())
        WLRIM.loc[:,:] = NoValue

        for i in range(len(WLRIM.columns)):
            f = np.loadtxt(Path + str(int(WLRIM.columns[i])) + ".txt",
                       delimiter = ",")

            f1 = list(range(int(f[0,0]),int(f[-1,0])+1))
            f2 = list()
            for j in range(len(f1)):
                # if the index exist in the original list
                if f1[j] in f[:,0]:
                    # put the coresponding value in f2
                    f2.append(f[np.where(f[:,0] == f1[j])[0][0],1])
                else:
                    # if it does not exist put zero
                    f2.append(0)

            if Shift:
                f2[ShiftSteps:-1] = f2[0:-(ShiftSteps+1)]
                # f2[1:-1] = f2[0:-2]

            # WLRIM.loc[:,WLRIM.columns[i]].loc[ind[f1[0]-1]:ind[f1[-1]-1]] = f2
            WLRIM.loc[ind[f1[0]-1]:ind[f1[-1]-1],WLRIM.columns[i]] = f2

        self.WLRIM = WLRIM[:]

    def ReadCalirationResult(self, SubID, Path = ''):
        """
        =============================================================================
          ReadCalirationResult(SubID, FromDay = [], ToDay = [], Path = '', FillMissing = True)
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
        hasattr(self,"QGauges"),"Please read the discharge gauges first"
        hasattr(self,"WlGauges"),"Please read the water level gauges first"
        
        if not hasattr(self, "CalibrationQ"):
            indD = pd.date_range(self.start, self.end, freq = "D")[:-1]
            self.CalibrationQ = pd.DataFrame(index = indD)
        if not hasattr(self, "CalibrationWL"):
            indD = pd.date_range(self.start, self.end, freq = "D")[:-1]
            self.CalibrationWL = pd.DataFrame(index = indD)
        
        ind = pd.date_range(self.start, self.end, freq = "H")[:-1]
        q = pd.read_csv(Path + str(SubID) + "_q.txt", header = None)
        wl = pd.read_csv(Path + str(SubID) + "_wl.txt", header = None)
        
        q.index = ind
        wl.index = ind
        
        self.CalibrationQ[SubID] = q[0].resample('D').mean()
        self.CalibrationWL[SubID] = wl[0].resample('D').mean()
        
        
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
        self.rivernetwork = pd.read_csv(Path, delimiter = ',') #,header = None
        # self.rivernetwork.columns = ['SubID','US','DS']

    def GetAnnualMax(self, option=1, CorespondingTo=dict(MaxObserved=" ", TimeWindow=0)):
        """
        ========================================================
              GetAnnualMax(option=1, option=1, CorespondingTo=dict(MaxObserved=" ", TimeWindow=0))
        ========================================================
        GetAnnualMax method get the max annual time series out of time series of any
        temporal resolution, the code assumes that the hydrological year is
        1-Nov/31-Oct (Petrow and Merz, 2009, JoH).

        Parameters
        ----------
            option : [integer], optional
                1 for the historical observed Discharge data, 2 for the historical observed water level data,
                3 for the rainfall-runoff data, 4 for the rim discharge result,
                5 for the rim water level result. The default is 1.

            CorespondingTo: [Dict], optional
                -if you want to extract the max annual values from the observed discharge
                    time series (CorespondingTo=dict(MaxObserved = "Q") and then extract the
                    values of the same dates in the result time series
                    the same for observed water level time series (CorespondingTo=dict(MaxObserved = "WL").
                    or if you just want to extract the max annual time values from each time series
                    (CorespondingTo=dict(MaxObserved = " ").The default is " ".

                - if you want to extract some values before and after the coresponding
                    date and then take the max value of all extracted values specify the
                    number of days using the keyword Window CorespondingTo=dict(TimeWindow =  1)

        Returns
        -------
            AnnualMaxObsQ: [dataframe attribute]
                when using Option=1
            AnnualMaxObsWL: [dataframe attribute]
                when using option = 2
            AnnualMaxRRM: [dataframe attribute]
                when using option = 3
            AnnualMaxRIMQ: [dataframe attribute]
                when using option = 4
            AnnualMaxRIMWL: [dataframe attribute]
                when using option = 5
        """
        if option == 1:
            assert hasattr(self, "QGauges"), "please read the observed Discharge data first with the ReadObservedQ method"
            columns = self.QGauges.columns.tolist()
        elif option == 2:
            assert hasattr(self, "WLGauges"), "please read the observed Water level data first with the ReadObservedWL method"
            columns = self.WLGauges.columns.tolist()
        elif option == 3:
            assert hasattr(self, "QRRM"), "please read the Rainfall-runoff data first with the ReadRRM method"
            columns = self.QRRM.columns.tolist()
        elif option == 4:
            assert hasattr(self, "QRIM"), "please read the RIM results first with the ReadRIMQ method"
            columns = self.QRIM.columns.tolist()
        else:
            assert hasattr(self, "WLRIM"), "please read the RIM results first with the ReadRIMWL method"
            columns = self.WLRIM.columns.tolist()


        if CorespondingTo['MaxObserved'] == "WL":
            assert hasattr(self, "WLGauges"), "please read the observed Water level data first with the ReadObservedWL method"

            startdate = self.WLGauges.index[0]
            AnnualMax = self.WLGauges.loc[:,self.WLGauges.columns[0]].resample('A-OCT').max()
            self.AnnualMaxDates = pd.DataFrame(index = AnnualMax.index, columns = self.WLGauges.columns)

            # get the dates when the max value happen every year
            for i in range(len(self.WLGauges.columns)):
                sub = self.WLGauges.columns[i]
                for j in range(len(AnnualMax)):
                    if j == 0:
                        f = self.WLGauges.loc[startdate:AnnualMax.index[j],sub]
                        self.AnnualMaxDates.loc[AnnualMax.index[j],sub] = f.index[f.argmax()]
                    else:
                        f = self.WLGauges.loc[AnnualMax.index[j-1]:AnnualMax.index[j],sub]
                        self.AnnualMaxDates.loc[AnnualMax.index[j],sub] = f.index[f.argmax()]

            # extract the values at the dates of the previous max value
            AnnualMax = pd.DataFrame(index = self.AnnualMaxDates.index, columns = columns)

            # Extract time series
            for i in range(len(columns)):
                Sub = columns[i]
                QTS = list()

                if option ==1:
                    # QTS = self.QGauges.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = CorespondingTo['TimeWindow'])
                        EndDate = date + dt.timedelta(days = CorespondingTo['TimeWindow'])
                        QTS.append(self.QGauges.loc[Startdate:EndDate, Sub].max())
                elif option ==2:
                    # QTS = self.WLGauges.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = 1)
                        EndDate = date + dt.timedelta(days = 1)
                        QTS.append(self.WLGauges.loc[Startdate:EndDate, Sub].max())
                elif option ==3:
                    # QTS = self.QRRM.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = CorespondingTo['TimeWindow'])
                        EndDate = date + dt.timedelta(days = CorespondingTo['TimeWindow'])
                        QTS.append(self.QRRM.loc[Startdate:EndDate, Sub].max())
                elif option ==4:
                    # QTS = self.QRIM.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = CorespondingTo['TimeWindow'])
                        EndDate = date + dt.timedelta(days = CorespondingTo['TimeWindow'])
                        QTS.append(self.QRIM.loc[Startdate:EndDate, Sub].max())
                else:
                    # QTS = self.WLRIM.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = CorespondingTo['TimeWindow'])
                        EndDate = date + dt.timedelta(days = CorespondingTo['TimeWindow'])
                        QTS.append(self.WLRIM.loc[Startdate:EndDate, Sub].max())

                AnnualMax.loc[:, Sub] = QTS

        elif CorespondingTo['MaxObserved'] == "Q":
            assert hasattr(self, "QGauges"), "please read the observed Discharge data first with the ReadObservedQ method"

            startdate = self.QGauges.index[0]
            AnnualMax = self.QGauges.loc[:,self.QGauges.columns[0]].resample('A-OCT').max()
            self.AnnualMaxDates = pd.DataFrame(index = AnnualMax.index, columns = self.QGauges.columns)

            # get the date when the max value happen every year
            for i in range(len(self.QGauges.columns)):
                sub = self.QGauges.columns[i]
                for j in range(len(AnnualMax)):
                    if j == 0:
                        f = self.QGauges.loc[startdate:AnnualMax.index[j],sub]
                        self.AnnualMaxDates.loc[AnnualMax.index[j],sub] = f.index[f.argmax()]
                    else:
                        f = self.QGauges.loc[AnnualMax.index[j-1]:AnnualMax.index[j],sub]
                        self.AnnualMaxDates.loc[AnnualMax.index[j],sub] = f.index[f.argmax()]

            # extract the values at the dates of the previous max value
            AnnualMax = pd.DataFrame(index = self.AnnualMaxDates.index, columns = columns)
            # Extract time series
            for i in range(len(columns)):
                Sub = columns[i]
                QTS = list()

                if option ==1:
                    # QTS = self.QGauges.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = CorespondingTo['TimeWindow'])
                        EndDate = date + dt.timedelta(days = CorespondingTo['TimeWindow'])
                        QTS.append(self.QGauges.loc[Startdate:EndDate, Sub].max())

                elif option ==2:
                    # QTS = self.WLGauges.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = CorespondingTo['TimeWindow'])
                        EndDate = date + dt.timedelta(days = CorespondingTo['TimeWindow'])
                        QTS.append(self.WLGauges.loc[Startdate:EndDate, Sub].max())

                elif option ==3:
                    # QTS = self.QRRM.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = CorespondingTo['TimeWindow'])
                        EndDate = date + dt.timedelta(days = CorespondingTo['TimeWindow'])
                        QTS.append(self.QRRM.loc[Startdate:EndDate, Sub].max())

                elif option ==4:
                    # QTS = self.QRIM.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = CorespondingTo['TimeWindow'])
                        EndDate = date + dt.timedelta(days = CorespondingTo['TimeWindow'])
                        QTS.append(self.QRIM.loc[Startdate:EndDate, Sub].max())
                else:
                    # QTS = self.WLRIM.loc[self.AnnualMaxDates.loc[:,Sub].values, Sub].values
                    for j in range(len(self.AnnualMaxDates.loc[:,Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind,Sub]
                        Startdate = date - dt.timedelta(days = CorespondingTo['TimeWindow'])
                        EndDate = date + dt.timedelta(days = CorespondingTo['TimeWindow'])
                        QTS.append(self.WLRIM.loc[Startdate:EndDate, Sub].max())

                # resample to annual time step
                AnnualMax.loc[:, Sub] = QTS
        else :
            AnnualMax = pd.DataFrame(columns = columns)
            # Extract time series
            for i in range(len(columns)):
                Sub = columns[i]
                if option ==1:
                    QTS = self.QGauges.loc[:, Sub]
                elif option ==2:
                    QTS = self.WLGauges.loc[:, Sub]
                elif option ==3:
                    QTS = self.QRRM.loc[:, Sub]
                elif option ==4:
                    QTS = self.QRIM.loc[:, Sub]
                else:
                    QTS = self.WLRIM.loc[:, Sub]
                # resample to annual time step
                AnnualMax.loc[:, Sub] = QTS.resample('A-OCT').max().values

            AnnualMax.index = QTS.resample('A-OCT').indices.keys()

        if option ==1:
            self.AnnualMaxObsQ = AnnualMax
        elif option ==2:
            self.AnnualMaxObsWL = AnnualMax
        elif option ==3:
            self.AnnualMaxRRM = AnnualMax
        elif option ==4:
            self.AnnualMaxRIMQ = AnnualMax #AnnualMaxRIM
        else:
            self.AnnualMaxRIMWL = AnnualMax
            
    
    def CalculateProfile(self, Segmenti, BedlevelDS, Manning, BC_slope):
        """
        ===========================================================================
            CalculateProfile(Segmenti, BedlevelDS, Manning)
        ===========================================================================
        CalculateProfile method takes the river segment ID and the calibration parameters
        (last downstream cross-section bed level and the manning coefficient) and 
        calculates the new profiles
        
        Parameters
        ----------
        Segmenti : [Integer]
            cross-sections segment ID .
        BedlevelDS : [Float]
            the bed level of the last cross section in the segment.
        Manning : [float]
            manning coefficient.

        Returns
        -------
        crosssection:[dataframe attribute]
            crosssection attribute will be updated with the newly calculated 
            profile for the given segment
        slope:[dataframe attribute]
            slope attribute will be updated with the newly calculated average
            slope for the given segment
        """
    
        levels = pd.DataFrame(columns=['segment','bedlevelUS','bedlevelDS'])
        
        # change cross-section
        bedlevel = self.crosssections.loc[self.crosssections["segment"]==Segmenti,'gl'].values
        # get the bedlevel of the last cross section in the segment as a calibration parameter
        levels.loc[Segmenti,'bedlevelDS'] = BedlevelDS
        levels.loc[Segmenti,'bedlevelUS'] = bedlevel[0]
        
        NoDistances = len(bedlevel)-1
        # AvgSlope = ((levels.loc[Segmenti,'bedlevelUS'] - levels.loc[Segmenti,'bedlevelDS'] )/ (500 * NoDistances)) *-500
        # change in the bed level of the last XS
        AverageDelta = (levels.loc[Segmenti,'bedlevelDS'] - bedlevel[-1])/ NoDistances
        
        # calculate the new bed levels 
        bedlevelNew = np.zeros(len(bedlevel))
        bedlevelNew[len(bedlevel)-1] = levels.loc[Segmenti,'bedlevelDS']
        bedlevelNew[0] = levels.loc[Segmenti,'bedlevelUS']
        
        for i in range(len(bedlevel)-1):
            # bedlevelNew[i] = levels.loc[Segmenti,'bedlevelDS'] + (len(bedlevel) - i -1) * abs(AvgSlope)
            bedlevelNew[i] = bedlevel[i] + i * AverageDelta
        
        self.crosssections.loc[self.crosssections["segment"]==Segmenti,'gl'] = bedlevelNew
        
        # change manning
        self.crosssections.loc[self.crosssections["segment"]==Segmenti,'m'] = Manning
        
        ## change slope
        # self.slope.loc[self.slope['segment']==Segmenti, 'slope'] = AvgSlope
        self.slope.loc[self.slope['segment']==Segmenti, 'slope'] = BC_slope
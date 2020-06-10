# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:27:32 2020

@author: mofarrag
"""
import pandas as pd
import datetime as dt
import numpy as np

datafn = lambda x: dt.datetime.strptime(x,"%Y-%m-%d")

class RIMCalibration():

    def __init__(self, name, Version):
        self.name = name
        self.Version = Version

    def ReadObservedWL(self, GaugesTable, Path, StartDate, EndDate, NoValue):
        """
        ============================================================================
            ReadObservedWL( WLGauges, Path, StartDate, EndDate, NoValue)
        ============================================================================

        Parameters
        ----------
            1-GaugesTable : [dataframe]
                Dataframe contains columns [swimid,xsid,datum(m)].
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
            1-WL: [dataframe attribute].


        """

        ind = pd.date_range(StartDate, EndDate)
        columns = GaugesTable['swimid'].tolist()

        WLGauges = pd.DataFrame(index = ind)
        # WLGaugesf.loc[:,:] = NoValue
        WLGauges.loc[:,0] = ind

        for i in range(len(columns)):
            f = pd.read_csv(Path + str(int(columns[i])) + ".txt",
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
            f.loc[f[1]!= NoValue,1] = (f.loc[f[1]!= NoValue,1] / 100) + GaugesTable.loc[i,'datum(m)']
            f = f.rename(columns={1:columns[i]})


            # assign the values in the dateframe
            # WLGauges.loc[:,WLGauges.columns[i]].loc[f[0][0]:f[0][len(f)-1]] = f[1].tolist()
            # use merge as there are some gaps in the middle
            WLGauges = WLGauges.merge(f, on=0, how='left', sort=False)

        WLGauges.replace(to_replace = np.nan, value = NoValue, inplace=True)
        WLGauges.index = ind
        del WLGauges[0]
        self.WLGauges = WLGauges

        GaugesTable.index = GaugesTable['swimid'].tolist()
        GaugesTable['start'] = 0
        GaugesTable['end'] = 0
        for i in range(len(columns)):
            st1 = WLGauges[columns[i]][WLGauges[columns[i]] != NoValue].index[0]
            end1 = WLGauges[columns[i]][WLGauges[columns[i]] != NoValue].index[-1]
            GaugesTable.loc[GaugesTable.loc[:,'swimid'] == columns[i],'start'] = st1
            GaugesTable.loc[GaugesTable.loc[:,'swimid'] == columns[i],'end'] = end1

        self.WLGaugesTable = GaugesTable


    def ReadObservedQ(self, CalibratedSubs, Path, StartDate, EndDate, NoValue):
        """
        ========================================================================
            ReadObservedQ(CalibratedSubs, Path, StartDate, EndDate, NoValue)
        ========================================================================
        ReadObservedQ method reads discharge data and store it in a dataframe
        attribute "QGauges"

        Parameters
        ----------
        CalibratedSubs : [DATAFRAME]
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

        ind = pd.date_range(StartDate, EndDate)
        GRDC = pd.DataFrame(index = ind)

        for i in range(len(CalibratedSubs[0])):
            GRDC.loc[:,int(CalibratedSubs[0][i])] = np.loadtxt(Path +
                      str(int(CalibratedSubs[0][i])) + '.txt') #,skiprows = 0
        self.QGauges = GRDC

        GaugesTable = pd.DataFrame(index = CalibratedSubs[0])
        # GaugesTable['SubID'] = CalibratedSubs[0]
        GaugesTable['start'] = 0
        GaugesTable['end'] = 0

        for i in range(len(CalibratedSubs[0])):
            st1 = GRDC[CalibratedSubs[0][i]][GRDC[CalibratedSubs[0][i]] != NoValue].index[0]
            end1 = GRDC[CalibratedSubs[0][i]][GRDC[CalibratedSubs[0][i]] != NoValue].index[-1]
            # GaugesTable.loc[GaugesTable.loc[:,'SubID'] == CalibratedSubs[0][i],'start'] = st1
            # GaugesTable.loc[GaugesTable.loc[:,'SubID'] == CalibratedSubs[0][i],'end'] = end1
            GaugesTable.loc[CalibratedSubs[0][i],'start'] = st1
            GaugesTable.loc[CalibratedSubs[0][i],'end'] = end1

        self.QGaugesTable = GaugesTable

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

            QRIM.loc[:,QRIM.columns[i]].loc[ind[f1[0]-1]:ind[f1[-1]-1]] = f2

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

        WLRIM = pd.DataFrame(index = ind, columns = WLGaugesTable['swimid'].tolist())
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

            WLRIM.loc[:,WLRIM.columns[i]].loc[ind[f1[0]-1]:ind[f1[-1]-1]] = f2

        self.WLRIM = WLRIM[:]


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

    def GetAnnualMax(self, option=1):
        """
        ========================================================
              GetAnnualMax(option=1)
        ========================================================
        GetAnnualMax method get the max annual time series out of time series of any
        temporal resolution, the code assumes that the hydrological year is
        1-Nov/31-Oct (Petrow and Merz, 2009, JoH).

        Parameters
        ----------
        option : [integer], optional
            1 for the historical observed data, 2 for the rainfall-runoff data
            3 for the rim result. The default is 1.

        Returns
        -------
        None.

        """
        if option == 1:
            assert hasattr(self, "QGauges"), "please read the observed data first with the ReadObservedQ method"
            columns = self.QGauges.columns.tolist()
        elif option == 2:
            assert hasattr(self, "QRRM"), "please read the observed data first with the ReadRRM method"
            columns = self.QRRM.columns.tolist()
        else :#option == 3:
            assert hasattr(self, "QRIM"), "please read the observed data first with the ReadRIMQ method"
            columns = self.QRIM.columns.tolist()

        AnnualMax = pd.DataFrame(columns = columns)
        # RIM2 results
        for i in range(len(columns)):
            Sub = columns[i]
            if option ==1:
                QTS = self.QGauges.loc[:, Sub]
            elif option ==2:
                QTS = self.QRRM.loc[:, Sub]
            else:
                QTS = self.QRIM.loc[:, Sub]

            AnnualMax.loc[:, Sub] = QTS.resample('A-OCT').max().values

        AnnualMax.index = QTS.resample('A-OCT').indices.keys()

        if option ==1:
            self.AnnualMaxObs = AnnualMax
        elif option ==2:
            self.AnnualMaxRRM = AnnualMax
        else:
            self.AnnualMaxRIM = AnnualMax


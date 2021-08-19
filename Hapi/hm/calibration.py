import pandas as pd
import datetime as dt
import numpy as np
from Hapi.hm.river import River

datafn = lambda x: dt.datetime.strptime(x,"%Y-%m-%d")


class Calibration(River):
    """Hydraulic model Calibration.
    
    Hydraulic model calibration class

    """
    
    def __init__(self, name, version=3, start="1950-1-1", days=36890,
                 fmt="%Y-%m-%d"):
        """HMCalibration.
        
        To instantiate the HMCalibration object you have to provide the following
        arguments

        Parameters
        ----------
        name : [str]
            name of the catchment.
        version : [integer], optional
            The version of the model. The default is 3.
        start : [str], optional
            starting date. The default is "1950-1-1".
        days : [integer], optional
            length of the simulation. The default is 36890.
        fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"
        Returns
        -------
        None.

        """
        self.name = name
        self.version = version

        self.start = dt.datetime.strptime(start,fmt)
        self.end = self.start + dt.timedelta(days = days)
        self.days = days

        Ref_ind = pd.date_range(self.start, self.end, freq='D')
        self.ReferenceIndex = pd.DataFrame(index = list(range(1,days+1)))
        self.ReferenceIndex['date'] = Ref_ind[:-1]

    def ReadGaugesTable(self,Path):
        """ReadGaugesTable.
        
        ReadGaugesTable reads the table of the gauges

        Parameters
        ----------
        Path : [String]
            the path to the text file of the gauges table.

        Returns
        -------
        GaugesTable: [dataframe attribute]
            the table will be read in a dataframe attribute

        """
        self.GaugesTable = pd.read_csv(Path)


    def ReadObservedWL(self, path, start, end, novalue, column='oid', fmt="%Y-%m-%d"):
        """ReadObservedWL.
        
        read the water level data of the gauges.

        Parameters
        ----------
            1-GaugesTable : [dataframe]
                Dataframe contains columns [id,xsid,datum(m)].
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
        if type(start) == str:
            start = dt.datetime.strptime(start,fmt)
        if type(end) == str:
            end = dt.datetime.strptime(end,fmt)
            
        ind = pd.date_range(start, end)
        columns = self.GaugesTable[column].tolist()

        WLGauges = pd.DataFrame(index = ind)
        # WLGaugesf.loc[:,:] = NoValue
        WLGauges.loc[:,0] = ind

        for i in range(len(columns)):
            if self.GaugesTable.loc[i,'waterlevel'] == 1:
                name = self.GaugesTable.loc[i,column]
                f = pd.read_csv(path + str(int(name)) + ".txt",
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
                f.loc[f[1]!= novalue,1] = (f.loc[f[1]!= novalue,1] / 100) + self.GaugesTable.loc[i,'datum(m)']
                f = f.rename(columns={1:columns[i]})

                # assign the values in the dateframe
                # WLGauges.loc[:,WLGauges.columns[i]].loc[f[0][0]:f[0][len(f)-1]] = f[1].tolist()
                # use merge as there are some gaps in the middle
                WLGauges = WLGauges.merge(f, on=0, how='left', sort=False)

        WLGauges.replace(to_replace = np.nan, value = novalue, inplace=True)
        WLGauges.index = ind
        del WLGauges[0]
        self.WLGauges = WLGauges

        # GaugesTable.index = GaugesTable['id'].tolist()
        self.GaugesTable['WLstart'] = 0
        self.GaugesTable['WLend'] = 0
        for i in range(len(columns)):
            if self.GaugesTable.loc[i,'waterlevel'] == 1:
                st1 = WLGauges[columns[i]][WLGauges[columns[i]] != novalue].index[0]
                end1 = WLGauges[columns[i]][WLGauges[columns[i]] != novalue].index[-1]
                self.GaugesTable.loc[i,'WLstart'] = st1
                self.GaugesTable.loc[i,'WLend'] = end1


    def ReadObservedQ(self, path, start, end, novalue, column='oid'): #Gauges,
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

        # GaugesTable = pd.DataFrame(index = Gauges['id'])
        # GaugesTable['start'] = 0
        # GaugesTable['end'] = 0

        # for i in range(len(Gauges[ID])):
        #     st1 = GRDC[Gauges['id'][i]][GRDC[Gauges['id'][i]] != NoValue].index[0]
        #     end1 = GRDC[Gauges['id'][i]][GRDC[Gauges['id'][i]] != NoValue].index[-1]
        #     # GaugesTable.loc[GaugesTable.loc[:,'SubID'] == Gauges[0][i],'start'] = st1
        #     # GaugesTable.loc[GaugesTable.loc[:,'SubID'] == Gauges[0][i],'end'] = end1
        #     GaugesTable.loc[Gauges['id'][i],'start'] = st1
        #     GaugesTable.loc[Gauges['id'][i],'end'] = end1

        # self.QGaugesTable = GaugesTable

        ind = pd.date_range(start, end)
        QGauges = pd.DataFrame(index = ind)
        # ID = Gauges.columns[0]
        # columns = Gauges['gid'].tolist()
        for i in range(len(self.GaugesTable)):
            if self.GaugesTable.loc[i,'discharge'] == 1:
                name = self.GaugesTable.loc[i,column]
                try:
                    QGauges.loc[:,int(name)] = np.loadtxt(path +
                              str(int(name)) + '.txt') #,skiprows = 0
                except:
                    print(str(i) + "-" + path + str(int(name)) + '.txt')
        self.QGauges = QGauges

        # Gauges = pd.DataFrame(index = Gauges['id'])
        self.GaugesTable['Qstart'] = 0
        self.GaugesTable['Qend'] = 0

        for i in range(len(self.GaugesTable)):
            if self.GaugesTable.loc[i,'discharge'] == 1:
                ii = self.GaugesTable.loc[i,column]
                st1 = QGauges[ii][QGauges[ii] != novalue].index[0]
                end1 = QGauges[ii][QGauges[ii] != novalue].index[-1]
                self.GaugesTable.loc[i,'Qstart'] = st1
                self.GaugesTable.loc[i,'Qend'] = end1


    def ReadRRM(self, path, start, end, column='oid'): #Qgauges,
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

        ind = pd.date_range(start,end)
        QSWIM = pd.DataFrame(index = ind)


        for i in range(len(self.GaugesTable[column])):
            # read SWIM data
            # only at the begining to get the length of the time series
            QSWIM.loc[:,int(self.GaugesTable.loc[i,column])] = np.loadtxt(path +
                  str(int(self.GaugesTable.loc[i,column])) + '.txt')#,skiprows = 0
        self.QRRM = QSWIM


    def ReadHMQ(self, path, start, days, novalue, addHQ2=False, #, column='oid'
                 shift=False, shiftsteps=0, column='oid'):
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
        if addHQ2 and self.version == 1:
            assert hasattr(self,"rivernetwork"), "please read the traceall file using the RiverNetwork method"
            assert hasattr(self, "RP"), "please read the HQ file first using ReturnPeriod method"
        end = start + dt.timedelta(days = days-1)
        ind = pd.date_range(start,end)
        QRIM = pd.DataFrame(index = ind, columns = self.GaugesTable.loc[self.GaugesTable['discharge']==1,column].tolist())
        # for RIM1.0 don't fill with -9 as the empty days will be filled with 0 so to get
        # the event days we have to filter 0 and -9
        if self.version == 1:
            QRIM.loc[:,:] = 0
        else:
            QRIM.loc[:,:] = novalue

        # fill non modelled time steps with zeros
        for i in range(len(self.GaugesTable[column])):
            f = np.loadtxt( path + str(int(QRIM.columns[i])) + ".txt",
                       delimiter = ",")
            f1 = list(range(int(f[0,0]),int(f[-1,0])+1))
            f2 = list()

            if addHQ2 and self.version == 1:
                USnode = self.rivernetwork.loc[np.where(self.rivernetwork['id'] == self.GaugesTable.loc[i,column])[0][0],'us']
                CutValue = self.RP.loc[np.where(self.RP['node'] == USnode)[0][0],'HQ2']


            for j in range(len(f1)):
                # if the index exist in the original list
                if f1[j] in f[:,0]:
                    # put the coresponding value in f2
                    f2.append(f[np.where(f[:,0] == f1[j])[0][0],1])
                else:
                    # if it does not exist put zero
                    if addHQ2 and self.version == 1:
                        f2.append(CutValue)
                    else:
                        f2.append(0)

            if shift:
                f2[shiftsteps:-1] = f2[0:-(shiftsteps+1)]

            # QRIM.loc[:,QRIM.columns[i]].loc[ind[f1[0]-1]:ind[f1[-1]-1]] = f2
            QRIM.loc[ind[f1[0]-1]:ind[f1[-1]-1],QRIM.columns[i]] = f2

        self.QRIM = QRIM[:]


    def ReadHMWL(self, path, start, days, novalue, shift=False, shiftsteps=0,
                  column='oid'): #WLGaugesTable,
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

        end = start + dt.timedelta(days = days-1)
        ind = pd.date_range(start,end)

        WLRIM = pd.DataFrame(index = ind, columns = self.GaugesTable.loc[self.GaugesTable['waterlevel']==1,column].tolist())
        WLRIM.loc[:,:] = novalue

        for i in range(len(WLRIM.columns)):
            f = np.loadtxt(path + str(int(WLRIM.columns[i])) + ".txt",
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

            if shift:
                f2[shiftsteps:-1] = f2[0:-(shiftsteps+1)]

            
            WLRIM.loc[ind[f1[0]-1]:ind[f1[-1]-1],WLRIM.columns[i]] = f2

        self.WLRIM = WLRIM[:]


    def ReadCalirationResult(self, SubID, Path = ''):
        """
        =============================================================================
          ReadCalirationResult(SubID, FromDay = [], ToDay = [], Path = '', FillMissing = True)
        =============================================================================
        ReadCalirationResult method reads the 1D results and fill the missing days in the middle

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
        q = pd.read_csv(Path + str(SubID) + "_q.txt", header = None, delimiter=r'\s+')
        wl = pd.read_csv(Path + str(SubID) + "_wl.txt", header = None, delimiter=r'\s+')

        q.index = ind
        wl.index = ind

        self.CalibrationQ[SubID] = q[1].resample('D').mean()
        self.CalibrationWL[SubID] = wl[1].resample('D').mean()


    # def ReturnPeriod(self,Path):
    #     """
    #     ==========================================
    #          ReturnPeriod(Path)
    #     ==========================================
    #     ReturnPeriod method reads the HQ file which contains all the computational nodes
    #     with HQ2, HQ10, HQ100
    #     Parameters
    #     ----------
    #         1-Path : [String]
    #             path to the HQ.csv file including the file name and extention
    #             "RIM1Files + "/HQRhine.csv".

    #     Returns
    #     -------
    #         1-RP:[data frame attribute]
    #             containing the river computational node and calculated return period
    #             for with columns ['node','HQ2','HQ10','HQ100']
    #     """
    #     self.RP = pd.read_csv(Path, delimiter = ",",header = None)
    #     self.RP.columns = ['node','HQ2','HQ10','HQ100']

    # def RiverNetwork(self, Path):
    #     """
    #     =====================================================
    #           RiverNetwork(Path)
    #     =====================================================
    #     RiverNetwork method rad the table of each computational node followed by
    #     upstream and then downstream node (TraceAll file)

    #     ==============   ====================================================
    #     Keyword          Description
    #     ==============   ====================================================
    #     1-Path :         [String] path to the Trace.txt file including the file name and extention
    #                         "path/Trace.txt".

    #     Returns
    #     -------
    #         1-rivernetwork:[data frame attribute]
    #             containing the river network with columns ['SubID','US','DS']
    #     """
    #     self.rivernetwork = pd.read_csv(Path, delimiter = ',') #,header = None
    #     # self.rivernetwork.columns = ['SubID','US','DS']

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

        levels = pd.DataFrame(columns=['id','bedlevelUS','bedlevelDS'])

        # change cross-section
        bedlevel = self.crosssections.loc[self.crosssections["id"]==Segmenti,'gl'].values
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

        self.crosssections.loc[self.crosssections["id"]==Segmenti,'gl'] = bedlevelNew

        # change manning
        self.crosssections.loc[self.crosssections["id"]==Segmenti,'m'] = Manning

        ## change slope
        # self.slope.loc[self.slope['id']==Segmenti, 'slope'] = AvgSlope
        self.slope.loc[self.slope['id']==Segmenti, 'slope'] = BC_slope

    # def ReadCrossSections(self,Path):
    #     """
    #     ===========================================
    #       CrossSections(self,Path)
    #     ===========================================
    #     CrossSections method reads the cross section data of the river and assign it
    #     to an attribute "Crosssections" of type dataframe
    #     """
    #     if self.version == 1 or self.version == 2:
    #         self.crosssections = pd.read_csv(Path, delimiter = ',', skiprows =1  )
    #     else:
    #         self.crosssections = pd.read_csv(Path, delimiter = ',')

    def SmoothBedLevel(self, segmenti):
        """
        ================================================
               SmoothXS(segmenti)
        =================================================
        SmoothBedLevel method smoothes the bed level of a given segment ID by
        calculating the moving average of three cross sections

        Parameters
        ----------
        segmenti : [Integer]
            segment ID.

        Returns
        -------
        crosssections: [dataframe attribute]
            the "gl" column in the crosssections attribute will be smoothed

        """
        assert hasattr(self,"crosssections"), "please read the cross section first"
        g = self.crosssections.loc[self.crosssections['id']==segmenti,:].index[0]

        segment = self.crosssections.loc[self.crosssections['id']==segmenti,:]
        segment.index = range(len(segment))
        segment['glnew'] = 0
        # the bed level at the beginning and end of the egment
        segment.loc[0,'glnew'] = segment.loc[0,'gl']
        segment.loc[len(segment)-1,'glnew'] = segment.loc[len(segment)-1,'gl']
        # calculate the average of three XS bed level
        for j in range(1,len(segment)-1):
            segment.loc[j,'glnew'] = (segment.loc[j-1,'gl'] + segment.loc[j,'gl'] + segment.loc[j+1,'gl'])/3
        # calculate the difference in the bed level and take it from the bankful depth
        segment['diff'] = segment['glnew'] - segment['gl']
        segment['dbf'] = segment['dbf'] - segment['diff']
        segment['gl'] = segment['glnew']
        del segment['glnew'], segment['diff']

        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections['id']==segmenti,:] = segment
        # g = g + len(segment)

    def SmoothBankLevel(self,segmenti):
        """
        ========================================================
              SmoothBankLevel(segmenti)
        ========================================================
        SmoothBankLevel method smoothes the bankfull depth for a given segment

        Parameters
        ----------
        segmenti : [Integer]
            segment ID.

        Returns
        -------
        crosssections: [dataframe attribute]
            the "dbf" column in the crosssections attribute will be smoothed

        """

        self.crosssections['banklevel'] = self.crosssections['dbf'] + self.crosssections['gl']

        # g = 0



        # for i in range(len(segments)):
        # i=30
        g = self.crosssections.loc[self.crosssections['id']==segmenti,:].index[0]
        #---
        # segmenti = segments[i]
        segment = self.crosssections.loc[self.crosssections['id']==segmenti,:]
        segment.index = range(len(segment))
        segment['banklevelnew'] = 0
        segment.loc[0,'banklevelnew'] = segment.loc[0,'banklevel']
        segment.loc[len(segment)-1,'banklevelnew'] = segment.loc[len(segment)-1,'banklevel']

        for j in range(1,len(segment)-1):
            segment.loc[j,'banklevelnew'] = (segment.loc[j-1,'banklevel'] + segment.loc[j,'banklevel'] + segment.loc[j+1,'banklevel'])/3

        segment['diff'] = segment['banklevelnew'] - segment['banklevel']
        segment['dbf'] = segment['dbf'] + segment['diff']

        del self.crosssections['banklevel']
        segment.index = range(g, g + len(segment))

        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections['id']==segmenti,:] = segment
        # g = g + len(segment)
        # end of loop

    def SmoothFloodplainHeight(self,segmenti):
        """
        ========================================================
              SmoothFloodplainHeight(segmenti)
        ========================================================
        SmoothFloodplainHeight method smoothes the Floodplain Height the point 5 and 6
        in the cross section for a given segment

        Parameters
        ----------
        segmenti : [Integer]
            segment ID.

        Returns
        -------
        crosssections: [dataframe attribute]
            the "hl" and "hr" column in the crosssections attribute will be smoothed

        """

        self.crosssections['banklevel'] = self.crosssections['dbf'] + self.crosssections['gl']
        self.crosssections['fpl'] = self.crosssections['hl'] + self.crosssections['banklevel']
        self.crosssections['fpr'] = self.crosssections['hr'] + self.crosssections['banklevel']

        # g = 0

        # for i in range(len(segments)):
        # i=30
        g = self.crosssections.loc[self.crosssections['id']==segmenti,:].index[0]
        #------

        # segmenti = segments[i]
        segment = self.crosssections.loc[self.crosssections['id']==segmenti,:]
        segment.index = range(len(segment))

        segment['fplnew'] = 0
        segment['fprnew'] = 0
        segment.loc[0,'fplnew'] = segment.loc[0,'fpl']
        segment.loc[len(segment)-1,'fplnew'] = segment.loc[len(segment)-1,'fpl']

        segment.loc[0,'fprnew'] = segment.loc[0,'fpr']
        segment.loc[len(segment)-1,'fprnew'] = segment.loc[len(segment)-1,'fpr']

        for j in range(1,len(segment)-1):
            segment.loc[j,'fplnew'] = (segment.loc[j-1,'fpl'] + segment.loc[j,'fpl'] + segment.loc[j+1,'fpl'])/3
            segment.loc[j,'fprnew'] = (segment.loc[j-1,'fpr'] + segment.loc[j,'fpr'] + segment.loc[j+1,'fpr'])/3

        segment['diff0'] = segment['fplnew'] - segment['fpl']
        segment['diff1'] = segment['fprnew'] - segment['fpr']


        segment['hl'] = segment['hl'] + segment['diff0']
        segment['hr'] = segment['hr'] + segment['diff1']

        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections['id']==segmenti,:] = segment

        del self.crosssections['banklevel'], self.crosssections['fpr'], self.crosssections['fpl']
        # g = g + len(segment)
        # end of loop

    def SmoothBedWidth(self,segmenti):
        """
        ========================================================
              SmoothBedWidth(segmenti)
        ========================================================
        SmoothBedWidth method smoothes the Bed Width the in the cross section
        for a given segment

        Parameters
        ----------
        segmenti : [Integer]
            segment ID.

        Returns
        -------
        crosssections: [dataframe attribute]
            the "b" column in the crosssections attribute will be smoothed

        """
        # for i in range(len(segments)):
        # i=30
        g = self.crosssections.loc[self.crosssections['id']==segmenti,:].index[0]
        #------

        # segmenti = segments[i]
        segment = self.crosssections.loc[self.crosssections['id']==segmenti,:]
        segment.index = range(len(segment))
        segment['bnew'] = 0
        segment.loc[0,'bnew'] = segment.loc[0,'b']
        segment.loc[len(segment)-1,'bnew'] = segment.loc[len(segment)-1,'b']

        for j in range(1,len(segment)-1):
            segment.loc[j,'bnew'] = (segment.loc[j-1,'b'] + segment.loc[j,'b'] + segment.loc[j+1,'b'])/3

        segment['b'] = segment['bnew']
        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections['id']==segmenti,:] = segment
        # g = g + len(segment)
        # end of loop

    def DownWardBedLevel(self,segmenti, height):
        """
        ========================================================
              SmoothBedWidth(segmenti)
        ========================================================
        SmoothBedWidth method smoothes the Bed Width the in the cross section
        for a given segment

        Parameters
        ----------
        segmenti : [Integer]
            segment ID.

        Returns
        -------
        crosssections: [dataframe attribute]
            the "b" column in the crosssections attribute will be smoothed

        """
        g = self.crosssections.loc[self.crosssections['id']==segmenti,:].index[0]

        segment = self.crosssections.loc[self.crosssections['id']==segmenti,:]
        segment.index = range(len(segment))


        for j in range(1,len(segment)):
            if segment.loc[j-1,'gl'] - segment.loc[j,'gl'] < height:
                segment.loc[j,'gl'] = segment.loc[j-1,'gl'] - height

        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections['id']==segmenti,:] = segment







    def SmoothMaxSlope(self,segmenti, SlopePercentThreshold = 1.5):
        """
        ========================================================
              SmoothMaxSlope(segmenti,SlopePercentThreshold = 1.5)
        ========================================================
        SmoothMaxSlope method smoothes the bed level the in the cross section
        for a given segment

        As now the slope is not very smoothed as it was when using the average slope
        everywhere, when the the difference between two consecutive slopes is very high,
        the difference is reflected in the calculated discharge from both cross section

        Qout is very high
        Qin is smaller compared to Qout3
        and from the continuity equation the amount of water that stays at the cross-section
        is very few water(Qin3-Qout3), less than the minimum depth

        then the minimum depth is assigned at the cross-section, applying the minimum
        depth in all time steps will make the delta A / delta t equals zero
        As a result, the integration of delta Q/delta x will give a constant discharge
        for all the downstream cross-section.

        To overcome this limitation, a manual check is performed during the calibration
        process by visualizing the hydrographs of the first and last cross-section in
        the sub-basin and the water surface profile to make sure that the algorithm
        does not give a constant discharge.

        Parameters
        ----------
        segmenti : [Integer]
            segment ID.
        SlopePercentThreshold  : [Float]
             the percent of change in slope between three successive  cross sections
             The default is 1.5.

        Returns
        -------
        crosssections: [dataframe attribute]
            the "gl" column in the crosssections attribute will be smoothed

        """
        # segments = list(set(XS['id']))
        # SlopePercentThreshold = 1.5
        # g = 0

        # for i in range(len(segments)):
        #-------
        # i=30
        g = self.crosssections.loc[self.crosssections['id']==segmenti,:].index[0]
        #-------
        # segmenti = segments[i]
        segment = self.crosssections.loc[self.crosssections['id']==segmenti,:]
        segment.index = range(len(segment))
        # slope must be positive due to the smoothing
        slopes = [(segment.loc[k,'gl']-segment.loc[k+1,'gl'])/500 for k in range(len(segment)-1)]
        # if percent is -ve means second slope is steeper
        precent = [(slopes[k] - slopes[k+1])/ slopes[k] for k in range(len(slopes)-1)]

        # at row 1 in precent list is difference between row 1 and row 2 in slopes list
        # and slope in row 2 is the steep slope,
        # slope at row 2 is the difference
        # between gl in row 2 and row 3 in the segment dataframe, and gl row 3 is very
        # and we want to elevate it to reduce the slope
        for j in range(len(precent)):
            if precent[j] < 0 and abs(precent[j]) >= SlopePercentThreshold:
                print(j)
                # get the calculated slope based on the slope percent threshold
                slopes[j+1] = slopes[j] - (-SlopePercentThreshold * slopes[j])
                segment.loc[j+2,'gl'] = segment.loc[j+1,'gl'] - slopes[j+1] * 500
                # recalculate all the slopes again
                slopes = [(segment.loc[k,'gl']-segment.loc[k+1,'gl'])/500 for k in range(len(segment)-1)]
                precent = [(slopes[k] - slopes[k+1])/ slopes[k] for k in range(len(slopes)-1)]

        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections['id']==segmenti,:] = segment
        # g = g + len(segment)

    def CheckFloodplain(self):
        """
        =================================================
               CheckFloodplain(self)
        =================================================
        CheckFloodplain method check if the dike levels is higher than the
        floodplain height (point 5 and 6 has to be lower than point 7 and 8
                           in the cross sections)

        Returns
        -------
        crosssection : [dataframe attribute]
            the "zl" and "zr" column in the "crosssections" attribute will be updated
        """
        assert hasattr(self, "crosssections"), "please read the cross section first or copy it to the Calibration object"
        for i in range(len(self.crosssections)):
            BankLevel = self.crosssections.loc[i,'gl']+ self.crosssections.loc[i,'dbf']

            if BankLevel + self.crosssections.loc[i,'hl'] > self.crosssections.loc[i,'zl']:
                self.crosssections.loc[i,'zl'] = BankLevel + self.crosssections.loc[i,'hl'] + 0.5
            if BankLevel + self.crosssections.loc[i,'hr'] > self.crosssections.loc[i,'zr']:
                self.crosssections.loc[i,'zr'] = BankLevel + self.crosssections.loc[i,'hr'] + 0.5

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
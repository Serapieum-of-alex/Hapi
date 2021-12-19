import datetime as dt
from typing import Union  # List, Optional,

import numpy as np
import pandas as pd
from loguru import logger

from Hapi.hm.river import River

datafn = lambda x: dt.datetime.strptime(x, "%Y-%m-%d")


class Calibration(River):
    """Hydraulic model Calibration.

    Hydraulic model calibration class

    """

    def __init__(self, name: str, version: int=3, start: Union[str, dt.datetime]="1950-1-1",
                 days=36890, fmt="%Y-%m-%d", rrmstart: str="", rrmdays: int=36890,):
        """HMCalibration.

        To instantiate the HMCalibration object you have to provide the
        following arguments

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
            (default number of days are equivalent to 100 years)
        fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"
        rrmstart : [str], optional
            the start date of the rainfall-runoff data. The default is
            "1950-1-1".
        rrmdays : [integer], optional
            the length of the data of the rainfall-runoff data in days.
            The default is 36890.
        Returns
        -------
        None.

        """
        self.name = name
        self.version = version
        if isinstance(start, str):
            self.start = dt.datetime.strptime(start, fmt)
        self.end = self.start + dt.timedelta(days=days)
        self.days = days

        Ref_ind = pd.date_range(self.start, self.end, freq="D")
        self.ReferenceIndex = pd.DataFrame(index=list(range(1, days + 1)))
        self.ReferenceIndex["date"] = Ref_ind[:-1]

        if rrmstart == "":
            self.rrmstart = self.start
        else:
            try:
                self.rrmstart = dt.datetime.strptime(rrmstart, fmt)
            except ValueError:
                msg = (
                    "plese check the fmt ({0}) you entered as it is different from the"
                    " rrmstart data ({1})"
                )
                print(msg.format(fmt, rrmstart))
                return

        self.rrmend = self.rrmstart + dt.timedelta(days=rrmdays)
        ref_ind = pd.date_range(self.rrmstart, self.rrmend, freq="D")
        self.rrmreferenceindex = pd.DataFrame(index=list(range(1, rrmdays + 1)))
        self.rrmreferenceindex["date"] = ref_ind[:-1]

        self.QHM = None #ReadHMQ
        self.WLHM = None #ReadHMWL
        self.QRRM = None # ReadRRM
        self.QRRM2 = None # ReadRRM
        self.RRM_Gauges = [] # ReadRRM

        self.GaugesTable = None
        self.QGauges = None
        self.WLGauges = None

        self.CalibrationQ = None
        self.CalibrationWL = None
        self.AnnualMaxObsQ = None
        self.AnnualMaxObsWL = None
        self.AnnualMaxRRM = None
        self.AnnualMaxRIMQ = None
        self.AnnualMaxRIMWL = None


    def ReadGaugesTable(self, path: str):
        """ReadGaugesTable.

        ReadGaugesTable reads the table of the gauges

        Parameters
        ----------
        path : [String]
            the path to the text file of the gauges table.

        Returns
        -------
        GaugesTable: [dataframe attribute]
            the table will be read in a dataframe attribute

        """
        self.GaugesTable = pd.read_csv(path)


    def ReadObservedWL(self, path: str, start: Union[str, dt.datetime],
                       end: Union[str, dt.datetime], novalue: Union[int, float],
                       column="oid", fmt="%Y-%m-%d", file_extension: str=".txt",
                       gauge_date_format="%Y-%m-%d"):
        """ReadObservedWL.

        read the water level data of the gauges.

        Parameters
        ----------
            2-path : [String]
                  path to the folder containing the text files of the water
                  level gauges
            3-start : [datetime object/str]
                the starting date of the water level time series.
            4-end : [datetime object/str]
                the end date of the water level time series.
            5-novalue : [integer/float]
                value used to fill the missing values.
            6-column : [str/numeric]
                the id of the column containing the name of the files.default
                is "oid".
            7-fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"

        Returns
        -------
            1-WLGauges: [dataframe attiribute].
                dataframe containing the data of the water level gauges and
                the index as the time series from the StartDate till the end
                and the gaps filled with the NoValue
            2-GaugesTable:[dataframe attiribute].
                in the the GaugesTable dataframe two new columns are inserted
                ["WLstart", "WLend"] for the start and end date of the time
                series.
        """
        if isinstance(start, str):
            start = dt.datetime.strptime(start, fmt)
        if isinstance(end, str):
            end = dt.datetime.strptime(end, fmt)

        columns = self.GaugesTable[column].tolist()

        ind = pd.date_range(start, end)
        Gauges = pd.DataFrame(index=ind)
        Gauges.loc[:, 0] = ind
        print("Reading water level gauges data")
        for i in range(len(columns)):
            if self.GaugesTable.loc[i, "waterlevel"] == 1:
                name = self.GaugesTable.loc[i, column]
                try:
                    f = pd.read_csv(
                        path + str(int(name)) + file_extension, delimiter=",", header=0
                    )
                    f.columns = [0, 1]
                    f[0] = f[0].map(lambda x: dt.datetime.strptime(x, gauge_date_format))
                    # sort by date as some values are missed up
                    f.sort_values(by=[0], ascending=True, inplace=True)
                    # filter to the range we want
                    f = f.loc[f[0] >= ind[0], :]
                    f = f.loc[f[0] <= ind[-1], :]
                    # reindex
                    f.index = list(range(len(f)))
                    # add datum and convert to meter
                    f.loc[f[1] != novalue, 1] = (
                                                        f.loc[f[1] != novalue, 1] / 100
                                                ) + self.GaugesTable.loc[i, "datum(m)"]
                    f = f.rename(columns={1: columns[i]})

                    # use merge as there are some gaps in the middle
                    Gauges = Gauges.merge(f, on=0, how="left", sort=False)

                    print(f"{i} - {path}{name}{file_extension} is read")

                except FileNotFoundError:
                    print(f"{i} - {path}{name}{file_extension} has a problem")
                    return

        Gauges.replace(to_replace=np.nan, value=novalue, inplace=True)
        Gauges.index = ind
        del Gauges[0]
        self.WLGauges = Gauges

        self.GaugesTable["WLstart"] = 0
        self.GaugesTable["WLend"] = 0
        for i in range(len(columns)):
            if self.GaugesTable.loc[i, "waterlevel"] == 1:
                st1 = self.WLGauges[columns[i]][self.WLGauges[columns[i]] != novalue].index[0]
                end1 = self.WLGauges[columns[i]][self.WLGauges[columns[i]] != novalue].index[-1]
                self.GaugesTable.loc[i, "WLstart"] = st1
                self.GaugesTable.loc[i, "WLend"] = end1

    # @staticmethod
    # def readfile(path,date_format):
    #
    #     ind = pd.date_range(start, end)
    #     Gauges = pd.DataFrame(index=ind)
    #     Gauges.loc[:, 0] = ind
    #     print("Reading discharge gauges data")
    #     for i in range(len(self.GaugesTable)):
    #         if self.GaugesTable.loc[i, "discharge"] == 1:
    #             name = self.GaugesTable.loc[i, column]
    #             try:
    #                 f = pd.read_csv(path, delimiter=",", header=0)
    #                 print(f"{i} - {path} is read")
    #
    #             except FileNotFoundError:
    #                 print(f"{i} - {path} has a problem")
    #                 return
    #             f.columns = [0, 1]
    #             f[0] = f[0].map(lambda x: dt.datetime.strptime(x, date_format))
    #             # sort by date as some values are missed up
    #             f.sort_values(by=[0], ascending=True, inplace=True)
    #             # filter to the range we want
    #             f = f.loc[f[0] >= ind[0], :]
    #             f = f.loc[f[0] <= ind[-1], :]
    #             # reindex
    #             f.index = list(range(len(f)))
    #             f = f.rename(columns={1: name})
    #
    #             # use merge as there are some gaps in the middle
    #             Gauges = Gauges.merge(f, on=0, how="left", sort=False)
    #
    #     Gauges.replace(to_replace=np.nan, value=novalue, inplace=True)
    #     Gauges.index = ind
    #     del Gauges[0]


    def ReadObservedQ(self, path: str, start: Union[str, dt.datetime],
                      end: Union[str, dt.datetime], novalue: Union[int, float],
                      column: str="oid", fmt: str="%Y-%m-%d", file_extension: str=".txt",
                      gauge_date_format="%Y-%m-%d"):
        """ReadObservedQ.

        ReadObservedQ method reads discharge data and store it in a dataframe
        attribute "QGauges"

        Parameters
        ----------
        1-path : [String]
            path to the folder where files for the gauges exist.
        2-start : [datetime object]
            starting date of the time series.
        3-end : [datetime object]
            ending date of the time series.
        4-novalue : [numeric]
            value stored in gaps.
        5-column : [string/numeric]
            name of the column that contains the gauges file name
        7-fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"

        Returns
        -------
        QGauges:[dataframe attribute]
            dataframe containing the hydrograph of each gauge under a column
             by the name of  gauge.
        GaugesTable:[dataframe attribute]
            in the GaugesTable dataframe two new columns are inserted
            ["Qstart", "Qend"] containing the start and end date of the
            discharge time series.
        """
        if isinstance(start, str):
            start = dt.datetime.strptime(start, fmt)
        if isinstance(end, str):
            end = dt.datetime.strptime(end, fmt)

        ind = pd.date_range(start, end)
        Gauges = pd.DataFrame(index=ind)
        Gauges.loc[:, 0] = ind
        print("Reading discharge gauges data")
        for i in range(len(self.GaugesTable)):
            if self.GaugesTable.loc[i, "discharge"] == 1:
                name = self.GaugesTable.loc[i, column]
                try:
                    f = pd.read_csv(
                        path + str(int(name)) + file_extension, delimiter=",", header=0
                    )
                    print(f"{i} - {path}{name}{file_extension} is read")

                except FileNotFoundError:
                    print(f"{i} - {path}{name}{file_extension} has a problem")
                    return
                f.columns = [0,1]
                f[0] = f[0].map(lambda x: dt.datetime.strptime(x, gauge_date_format))
                # sort by date as some values are missed up
                f.sort_values(by=[0], ascending=True, inplace=True)
                # filter to the range we want
                f = f.loc[f[0] >= ind[0], :]
                f = f.loc[f[0] <= ind[-1], :]
                # reindex
                f.index = list(range(len(f)))
                f = f.rename(columns={1: name})

                # use merge as there are some gaps in the middle
                Gauges = Gauges.merge(f, on=0, how="left", sort=False)

        Gauges.replace(to_replace=np.nan, value=novalue, inplace=True)
        Gauges.index = ind
        del Gauges[0]
        # try:
        #     QGauges.loc[:, int(name)] = np.loadtxt(
        #         path + str(int(name)) + file_extension
        #     )  # ,skiprows = 0
        #
        #     print(f"{i} - {path}{name}{file_extension} is read")
        # except FileNotFoundError:
        #     print(f"{i} - {path}{name}{file_extension} has a problem")
        #     return
        # name = self.GaugesTable.loc[i, column]

        self.QGauges = Gauges
        self.GaugesTable["Qstart"] = 0
        self.GaugesTable["Qend"] = 0

        for i in range(len(self.GaugesTable)):
            if self.GaugesTable.loc[i, "discharge"] == 1:
                ii = self.GaugesTable.loc[i, column]
                st1 = self.QGauges[ii][self.QGauges[ii] != novalue].index[0]
                end1 = self.QGauges[ii][self.QGauges[ii] != novalue].index[-1]
                self.GaugesTable.loc[i, "Qstart"] = st1
                self.GaugesTable.loc[i, "Qend"] = end1


    def ReadRRM(self, path: str, fromday: Union[str, int]="", today: Union[str, int]="",
                column="oid", fmt="%Y-%m-%d", location=1, path2=""):
        """ReadRRM.

        ReadRRM method reads the discharge results of the rainfall runoff
        model and store it in a dataframe attribute "QRRM"

        Parameters
        ----------
        1-path : [String]
            path to the folder where files for the gauges exist.
        2-start : [datetime object]
            starting date of the time series.
        3-end : [datetime object]
            ending date of the time series.
        4-column : [string]
            name of the column that contains the gauges file name. default is
            'oid'.
        5-fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"

        Returns
        -------
        1-QRRM : [dataframe]
            rainfall-runoff discharge time series stored in a dataframe with
            the columns as the gauges id and the index are the time.
        """
        gauges = self.GaugesTable.loc[self.GaugesTable["discharge"] == 1, column].tolist()

        self.QRRM = pd.DataFrame()
        if location == 2:
            # create a dataframe for the 2nd time series of the rainfall runoff
            # model at the second location
            self.QRRM2 = pd.DataFrame()

        self.RRM_Gauges = []
        if path == "":
            path = self.rrmpath

        if location == 2:
            assert (
                not path2 == ""
            ), "path2 argument has to be given fot the location of the 2nd rainfall run-off time series"

        if location == 1:
            for i in range(len(gauges)):
                station_id = gauges[i]
                try:
                    self.QRRM[station_id] = self.ReadRRMResults(
                        self.version,
                        self.rrmreferenceindex,
                        path,
                        station_id,
                        fromday,
                        today,
                        date_format=fmt,
                    )[station_id].tolist()
                    print(f"{i} - {path}{station_id}.txt is read")
                    self.RRM_Gauges.append(station_id)
                except FileNotFoundError:
                    print(f"{i} - {path}{station_id}.txt does not exist or have a problem")
        else:
            for i in range(len(gauges)):
                station_id = gauges[i]
                try:
                    self.QRRM[station_id] = self.ReadRRMResults(
                        self.version,
                        self.rrmreferenceindex,
                        path,
                        station_id,
                        fromday,
                        today,
                        date_format=fmt,
                    )[station_id].tolist()
                    self.QRRM2[station_id] = self.ReadRRMResults(
                        self.version,
                        self.rrmreferenceindex,
                        path2,
                        station_id,
                        fromday,
                        today,
                        date_format=fmt,
                    )[station_id].tolist()
                    print(f"{i} - {path}{station_id}.txt is read")
                    self.RRM_Gauges.append(station_id)
                except FileNotFoundError:
                    print(f"{i} - {path}{station_id}.txt does not exist or have a problem")
        # print("RRM time series for the gauge " + str(station_id) + " is read")

        if fromday == "":
            fromday = 1
        if today == "":
            today = len(self.QRRM[self.QRRM.columns[0]])

        start = self.ReferenceIndex.loc[fromday, "date"]
        end = self.ReferenceIndex.loc[today, "date"]

        self.QRRM.index = pd.date_range(start, end, freq="D")
        self.QRRM2.index = pd.date_range(start, end, freq="D")


    def ReadHMQ(self, path: str, fromday: Union[str, int]="", today: Union[str, int]="",
                novalue: Union[int, float]=-9, addHQ2: bool = False, shift: bool =False,
                shiftsteps: int = 0, column: str = "oid", fmt: str = "%Y-%m-%d",
    ):
        """ReadRIMQ.

        Read Hydraulic model discharge time series.

        Parameters
        ----------
        path : [String]
            path to the folder where files for the gauges exist.
        fromday : [datetime object/str]
            starting date of the time series.
        today : [integer]
            length of the simulation (how many days after the start date) .
        novalue : [numeric value]
            the value used to fill the gaps in the time series or to fill the
            days that is not simulated (discharge is less than threshold).
        addHQ2 : [bool]
            for version 1 the HQ2 can be added to the simulated hydrograph to
            compare it with the gauge data.default is False.
        shift : [bool]
            boolean value to decide whither to shift the time series or not.
            default is False.
        shiftsteps : [integer]
            number of time steps to shift the time series, could be negative
            integer to shift the time series beackward. default is 0.
        column : [string]
            name of the column that contains the gauges file name. default is
            'oid'.
        fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"

        Returns
        -------
        QHM : [dataframe attribute]
            dataframe containing the simulated hydrograph for each river
            segment in the catchment.
        """
        if addHQ2 and self.version == 1:
            msg = "please read the traceall file using the RiverNetwork method"
            assert hasattr(self, "rivernetwork"), msg
            msg = "please read the HQ file first using ReturnPeriod method"
            assert hasattr(self, "RP"), msg

        gauges = self.GaugesTable.loc[self.GaugesTable["discharge"] == 1, column].tolist()

        self.QHM = pd.DataFrame()

        # for RIM1.0 don't fill with -9 as the empty days will be filled
        # with 0 so to get the event days we have to filter 0 and -9
        # if self.version == 1:
        #     QHM.loc[:, :] = 0
        # else:
        #     QHM.loc[:, :] = novalue

        # fill non modelled time steps with zeros
        for i in range(len(gauges)):
            nodeid = gauges[i]
            self.QHM[nodeid] = self.ReadRRMResults(self.version,
                                                   self.ReferenceIndex,
                                                   path,
                                                   nodeid,
                                                   fromday='',
                                                   today='',
                                                   date_format=fmt,
                                                   )[nodeid].tolist()
            print(f"{i} - {path}{nodeid}.txt is read")

            if addHQ2 and self.version == 1:
                USnode = self.rivernetwork.loc[
                    np.where(
                        self.rivernetwork["id"] == self.GaugesTable.loc[i, column]
                    )[0][0],
                    "us",
                ]

                CutValue = self.RP.loc[np.where(self.RP["node"] == USnode)[0][0], "HQ2"]

            # for j in range(len(f1)):
            #     # if the index exist in the original list
            #     if f1[j] in f[:, 0]:
            #         # put the coresponding value in f2
            #         f2.append(f[np.where(f[:, 0] == f1[j])[0][0], 1])
            #     else:
            #         # if it does not exist put zero
            #         if addHQ2 and self.version == 1:
            #             f2.append(CutValue)
            #         else:
            #             f2.append(0)

            # if shift:
            #     f2[shiftsteps:-1] = f2[0 : -(shiftsteps + 1)]

            # QHM.loc[ind[f1[0] - 1] : ind[f1[-1] - 1], QHM.columns[i]] = f2
        if fromday == "":
            fromday = 1
        if today == "":
            today = len(self.QHM[self.QHM.columns[0]])

        start = self.ReferenceIndex.loc[fromday, "date"]
        end = self.ReferenceIndex.loc[today, "date"]

        self.QHM.index = pd.date_range(start, end, freq="D")


    def ReadHMWL(self, path: str, fromday: Union[str, int]="", today: Union[str, int]="",
                 novalue: Union[int, float]=-9, shift=False, shiftsteps=0,
                 column: str = "oid", fmt: str = "%Y-%m-%d"):
        """ReadRIMWL

        Parameters
        ----------
        1-path : [String]
            path to the folder where files for the gauges exist.
        2-start : [datetime object/str]
            starting date of the time series.
        3-days : [integer]
            length of the simulation (how many days after the start date) .
        4-novalue : [numeric value]
            the value used to fill the gaps in the time series or to fill the
            days that is not simulated (discharge is less than threshold).
        5-shift : [bool]
            boolean value to decide whither to shift the time series or not.
            default is False.
        6-shiftsteps : [integer]
            number of time steps to shift the time series, could be negative
            integer to shift the time series beackward. default is 0.
        7-column : [string]
            name of the column that contains the gauges file name. default is
            'oid'.
        8-fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"

        Returns
        -------
        WLHM : [dataframe attribute]
            dataframe containing the simulated water level hydrograph for
            each river segment in the catchment.

        """
        gauges = self.GaugesTable.loc[self.GaugesTable["waterlevel"] == 1, column].tolist()
        self.WLHM = pd.DataFrame()
        # WLHM.loc[:, :] = novalue
        for i in range(len(gauges)):
            nodeid = gauges[i]
            self.WLHM[nodeid] = self.ReadRRMResults(self.version,
                                                   self.ReferenceIndex,
                                                   path,
                                                   nodeid,
                                                   fromday='',
                                                   today='',
                                                   date_format=fmt,
                                                   )[nodeid].tolist()
            print(f"{i} - {path}{nodeid}.txt is read")
        # for i in range(len(WLHM.columns)):
        #     f = np.loadtxt(path + str(int(WLHM.columns[i])) + ".txt", delimiter=",")
        #
        #     f1 = list(range(int(f[0, 0]), int(f[-1, 0]) + 1))
        #     f2 = list()
        #     for j in range(len(f1)):
        #         # if the index exist in the original list
        #         if f1[j] in f[:, 0]:
        #             # put the coresponding value in f2
        #             f2.append(f[np.where(f[:, 0] == f1[j])[0][0], 1])
        #         else:
        #             # if it does not exist put zero
        #             f2.append(0)
        #
        #     if shift:
        #         f2[shiftsteps:-1] = f2[0 : -(shiftsteps + 1)]

            # WLHM.loc[ind[f1[0] - 1] : ind[f1[-1] - 1], WLHM.columns[i]] = f2
        if fromday == "":
            fromday = 1
        if today == "":
            today = len(self.WLHM[self.WLHM.columns[0]])

        start = self.ReferenceIndex.loc[fromday, "date"]
        end = self.ReferenceIndex.loc[today, "date"]

        self.WLHM.index = pd.date_range(start, end, freq="D")


    def ReadCalirationResult(self, subid, path: str=""):
        """ReadCalirationResult.

        ReadCalirationResult method reads the 1D results and fill the missing
        days in the middle

        Parameters
        ----------
        1-subid : [integer]
            ID of the sub-basin you want to read its data.
        4-path : [String], optional
            Path to read the results from. The default is ''.

        Returns
        -------
        1-CalibrationQ : [dataframe]
            the discharge time series of the  calibrated gauges
        2-CalibrationWL : [dataframe]
            the water level time series of the  calibrated gauges
        """
        hasattr(self, "QGauges"), "Please read the discharge gauges first"
        hasattr(self, "WlGauges"), "Please read the water level gauges first"

        if not hasattr(self, "CalibrationQ"):
            indD = pd.date_range(self.start, self.end, freq="D")[:-1]
            self.CalibrationQ = pd.DataFrame(index=indD)
        if not hasattr(self, "CalibrationWL"):
            indD = pd.date_range(self.start, self.end, freq="D")[:-1]
            self.CalibrationWL = pd.DataFrame(index=indD)

        ind = pd.date_range(self.start, self.end, freq="H")[:-1]
        q = pd.read_csv(path + str(subid) + "_q.txt", header=None, delimiter=r"\s+")
        wl = pd.read_csv(path + str(subid) + "_wl.txt", header=None, delimiter=r"\s+")

        q.index = ind
        wl.index = ind

        self.CalibrationQ[subid] = q[1].resample("D").mean()
        self.CalibrationWL[subid] = wl[1].resample("D").mean()

    def GetAnnualMax(
        self, option=1, CorespondingTo=dict(MaxObserved=" ", TimeWindow=0)
    ):
        """GetAnnualMax.

        GetAnnualMax method get the max annual time series out of time series
        of any temporal resolution, the code assumes that the hydrological
        year is 1-Nov/31-Oct (Petrow and Merz, 2009, JoH).

        Parameters
        ----------
        1-option : [integer], optional
            1 for the historical observed Discharge data, 2 for the historical
            observed water level data, 3 for the rainfall-runoff data,
            4 for the rim discharge result, 5 for the rim water level result.
            The default is 1.

        2-CorespondingTo: [Dict], optional
            -if you want to extract the max annual values from the observed
            discharge time series (CorespondingTo=dict(MaxObserved = "Q") and
            then extract the values of the same dates in the result time
            series the same for observed water level time series
            (CorespondingTo=dict(MaxObserved = "WL").
             or if you just want to extract the max annual time values from
             each time series (CorespondingTo=dict(MaxObserved = " ").
             The default is " ".

            - if you want to extract some values before and after the
            coresponding date and then take the max value of all extracted
            values specify the number of days using the keyword Window
            CorespondingTo=dict(TimeWindow =  1)

        Returns
        -------
        1-AnnualMaxObsQ: [dataframe attribute]
            when using Option=1
        2-AnnualMaxObsWL: [dataframe attribute]
            when using option = 2
        3-AnnualMaxRRM: [dataframe attribute]
            when using option = 3
        4-AnnualMaxRIMQ: [dataframe attribute]
            when using option = 4
        5-AnnualMaxRIMWL: [dataframe attribute]
            when using option = 5
        """
        if option == 1:
            msg = """ please read the observed Discharge data first with the
            ReadObservedQ method """
            assert hasattr(self, "QGauges"), "{0}".format(msg)
            columns = self.QGauges.columns.tolist()
        elif option == 2:
            msg = """please read the observed Water level data first with the
            ReadObservedWL method"""
            assert hasattr(self, "WLGauges"), "{0}".format(msg)
            columns = self.WLGauges.columns.tolist()
        elif option == 3:
            msg = """ please read the Rainfall-runoff data first with the
            ReadRRM method"""
            assert hasattr(self, "QRRM"), "{0}".format(msg)
            columns = self.QRRM.columns.tolist()
        elif option == 4:
            msg = """ please read the RIM results first with the ReadRIMQ
            method """
            assert hasattr(self, "QHM"), "{0}".format(msg)
            columns = self.QHM.columns.tolist()
        else:
            msg = "please read the RIM results first with the ReadRIMWL method"
            assert hasattr(self, "WLHM"), "{0}".format(msg)
            columns = self.WLHM.columns.tolist()

        if CorespondingTo["MaxObserved"] == "WL":
            msg = """ please read the observed Water level data first with the
            ReadObservedWL method"""
            assert hasattr(self, "WLGauges"), "{0}".format(msg)

            startdate = self.WLGauges.index[0]
            AnnualMax = (
                self.WLGauges.loc[:, self.WLGauges.columns[0]].resample("A-OCT").max()
            )
            self.AnnualMaxDates = pd.DataFrame(
                index=AnnualMax.index, columns=self.WLGauges.columns
            )

            # get the dates when the max value happen every year
            for i in range(len(self.WLGauges.columns)):
                sub = self.WLGauges.columns[i]
                for j in range(len(AnnualMax)):
                    if j == 0:
                        f = self.WLGauges.loc[startdate : AnnualMax.index[j], sub]
                        self.AnnualMaxDates.loc[AnnualMax.index[j], sub] = f.index[
                            f.argmax()
                        ]
                    else:
                        f = self.WLGauges.loc[
                            AnnualMax.index[j - 1] : AnnualMax.index[j], sub
                        ]
                        self.AnnualMaxDates.loc[AnnualMax.index[j], sub] = f.index[
                            f.argmax()
                        ]

            # extract the values at the dates of the previous max value
            AnnualMax = pd.DataFrame(index=self.AnnualMaxDates.index, columns=columns)

            # Extract time series
            for i in range(len(columns)):
                Sub = columns[i]
                QTS = list()

                if option == 1:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.QGauges.loc[start:end, Sub].max())
                elif option == 2:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=1)
                        end = date + dt.timedelta(days=1)
                        QTS.append(self.WLGauges.loc[start:end, Sub].max())
                elif option == 3:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.QRRM.loc[start:end, Sub].max())
                elif option == 4:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.QHM.loc[start:end, Sub].max())
                else:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.WLHM.loc[start:end, Sub].max())

                AnnualMax.loc[:, Sub] = QTS

        elif CorespondingTo["MaxObserved"] == "Q":
            msg = """ please read the observed Discharge data first with the
            ReadObservedQ method """
            assert hasattr(self, "QGauges"), "{0}".format(msg)

            startdate = self.QGauges.index[0]
            AnnualMax = (
                self.QGauges.loc[:, self.QGauges.columns[0]].resample("A-OCT").max()
            )
            self.AnnualMaxDates = pd.DataFrame(
                index=AnnualMax.index, columns=self.QGauges.columns
            )

            # get the date when the max value happen every year
            for i in range(len(self.QGauges.columns)):
                sub = self.QGauges.columns[i]
                for j in range(len(AnnualMax)):
                    if j == 0:
                        f = self.QGauges.loc[startdate : AnnualMax.index[j], sub]
                        self.AnnualMaxDates.loc[AnnualMax.index[j], sub] = f.index[
                            f.argmax()
                        ]
                    else:
                        f = self.QGauges.loc[
                            AnnualMax.index[j - 1] : AnnualMax.index[j], sub
                        ]
                        self.AnnualMaxDates.loc[AnnualMax.index[j], sub] = f.index[
                            f.argmax()
                        ]

            # extract the values at the dates of the previous max value
            AnnualMax = pd.DataFrame(index=self.AnnualMaxDates.index, columns=columns)
            # Extract time series
            for i in range(len(columns)):
                Sub = columns[i]
                QTS = list()

                if option == 1:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.QGauges.loc[start:end, Sub].max())

                elif option == 2:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.WLGauges.loc[start:end, Sub].max())

                elif option == 3:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.QRRM.loc[start:end, Sub].max())

                elif option == 4:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.QHM.loc[start:end, Sub].max())
                else:

                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.WLHM.loc[start:end, Sub].max())

                # resample to annual time step
                AnnualMax.loc[:, Sub] = QTS
        else:
            AnnualMax = pd.DataFrame(columns=columns)
            # Extract time series
            for i in range(len(columns)):
                Sub = columns[i]
                if option == 1:
                    QTS = self.QGauges.loc[:, Sub]
                elif option == 2:
                    QTS = self.WLGauges.loc[:, Sub]
                elif option == 3:
                    QTS = self.QRRM.loc[:, Sub]
                elif option == 4:
                    QTS = self.QHM.loc[:, Sub]
                else:
                    QTS = self.WLHM.loc[:, Sub]
                # resample to annual time step
                AnnualMax.loc[:, Sub] = QTS.resample("A-OCT").max().values

            AnnualMax.index = QTS.resample("A-OCT").indices.keys()

        if option == 1:
            self.AnnualMaxObsQ = AnnualMax
        elif option == 2:
            self.AnnualMaxObsWL = AnnualMax
        elif option == 3:
            self.AnnualMaxRRM = AnnualMax
        elif option == 4:
            self.AnnualMaxRIMQ = AnnualMax
        else:
            self.AnnualMaxRIMWL = AnnualMax

    def CalculateProfile(self, Segmenti: int, BedlevelDS: float, Manning: float, BC_slope: float):
        """CalculateProfile.

        CalculateProfile method takes the river segment ID and the calibration
        parameters (last downstream cross-section bed level and the manning
        coefficient) and calculates the new profiles.

        Parameters
        ----------
        1-Segmenti : [Integer]
            cross-sections segment ID .
        2-BedlevelDS : [Float]
            the bed level of the last cross section in the segment.
        3-Manning : [float]
            manning coefficient.
        4-BC_slope : [float]
            slope of the BC.

        Returns
        -------
        1-crosssection:[dataframe attribute]
            crosssection attribute will be updated with the newly calculated
            profile for the given segment
        2-slope:[dataframe attribute]
            slope attribute will be updated with the newly calculated average
            slope for the given segment
        """
        levels = pd.DataFrame(columns=["id", "bedlevelUS", "bedlevelDS"])

        # change cross-section
        bedlevel = self.crosssections.loc[
            self.crosssections["id"] == Segmenti, "gl"
        ].values
        # get the bedlevel of the last cross section in the segment
        # as a calibration parameter
        levels.loc[Segmenti, "bedlevelDS"] = BedlevelDS
        levels.loc[Segmenti, "bedlevelUS"] = bedlevel[0]

        NoDistances = len(bedlevel) - 1
        # AvgSlope = ((levels.loc[Segmenti,'bedlevelUS'] -
        #      levels.loc[Segmenti,'bedlevelDS'] )/ (500 * NoDistances)) *-500
        # change in the bed level of the last XS
        AverageDelta = (levels.loc[Segmenti, "bedlevelDS"] - bedlevel[-1]) / NoDistances

        # calculate the new bed levels
        bedlevelNew = np.zeros(len(bedlevel))
        bedlevelNew[len(bedlevel) - 1] = levels.loc[Segmenti, "bedlevelDS"]
        bedlevelNew[0] = levels.loc[Segmenti, "bedlevelUS"]

        for i in range(len(bedlevel) - 1):
            # bedlevelNew[i] = levels.loc[Segmenti,'bedlevelDS'] + (len(bedlevel) - i -1) * abs(AvgSlope)
            bedlevelNew[i] = bedlevel[i] + i * AverageDelta

        self.crosssections.loc[self.crosssections["id"] == Segmenti, "gl"] = bedlevelNew
        # change manning
        self.crosssections.loc[self.crosssections["id"] == Segmenti, "m"] = Manning
        ## change slope
        try:
            # self.slope.loc[self.slope['id']==Segmenti, 'slope'] = AvgSlope
            self.slope.loc[self.slope["id"] == Segmenti, "slope"] = BC_slope
        except AttributeError:
            logger.debug(f"The Given river segment- {Segmenti} does not have a slope")


    def SmoothBedLevel(self, segmenti):
        """SmoothXS

        SmoothBedLevel method smoothes the bed level of a given segment ID by
        calculating the moving average of three cross sections

        Parameters
        ----------
        1-segmenti : [Integer]
            segment ID.

        Returns
        -------
        1-crosssections: [dataframe attribute]
            the "gl" column in the crosssections attribute will be smoothed
        """
        msg = "please read the cross section first"
        assert hasattr(self, "crosssections"), "{0}".format(msg)
        g = self.crosssections.loc[self.crosssections["id"] == segmenti, :].index[0]

        segment = self.crosssections.loc[self.crosssections["id"] == segmenti, :].copy()

        segment.index = range(len(segment))
        segment.loc[:, "glnew"] = 0
        # the bed level at the beginning and end of the egment
        segment.loc[0, "glnew"] = segment.loc[0, "gl"]
        segment.loc[len(segment) - 1, "glnew"] = segment.loc[len(segment) - 1, "gl"]

        # calculate the average of three XS bed level
        for j in range(1, len(segment) - 1):
            segment.loc[j, "glnew"] = (
                segment.loc[j - 1, "gl"]
                + segment.loc[j, "gl"]
                + segment.loc[j + 1, "gl"]
            ) / 3
        # calculate the difference in the bed level and take it from
        # the bankful depth
        segment.loc[:, "diff"] = segment.loc[:, "glnew"] - segment.loc[:, "gl"]
        segment.loc[:, "dbf"] = segment.loc[:, "dbf"] - segment.loc[:, "diff"]
        segment.loc[:, "gl"] = segment.loc[:, "glnew"]
        del segment["glnew"], segment["diff"]

        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections["id"] == segmenti, :] = segment

    def SmoothBankLevel(self, segmenti):
        """SmoothBankLevel.

        SmoothBankLevel method smoothes the bankfull depth for a given segment

        Parameters
        ----------
        1-segmenti : [Integer]
            segment ID.

        Returns
        -------
        1-crosssections: [dataframe attribute]
            the "dbf" column in the crosssections attribute will be smoothed
        """
        self.crosssections.loc[:, "banklevel"] = (
            self.crosssections.loc[:, "dbf"] + self.crosssections.loc[:, "gl"]
        )

        g = self.crosssections.loc[self.crosssections["id"] == segmenti, :].index[0]

        segment = self.crosssections.loc[self.crosssections["id"] == segmenti, :].copy()
        segment.index = range(len(segment))
        segment.loc[:, "banklevelnew"] = 0
        segment.loc[0, "banklevelnew"] = segment.loc[0, "banklevel"]
        segment.loc[len(segment) - 1, "banklevelnew"] = segment.loc[
            len(segment) - 1, "banklevel"
        ]

        for j in range(1, len(segment) - 1):
            segment.loc[j, "banklevelnew"] = (
                segment.loc[j - 1, "banklevel"]
                + segment.loc[j, "banklevel"]
                + segment.loc[j + 1, "banklevel"]
            ) / 3

        segment.loc[:, "diff"] = (
            segment.loc[:, "banklevelnew"] - segment.loc[:, "banklevel"]
        )
        segment.loc[:, "dbf"] = segment.loc[:, "dbf"] + segment.loc[:, "diff"]

        del self.crosssections["banklevel"]
        segment.index = range(g, g + len(segment))

        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections["id"] == segmenti, :] = segment


    def SmoothFloodplainHeight(self, segmenti):
        """SmoothFloodplainHeight.

        SmoothFloodplainHeight method smoothes the Floodplain Height the
        point 5 and 6 in the cross section for a given segment

        Parameters
        ----------
        1-segmenti : [Integer]
            segment ID.

        Returns
        -------
        1-crosssections: [dataframe attribute]
            the "hl" and "hr" column in the crosssections attribute will be
            smoothed.
        """
        self.crosssections.loc[:, "banklevel"] = (
            self.crosssections.loc[:, "dbf"] + self.crosssections.loc[:, "gl"]
        )
        self.crosssections.loc[:, "fpl"] = (
            self.crosssections.loc[:, "hl"] + self.crosssections.loc[:, "banklevel"]
        )
        self.crosssections.loc[:, "fpr"] = (
            self.crosssections.loc[:, "hr"] + self.crosssections.loc[:, "banklevel"]
        )

        g = self.crosssections.loc[self.crosssections["id"] == segmenti, :].index[0]

        segment = self.crosssections.loc[self.crosssections["id"] == segmenti, :].copy()
        segment.index = range(len(segment))

        segment.loc[:, "fplnew"] = 0
        segment.loc[:, "fprnew"] = 0
        segment.loc[0, "fplnew"] = segment.loc[0, "fpl"]
        segment.loc[len(segment) - 1, "fplnew"] = segment.loc[len(segment) - 1, "fpl"]

        segment.loc[0, "fprnew"] = segment.loc[0, "fpr"]
        segment.loc[len(segment) - 1, "fprnew"] = segment.loc[len(segment) - 1, "fpr"]

        for j in range(1, len(segment) - 1):
            segment.loc[j, "fplnew"] = (
                segment.loc[j - 1, "fpl"]
                + segment.loc[j, "fpl"]
                + segment.loc[j + 1, "fpl"]
            ) / 3
            segment.loc[j, "fprnew"] = (
                segment.loc[j - 1, "fpr"]
                + segment.loc[j, "fpr"]
                + segment.loc[j + 1, "fpr"]
            ) / 3

        segment.loc[:, "diff0"] = segment.loc[:, "fplnew"] - segment.loc[:, "fpl"]
        segment.loc[:, "diff1"] = segment.loc[:, "fprnew"] - segment.loc[:, "fpr"]

        segment.loc[:, "hl"] = segment.loc[:, "hl"] + segment.loc[:, "diff0"]
        segment.loc[:, "hr"] = segment.loc[:, "hr"] + segment.loc[:, "diff1"]

        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections["id"] == segmenti, :] = segment

        del (
            self.crosssections["banklevel"],
            self.crosssections["fpr"],
            self.crosssections["fpl"],
        )

    def SmoothBedWidth(self, segmenti):
        """SmoothBedWidth.

        SmoothBedWidth method smoothes the Bed Width the in the cross section
        for a given segment

        Parameters
        ----------
        1-segmenti : [Integer]
            segment ID.

        Returns
        -------
        1-crosssections: [dataframe attribute]
            the "b" column in the crosssections attribute will be smoothed
        """
        g = self.crosssections.loc[self.crosssections["id"] == segmenti, :].index[0]
        segment = self.crosssections.loc[self.crosssections["id"] == segmenti, :].copy()
        segment.index = range(len(segment))
        segment.loc[:, "bnew"] = 0
        segment.loc[0, "bnew"] = segment.loc[0, "b"]
        segment.loc[len(segment) - 1, "bnew"] = segment.loc[len(segment) - 1, "b"]

        for j in range(1, len(segment) - 1):
            segment.loc[j, "bnew"] = (
                segment.loc[j - 1, "b"] + segment.loc[j, "b"] + segment.loc[j + 1, "b"]
            ) / 3

        segment.loc[:, "b"] = segment.loc[:, "bnew"]
        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections["id"] == segmenti, :] = segment


    def DownWardBedLevel(self, segmenti: int, height: Union[int, float]):
        """SmoothBedWidth.

        SmoothBedWidth method smoothes the Bed Width the in the cross section
        for a given segment

        Parameters
        ----------
        segmenti : [Integer]
            segment ID.
        height : []

        Returns
        -------
        crosssections: [dataframe attribute]
            the "b" column in the crosssections attribute will be smoothed


        """
        g = self.crosssections.loc[self.crosssections["id"] == segmenti, :].index[0]

        segment = self.crosssections.loc[self.crosssections["id"] == segmenti, :].copy()
        segment.index = range(len(segment))

        for j in range(1, len(segment)):
            if segment.loc[j - 1, "gl"] - segment.loc[j, "gl"] < height:
                segment.loc[j, "gl"] = segment.loc[j - 1, "gl"] - height

        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections["id"] == segmenti, :] = segment


    def SmoothMaxSlope(self, segmenti, SlopePercentThreshold=1.5):
        """SmoothMaxSlope.

        SmoothMaxSlope method smoothes the bed level the in the cross section
        for a given segment

        As now the slope is not very smoothed as it was when using the average
        slope everywhere, when the the difference between two consecutive
        slopes is very high, the difference is reflected in the calculated
        discharge from both cross section

        Qout is very high
        Qin is smaller compared to Qout3
        and from the continuity equation the amount of water that stays at the
        cross-section is very few water(Qin3-Qout3), less than the minimum
        depth

        then the minimum depth is assigned at the cross-section, applying the
        minimum depth in all time steps will make the delta A / delta t equals
        zero As a result, the integration of delta Q/delta x will give a
        constant discharge for all the downstream cross-section.

        To overcome this limitation, a manual check is performed during the
        calibration process by visualizing the hydrographs of the first and
        last cross-section in the sub-basin and the water surface profile to
        make sure that the algorithm does not give a constant discharge.

        Parameters
        ----------
        1-segmenti : [Integer]
            segment ID.
        2-SlopePercentThreshold  : [Float]
             the percent of change in slope between three successive  cross
             sections. The default is 1.5.

        Returns
        -------
        1-crosssections: [dataframe attribute]
            the "gl" column in the crosssections attribute will be smoothed
        """
        g = self.crosssections.loc[self.crosssections["id"] == segmenti, :].index[0]

        segment = self.crosssections.loc[self.crosssections["id"] == segmenti, :].copy()
        segment.index = range(len(segment))
        # slope must be positive due to the smoothing
        slopes = [
            (segment.loc[k, "gl"] - segment.loc[k + 1, "gl"]) / 500
            for k in range(len(segment) - 1)
        ]
        # if percent is -ve means second slope is steeper
        precent = [
            (slopes[k] - slopes[k + 1]) / slopes[k] for k in range(len(slopes) - 1)
        ]

        # at row 1 in precent list is difference between row 1 and row 2
        # in slopes list and slope in row 2 is the steep slope,
        # slope at row 2 is the difference
        # between gl in row 2 and row 3 in the segment dataframe, and gl row
        # 3 is very and we want to elevate it to reduce the slope
        for j in range(len(precent)):
            if precent[j] < 0 and abs(precent[j]) >= SlopePercentThreshold:
                print(j)
                # get the calculated slope based on the slope percent threshold
                slopes[j + 1] = slopes[j] - (-SlopePercentThreshold * slopes[j])
                segment.loc[j + 2, "gl"] = (
                    segment.loc[j + 1, "gl"] - slopes[j + 1] * 500
                )
                # recalculate all the slopes again
                slopes = [
                    (segment.loc[k, "gl"] - segment.loc[k + 1, "gl"]) / 500
                    for k in range(len(segment) - 1)
                ]
                precent = [
                    (slopes[k] - slopes[k + 1]) / slopes[k]
                    for k in range(len(slopes) - 1)
                ]

        segment.index = range(g, g + len(segment))
        # copy back the segment to the whole XS df
        self.crosssections.loc[self.crosssections["id"] == segmenti, :] = segment

    def CheckFloodplain(self):
        """CheckFloodplain.

        CheckFloodplain method check if the dike levels is higher than the
        floodplain height (point 5 and 6 has to be lower than point 7 and 8
                           in the cross sections)

        Returns
        -------
        crosssection : [dataframe attribute]
            the "zl" and "zr" column in the "crosssections" attribute will be
            updated
        """
        msg = """please read the cross section first or copy it to the
        Calibration object"""
        assert hasattr(self, "crosssections"), "{0}".format(msg)
        for i in range(len(self.crosssections)):
            BankLevel = (
                self.crosssections.loc[i, "gl"] + self.crosssections.loc[i, "dbf"]
            )

            if (
                BankLevel + self.crosssections.loc[i, "hl"]
                > self.crosssections.loc[i, "zl"]
            ):
                self.crosssections.loc[i, "zl"] = (
                    BankLevel + self.crosssections.loc[i, "hl"] + 0.5
                )
            if (
                BankLevel + self.crosssections.loc[i, "hr"]
                > self.crosssections.loc[i, "zr"]
            ):
                self.crosssections.loc[i, "zr"] = (
                    BankLevel + self.crosssections.loc[i, "hr"] + 0.5
                )

    def ListAttributes(self):
        """ListAttributes.

        Print Attributes List
        """
        print("\n")
        print(
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
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")

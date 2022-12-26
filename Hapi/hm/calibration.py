"""Hydaulic model calibration related function module."""
import datetime as dt
from typing import Any, Union  # , Optional,Tuple,

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statista.metrics as pf
from geopandas import GeoDataFrame
from loguru import logger
from matplotlib.figure import Figure

# from pandas import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.frame import DataFrame

from Hapi.hapi_warnings import SilenceNumpyWarning, SilenceShapelyWarning
from Hapi.hm.river import River

datafn = lambda x: dt.datetime.strptime(x, "%Y-%m-%d")

SilenceNumpyWarning()
SilenceShapelyWarning()


class Calibration(River):
    """Hydraulic model Calibration.

    Hydraulic model calibration class
    """

    def __init__(
        self,
        name: str,
        version: int = 3,
        start: Union[str, dt.datetime] = "1950-1-1",
        days: int = 36890,
        fmt: str = "%Y-%m-%d",
        rrmstart: str = "",
        rrmdays: int = 36890,
        novalue: int = -9,
        gauge_id_col: Any = "oid",
    ):
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
        gauge_id_col: [Any]
            the name of the column where the used id of the gauges is stored. Default is 'oid'

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
        self.novalue = novalue
        self.gauge_id_col = gauge_id_col

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
                logger.debug(msg.format(fmt, rrmstart))
                return

        self.rrmend = self.rrmstart + dt.timedelta(days=rrmdays)
        ref_ind = pd.date_range(self.rrmstart, self.rrmend, freq="D")
        self.rrmreferenceindex = pd.DataFrame(index=list(range(1, rrmdays + 1)))
        self.rrmreferenceindex["date"] = ref_ind[:-1]

        self.q_hm = None  # ReadHMQ
        self.WLHM = None  # ReadHMWL
        self.q_rrm = None  # ReadRRM
        self.QRRM2 = None  # ReadRRM
        self.rrm_gauges = None  # ReadRRM

        self.hm_gauges = None
        self.q_gauges = None
        self.WLGauges = None

        self.CalibrationQ = None
        self.CalibrationWL = None
        self.annual_max_obs_q = None
        self.annual_max_obs_wl = None
        self.annual_max_rrm = None
        self.annual_max_hm_q = None
        self.annual_max_hm_wl = None
        self.AnnualMaxDates = None
        self.MetricsHMvsRRM = None
        self.MetricsRRMvsObs = None
        self.MetricsHMWLvsObs = None
        self.MetricsHMQvsObs = None
        self.WLgaugesList = None
        self.QgaugesList = None

    def readGaugesTable(self, path: str):
        """ReadGaugesTable.

        readGaugesTable reads the table of the gauges

        Parameters
        ----------
        path : [String]
            the path to the text file of the gauges table. the file can be geojson or a csv file.
        >>> "gauges.geojson"
        {
        "type": "FeatureCollection", "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3035" } },
        "features": [
        { "type": "Feature", "properties": { "gid": 149, "name": "station 1", "oid": 23800100, "river": "Nile",
        "id": 1, "xsid": 16100, "datum(m)": 252.36, "discharge": 1, "waterlevel": 1 }, "geometry": { "type": "Point", "coordinates": [ 4278240.4259, 2843958.863 ] } },
        { "type": "Feature", "properties": { "gid": 106, "name": "station 2", "oid": 23800500, "river": "Nile",
        "id": 2, "xsid": 16269, "datum(m)": 159.37, "discharge": 1, "waterlevel": 1 }, "geometry": { "type": "Point", "coordinates": [ 4259614.333, 2884750.556 ] } },
        { "type": "Feature", "properties": { "gid": 158, "name": "station 3", "oid": 23800690, "river": "Nile",
        "id": 4, "xsid": 16581, "datum(m)": 119.71, "discharge": 1, "waterlevel": 1}, "geometry": { "type": "Point", "coordinates": [ 4248756.490, 2924872.503 ] } },

        Returns
        -------
        GaugesTable: [dataframe attribute]
            the table will be read in a dataframe attribute

        Examples
        --------
        >>> import Hapi.hm.calibration as RC
        >>> Calib = RC.Calibration("Hydraulic model", gauge_id_col="id")
        >>> Calib.readGaugesTable("path/to/gauges.geojson")
        >>> Calib.hm_gauges
                gid  ...                         geometry
            0   149  ...  POINT (4278240.426 2843958.864)
            1   106  ...  POINT (4259614.334 2884750.556)
        """
        try:
            self.hm_gauges = gpd.read_file(path, driver="GeoJSON")
        except fiona.errors.DriverError:
            self.hm_gauges = pd.read_csv(path)
        # sort the gauges table based on the segment
        self.hm_gauges.sort_values(by="id", inplace=True, ignore_index=True)

    def GetGauges(self, subid: int, gaugei: int = 0) -> DataFrame:
        """GetGauges. Get_Gauge_ID get the id of the station for a given river segment.

        parameters:
        ----------
        subid: [int]
            the river segment id

        return:
        -------
        id: [list/int]
            if the river segment contains more than one gauges the function
            returns a list of ids, otherwise it returns the id.
        gauge name: [str]
            name of the gauge
        gauge xs: [int]
            the nearest cross section to the gauge
        """
        gauges = self.hm_gauges.loc[self.hm_gauges["id"] == subid, :].reset_index()
        if len(gauges) == 0:
            raise KeyError(
                "The given river segment does not have gauges in the gauge table"
            )
        elif len(gauges) > 1:
            f = gauges.loc[gaugei, :].to_frame()
            gauge = pd.DataFrame(index=[0], columns=f.index.to_list())
            gauge.loc[0, :] = f.loc[f.index.to_list(), gaugei].values.tolist()
            return gauge
        else:
            return gauges
        # stationname = gauges.loc[:, column].values.tolist()
        # gaugename = str(gauges.loc[gaugei, 'name'])
        # gaugexs = gauges.loc[gaugei, 'xsid']
        # segment_xs = str(subid) + "_" + str(gaugexs)

        # stationname, gaugename, gaugexs

    def readObservedWL(
        self,
        path: str,
        start: Union[str, dt.datetime],
        end: Union[str, dt.datetime],
        novalue: Union[int, float],
        fmt="%Y-%m-%d",
        file_extension: str = ".txt",
        gauge_date_format="%Y-%m-%d",
    ):
        """ReadObservedWL.

        read the water level data of the gauges.

        Parameters
        ----------
        path : [String]
              path to the folder containing the text files of the water
              level gauges
        start : [datetime object/str]
            the starting date of the water level time series.
        end : [datetime object/str]
            the end date of the water level time series.
        novalue : [integer/float]
            value used to fill the missing values.
        fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"
        file_extension: [str]
            extension of the files. Default is ".txt"
        gauge_date_format: [str]
            format of the date in the first column in the gauges files. Default is "%Y-%m-%d".

        Returns
        -------
        WLGauges: [dataframe attiribute].
            dataframe containing the data of the water level gauges and
            the index as the time series from the StartDate till the end
            and the gaps filled with the NoValue
        hm_gauges:[dataframe attiribute].
            in the the hm_gauges dataframe two new columns are inserted
            ["WLstart", "WLend"] for the start and end date of the time
            series.
        """
        if isinstance(start, str):
            start = dt.datetime.strptime(start, fmt)
        if isinstance(end, str):
            end = dt.datetime.strptime(end, fmt)

        columns = self.hm_gauges[self.gauge_id_col].tolist()

        ind = pd.date_range(start, end)
        Gauges = pd.DataFrame(index=ind)
        Gauges.loc[:, 0] = ind
        logger.debug("Reading water level gauges data")
        for i in range(len(columns)):
            if self.hm_gauges.loc[i, "waterlevel"] == 1:
                name = self.hm_gauges.loc[i, self.gauge_id_col]
                try:
                    f = pd.read_csv(
                        path + str(int(name)) + file_extension, delimiter=",", header=0
                    )
                    f.columns = [0, 1]
                    f[0] = f[0].map(
                        lambda x: dt.datetime.strptime(x, gauge_date_format)
                    )
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
                    ) + self.hm_gauges.loc[i, "datum(m)"]
                    f = f.rename(columns={1: columns[i]})

                    # use merge as there are some gaps in the middle
                    Gauges = Gauges.merge(f, on=0, how="left", sort=False)

                    logger.debug(f"{i} - {path}{name}{file_extension} is read")

                except FileNotFoundError:
                    logger.debug(f"{i} - {path}{name}{file_extension} has a problem")
                    return

        Gauges.replace(to_replace=np.nan, value=novalue, inplace=True)
        Gauges.index = ind
        del Gauges[0]
        self.WLGauges = Gauges

        self.hm_gauges["WLstart"] = 0
        self.hm_gauges["WLend"] = 0
        for i in range(len(columns)):
            if self.hm_gauges.loc[i, "waterlevel"] == 1:
                st1 = self.WLGauges[columns[i]][
                    self.WLGauges[columns[i]] != novalue
                ].index[0]
                end1 = self.WLGauges[columns[i]][
                    self.WLGauges[columns[i]] != novalue
                ].index[-1]
                self.hm_gauges.loc[i, "WLstart"] = st1
                self.hm_gauges.loc[i, "WLend"] = end1

    # @staticmethod
    # def readfile(path,date_format):
    #
    #     ind = pd.date_range(start, end)
    #     Gauges = pd.DataFrame(index=ind)
    #     Gauges.loc[:, 0] = ind
    #     logger.debug("Reading discharge gauges data")
    #     for i in range(len(self.hm_gauges)):
    #         if self.hm_gauges.loc[i, "discharge"] == 1:
    #             name = self.hm_gauges.loc[i, column]
    #             try:
    #                 f = pd.read_csv(path, delimiter=",", header=0)
    #                 logger.debug(f"{i} - {path} is read")
    #
    #             except FileNotFoundError:
    #                 logger.debug(f"{i} - {path} has a problem")
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

    def readObservedQ(
        self,
        path: str,
        start: Union[str, dt.datetime],
        end: Union[str, dt.datetime],
        novalue: Union[int, float],
        fmt: str = "%Y-%m-%d",
        file_extension: str = ".txt",
        gauge_date_format="%Y-%m-%d",
    ):
        """readObservedQ.

            ReadObservedQ method reads discharge data and store it in a dataframe
            attribute "q_gauges"

        Parameters
        ----------
        path : [String]
            path to the folder where files for the gauges exist.
        start : [datetime object]
            starting date of the time series.
        end : [datetime object]
            ending date of the time series.
        novalue : [numeric]
            value stored in gaps.
        fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"
        file_extension: [str]
            extension of the files. Default is ".txt"
        gauge_date_format: [str]
            format of the date in the first column in the gauges files. Default is "%Y-%m-%d".

        Returns
        -------
        q_gauges:[dataframe attribute]
            dataframe containing the hydrograph of each gauge under a column
             by the name of  gauge.
        hm_gauges:[dataframe attribute]
            in the hm_gauges dataframe two new columns are inserted
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
        logger.debug("Reading discharge gauges data")
        for i in range(len(self.hm_gauges)):
            if self.hm_gauges.loc[i, "discharge"] == 1:
                name = self.hm_gauges.loc[i, self.gauge_id_col]
                try:
                    f = pd.read_csv(
                        path + str(int(name)) + file_extension, delimiter=",", header=0
                    )
                    logger.info(f"{i} - {path}{name}{file_extension} is read")

                except FileNotFoundError:
                    logger.debug(f"{i} - {path}{name}{file_extension} has a problem")
                    continue
                f.columns = [0, 1]
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
        #     q_gauges.loc[:, int(name)] = np.loadtxt(
        #         path + str(int(name)) + file_extension
        #     )  # ,skiprows = 0
        #
        #     logger.debug(f"{i} - {path}{name}{file_extension} is read")
        # except FileNotFoundError:
        #     logger.debug(f"{i} - {path}{name}{file_extension} has a problem")
        #     return
        # name = self.hm_gauges.loc[i, column]

        self.q_gauges = Gauges
        self.hm_gauges["Qstart"] = 0
        self.hm_gauges["Qend"] = 0

        for i in range(len(self.hm_gauges)):
            if self.hm_gauges.loc[i, "discharge"] == 1:
                ii = self.hm_gauges.loc[i, self.gauge_id_col]
                st1 = self.q_gauges[ii][self.q_gauges[ii] != novalue].index[0]
                end1 = self.q_gauges[ii][self.q_gauges[ii] != novalue].index[-1]
                self.hm_gauges.loc[i, "Qstart"] = st1
                self.hm_gauges.loc[i, "Qend"] = end1

    def readRRM(
        self,
        path: str,
        fromday: Union[str, int] = "",
        today: Union[str, int] = "",
        fmt: str = "%Y-%m-%d",
        location: int = 1,
        path2: str = "",
    ):
        """ReadRRM.

            ReadRRM method reads the discharge results of the rainfall runoff
            model and store it in a dataframe attribute "q_rrm"

        Parameters
        ----------
        path : [String]
            path to the folder where files for the gauges exist.
        fromday : [datetime object]
            starting date of the time series.
        today : [datetime object]
            ending date of the time series.
        fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"
        location: [int]
            the RRM hydrographs for a 2nd location
        path2 : [str]
            directory where the RRM hydrographs for the 2nd location are saved

        Returns
        -------
        q_rrm : [dataframe]
            rainfall-runoff discharge time series stored in a dataframe with
            the columns as the gauges id and the index are the time.
        rrm_gauges: [list]
            list og gauges id
        """
        gauges = self.hm_gauges.loc[
            self.hm_gauges["discharge"] == 1, self.gauge_id_col
        ].tolist()

        self.q_rrm = pd.DataFrame()
        if location == 2:
            # create a dataframe for the 2nd time series of the rainfall runoff
            # model at the second location
            self.QRRM2 = pd.DataFrame()

        self.rrm_gauges = []
        if path == "":
            path = self.rrmpath

        if location == 2:
            if path2 == "":
                raise ValueError(
                    "path2 argument has to be given for the location of the 2nd rainfall run-off time "
                    "series"
                )

        if location == 1:
            for i in range(len(gauges)):
                station_id = gauges[i]
                try:
                    self.q_rrm[station_id] = self.readRRMResults(
                        self.version,
                        self.rrmreferenceindex,
                        path,
                        station_id,
                        fromday,
                        today,
                        date_format=fmt,
                    )[station_id].tolist()
                    logger.info(f"{i} - {path}{station_id}.txt is read")
                    self.rrm_gauges.append(station_id)
                except FileNotFoundError:
                    logger.info(
                        f"{i} - {path}{station_id}.txt does not exist or have a problem"
                    )
        else:
            for i in range(len(gauges)):
                station_id = gauges[i]
                try:
                    self.q_rrm[station_id] = self.readRRMResults(
                        self.version,
                        self.rrmreferenceindex,
                        path,
                        station_id,
                        fromday,
                        today,
                        date_format=fmt,
                    )[station_id].tolist()
                    self.QRRM2[station_id] = self.readRRMResults(
                        self.version,
                        self.rrmreferenceindex,
                        path2,
                        station_id,
                        fromday,
                        today,
                        date_format=fmt,
                    )[station_id].tolist()
                    logger.info(f"{i} - {path}{station_id}.txt is read")
                    self.rrm_gauges.append(station_id)
                except FileNotFoundError:
                    logger.info(
                        f"{i} - {path}{station_id}.txt does not exist or have a problem"
                    )
        # logger.debug("RRM time series for the gauge " + str(station_id) + " is read")

        if fromday == "":
            fromday = 1
        if today == "":
            today = len(self.q_rrm[self.q_rrm.columns[0]])

        start = self.ReferenceIndex.loc[fromday, "date"]
        end = self.ReferenceIndex.loc[today, "date"]

        if location == 1:
            self.q_rrm.index = pd.date_range(start, end, freq="D")
        else:
            self.q_rrm.index = pd.date_range(start, end, freq="D")
            self.QRRM2.index = pd.date_range(start, end, freq="D")

    def readHMQ(
        self,
        path: str,
        fromday: Union[str, int] = "",
        today: Union[str, int] = "",
        novalue: Union[int, float] = -9,
        addHQ2: bool = False,
        shift: bool = False,
        shiftsteps: int = 0,
        fmt: str = "%Y-%m-%d",
    ):
        """ReadHMQ.

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
        fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"

        Returns
        -------
        q_hm : [dataframe attribute]
            dataframe containing the simulated hydrograph for each river
            segment in the catchment.
        """
        if addHQ2 and self.version == 1:
            msg = "please read the traceall file using the RiverNetwork method"
            assert hasattr(self, "rivernetwork"), msg
            msg = "please read the HQ file first using ReturnPeriod method"
            assert hasattr(self, "RP"), msg

        gauges = self.hm_gauges.loc[
            self.hm_gauges["discharge"] == 1, self.gauge_id_col
        ].tolist()
        self.QgaugesList = gauges
        self.q_hm = pd.DataFrame()

        # for RIM1.0 don't fill with -9 as the empty days will be filled
        # with 0 so to get the event days we have to filter 0 and -9
        # if self.version == 1:
        #     q_hm.loc[:, :] = 0
        # else:
        #     q_hm.loc[:, :] = novalue

        # fill non modelled time steps with zeros
        for i in range(len(gauges)):
            nodeid = gauges[i]
            self.q_hm[nodeid] = self.readRRMResults(
                self.version,
                self.ReferenceIndex,
                path,
                nodeid,
                fromday="",
                today="",
                date_format=fmt,
            )[nodeid].tolist()
            logger.debug(f"{i} - {path}{nodeid}.txt is read")

            if addHQ2 and self.version == 1:
                USnode = self.rivernetwork.loc[
                    np.where(
                        self.rivernetwork["id"]
                        == self.hm_gauges.loc[i, self.gauge_id_col]
                    )[0][0],
                    "us",
                ]

                CutValue = self.RP.loc[np.where(self.RP["node"] == USnode)[0][0], "HQ2"]
                print(CutValue)

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

            # q_hm.loc[ind[f1[0] - 1] : ind[f1[-1] - 1], q_hm.columns[i]] = f2
        if fromday == "":
            fromday = 1
        if today == "":
            today = len(self.q_hm[self.q_hm.columns[0]])

        start = self.ReferenceIndex.loc[fromday, "date"]
        end = self.ReferenceIndex.loc[today, "date"]

        self.q_hm.index = pd.date_range(start, end, freq="D")

    def ReadHMWL(
        self,
        path: str,
        fromday: Union[str, int] = "",
        today: Union[str, int] = "",
        novalue: Union[int, float] = -9,
        shift=False,
        shiftsteps=0,
        column: str = "oid",
        fmt: str = "%Y-%m-%d",
    ):
        """ReadRIMWL.

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
        gauges = self.hm_gauges.loc[
            self.hm_gauges["waterlevel"] == 1, self.gauge_id_col
        ].tolist()
        self.WLgaugesList = gauges

        self.WLHM = pd.DataFrame()
        for i in range(len(gauges)):
            nodeid = gauges[i]
            self.WLHM[nodeid] = self.readRRMResults(
                self.version,
                self.ReferenceIndex,
                path,
                nodeid,
                fromday="",
                today="",
                date_format=fmt,
            )[nodeid].tolist()
            logger.debug(f"{i} - {path}{nodeid}.txt is read")
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

    def readCalirationResult(self, subid: int, path: str = ""):
        """ReadCalirationResult.

        ReadCalirationResult method reads the 1D results and fill the missing
        days in the middle

        Parameters
        ----------
        subid : [integer]
            ID of the sub-basin you want to read its data.
        path : [String], optional
            Path to read the results from. The default is ''.

        Returns
        -------
        CalibrationQ : [dataframe]
            the discharge time series of the  calibrated gauges
        CalibrationWL : [dataframe]
            the water level time series of the  calibrated gauges
        """
        hasattr(self, "q_gauges"), "Please read the discharge gauges first"
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

    def getAnnualMax(
        self, option=1, CorespondingTo=dict(MaxObserved=" ", TimeWindow=0)
    ):
        """getAnnualMax.

        GetAnnualMax method get the max annual time series out of time series
        of any temporal resolution, the code assumes that the hydrological
        year is 1-Nov/31-Oct (Petrow and Merz, 2009, JoH).

        Parameters
        ----------
        option : [integer], optional
            - 1 for the historical observed Discharge data.
            - 2 for the historical observed water level data.
            - 3 for the rainfall-runoff data.
            - 4 for the hm discharge result.
            - 5 for the hm water level result.
            The default is 1.

        CorespondingTo: [Dict], optional
            - if you want to extract the max annual values from the observed
            discharge time series (CorespondingTo=dict(MaxObserved = "Q") and
            then extract the values of the same dates in the simulated time
            series. The same for observed water level time series
            (CorespondingTo=dict(MaxObserved = "WL").
            or if you just want to extract the max annual values from
            each time series (CorespondingTo=dict(MaxObserved = " ").
            The default is " ".

            - if you want to extract some values before and after the
            coresponding date and then take the max value of all extracted
            values specify the number of days using the keyword Window
            CorespondingTo=dict(TimeWindow =  1)

        Returns
        -------
        annual_max_obs_q: [dataframe attribute]
            when using Option=1
        annual_max_obs_wl: [dataframe attribute]
            when using option = 2
        annual_max_rrm: [dataframe attribute]
            when using option = 3
        AnnualMaxRIMQ: [dataframe attribute]
            when using option = 4
        annual_max_hm_wl: [dataframe attribute]
            when using option = 5
        AnnualMaxDates : [dataframe attribute]
        """
        if option == 1:
            if not isinstance(self.q_gauges, DataFrame):
                raise ValueError(
                    "please read the observed Discharge data first with the"
                    "ReadObservedQ method "
                )
            columns = self.q_gauges.columns.tolist()
        elif option == 2:
            if not isinstance(self.WLGauges, DataFrame):
                raise ValueError(
                    "please read the observed Water level data first with the "
                    "ReadObservedWL method"
                )
            columns = self.WLGauges.columns.tolist()
        elif option == 3:
            if not isinstance(self.q_rrm, DataFrame):
                raise ValueError(
                    "please read the Rainfall-runoff data first with the "
                    "ReadRRM method"
                )
            columns = self.q_rrm.columns.tolist()
        elif option == 4:
            if not isinstance(self.q_hm, DataFrame):
                raise ValueError(
                    "please read the RIM results first with the ReadRIMQ method "
                )
            columns = self.q_hm.columns.tolist()
        else:
            if not isinstance(self.WLHM, DataFrame):
                raise ValueError(
                    "please read the RIM results first with the ReadRIMWL method"
                )
            columns = self.WLHM.columns.tolist()

        if CorespondingTo["MaxObserved"] == "WL":
            if not isinstance(self.WLGauges, DataFrame):
                raise ValueError(
                    "please read the observed Water level data first with the "
                    "ReadObservedWL method"
                )

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
                        QTS.append(self.q_gauges.loc[start:end, Sub].max())
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
                        QTS.append(self.q_rrm.loc[start:end, Sub].max())
                elif option == 4:
                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.q_hm.loc[start:end, Sub].max())
                else:
                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.WLHM.loc[start:end, Sub].max())

                AnnualMax.loc[:, Sub] = QTS

        elif CorespondingTo["MaxObserved"] == "Q":
            if not isinstance(self.q_gauges, DataFrame):
                raise ValueError(
                    "please read the observed Discharge data first with the"
                    "ReadObservedQ method"
                )
            startdate = self.q_gauges.index[0]
            AnnualMax = (
                self.q_gauges.loc[:, self.q_gauges.columns[0]].resample("A-OCT").max()
            )
            self.AnnualMaxDates = pd.DataFrame(
                index=AnnualMax.index, columns=self.q_gauges.columns
            )

            # get the date when the max value happen every year
            for i in range(len(self.q_gauges.columns)):
                sub = self.q_gauges.columns[i]
                for j in range(len(AnnualMax)):
                    if j == 0:
                        f = self.q_gauges.loc[startdate : AnnualMax.index[j], sub]
                        self.AnnualMaxDates.loc[AnnualMax.index[j], sub] = f.index[
                            f.argmax()
                        ]
                    else:
                        f = self.q_gauges.loc[
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
                        QTS.append(self.q_gauges.loc[start:end, Sub].max())

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
                        QTS.append(self.q_rrm.loc[start:end, Sub].max())

                elif option == 4:
                    for j in range(len(self.AnnualMaxDates.loc[:, Sub])):
                        ind = self.AnnualMaxDates.index[j]
                        date = self.AnnualMaxDates.loc[ind, Sub]
                        start = date - dt.timedelta(days=CorespondingTo["TimeWindow"])
                        end = date + dt.timedelta(days=CorespondingTo["TimeWindow"])
                        QTS.append(self.q_hm.loc[start:end, Sub].max())
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
                    QTS = self.q_gauges.loc[:, Sub]
                elif option == 2:
                    QTS = self.WLGauges.loc[:, Sub]
                elif option == 3:
                    QTS = self.q_rrm.loc[:, Sub]
                elif option == 4:
                    QTS = self.q_hm.loc[:, Sub]
                else:
                    QTS = self.WLHM.loc[:, Sub]
                # resample to annual time step
                AnnualMax.loc[:, Sub] = QTS.resample("A-OCT").max().values

            AnnualMax.index = QTS.resample("A-OCT").indices.keys()

        if option == 1:
            self.annual_max_obs_q = AnnualMax
        elif option == 2:
            self.annual_max_obs_wl = AnnualMax
        elif option == 3:
            self.annual_max_rrm = AnnualMax
        elif option == 4:
            self.annual_max_hm_q = AnnualMax
        else:
            self.annual_max_hm_wl = AnnualMax

    def CalculateProfile(
        self, Segmenti: int, BedlevelDS: float, Manning: float, BC_slope: float
    ):
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
        """SmoothXS.

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
                logger.debug(j)
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

    @staticmethod
    def Metrics(
        df1: DataFrame,
        df2: DataFrame,
        gauges: list,
        novalue: int,
        start: str = "",
        end: str = "",
        shift: int = 0,
        fmt: str = "%Y-%m-%d",
    ) -> DataFrame:
        """

        Parameters
        ----------
        df1: [DataFrame]
            first dataframe, with columns as the gauges id and rows as the time series
        df2: [DataFrame]
            second dataframe, with columns as the gauges id and rows as the time series
        gauges: [list]
            list of gauges ids
        novalue:
            the value used to fill the missing values
        start:
            start date
        end:
            end date
        shift:
            shift in the days
        fmt:
            date format

        Returns
        -------
        GeoDataFrame:
            with the following columns ["start", "end", "rmse", "KGE", "WB", "NSE", "NSEModified"]
        """
        Metrics = gpd.GeoDataFrame(
            index=gauges,
            columns=["start", "end", "rmse", "KGE", "WB", "NSE", "NSEModified"],
        )

        for i in range(len(gauges)):
            # get the index of the first value in the column that is not -9 or Nan
            st1 = df1.loc[:, df1.columns[i]][
                df1.loc[:, df1.columns[i]] != novalue
            ].index[0]
            st2 = df2.loc[:, df2.columns[i]][
                df2.loc[:, df2.columns[i]] != novalue
            ].index[0]

            Metrics.loc[gauges[i], "start"] = max(st1, st2)
            end1 = df1[df1.columns[i]][df1[df1.columns[i]] != novalue].index[-1]
            end2 = df2[df2.columns[i]][df2[df2.columns[i]] != novalue].index[-1]
            Metrics.loc[gauges[i], "end"] = min(end1, end2)

        # manually adjust and start or end date to calculate the performance between
        # two dates
        if start != "":
            Metrics.loc[:, "start"] = dt.datetime.strptime(start, fmt)
        if end != "":
            Metrics.loc[:, "end"] = dt.datetime.strptime(end, fmt)

        # calculate th metrics
        for i in range(len(gauges)):
            start_date = Metrics.loc[gauges[i], "start"]
            end_date = Metrics.loc[gauges[i], "end"]
            obs = df1.loc[start_date : end_date, gauges[i]].values.tolist()
            sim = df2.loc[start_date : end_date, gauges[i]].values.tolist()

            # shift hm result
            sim[shift:-shift] = sim[0 : -shift * 2]

            Metrics.loc[gauges[i], "rmse"] = round(pf.RMSE(obs, sim), 0)
            Metrics.loc[gauges[i], "KGE"] = round(pf.KGE(obs, sim), 3)
            Metrics.loc[gauges[i], "NSE"] = round(pf.NSE(obs, sim), 3)
            Metrics.loc[gauges[i], "NSEModified"] = round(pf.NSEHF(obs, sim), 3)
            Metrics.loc[gauges[i], "WB"] = round(pf.WB(obs, sim), 0)
            Metrics.loc[gauges[i], "MBE"] = round(pf.MBE(obs, sim), 3)
            Metrics.loc[gauges[i], "MAE"] = round(pf.MAE(obs, sim), 3)

        return Metrics

    def HMvsRRM(
        self, start: str = "", end: str = "", fmt: str = "%Y-%m-%d", shift: int = 0
    ):
        """HMvsRRM.

            HM_vs_RRM calculate the performance metrics between the hydraulic model simulated
            discharge and the rainfall-runoff model simulated discharge

        Parameters
        ----------
        start: [str]
            the stating date for the period you want to calculate the error for.
        end: [str]
            the end date for the period you want to calculate the error for.
        fmt: [str]
            format of the given date
        shift: [int]
            if there is a shift in the clculated timeseries by one or more time
            steps and you want to fix this problem in calculating the metrics by
            shifting the calculated timeseries by one or more time steps.

        Returns
        -------
        MetricsHM_RRM: [dataframe]
            dataframe with the gauges id as rows and ['start', 'end', 'rmse', 'KGE', 'WB', 'NSE',
            'NSEModefied'], as columns.
        """
        if not isinstance(self.hm_gauges, DataFrame) and not isinstance(
            self.hm_gauges, GeoDataFrame
        ):
            raise ValueError(
                "rrm_gauges variable does not exist please read the RRM results "
                "with 'ReadRRM' method"
            )
        if not isinstance(self.rrm_gauges, list):
            raise ValueError(
                "rrm_gauges variable does not exist please read the RRM results "
                "with 'ReadRRM' method"
            )
        if not isinstance(self.q_rrm, DataFrame):
            raise ValueError("please read the RRM results with the 'ReadRRM' method")
        if not isinstance(self.q_hm, DataFrame):
            raise ValueError("please read the HM results with the 'ReadHMQ' method")
        ### HM vs RRM
        self.MetricsHMvsRRM = self.Metrics(
            self.q_rrm, self.q_hm, self.rrm_gauges, self.novalue, start, end, shift, fmt
        )
        # get the point geometry from the hm_gauges
        self.MetricsHMvsRRM = self.hm_gauges.merge(
            self.MetricsHMvsRRM,
            left_on=self.gauge_id_col,
            right_index=True,
            how="left",
            sort=False,
        )#.set_geometry("geometry")
        self.MetricsHMvsRRM.index = self.MetricsHMvsRRM[self.gauge_id_col]
        self.MetricsHMvsRRM.index.name = "index"
        if isinstance(self.hm_gauges, GeoDataFrame):
            self.MetricsHMvsRRM.crs = self.hm_gauges.crs

    def RRMvsObserved(
        self, start: str = "", end: str = "", fmt: str = "%Y-%m-%d", shift: int = 0
    ):
        """RRM_vs_observed.

            HM_vs_RRM calculate the performance metrics between the hydraulic model simulated
            discharge and the rainfall-runoff model simulated discharge

        Parameters
        ----------
        start: [str]
            the stating date for the period you want to calculate the error for.
        end: [str]
            the end date for the period you want to calculate the error for.
        fmt: [str]
            format of the given date
        shift: [int]
            if there is a shift in the clculated timeseries by one or more time
            steps and you want to fix this problem in calculating the metrics by
            shifting the calculated timeseries by one or more time steps.

        Returns
        -------
        MetricsHM_RRM: [dataframe]
            dataframe with the gauges id as rows and ['start', 'end', 'rmse', 'KGE', 'WB', 'NSE',
            'NSEModefied'], as columns.
        """
        if not isinstance(self.rrm_gauges, list):
            raise ValueError(
                "rrm_gauges variable does not exist please read the RRM results "
                "with 'ReadRRM' method"
            )
        if not isinstance(self.q_rrm, DataFrame):
            raise ValueError("please read the RRM results with the 'ReadRRM' method")

        if not isinstance(self.q_gauges, DataFrame):
            raise ValueError(
                "q_gauges variable does not exist please read the gauges data "
                "with 'ReadObservedQ' method"
            )

        ### RRM vs observed
        self.MetricsRRMvsObs = self.Metrics(
            self.q_gauges,
            self.q_rrm,
            self.rrm_gauges,
            self.novalue,
            start,
            end,
            shift,
            fmt,
        )

        self.MetricsRRMvsObs = self.hm_gauges.merge(
            self.MetricsRRMvsObs,
            left_on=self.gauge_id_col,
            right_index=True,
            how="left",
            sort=False,
        )

        self.MetricsRRMvsObs.index = self.MetricsRRMvsObs[self.gauge_id_col]
        self.MetricsRRMvsObs.index.name = "index"
        if isinstance(self.hm_gauges, GeoDataFrame):
            self.MetricsRRMvsObs.crs = self.hm_gauges.crs

    def HMQvsObserved(
        self,
        start: str = "",
        end: str = "",
        fmt: str = "%Y-%m-%d",
        shift: int = 0,
    ):
        """HMQvsObserved.

            HM_vs_RRM calculate the performance metrics between the hydraulic model simulated
            discharge and the rainfall-runoff model simulated discharge

        Parameters
        ----------
        start: [str]
            the stating date for the period you want to calculate the error for.
        end: [str]
            the end date for the period you want to calculate the error for.
        fmt: [str]
            format of the given date
        shift: [int]
            if there is a shift in the clculated timeseries by one or more time
            steps and you want to fix this problem in calculating the metrics by
            shifting the calculated timeseries by one or more time steps.

        Returns
        -------
        MetricsHMQvsObs: [dataframe]
            dataframe with the gauges id as rows and ['start', 'end', 'rmse', 'KGE', 'WB', 'NSE',
            'NSEModefied'], as columns.
        """
        if not isinstance(self.q_gauges, DataFrame):
            raise ValueError(
                "q_gauges variable does not exist please read the gauges data "
                "with 'ReadObservedQ' method"
            )

        if not isinstance(self.q_hm, DataFrame):
            raise ValueError(
                "q_hm variable does not exist please read the HM results "
                "with 'ReadHMQ' method"
            )

        ### HM Q vs observed
        self.MetricsHMQvsObs = self.Metrics(
            self.q_gauges,
            self.q_hm,
            self.QgaugesList,
            self.novalue,
            start,
            end,
            shift,
            fmt,
        )

        self.MetricsHMQvsObs = self.hm_gauges.merge(
            self.MetricsHMQvsObs,
            left_on=self.gauge_id_col,
            right_index=True,
            how="left",
            sort=False,
        )

        self.MetricsHMQvsObs.index = self.MetricsHMQvsObs[self.gauge_id_col]
        self.MetricsHMQvsObs.index.name = "index"
        if isinstance(self.hm_gauges, GeoDataFrame):
            self.MetricsHMQvsObs.crs = self.hm_gauges.crs

    def HMWLvsObserved(
        self,
        start: str = "",
        end: str = "",
        fmt: str = "%Y-%m-%d",
        shift: int = 0,
    ):
        """HMWLvsObserved.

            HM_vs_RRM calculate the performance metrics between the hydraulic model simulated
            discharge and the rainfall-runoff model simulated discharge

        Parameters
        ----------
        start: [str]
            the stating date for the period you want to calculate the error for.
        end: [str]
            the end date for the period you want to calculate the error for.
        fmt: [str]
            format of the given date
        shift: [int]
            if there is a shift in the clculated timeseries by one or more time
            steps and you want to fix this problem in calculating the metrics by
            shifting the calculated timeseries by one or more time steps.

        Returns
        -------
        MetricsHMWLvsObs: [dataframe]
            dataframe with the gauges id as rows and ['start', 'end', 'rmse', 'KGE', 'WB', 'NSE',
            'NSEModefied'], as columns.
        """
        if not isinstance(self.WLGauges, DataFrame):
            raise ValueError(
                "WLGauges variable does not exist please read the water level gauges "
                "with 'ReadObservedWL' method"
            )

        if not isinstance(self.WLHM, DataFrame):
            raise ValueError(
                "WLHM variable does not exist please read the water level simulated by the HM "
                "with 'ReadHMWL' method"
            )

        ### HM WL vs observed
        self.MetricsHMWLvsObs = self.Metrics(
            self.WLGauges,
            self.WLHM,
            self.WLgaugesList,
            self.novalue,
            start,
            end,
            shift,
            fmt,
        )

        self.MetricsHMWLvsObs = self.hm_gauges.merge(
            self.MetricsHMWLvsObs,
            left_on=self.gauge_id_col,
            right_index=True,
            how="left",
            sort=False,
        )

        self.MetricsHMWLvsObs.index = self.MetricsHMWLvsObs[self.gauge_id_col]
        self.MetricsHMWLvsObs.index.name = "index"
        if isinstance(self.hm_gauges, GeoDataFrame):
            self.MetricsHMWLvsObs.crs = self.hm_gauges.crs

    def InspectGauge(
        self,
        subid: int,
        gaugei: int = 0,
        start: str = "",
        end: str = "",
        fmt: str = "%Y-%m-%d",
    ) -> Union[
        tuple[DataFrame, Figure, tuple[Any, Any]], tuple[DataFrame, Figure, Any]
    ]:
        """InspectGauge.

            InspectGauge returns the metrices of the gauge simulated discharge and water level
            and plot it

        parameters
        ----------
        subid: [int]
            river segment id
        gaugei: [int]
            if the river segment has more than one gauge, gaugei is the gauge order
        start: [str]
            start date, if not given it will be taken from the already calculated Metrics table
        end: [str]
            end date, if not given it will be taken from the already calculated Metrics table
        fmt : [str]
            format of the given dates. The default is "%Y-%m-%d"
        Returns
        -------
        summary: [DataFrame]
            performance metrix
        """
        if not isinstance(self.MetricsHMvsRRM, DataFrame) and not isinstance(
            self.MetricsHMvsRRM, GeoDataFrame
        ):
            raise ValueError(
                "please calculate first the MetricsHMvsRRM by the method HMvsRRM"
            )

        gauge = self.GetGauges(subid, gaugei)
        gauge_id = gauge.loc[0, self.gauge_id_col]
        gaugename = str(gauge.loc[0, "name"])

        summary = pd.DataFrame(
            index=["HM-RRM", "RRM-Observed", "HM-Q-Observed", "HM-WL-Observed"],
            columns=self.MetricsHMvsRRM.columns,
        )
        # for each gauge in the segment
        if isinstance(self.MetricsHMQvsObs, DataFrame) or isinstance(
            self.MetricsHMQvsObs, GeoDataFrame
        ):
            summary.loc["HM-Q-Observed", :] = self.MetricsHMQvsObs.loc[gauge_id, :]

        if gauge.loc[0, "waterlevel"] == 1 and gauge.loc[0, "discharge"] == 1:
            fig, (ax1, ax2) = plt.subplots(
                ncols=1, nrows=2, sharex=True, figsize=(15, 8)
            )
        else:
            fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(15, 8))

        if gauge_id in self.rrm_gauges:
            # there are RRM simulated data
            summary.loc["HM-RRM", :] = self.MetricsHMvsRRM.loc[gauge_id, :]
            if isinstance(self.MetricsRRMvsObs, DataFrame) or isinstance(
                self.MetricsRRMvsObs, GeoDataFrame
            ):
                summary.loc["RRM-Observed", :] = self.MetricsRRMvsObs.loc[gauge_id, :]

            if start == "":
                start_1 = self.MetricsHMvsRRM.loc[gauge_id, "start"]
            else:
                s1 = dt.datetime.strptime(start, fmt)
                s2 = self.MetricsHMvsRRM.loc[gauge_id, "start"]
                start_1 = max(s1, s2)

            if end == "":
                end_1 = self.MetricsHMvsRRM.loc[gauge_id, "end"]
            else:
                e1 = dt.datetime.strptime(end, fmt)
                e2 = self.MetricsHMvsRRM.loc[gauge_id, "end"]
                end_1 = min(e1, e2)

            ax1.plot(self.q_hm[gauge_id].loc[start_1:end_1], label="HM", zorder=5)
            ax1.plot(self.q_gauges[gauge_id].loc[start_1:end_1], label="Observed")
            ax1.plot(self.q_rrm[gauge_id].loc[start_1:end_1], label="RRM")
            ax1.set_ylabel("Discharge m3/s", fontsize=12)
            ax1.legend(fontsize=15)
            # SimMax = max(self.q_hm[gauge_id].loc[start:end])
            # ObsMax = max(self.q_rrm[gauge_id].loc[start:end])
            # pos = max(SimMax, ObsMax)
        if gauge.loc[0, "waterlevel"] == 1:
            # there are water level observed data
            summary.loc["HM-WL-Observed", :] = self.MetricsHMWLvsObs.loc[gauge_id, :]

            if start == "":
                start_2 = self.MetricsHMWLvsObs.loc[gauge_id, "start"]
            else:
                s1 = dt.datetime.strptime(start, fmt)
                s2 = self.MetricsHMWLvsObs.loc[gauge_id, "start"]
                start_2 = max(s1, s2)

            if end == "":
                end_2 = self.MetricsHMWLvsObs.loc[gauge_id, "end"]
            else:
                e1 = dt.datetime.strptime(end, fmt)
                e2 = self.MetricsHMWLvsObs.loc[gauge_id, "end"]
                end_2 = min(e1, e2)

            ax2.plot(self.WLHM[gauge_id].loc[start_2:end_2], label="HM", linewidth=2)
            ax2.plot(
                self.WLGauges[gauge_id].loc[start_2:end_2],
                label="Observed",
                linewidth=2,
            )
            ax2.set_ylabel("Water Level m", fontsize=12)
            ax2.legend(fontsize=15)

            # SimMax = max(self.WLHM[gauge_id].loc[start_2:end_2])
            # ObsMax = max(self.WLGauges[gauge_id].loc[start_2: end_2])
            # pos = max(SimMax, ObsMax)
        # plt.legend(fontsize=15)
        ax1.set_title(gaugename, fontsize=30)
        ax1.set_title(gaugename, fontsize=30)

        if gauge.loc[0, "waterlevel"] == 1:
            return summary, fig, (ax1, ax2)
        else:
            return summary, fig, ax1

    @staticmethod
    def PrepareToSave(df: DataFrame) -> DataFrame:
        """PrepareToSave.

            PrepareToSave convert all the dates in the dataframe into string

        Parameters
        ----------
        df: [dataframe]
            the
        Returns
        -------
        Dataframe
        """
        df.drop(["start", "end"], axis=1, inplace=True)
        if "Qstart" in df.columns.tolist():
            start = df["Qstart"].tolist()
        else:
            start = df["WLstart"].tolist()

        for i in range(len(start)):
            if "Qstart" in df.columns.tolist():
                if isinstance(df.loc[df.index[i], "Qstart"], Timestamp):
                    df.loc[df.index[i], "Qstart"] = str(
                        df.loc[df.index[i], "Qstart"].date()
                    )
                if isinstance(df.loc[df.index[i], "Qend"], Timestamp):
                    df.loc[df.index[i], "Qend"] = str(
                        df.loc[df.index[i], "Qend"].date()
                    )

            if "WLstart" in df.columns.tolist():
                if isinstance(df.loc[df.index[i], "WLstart"], Timestamp):
                    df.loc[df.index[i], "WLstart"] = str(
                        df.loc[df.index[i], "WLstart"].date()
                    )
                if isinstance(df.loc[df.index[i], "WLend"], Timestamp):
                    df.loc[df.index[i], "WLend"] = str(
                        df.loc[df.index[i], "WLend"].date()
                    )
        return df

    def SaveMetices(self, path):
        """SaveMetices.

            SaveMetices saves the calculated metrics

        Parameters
        ----------
        path: [str]

        Returns
        -------
        None
        """
        if isinstance(self.MetricsHMvsRRM, GeoDataFrame) or isinstance(
            self.MetricsHMvsRRM, DataFrame
        ):
            df = self.PrepareToSave(self.MetricsHMvsRRM.copy())
            if isinstance(self.MetricsHMvsRRM, GeoDataFrame):
                df.to_file(path + "MetricsHM_Q_RRM.geojson", driver="GeoJSON")
            if isinstance(self.MetricsHMvsRRM, DataFrame):
                df.to_csv(path + "MetricsHM_Q_RRM.geojson.csv")

        if isinstance(self.MetricsHMQvsObs, GeoDataFrame) or isinstance(
            self.MetricsHMQvsObs, DataFrame
        ):
            df = self.PrepareToSave(self.MetricsHMQvsObs.copy())
            if isinstance(self.MetricsHMQvsObs, GeoDataFrame):
                df.to_file(path + "MetricsHM_Q_Obs.geojson", driver="GeoJSON")
            if isinstance(self.MetricsHMQvsObs, DataFrame):
                df.to_csv(path + "MetricsHM_Q_Obs.geojson.csv")

        if isinstance(self.MetricsRRMvsObs, GeoDataFrame) or isinstance(
            self.MetricsRRMvsObs, DataFrame
        ):
            df = self.PrepareToSave(self.MetricsRRMvsObs.copy())
            if isinstance(self.MetricsRRMvsObs, GeoDataFrame):
                df.to_file(path + "MetricsRRM_Q_Obs.geojson", driver="GeoJSON")
            if isinstance(self.MetricsRRMvsObs, DataFrame):
                df.to_csv(path + "MetricsRRM_Q_Obs.geojson.csv")

        if isinstance(self.MetricsHMWLvsObs, GeoDataFrame) or isinstance(
            self.MetricsHMWLvsObs, DataFrame
        ):
            df = self.PrepareToSave(self.MetricsHMWLvsObs.copy())
            if isinstance(self.MetricsHMWLvsObs, GeoDataFrame):
                df.to_file(path + "MetricsHM_WL_Obs.geojson", driver="GeoJSON")
            if isinstance(self.MetricsHMWLvsObs, DataFrame):
                df.to_csv(path + "MetricsHM_WL_Obs.geojson.csv")

    def ListAttributes(self):
        """ListAttributes.

        Print Attributes List
        """
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

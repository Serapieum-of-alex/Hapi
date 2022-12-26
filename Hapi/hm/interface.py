"""Created on Wed Mar  3 12:40:23 2021.

@author: mofarrag
"""
import datetime as dt
from typing import Any, Union, Optional, Union

import pandas as pd
from loguru import logger
from pandas import DataFrame
from joblib import Parallel, delayed, cpu_count

from Hapi.hm.river import River


class Interface(River):
    """Interface.

    Interface between the Rainfall runoff model and the Hydraulic model

    Methods
    -------
        1- ReadLateralsTable
        2- ReadLaterals
        3- ReadBoundaryConditionsTable
        4- ReadBoundaryConditions
        5- ListAttributes
    """

    Laterals: DataFrame

    def __init__(
        self,
        name: Any,
        version: int = 3,
        start: str = "1952-1-1",
        days: int = 36890,
        fmt: str = "%Y-%m-%d",
    ):
        if not isinstance(start, str):
            raise ValueError("start argument has to be string")
        if not isinstance(version, int):
            raise ValueError("version argument has to be integer number")
        if not isinstance(days, int):
            raise ValueError("number of days has to be integer number")
        if not isinstance(fmt, str):
            raise ValueError("date format 'fmt' has to be a string")

        self.name = name
        self.version = version
        self.start = dt.datetime.strptime(start, fmt)
        self.end = self.start + dt.timedelta(days=days)
        Ref_ind = pd.date_range(self.start, self.end, freq="D")
        self.ReferenceIndex = pd.DataFrame(index=list(range(1, days + 1)))
        self.ReferenceIndex["date"] = Ref_ind[:-1]

        self.LateralsTable = None
        self.routedRRM = None
        self.BCTable = None
        self.BC = None
        pass

    def readLateralsTable(
        self, path: str, prefix: str = "lf_xsid", suffix: str = ".txt"
    ) -> None:
        """ReadLateralsTable.

        ReadLateralsTable method reads the laterals file
            laterals file : file contains the xsid of the cross-sections that
            has laterals
        if the user has already read te cross section file, the methos is going
        to add column to the crosssection dataframe attribute and is going to add
        a value of 1 to the xs that has laterals

        Parameters
        ----------
        path : [String], optional
            Path to read the results from.
        suffix: [str]
            any
        prefix: []


        Returns
        -------
        IF: [dataframe attribute]
            dataframe with two columns ["filename", "sxid"]
        """
        try:
            self.LateralsTable = pd.read_csv(path, skiprows=[0], header=None)
        except pd.errors.EmptyDataError:
            self.LateralsTable = pd.DataFrame()
            logger.warning("The Lateral table file is empty")
            return

        self.LateralsTable.columns = ["filename"]
        l1 = len(prefix)
        l2 = len(suffix)
        self.LateralsTable["xsid"] = [
            int(i[l1 : len(i) - l2])
            for i in self.LateralsTable[self.LateralsTable.columns[0]]
        ]

        if hasattr(self, "crosssections"):
            self.crosssections["lateral"] = 0
            for i in range(len(self.crosssections)):
                if (
                    self.crosssections.loc[i, "xsid"]
                    in self.LateralsTable["xsid"].tolist()
                ):
                    self.crosssections.loc[i, "lateral"] = 1
        else:
            raise ValueError(
                "Please read the cross section file first using the method 'ReadCrossSections'"
            )

    def _readRRMwrapper(
            self,
            table: DataFrame,
            fromday: Union[str, int] = "",
            today: Union[str, int] = "",
            path: str = "",
            date_format: str = "'%Y-%m-%d'",
            prefix: str = "lf_xsid",
            cores: Optional[Union[int, bool]]=None,
    ):
        """_readRRMwrapper.

            wrapper to read any time series results from the rainfall-runoff model

        Parameters
        ----------
        table: [DataFrame]
            LateralsTable, or BCTable
        fromday : [integer], optional
                the day you want to read the result from, the first day is 1 not zero.The default is ''.
        today : [integer], optional
                the day you want to read the result to.
        path : [String], optional
            path to read the results from. The default is ''.
        date_format : TYPE, optional
            format of the given date string. The default is "'%Y-%m-%d'".
        cores: Optional[Union[int, bool]]
            if True the code will use all the cores except one, if integer the code will use the given number
            of cores for parallel io. Default is
            None.

        Returns
        -------
        DataFrame : [dataframe attribute].
            dataframe contains the hydrograph at the location if the xs with xsid as a column name
            and a column 'total' contains the sum of all the hydrographs.
        """
        # rainfall-runoff time series
        rrm_ts = pd.DataFrame()

        if cores:
            if isinstance(cores, bool):
                cores = cpu_count() - 1

            node_ids  = table.loc[:, "xsid"].to_list()
            fnames = [f"{prefix}{NodeID}" for NodeID in node_ids]
            func = self.readRRMResults
            results = Parallel(n_jobs=cores)(delayed(func)(self.version, self.ReferenceIndex, path, fname,
                                                           fromday, today, date_format) for fname in fnames)
            # results is a list of dataframes that have the same length (supposedly)
            results = [results[i][fnames[i]].to_list() for i in range(len(fnames))]
            for i in range(len(node_ids)):
                node_id = node_ids[i]
                rrm_ts[node_id] = results[i]
        else:
            for i in range(len(table)):
                node_id = table.loc[i, "xsid"]
                fname = f"{prefix}{node_id}"

                rrm_ts[node_id] = self.readRRMResults(
                    self.version,
                    self.ReferenceIndex,
                    path,
                    fname,
                    fromday,
                    today,
                    date_format,
                )[fname].tolist()
                logger.info(f"Lateral file {fname} is read")

        rrm_ts["total"] = rrm_ts.sum(axis=1)
        if fromday == "":
            fromday = 1
        if today == "":
            today = len(rrm_ts[rrm_ts.columns[0]])

        start = self.ReferenceIndex.loc[fromday, "date"]
        end = self.ReferenceIndex.loc[today, "date"]
        rrm_ts.index = pd.date_range(start, end, freq="D")

        return rrm_ts


    def readLaterals(
            self,
            fromday: Union[str, int] = "",
            today: Union[str, int] = "",
            path: str = "",
            date_format: str = "'%Y-%m-%d'",
            cores: Optional[Union[int, bool]]=None,
            prefix: str = "lf_xsid",
            laterals: Optional[bool]= True,
    ):
        """readLaterals.
            TODO: rename this function as it is better to name if readRRMoutputs
            - read the laterals at the location of cross-sections (if laterals=True)
            - read the routed hydrograph by the rainfall-runoff model at the location
            of the lateral cross-sections (if laterals=False)

        Parameters
        ----------
        fromday : [integer], optional
                the day you want to read the result from, the first day is 1 not zero.The default is ''.
        today : [integer], optional
                the day you want to read the result to.
        path : [String], optional
            path to read the results from. The default is ''.
        date_format : TYPE, optional
            format of the given date string. The default is "'%Y-%m-%d'".
        cores: Optional[Union[int, bool]]
            if True the code will use all the cores except one, if integer the code will use the given number
            of cores for parallel io. Default is
            None.
        prefix: [str]
            prefix used to distinguish the boundary condition files, Default is "lf_xsid".
        laterals: Optional[bool]
            True if you want to read the laterals, false if you want to read the routedRRM
            Default is True.

        Returns
        -------
        Laterals : [dataframe attribute].
            dataframe contains the hydrograph of each of the laterals at the location if the xs
            with xsid as a column name and a column 'total' contains the
            sum of all the hydrographs. this attribut will be careated only if the laterals
            argument is True [default]
        routedRRM: [dataframe attribute]
            read the routed hydrograph by the rainfall-runoff model at the location of the lateral
            cross-sections
            dataframe contains the hydrograph of each of the laterals at the location if the xs
            with xsid as a column name and a column 'total' contains the
            sum of all the hydrographs. this attribut will be careated only if the laterals
            argument is False
        """
        if not isinstance(self.LateralsTable, DataFrame):
            raise ValueError(
                "Please read the laterals table first using the 'ReadLateralsTable' method"
            )


        if len(self.LateralsTable) > 0:
            rrm_df = self._readRRMwrapper(
                self.LateralsTable,
                fromday=fromday,
                today=today,
                path=path,
                date_format=date_format,
                prefix= prefix,
                cores=cores
            )

            if laterals:
                self.Laterals = rrm_df
            else:
                self.routedRRM = rrm_df
        else:
            logger.info("There are no Laterals table please check")


    def readBoundaryConditionsTable(self, path, prefix="bc_xsid", suffix=".txt"):
        """readBoundaryConditionsTable.

        ReadLateralsTable method reads the laterals file
            laterals file : file contains the xsid of the cross-sections that
            has laterals
        if the user has already read te cross section file, the methos is going
        to add column to the crosssection dataframe attribute and is going to add
        a value of 1 to the xs that has laterals

        Parameters
        ----------
        path : [String], optional
            path to read the results from.
        suffix: [string]
            if the lateral files has a suffix in their names
        prefix: [string]
            `if the lateral files has a prefix in their names

        Returns
        -------
        None.
        """
        self.BCTable = pd.read_csv(path, skiprows=[0], header=None)
        self.BCTable.columns = ["filename"]
        l1 = len(prefix)
        l2 = len(suffix)
        self.BCTable["xsid"] = [
            int(i[l1 : len(i) - l2]) for i in self.BCTable[self.BCTable.columns[0]]
        ]

    def readBoundaryConditions(
            self,
            fromday: Union[str, int] = "",
            today: Union[str, int] = "",
            path: str = "",
            date_format: str = "'%Y-%m-%d'",
            prefix: str = "bc_xsid",
            cores: Optional[Union[int, bool]] = None,
    ):
        """ReadUSHydrograph.

        Read the boundary condition

        Parameters
        ----------
        fromday : [integer], optional
                the day you want to read the result from, the first day is 1 not zero.The default is ''.
        today : [integer], optional
                the day you want to read the result to.
        path : [String], optional
            path to read the results from. The default is ''.
        date_format : "TYPE, optional
            DESCRIPTION. The default is "'%Y-%m-%d'".
        prefix: [str]
            prefix used to distinguish the boundary condition files, Default is "bc_xsid".
        cores: Optional[Union[int, bool]]
            if True the code will use all the cores except one, if integer the code will use the given number
            of cores for parallel io. Default is
            None.

        Returns
        -------
        USHydrographs : [dataframe attribute].
            dataframe contains the hydrograph of each of the upstream segments
            with segment id as a column name and a column 'total' contains the
            sum of all the hydrographs.
        """
        if not isinstance(self.BCTable, DataFrame):
            raise ValueError("Please read the lateras table first using the 'ReadLateralsTable' method")

        self.BC = pd.DataFrame()
        self.BC = self._readRRMwrapper(
            self.BCTable,
            fromday=fromday,
            today=today,
            path=path,
            date_format=date_format,
            prefix=prefix,
            cores=cores
        )


    def ListAttributes(self):
        """ListAttributes.

        Print Attributes List
        """
        logger.debug("\n")
        logger.debug(
            f"Attributes List of: {repr(self.__dict__['name'])} - {self.__class__.__name__} Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                logger.debug(str(key) + " : " + repr(self.__dict__[key]))

        logger.debug("\n")

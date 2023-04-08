"""1D riven Events."""
import os
from typing import Tuple, Any, Union
import datetime as dt

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
# from pyramids.raster import Raster
from pyramids.dataset import Dataset
from cleopatra.statistics import Statistic

class Event:
    """Event.

        The Event class reads all the results of the Hydraulic model to preform all kind of analysis on flood event
        basis and the overtopping.

    Methods
    -------
    indexToDate
    createEventIndex
    getAllEvents
    readOvertopping
    calculateVolumeError
    OverlayMaps
    readEventIndex
    histogram
    drop
    to_csv
    getEventBeginning
    getEventEnd
    prepareForPlotting
    listAttributes
    """

    def __init__(
        self,
        name: str,
        start: Union[str, dt.datetime] = "1950-1-1",
        event_index: DataFrame = None,
        left_overtopping_suffix: str = "_left.txt",
        right_overtopping_suffix: str = "_right.txt",
        depth_prefix: str = "DepthMax",
        duration_prefix: str = "Duration",
        return_period_prefix: str = "ReturnPeriod",
        compressed: bool = True,
    ):
        """Event. To instantiate the Event class you need to provide the following arguments.

        Parameters
        ----------
        name : [str]
            DESCRIPTION.
        start : [str], optional
            start date. The default is "1950-1-1".
        left_overtopping_suffix : [str], optional
            the prefix you used to name the overtopping form the left bank files.
            The default is "_left.txt".
        right_overtopping_suffix : TYPE, optional
            the prefix you used to name the overtopping form the right bank files.
            The default is "_right.txt".
        depth_prefix : [str], optional
            the prefix you used to name the Max depth raster result maps.
            The default is "DepthMax".
        duration_prefix : [str], optional
            the prefix you used to name the inundation duration raster result maps.
            The default is "Duration".
        return_period_prefix : [str], optional
            the prefix you used to name the Return Period raster result maps.
            The default is "ReturnPeriod".
        compressed : [bool], optional
            True if the result raster/ascii files are compressed. The default is True.

        Returns
        -------
        None.
        """
        # instance attribute
        self.name = name
        self.start = (
            start
            if not isinstance(start, str)
            else dt.datetime.strptime(start, "%Y-%m-%d")
        )
        self.reference_index = self.create_reference_index(self.start, freq="D")
        self._event_index = event_index
        self._left_overtopping_suffix = left_overtopping_suffix
        self._right_overtopping_suffix = right_overtopping_suffix
        self._depth_prefix = depth_prefix
        self._duration_prefix = duration_prefix
        self._return_period_prefix = return_period_prefix
        self._two_d_result_path = ""
        self.compressed = compressed

        # create dictionary to store any extracted values from maps
        self.extracted_values = dict()
        self.event_beginning = None
        self.end_days = None

    @staticmethod
    def create_reference_index(start: dt.datetime, days: int = 36890, freq: str = "D"):
        """Create Event Index."""
        end = start + dt.timedelta(days=days)
        Ref_ind = pd.date_range(start, end, freq=freq)
        # the last day is not in the results day Ref_ind[-1]
        # write the number of days + 1 as python does not include the last number in the range
        # 19723 days so write 19724
        reference_index = pd.DataFrame(index=list(range(1, days + 1)))
        reference_index["date"] = Ref_ind[:-1]

        return reference_index

    @property
    def event_index(self):
        """Event index."""
        return self._event_index

    @property
    def depth_prefix(self):
        """Depth Prefix."""
        return self._depth_prefix

    @depth_prefix.setter
    def depth_prefix(self, value):
        self._depth_prefix = value

    @property
    def duration_prefix(self):
        """Duration Prefix."""
        return self._duration_prefix

    @duration_prefix.setter
    def duration_prefix(self, value):
        self._duration_prefix = value

    @property
    def two_d_result_path(self):
        """2D result path."""
        return self._two_d_result_path

    @two_d_result_path.setter
    def two_d_result_path(self, value):
        self._two_d_result_path = value

    @staticmethod
    def ordinal_to_date(df: DataFrame, reference_index: DataFrame):
        """IndexToDate. get the date coresponding to a given index.

        Returns
        -------
        Date
        """
        # convert the index into date
        dateFn = lambda i: reference_index.loc[i, "date"]
        # get the date the column 'id'
        date = df.loc[:, "id"].to_frame().applymap(dateFn)
        df["date"] = date
        return df

    @classmethod
    def create_from_index(
        cls, name: str, path: str, start: str, freq: str = "D", fmt: str = "%Y-%m-%d"
    ):
        """create_from_index.

            create_from_index takes the path to the index file result from the 2D model and creates a data frame to
            start adding the components of the event_index table.

        Parameters
        ----------
        name: [str]
            name of the river.
        path: [str]
            path of the index file result from the 2D model.
        start: [str]
            start date of the simulation.
        freq: [str]
            temporal resolution of the simulation. Default is "D".
        fmt: [str]
            Default is "%Y-%m-%d".

        Returns
        -------
        event_index: [dataframe]
            this method creates an instance attribute of type dataframe with columns ['id','continue', 'IndDiff',
            'Duration']
        """
        # FIXME
        # if the flood event does not have overtopping for 1 day then continues to
        # overtop after the method considers it as two separate events however
        # it is the same event (if the gap is less than 10 days it is still
        # considered the same event)

        # read the index file (containing the id of the days where flood happens (2D
        # algorithm works))
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file you have entered does not exist: {path}")
        start = dt.datetime.strptime(start, fmt)

        reference_index = Event.create_reference_index(start=start, freq=freq)

        event_index = pd.read_csv(path, header=None)
        event_index.rename(columns={0: "id"}, inplace=True)
        # convert the index into date

        event_index = Event.ordinal_to_date(event_index, reference_index)

        event_index.loc[:, "continue"] = 0
        # index difference maybe different than the duration as there might be
        # a gap in the middle of the event
        event_index.loc[:, "IndDiff"] = 0
        event_index.loc[:, "Duration"] = 0

        # the first day in the index file is an event beginning
        event_beginning = event_index.loc[0, "date"]
        for i in range(1, len(event_index)):
            # if the day is previous day+1
            if event_index.loc[i, "id"] == event_index.loc[i - 1, "id"] + 1:
                # then the event continues
                event_index.loc[i, "continue"] = 1
                # increase the duration
                event_index.loc[i, "IndDiff"] = event_index.loc[i - 1, "IndDiff"] + 1

                event_index.loc[i, "Duration"] = (
                    event_index.loc[i, "date"] - event_beginning
                ).days + 1
            else:  # if not then the day is the start of another event
                event_beginning = event_index.loc[i, "date"]

        return cls(name, start=start, event_index=event_index)

    def get_all_events(self):
        """GetAllEvents. GetAllEvents methods returns the end day of all events.

        Returns
        -------
            None.
        """
        assert hasattr(self, "event_index"), "please read/Create the event_index"
        IDs = list()
        for i in range(len(self.event_index)):
            if self.event_index.loc[i, "continue"] == 0 and i != 0:
                IDs.append(self.event_index.loc[i - 1, "id"])

        self.end_days = IDs

    def create_from_overtopping(self, path: str, delimiter: str = r"\s+"):
        r"""Overtopping.

            - Overtopping method reads the overtopping file and check if the event_index
            dataframe has already need created by the CreateEventIndex method, it will add
            the overtopping to it, if not it will create the event_index dataframe.

        Parameters
        ----------
        path: [str]
            path including the file name and extention of the Overtopping
            file result from the 1D model, the file has the follwoing headers.
            >>>  Step	overtopping(m3/s)
            >>>  14342            850.8
            >>>  14373            893.4
            >>>  14374           1049.7
        delimiter: [str]
            Delimeter used in the overtopping file Default is space (r"\s+").

        Returns
        -------
        event_index: [DataFrame]
            this method creates an instance attribute of type
            dataframe with columns ['id','continue', 'IndDiff', 'Duration',
            'Overtopping', 'OvertoppingCum', 'Volume']
        """
        overtop_total = pd.read_csv(path, delimiter=delimiter)
        # FIXME
        # if the flood event does not have overtopping for 1 day then continues to
        # overtop after the method considers it as two separate events however
        # it is the same event (if the gap is less than 10 days it is still
        # considered the same event)
        if not isinstance(self._event_index, DataFrame):
            # create the dataframe if the user did not use the CreateEventIndex method to
            # create the _event_index dataframe
            self._event_index = pd.DataFrame()
            self._event_index["id"] = overtop_total["Step"]
            self._event_index = self.ordinal_to_date(
                self._event_index, self.reference_index
            )

            self._event_index.loc[:, "continue"] = 0
            # index difference maybe different than the duration as there might be
            # a gap in the middle of the event
            self._event_index.loc[:, "IndDiff"] = 0
            self._event_index.loc[:, "Duration"] = 0

            # the first day in the index file is an event beginning
            self.event_beginning = self._event_index.loc[0, "date"]
            for i in range(1, len(self._event_index)):
                # if the day is previous day+1
                if (
                    self._event_index.loc[i, "id"]
                    == self._event_index.loc[i - 1, "id"] + 1
                ):
                    # then the event continues
                    self._event_index.loc[i, "continue"] = 1
                    # increase the duration
                    self._event_index.loc[i, "IndDiff"] = (
                        self._event_index.loc[i - 1, "IndDiff"] + 1
                    )

                    self._event_index.loc[i, "Duration"] = (
                        self._event_index.loc[i, "date"] - self.event_beginning
                    ).days + 1
                else:  # if not then the day is the start of another event
                    self.event_beginning = self._event_index.loc[i, "date"]

        # store the overtoppiung data in the _event_index dataframe
        self._event_index["Overtopping"] = overtop_total["overtopping(m3/s)"]

        self._event_index.loc[0, "OvertoppingCum"] = self._event_index.loc[
            0, "Overtopping"
        ]
        for i in range(1, len(self._event_index)):
            if self._event_index.loc[i, "continue"] == 0:
                self._event_index.loc[i, "OvertoppingCum"] = self._event_index.loc[
                    i, "Overtopping"
                ]
            else:
                self._event_index.loc[i, "OvertoppingCum"] = (
                    self._event_index.loc[i, "Overtopping"]
                    + self._event_index.loc[i - 1, "OvertoppingCum"]
                )
        # the volume of water is m3/s for hourly stored and acumulated values
        # volume = overtopping * 60 *60 = m3
        self._event_index.loc[:, "Volume"] = (
            self._event_index.loc[:, "OvertoppingCum"] * 60 * 60
        )

    def calculate_volume_error(self, path):
        """VolumeError. VolumeError method reads the VoleError file, assign values to the the coresponding time step.

        Parameters
        ----------
        path : [str]
            a path to the folder includng the maps.

        Returns
        -------
        event_index: [dataframe].
            add columns ['DEMError','StepError','TooMuchWater'] to the event_index dataframe
        """
        # read the VolError file
        VolError = pd.read_csv(path, delimiter=r"\s+")
        self._event_index["DEMError"] = 0
        self._event_index["StepError"] = 0
        self._event_index["TooMuchWater"] = 0

        for i in range(len(VolError)):
            loc = np.where(VolError.loc[i, "step"] == self._event_index.loc[:, "id"])[
                0
            ][0]
            self._event_index.loc[
                loc, ["DEMError", "StepError", "TooMuchWater"]
            ] = VolError.loc[
                i, ["DEM_Error", "PreviousDepthError", "TOOMuchWaterError"]
            ].tolist()

        self._event_index["VolError"] = (
            self._event_index["StepError"]
            + self._event_index["DEMError"]
            + self._event_index["TooMuchWater"]
        )
        self._event_index["VolError2"] = self._event_index["VolError"] / 20

    # def overlay_maps(
    #     self,
    #     path: str,
    #     base_map: str,
    #     excluded_value,
    #     occupied_cells_only: bool,
    #     save_to: str,
    # ) -> Tuple[Dict, DataFrame]:
    #     """OverlayMaps.
    #
    #         - OverlayMaps method reads all the maps in the folder given by path input and overlay them with the
    #         basemap and for each value in the basemap it create a dictionary with the intersected values from all maps.
    #
    #     Parameters
    #     ----------
    #     path: [String]
    #         a path to the folder includng the maps.
    #     base_map: [String]
    #         a path includng the name of the ASCII and extention like
    #         path="data/cropped.asc"
    #     excluded_value: [Numeric]
    #         values you want to exclude from exteacted values
    #     occupied_cells_only: [Bool]
    #         if you want to count only cells that is not excluded_value.
    #     save_to: [String]
    #         a path to the folder to save a text file for each
    #         value in the base map including all the intersected values
    #         from other maps.
    #
    #     Returns
    #     -------
    #     extracted_values: [Dict]
    #         dictonary with a list of values in the basemap as keys and for each key a list of all the intersected
    #         values in the maps from the path
    #     NonZeroCells: [dataframe]
    #         dataframe with the first column as the "file" name and the second column is the number of cells in each map
    #     """
    #     self.depth_values, non_zero_cells = Raster.overlayMaps(
    #         path,
    #         base_map,
    #         self.depth_prefix,
    #         excluded_value,
    #         self.compressed,
    #         occupied_cells_only,
    #     )
    #
    #     # non_zero_cells dataframe with the first column as the "file" name and the second column
    #     # is the number of cells in each map
    #
    #     non_zero_cells["days"] = [
    #         int(i[len(self.depth_prefix) : -4])
    #         for i in non_zero_cells["files"].tolist()
    #     ]
    #     # get the numbe of inundated cells in the Event index data frame
    #     self._event_index["cells"] = 0
    #     event_days = self._event_index["id"].values
    #     for i, event_i in non_zero_cells.iterrows():
    #         # get the location in the _event_index dataframe
    #         day_i = event_i["days"]
    #         diff = abs(event_days - day_i)
    #         loc = diff.argmin()
    #         if diff[loc] > 20:
    #             df = pd.DataFrame()
    #             df.loc[0, ["id", "cells"]] = [day_i, event_i["cells"]]
    #             self._event_index = pd.concat([self._event_index, df]).reset_index(
    #                 drop=True
    #             )
    #         else:
    #             self._event_index.loc[loc, "cells"] = event_i["cells"]
    #
    #     self._event_index.sort_values(["id"], axis=0, inplace=True)
    #     # save depths of each sub-basin
    #     inundatedSubs = list(self.depth_values.keys())
    #     for i in range(len(inundatedSubs)):
    #         np.savetxt(
    #             f"{save_to}/{inundatedSubs[i]}.txt",
    #             self.depth_values[inundatedSubs[i]],
    #             fmt="%4.2f",
    #         )

    @classmethod
    def read_event_index(cls, name: str, path: str, start: str):
        """read_event_index.

            ReadEventIndex method reads the event_index table created using the "CreateEventIndex" or
            "Overtopping" methods.

        Parameters
        ----------
        name: [str]
            name of the river.
        path: [str]
            path of the index file result from the 2D model.
        start: [str]
            start date of the simulation.

        Returns
        -------
        event_index : [dataframe].
            dataframe of the event_index table
        """
        event_index = pd.read_csv(path)
        return cls(name, start=start, event_index=event_index)

    def histogram(
        self, Day: int, exclude_value: Any = 0, Map=1, upper_bound=None, lower_bound=None, **kwargs
    ):
        """histogram.

            - Histogram method extract values fro the event MaxDepth map and plot the histogram th emethod check
            first if you already extracted the values before then plot the histogram.

        Parameters
        ----------
        Day : [Integer]
            DESCRIPTION.
        exclude_value : [Integer]
            DESCRIPTION.
        Map : [integer], optional
            1 for the max depth maps, 2 for the duration map, 3 for the
            return period maps. The default is 1.
        upper_bound: [float, int]
            Default is 0.2
        lower_bound: [float, int]
            Default is 15
        """
        if hasattr(self, "extracted_values"):
            # get the list of event that then object has their Extractedvalues
            if Day not in list(self.extracted_values.keys()):
                # depth map
                if Map == 1:
                    path = f"{self.two_d_result_path}{self.depth_prefix}{Day}.tif"
                elif Map == 2:
                    path = f"{self.two_d_result_path}{self.duration_prefix}{Day}.tif"
                else:
                    path = (
                        f"{self.two_d_result_path}{self.return_period_prefix}{Day}.tif"
                    )

                dataset = Dataset.read_file(path)
                extracted_values = dataset.extract(exclude_value=exclude_value)
                # non_zero_cells = len(extracted_values)

                self.extracted_values[Day] = extracted_values

        extracted_values = self.extracted_values[Day]
        # filter values
        if lower_bound is not None:
            extracted_values = [j for j in extracted_values if j > lower_bound]
        if upper_bound is not None:
            extracted_values = [j for j in extracted_values if j < upper_bound]

        hist = Statistic(extracted_values)
        fig, ax, opts = hist.histogram(
            bins=15, color="#0504aa", alpha=0.7, rwidth=0.85, ylabel="Frequency", **kwargs
        )
        return fig, ax, opts

    def drop(self, DropList):
        """Drop Drop method deletes columns from the event_index dataframe.

        Parameters
        ----------
        DropList: [list]
            list of column names to delete from the event_index dataframe table

        Returns
        -------
        event_index: [datadrame]
            the event_index dataframe without the columns in the Droplist
        """
        dataframe = self._event_index.loc[:, :]
        columns = list(dataframe.columns)

        [columns.remove(i) for i in DropList]

        dataframe = dataframe.loc[:, columns]
        self._event_index = dataframe

    def to_csv(self, path: str):
        """Save.

            - Save method saves the event_index table.

        Parameters
        ----------
        path : [str]
            path to where you want to save the table.
        """
        self._event_index.to_csv(path, header=True, index=None)

    def get_event_start(self, loc: int) -> Tuple[int, int]:
        """GetEventBeginning.

            - event_beginning method returns the index of the beginning of the event in the event_index dataframe.

        Parameters
        ----------
        loc: [int]
            dataframe index of the day you want to trace back to get the begining

        Returns
        -------
        ind: [int]
            dataframe index of the beginning day of the event

        Example
        ------
        if you want to get the beginning of the event that has the highest overtopping
        >>> HighOvertoppingInd = event_index['Overtopping'].idxmax()
        >>> ind = event_beginning(HighOvertoppingInd)
        """
        # loc = np.where(self.event_index['id'] == day)[0][0]
        # get all the days in the same event before that day as the inundation in the maps may
        # happen due to any of the days before not in this day
        ind = self.event_index.index[loc - self.event_index.loc[loc, "IndDiff"]]
        day = self.event_index.loc[ind, "id"]
        return ind, day

        # # filter the dataframe and get only the 'indDiff' and 'id' columns
        # FilteredEvent = self.event_index.loc[:,['IndDiff','id']]
        # FilteredEvent['diff'] = FilteredEvent.index - ind
        # # get only days before the day you inputed
        # FilteredEvent = FilteredEvent[FilteredEvent['diff'] <=0 ]
        # # start to search from down to up till you get the first 0 in the IndDiff
        # for i in range(self.event_index['Duration'].max()):

        #     if FilteredEvent.loc[len(FilteredEvent)-1-i,'IndDiff'] == 0:
        #         break

        # return FilteredEvent.index[len(FilteredEvent)-1-i]

    def get_event_end(self, loc):
        """GetEventEnd. method returns the index of the beginning of the event in the event_index dataframe.

        Parameters
        ----------
        loc: [Integer]
            dataframe index of the day you want to trace back to get the begining

        Returns
        -------
        ind: [Integer]
            dataframe index of the beginning day of the event

        Example
        -------
        if you want to get the beginning of the event that has the highest overtopping
        >>> HighOvertoppingInd = event_index['Overtopping'].idxmax()
        >>> ind = event_beginning(HighOvertoppingInd)
        """
        # loc = np.where(self.event_index['id'] == day)[0][0]
        # get all the days in the same event before that day as the inundation in the maps may
        # happen due to any of the days before not in this day

        # filter the dataframe and get only the 'indDiff' and 'id' columns
        FilteredEvent = self.event_index.loc[:, ["continue", "id"]]
        # get only days before the day you inputed
        for i in range(loc + 1, len(FilteredEvent)):
            # start search from the following day
            if FilteredEvent.loc[i, "continue"] != 1:
                break

        ind = i - 1
        day = self.event_index.loc[ind, "id"]

        return ind, day

    def prepare_for_plotting(self, ColumnName):
        """PrepareForPlotting.

            - PrepareForPlotting takes a time series in the event_index dataframe and fill the days that does not
            exist in date column and fill it with zero to properly plot it without letting the graph mislead the
            viewer of connecting the data over the gap period.

        Parameters
        ----------
        ColumnName : [String]
            name of the columns you want.

        Returns
        -------
        DataFrame : [Dataframe]
            dataframe with a date column, and the required column
        """
        NewDataFrame = pd.DataFrame()
        NewDataFrame["date"] = self.reference_index["date"].tolist()
        NewDataFrame[ColumnName] = 0
        for i in range(len(self.event_index)):
            loc = np.where(NewDataFrame["date"] == self.event_index.loc[i, "date"])[0][
                0
            ]
            NewDataFrame.loc[loc, ColumnName] = self.event_index.loc[i, ColumnName]

        return NewDataFrame

    def ListAttributes(self):
        """Print Attributes List."""

        print("\n")
        print(
            f"Attributes List of: {repr(self.__dict__['name'])} - {self.__class__.__name__} Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")

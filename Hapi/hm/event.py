"""1D riven Events."""
import os
from typing import Iterator, Tuple, Any, Union, Dict, List
from pathlib import Path
import datetime as dt
import yaml
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pyramids.dataset import Dataset
from cleopatra.statistics import Statistic


class Catalog:
    """Flood catalog.

    - the catalog is a dictionary
        - keys can be a string or a numeric.
        - values can only be
            - numeric
            - lists:
                list of numerics
            - nested dictionary (so this inner dictionary can have only lists or numeric as values)

    - So the following is how the Catalog is structured, where we have two events 19, and 25, for each event
    there is some properties that we want to store for this event
    19:
    {
        day: 19
        depth: {5: [0.0199, 0.0099, 0.1400], 29: [0.4199, 0.3100, 0.3100]}
        reaches: [29.0, 5.0]
      }
    25:
    {
        day: 25
        depth: {3: [0.0199, 0.0099, 0.1400], 15: [0.4199, 0.3100, 0.3100]}
        reaches: [3.0, 15.0]
        }

    - the catalog will be saved to disk in a yaml file with the following format.
    19:
      day: 19
      depth:
        5:
        - 0.0199
        - 0.0099
        - 0.1400
        29:
        - 0.4199
        - 0.3100
        - 0.3100
      reaches:
      - 29.0
      - 5.0
    25:
      day: 25
      depth:
        3:
        - 0.0199
        - 0.0099
        - 0.1400
        15:
        - 0.4199
        - 0.3100
        - 0.3100
      reaches:
      - 3.0
      - 15.0
    """

    float_3 = lambda x: float(round(x, 3))

    def __init__(self, catalog: dict):
        self._catalog = catalog

    @property
    def catalog(self) -> Dict:
        """Catalog."""
        return self._catalog

    @property
    def events(self) -> List:
        """Event days."""
        return list(self._catalog.keys())

    def __str__(self):
        """Print event details."""
        message = f"""
                    Number of Events: {len(self)}
                    Events: {self.events}
                """
        return message

    def __repr__(self):
        """Print event details."""
        message = f"""
                    Number of Events: {len(self)}
                    Events: {self.events}
                """
        return message

    def __len__(self):
        """Length."""
        return len(self._catalog.keys())

    def __iter__(self) -> Iterator[Any]:
        """Iterate."""
        return iter(self._catalog.items())

    def __setitem__(self, key, value):
        """Set event."""
        if key in self.events:
            raise KeyError(f"event: {key} already exists")
        self._catalog.update({key: value})

    def __getitem__(self, item):
        """Get Event."""
        if item not in self.events:
            raise KeyError(
                f"there is no event: {item}, available events are: {self.events}"
            )
        data = self._catalog[item]
        return EventData(data)

    @classmethod
    def read_file(cls, path: str):
        """Read catalog.

        Parameters
        ----------
        path: str
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"the path you entered does not exist: {path.absolute()}"
            )

        with open(path.absolute(), "r") as stream:
            catalog = yaml.safe_load(stream)

        return cls(catalog)

    @staticmethod
    def _serialize(in_dict: dict):
        # convert the keys to int
        new_dict = {}
        for key, val in in_dict.items():
            try:
                key = int(key)
            except ValueError:
                pass

            if isinstance(val, list):
                new_dict[key] = list(map(float, in_dict[key]))  # Catalog.float_3
            elif isinstance(val, dict):
                new_dict[key] = Catalog._serialize(val)
            else:
                new_dict[key] = int(val) if val is not None else None

        return new_dict

    def to_file(self, path: str):
        """Save catalog.

        Parameters
        ----------
        path: path
        """
        cat = self._serialize(self.catalog)

        with open(path, "w") as outfile:
            yaml.dump(cat, outfile, default_flow_style=False)


@dataclass
class EventData:
    """Event Data in the catalog."""

    data: Dict

    def parse(self):
        """Parse event data."""
        self.day = self.data.get("day")
        self.depth = self.data.get("depth")
        self.reaches = self.data.get("reaches")
        self.overtopping = self._get_overtopping()

    def _get_overtopping(self):
        over_top = self.data.get("overtopping")
        df = pd.DataFrame(columns=["cross-sections", "overtopping"])
        df["cross-sections"] = list(map(int, over_top["cross-sections"]))
        df["overtopping"] = over_top["volume"]
        return df


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
        if event_index is not None:
            self._number_events = len(self.event_index.loc[:, "index"].unique())

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
    def numer_events(self):
        """Events Number."""
        return self._number_events

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

    def get_event_by_index(self, event_i: int) -> Dict[str, str]:
        """get_event_by_index.

            using the index get the event last day.

        Parameters
        ----------
        event_i: [int]
            the order of the event.

        Returns
        -------
        Dict:
            {'start': 35, 'end': 38, 'cells': 1023}
        """
        if event_i > self.numer_events:
            raise ValueError(f"There are only {self.numer_events}, given: {event_i}")
        # ind = self.event_index.loc[:, "index"].values
        # row = np.where(ind == event_i)[0][-1]
        # return self.event_index.loc[row, "id"]
        df = self.event_index[self.event_index["index"] == event_i].reset_index(
            drop=True
        )
        ind = len(df) - 1
        event_data = {
            "start": df.loc[0, "id"],
            "end": df.loc[ind, "id"],
        }
        if "cells" in df.columns:
            event_data["cells"] = df.loc[ind, "cells"]

        return event_data

    @classmethod
    def create_from_index(
        cls,
        name: str,
        path: str,
        start: str,
        freq: str = "D",
        fmt: str = "%Y-%m-%d",
        event_duration: int = 10,
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
        event_duration: [int]
            Default is 10.

        Returns
        -------
        event_index: [dataframe]
            this method creates an instance attribute of type dataframe with columns ['id','continue', 'ind_diff',
            'duration']
        """
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
        # index difference maybe different than the duration as there might be
        # a gap in the middle of the event
        cols = ["continue", "ind_diff", "duration", "index"]
        event_index.loc[:, cols] = 0

        # the first day in the index file is an event beginning
        k = 1
        event_index.loc[0, "index"] = k
        event_beginning = event_index.loc[0, "date"]
        for i in range(1, len(event_index)):
            # if the day is previous day+1
            if (
                event_index.loc[i, "id"]
                <= event_index.loc[i - 1, "id"] + event_duration
            ):
                # then the event continues
                event_index.loc[i, "continue"] = 1
                # increase the duration
                event_index.loc[i, "ind_diff"] = event_index.loc[i - 1, "ind_diff"] + 1

                event_index.loc[i, "duration"] = (
                    event_index.loc[i, "date"] - event_beginning
                ).days + 1
            else:  # if not then the day is the start of another event
                k = k + 1
                event_beginning = event_index.loc[i, "date"]

            event_index.loc[i, "index"] = k

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
            dataframe with columns ['id','continue', 'ind_diff', 'duration',
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
            self._event_index.loc[:, "ind_diff"] = 0
            self._event_index.loc[:, "duration"] = 0

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
                    self._event_index.loc[i, "ind_diff"] = (
                        self._event_index.loc[i - 1, "ind_diff"] + 1
                    )

                    self._event_index.loc[i, "duration"] = (
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

    def overlay_maps(
        self,
        path: str,
        base_map: str,
        excluded_value,
    ) -> Catalog:
        """Overlay_maps.

            - OverlayMaps method reads all the maps in the folder given by path input and overlay them with the
            basemap and for each value in the basemap it create a dictionary with the intersected values from all maps.

        Parameters
        ----------
        path: [String]
            a path to the folder includng the maps.
        base_map: [String]
            a path includng the name of the ASCII and extention like
            path="data/cropped.asc"
        excluded_value: [Numeric]
            values you want to exclude from exteacted values

        Returns
        -------
        extracted_values: [Dict]
            dictonary with a list of values in the basemap as keys and for each key a list of all the intersected
            values in the maps from the path
        """
        classes_map = Dataset.read_file(base_map)
        # get the numbe of inundated cells in the Event index data frame
        ind_unique = self.event_index.loc[:, "index"].unique()
        ind = self.event_index.loc[:, "index"].values
        event_catalog = {}

        for ind_i in ind_unique:
            event_catalog_i = {}
            row = np.where(ind == ind_i)[0][-1]
            # convert into int to be able to dump it to json
            event_i = self.event_index.loc[row, "id"]
            event_catalog_i["day"] = event_i
            inundation_map = f"{path}/{self.depth_prefix}{event_i}.tif"
            if os.path.exists(inundation_map):
                dataset = Dataset.read_file(inundation_map)
                depth_values = dataset.overlay(
                    classes_map,
                    excluded_value,
                )
                event_catalog_i["reaches"] = list(depth_values.keys())
                event_catalog_i["depth"] = depth_values
            else:
                event_catalog_i["reaches"] = []
                event_catalog_i["depth"] = {}

            event_catalog[event_i] = event_catalog_i

        catalog = Catalog(event_catalog)
        return catalog

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
        self,
        Day: int,
        exclude_value: Any = 0,
        map_type: int = 1,
        upper_bound=None,
        lower_bound=None,
        **kwargs,
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
        map_type : [integer], optional
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
                if map_type == 1:
                    path = f"{self.two_d_result_path}/{self.depth_prefix}{Day}.tif"
                elif map_type == 2:
                    path = f"{self.two_d_result_path}/{self.duration_prefix}{Day}.tif"
                else:
                    path = (
                        f"{self.two_d_result_path}/{self.return_period_prefix}{Day}.tif"
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
            bins=15,
            color="#0504aa",
            alpha=0.7,
            rwidth=0.85,
            ylabel="Frequency",
            **kwargs,
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
        ind = self.event_index.index[loc - self.event_index.loc[loc, "ind_diff"]]
        day = self.event_index.loc[ind, "id"]
        return ind, day

    def get_event_end(self, loc):
        """get_event_end.

            - method returns the index of the beginning of the event in the event_index dataframe.

        Parameters
        ----------
        loc: [int]
            dataframe index of the day you want to trace back to get the begining

        Returns
        -------
        ind: [int]
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
        filtered_event = self.event_index.loc[:, ["continue", "id"]]
        # get only days before the day you inputed
        for i in range(loc + 1, len(filtered_event)):
            # start search from the following day
            if filtered_event.loc[i, "continue"] != 1:
                break

        ind = i - 1
        day = self.event_index.loc[ind, "id"]

        return ind, day

    def prepare_for_plotting(self, column_name: str):
        """Prepare For Plotting.

            - PrepareForPlotting takes a time series in the event_index dataframe and fill the days that does not
            exist in date column and fill it with zero to properly plot it without letting the graph mislead the
            viewer of connecting the data over the gap period.

        Parameters
        ----------
        column_name : [str]
            name of the columns you want.

        Returns
        -------
        DataFrame : [Dataframe]
            dataframe with a date column, and the required column
        """
        df = pd.DataFrame()
        df["date"] = self.reference_index["date"].tolist()
        df[column_name] = 0
        for i in range(len(self.event_index)):
            loc = np.where(df["date"] == self.event_index.loc[i, "date"])[0][0]
            df.loc[loc, column_name] = self.event_index.loc[i, column_name]

        return df

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

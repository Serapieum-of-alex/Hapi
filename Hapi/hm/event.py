"""Created on Tue Feb  4 14:57:30 2020.

@author: mofarrag
"""
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pyramids.raster import Raster

# import os


class Event:
    """Event.

        The Event class reads all the results of the Hydraulic model to preform all kind of analysis on flood event
        basis and the overtopping.

    Methods
    -------
        1- IndexToDate
        2- CreateEventIndex
        3- GetAllEvents
        4- Overtopping
        5- VolumeError
        6- OverlayMaps
        7- ReadEventIndex
        8- Histogram
        9- Drop
        10- Save
        11- GetEventBeginning
        12- GetEventEnd
        13- PrepareForPlotting
        14- ListAttributes
    """

    def __init__(
        self,
        name,
        start="1950-1-1",
        days=36890,
        left_overtopping_suffix="_left.txt",
        right_overtopping_suffix="_right.txt",
        depth_prefix="DepthMax",
        duration_prefix="Duration",
        return_period_prefix="ReturnPeriod",
        compressed=True,
    ):
        """Event. To instantiate the Event class you need to provide the following arguments.

        Parameters
        ----------
        name : [str]
            DESCRIPTION.
        start : [str], optional
            start date. The default is "1950-1-1".
        days : integer, optional
            length of the simulation . The default is 36890.
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
        self.start = dt.datetime.strptime(start, "%Y-%m-%d")
        self.end = self.start + dt.timedelta(days=days)

        self.left_overtopping_suffix = left_overtopping_suffix
        self.right_overtopping_suffix = right_overtopping_suffix
        self.depth_prefix = depth_prefix
        self.duration_prefix = duration_prefix
        self.return_period_prefix = return_period_prefix
        self.two_d_result_path = ""
        self.compressed = compressed
        Ref_ind = pd.date_range(self.start, self.end, freq="D")

        # the last day is not in the results day Ref_ind[-1]
        # write the number of days + 1 as python does not include the last number in the range
        # 19723 days so write 19724
        self.reference_index = pd.DataFrame(index=list(range(1, days + 1)))
        self.reference_index["date"] = Ref_ind[:-1]
        # create dictionary to store any extracted values from maps
        self.extracted_values = dict()
        self.event_index = None
        self.event_beginning = None
        self.end_days = None

    # method
    def indexToDate(self):
        """IndexToDate. get the date coresponding to a given index.

        Returns
        -------
        Date
        """
        # convert the index into date
        dateFn = lambda i: self.reference_index.loc[i, "date"]
        # get the date the column 'id'
        date = self.event_index.loc[:, "id"].to_frame().applymap(dateFn)
        self.event_index["date"] = date

    def createEventIndex(self, path: str):
        """CreateEventIndex.

            CreateEventIndex takes the path to the index file result from the 2D model and creates a data frame to
            start adding the components of the event_index table.

        Parameters
        ----------
        path: [String]
            path including the file name and extention of the index file result from the 2D model

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
        EventDays = pd.read_csv(path, header=None)
        EventIndex = EventDays.rename(columns={0: "id"})
        # convert the index into date
        self.event_index = EventIndex.loc[:, :]
        self.indexToDate()

        self.event_index.loc[:, "continue"] = 0
        # index difference maybe different than the duration as there might be
        # a gap in the middle of the event
        self.event_index.loc[:, "IndDiff"] = 0
        self.event_index.loc[:, "Duration"] = 0

        # the first day in the index file is an event beginning
        self.event_beginning = self.event_index.loc[0, "date"]
        for i in range(1, len(self.event_index)):
            # if the day is previous day+1
            if self.event_index.loc[i, "id"] == self.event_index.loc[i - 1, "id"] + 1:
                # then the event continues
                self.event_index.loc[i, "continue"] = 1
                # increase the duration
                self.event_index.loc[i, "IndDiff"] = (
                    self.event_index.loc[i - 1, "IndDiff"] + 1
                )

                self.event_index.loc[i, "Duration"] = (
                    self.event_index.loc[i, "date"] - self.event_beginning
                ).days + 1
            else:  # if not then the day is the start of another event
                self.event_beginning = self.event_index.loc[i, "date"]

    def getAllEvents(self):
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

    def Overtopping(self, overtopping_path: str):
        """Overtopping. Overtopping method reads the overtopping file and check if the event_index dataframe has already need created by the CreateEventIndex method, it will add the overtopping to it, if not it will create the event_index dataframe.

        Inputs:
            1- overtopping_path:
                [String] path including the file name and extention of the Overtopping
                file result from the 1D model
        Outputs:
            1- event_index:
                [dataframe] this method creates an instance attribute of type
                dataframe with columns ['id','continue', 'IndDiff', 'Duration',
                'Overtopping', 'OvertoppingCum', 'Volume']
        """
        OverTopTotal = pd.read_csv(
            overtopping_path, delimiter=r"\s+"
        )  # , header = None
        # FIXME
        # if the flood event does not have overtopping for 1 day then continues to
        # overtop after the method considers it as two separate events however
        # it is the same event (if the gap is less than 10 days it is still
        # considered the same event)
        if not isinstance(self.event_index, DataFrame):
            # create the dataframe if the user did not use the CreateEventIndex method to
            # create the event_index dataframe
            self.event_index = pd.DataFrame()
            self.event_index["id"] = OverTopTotal["Step"]
            self.indexToDate()

            self.event_index.loc[:, "continue"] = 0
            # index difference maybe different than the duration as there might be
            # a gap in the middle of the event
            self.event_index.loc[:, "IndDiff"] = 0
            self.event_index.loc[:, "Duration"] = 0

            # the first day in the index file is an event beginning
            self.event_beginning = self.event_index.loc[0, "date"]
            for i in range(1, len(self.event_index)):
                # if the day is previous day+1
                if (
                    self.event_index.loc[i, "id"]
                    == self.event_index.loc[i - 1, "id"] + 1
                ):
                    # then the event continues
                    self.event_index.loc[i, "continue"] = 1
                    # increase the duration
                    self.event_index.loc[i, "IndDiff"] = (
                        self.event_index.loc[i - 1, "IndDiff"] + 1
                    )

                    self.event_index.loc[i, "Duration"] = (
                        self.event_index.loc[i, "date"] - self.event_beginning
                    ).days + 1
                else:  # if not then the day is the start of another event
                    self.event_beginning = self.event_index.loc[i, "date"]

        # store the overtoppiung data in the event_index dataframe
        self.event_index["Overtopping"] = OverTopTotal["overtopping(m3/s)"]

        self.event_index.loc[0, "OvertoppingCum"] = self.event_index.loc[
            0, "Overtopping"
        ]
        for i in range(1, len(self.event_index)):
            if self.event_index.loc[i, "continue"] == 0:
                self.event_index.loc[i, "OvertoppingCum"] = self.event_index.loc[
                    i, "Overtopping"
                ]
            else:
                self.event_index.loc[i, "OvertoppingCum"] = (
                    self.event_index.loc[i, "Overtopping"]
                    + self.event_index.loc[i - 1, "OvertoppingCum"]
                )
        # the volume of water is m3/s for hourly stored and acumulated values
        # volume = overtopping * 60 *60 = m3
        self.event_index.loc[:, "Volume"] = (
            self.event_index.loc[:, "OvertoppingCum"] * 60 * 60
        )

    def calculateVolumeError(self, path):
        """VolumeError. VolumeError method reads the VoleError file, assign values to the the coresponding time step.

        Parameters
        ----------
        path : [String]
            a path to the folder includng the maps.

        Returns
        -------
        event_index: [dataframe attribute].
            add columns ['DEMError','StepError','TooMuchWater'] to the event_index dataframe
        """
        # read the VolError file
        VolError = pd.read_csv(path, delimiter=r"\s+")
        self.event_index["DEMError"] = 0
        self.event_index["StepError"] = 0
        self.event_index["TooMuchWater"] = 0

        for i in range(len(VolError)):
            loc = np.where(VolError.loc[i, "step"] == self.event_index.loc[:, "id"])[0][
                0
            ]
            self.event_index.loc[
                loc, ["DEMError", "StepError", "TooMuchWater"]
            ] = VolError.loc[
                i, ["DEM_Error", "PreviousDepthError", "TOOMuchWaterError"]
            ].tolist()

        self.event_index["VolError"] = (
            self.event_index["StepError"]
            + self.event_index["DEMError"]
            + self.event_index["TooMuchWater"]
        )
        self.event_index["VolError2"] = self.event_index["VolError"] / 20

    def overlayMaps(self, path, BaseMapF, ExcludedValue, OccupiedCellsOnly, SavePath):
        """OverlayMaps. OverlayMaps method reads all the maps in the folder given by path input and overlay them with the basemap and for each value in the basemap it create a dictionary with the intersected values from all maps.

        Parameters
        ----------
        path: [String]
            a path to the folder includng the maps.
        BaseMapF: [String]
            a path includng the name of the ASCII and extention like
            path="data/cropped.asc"
        ExcludedValue: [Numeric]
            values you want to exclude from exteacted values
        OccupiedCellsOnly: [Bool]
            if you want to count only cells that is not ExcludedValue.
        SavePath: [String]
            a path to the folder to save a text file for each
            value in the base map including all the intersected values
            from other maps.

        Returns
        -------
        extracted_values: [Dict]
            dictonary with a list of values in the basemap as keys and for each key a list of all the intersected
            values in the maps from the path
        NonZeroCells: [dataframe]
            dataframe with the first column as the "file" name and the second column is the number of cells in each map
        """
        self.DepthValues, NonZeroCells = Raster.overlayMaps(
            path,
            BaseMapF,
            self.depth_prefix,
            ExcludedValue,
            self.compressed,
            OccupiedCellsOnly,
        )

        # NonZeroCells dataframe with the first column as the "file" name and the second column
        # is the number of cells in each map

        NonZeroCells["days"] = [
            int(i[len(self.depth_prefix) : -4]) for i in NonZeroCells["files"].tolist()
        ]
        # get the numbe of inundated cells in the Event index data frame
        self.event_index["cells"] = 0

        for i in range(len(NonZeroCells)):
            # get the location in the event_index dataframe
            try:
                loc = np.where(
                    NonZeroCells.loc[i, "days"] == self.event_index.loc[:, "id"]
                )[0][0]
            except IndexError:
                # if it does not find the event in the eventindex table ignore
                continue
            # store number of cells
            self.event_index.loc[loc, "cells"] = NonZeroCells.loc[i, "cells"]

        # save depths of each sub-basin
        inundatedSubs = list(self.DepthValues.keys())
        for i in range(len(inundatedSubs)):
            np.savetxt(
                SavePath + "/" + str(inundatedSubs[i]) + ".txt",
                self.DepthValues[inundatedSubs[i]],
                fmt="%4.2f",
            )

    def readEventIndex(self, path):
        """ReadEventIndex.

            ReadEventIndex method reads the event_index table created using the "CreateEventIndex" or
            "Overtopping" methods.

        Parameters
        ----------
        path : [str]
            path to the event_index file.

        Returns
        -------
        event_index : [dataframe].
            dataframe of the event_index table
        """
        EventIndex = pd.read_csv(path)
        self.event_index = EventIndex
        self.indexToDate()

    def histogram(
        self, Day, ExcludeValue, OccupiedCellsOnly, Map=1, filter1=0.2, filter2=15
    ):
        """Histogram Histogram method extract values fro the event MaxDepth map and plot the histogram th emethod check first if you already extracted the values before then plot the histogram.

        Parameters
        ----------
        Day : [Integer]
            DESCRIPTION.
        ExcludeValue : [Integer]
            DESCRIPTION.
        OccupiedCellsOnly : TYPE
            DESCRIPTION.
        Map : [integer], optional
            1 for the max depth maps, 2 for the duration map, 3 for the
            return period maps. The default is 1.
        filter1: [float, int]
            Default is 0.2
        filter2: [float, int]
            Default is 15

        Returns
        -------
        None.
        """
        # check if the object has the attribute extracted_values
        if hasattr(self, "extracted_values"):
            # get the list of event that then object has their Extractedvalues
            if Day not in list(self.extracted_values.keys()):
                # depth map
                if Map == 1:
                    path = (
                        self.two_d_result_path + self.depth_prefix + str(Day) + ".zip"
                    )
                elif Map == 2:
                    path = (
                        self.two_d_result_path
                        + self.duration_prefix
                        + str(Day)
                        + ".zip"
                    )
                else:
                    path = (
                        f"{self.two_d_result_path}{self.return_period_prefix}{Day}.zip"
                    )

                ExtractedValues, NonZeroCells = Raster.extractValues(
                    path, ExcludeValue, self.compressed, OccupiedCellsOnly
                )
                self.extracted_values[Day] = ExtractedValues

        ExtractedValues = self.extracted_values[Day]
        # filter values
        ExtractedValues = [j for j in ExtractedValues if j > filter1]
        ExtractedValues = [j for j in ExtractedValues if j < filter2]
        # plot
        # fig, ax1 = plt.subplots(fig_size=(10,8))
        # ax1.hist(extracted_values, bins=15, alpha = 0.4) #width = 0.2,

        n, bins, patches = plt.hist(
            x=ExtractedValues, bins=15, color="#0504aa", alpha=0.7, rwidth=0.85
        )
        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Value", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.ylabel("Frequency", fontsize=15)
        plt.tight_layout()
        # plt.title('Normal Distribution Histogram matplotlib',font_size=15)
        plt.show()
        return n, bins, patches

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
        dataframe = self.event_index.loc[:, :]
        columns = list(dataframe.columns)

        [columns.remove(i) for i in DropList]

        dataframe = dataframe.loc[:, columns]
        self.event_index = dataframe

    def save(self, path):
        """Save Save method saves the event_index table.

        Parameters
        ----------
        path : [str]
            path to where you want to save the table.

        Returns
        -------
        None.
        """
        self.event_index.to_csv(path, header=True, index=None)  # index_label = "Index"

    def getEventBeginning(self, loc):
        """GetEventBeginning. event_beginning method returns the index of the beginning of the event in the event_index dataframe.

        Parameters
        ----------
        loc: [Integer]
            dataframe index of the day you want to trace back to get the begining

        Returns
        -------
        ind: [Integer]
            dataframe index of the beginning day of the event

        Example
        ------
            1- if you want to get the beginning of the event that has the highest
            overtopping
            HighOvertoppingInd = event_index['Overtopping'].idxmax()
            ind = event_beginning(HighOvertoppingInd)
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

    def getEventEnd(self, loc):
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
        1- if you want to get the beginning of the event that has the highest
        overtopping
        HighOvertoppingInd = event_index['Overtopping'].idxmax()
        ind = event_beginning(HighOvertoppingInd)
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

    def prepareForPlotting(self, ColumnName):
        """PrepareForPlotting. PrepareForPlotting takes a time series in the event_index dataframe and fill the days that does not exist in date column and fill it with zero to properly plot it without letting the graph mislead the viewer of connecting the data over the gap period.

        Parameters
        ----------
        ColumnName : [String]
            name of the columns you want.

        Returns
        -------
        NewDataFrame : [Dataframe]
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


if __name__ == "__main__":
    x = Event("Event")

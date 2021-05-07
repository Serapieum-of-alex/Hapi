"""
Created on Tue Feb  4 14:57:30 2020

@author: mofarrag
"""
import os
import datetime as dt
import pandas as pd
from Hapi.gis.raster import Raster
import numpy as np
import matplotlib.pyplot as plt


class Event():
    # class attributes
    """
    =========================================
        Event
    =========================================
    The Event class reads all the results of the Hydraulic model to preform all
    kind of analysis on flood event basis and the overtopping.

    Methods:
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

    def __init__(self, name, start = "1950-1-1", days = 36890,
                 leftOvertopping_Suffix = "_left.txt",
                 RightOvertopping_Suffix = "_right.txt", DepthPrefix = "DepthMax",
                 DurationPrefix = "Duration", ReturnPeriodPrefix = "ReturnPeriod" ,
                 Compressed = True):
        """
        =============================================================================
            Event(self, name, start = "1950-1-1", days = 36890,
                  leftOvertopping_Suffix = "_left.txt",
                  RightOvertopping_Suffix = "_right.txt", DepthPrefix = "DepthMax",
                  DurationPrefix = "Duration", ReturnPeriodPrefix = "ReturnPeriod" ,
                  Compressed = True)
        =============================================================================
        To instantiate the Event class you need to provide the following arguments

        Parameters
        ----------
        name : [str]
            DESCRIPTION.
        start : [str], optional
            start date. The default is "1950-1-1".
        days : integer, optional
            length of the simulation . The default is 36890.
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

        Returns
        -------
        None.

        """
        # instance attribute
        self.name = name
        self.start = dt.datetime.strptime(start,"%Y-%m-%d")
        self.end = self.start + dt.timedelta(days = days)

        self.leftOvertopping_Suffix = leftOvertopping_Suffix
        self.RightOvertopping_Suffix = RightOvertopping_Suffix
        self.DepthPrefix = DepthPrefix
        self.DurationPrefix = DurationPrefix
        self.ReturnPeriodPrefix = ReturnPeriodPrefix
        self.TwoDResultPath = ''
        self.Compressed = Compressed
        Ref_ind = pd.date_range(self.start,self.end, freq='D')

        # the last day is not in the results day Ref_ind[-1]
        # write the number of days + 1 as python does not include the last number in the range
        # 19723 days so write 19724
        self.ReferenceIndex = pd.DataFrame(index = list(range(1,days+1)))
        self.ReferenceIndex['date'] = Ref_ind[:-1]
        # create dictionary to store any extracted values from maps
        self.ExtractedValues = dict()

    # method
    def IndexToDate(self):
        # convert the index into date
        dateFn = lambda x: self.ReferenceIndex.loc[x,'date']
        # get the date the column 'id'
        date = self.EventIndex.loc[:,'id'].to_frame().applymap(dateFn)
        self.EventIndex['date'] = date



    def CreateEventIndex(self, IndexPath):
        """
        ========================================================
             CreateEventIndex(IndexPath)
        ========================================================
        CreateEventIndex takes the path to the index file result from the 2D model
        and creates a data frame to start adding the components of the EventIndex
        table

        Inputs:
            1-IndexPath:
                [String] path including the file name and extention of the index
                file result from the 2D model
        Outputs:
            1- EventIndex:
                [dataframe] this method creates an instance attribute of type
                dataframe with columns ['id','continue', 'IndDiff', 'Duration']
        """
        #FIXME
        # if the flood event does not have overtopping for 1 day then continues to
        # overtop after the method considers it as two separate events however
        # it is the same event (if the gap is less than 10 days it is still
        # considered the same event)

        # read the index file (containing the id of the days where flood happens (2D
        # algorithm works))
        EventDays = pd.read_csv(IndexPath,header = None)
        EventIndex = EventDays.rename(columns={0:'id'})
        # convert the index into date
        self.EventIndex = EventIndex.loc[:,:]
        self.IndexToDate()

        self.EventIndex.loc[:,'continue'] = 0
        # index difference maybe different than the duration as there might be
        # a gap in the middle of the event
        self.EventIndex.loc[:,'IndDiff'] = 0
        self.EventIndex.loc[:,'Duration'] = 0

        # the first day in the index file is an event beginning
        self.EventBeginning = self.EventIndex.loc[0,'date']
        for i in range(1,len(self.EventIndex)):
            # if the day is previous day+1
            if self.EventIndex.loc[i,'id'] == self.EventIndex.loc[i-1,'id']+1:
                # then the event continues
                self.EventIndex.loc[i,'continue'] = 1
                # increase the duration
                self.EventIndex.loc[i,'IndDiff'] = self.EventIndex.loc[i-1,'IndDiff'] +1

                self.EventIndex.loc[i,'Duration'] = (self.EventIndex.loc[i,'date'] - self.EventBeginning).days + 1
            else: # if not then the day is the start of another event
                self.EventBeginning = self.EventIndex.loc[i,'date']

    def GetAllEvents(self):
        """
        =============================================================================
            GetAllEvents()
        =============================================================================
        GetAllEvents methods returns the end day of all events


        Returns
        -------
            None.

        """
        assert hasattr(self, "EventIndex"), "please read/Create the EventIndex"
        IDs = list()
        for i in range(len(self.EventIndex)):
            if self.EventIndex.loc[i,'continue'] == 0 and i != 0 :
                IDs.append(self.EventIndex.loc[i-1,'id'])

        self.EndDays = IDs

    def Overtopping(self,OvertoppingPath):
        """
        ===================================================
            Overtopping(self,OvertoppingPath)
        ===================================================
        Overtopping method reads the overtopping file and check if the EventIndex
        dataframe has already need created by the CreateEventIndex method, it
        will add the overtopping to it, if not it will create the EventIndex dataframe

        Inputs:
            1- OvertoppingPath:
                [String] path including the file name and extention of the Overtopping
                file result from the 1D model
        Outputs:
            1- EventIndex:
                [dataframe] this method creates an instance attribute of type
                dataframe with columns ['id','continue', 'IndDiff', 'Duration',
                'Overtopping', 'OvertoppingCum', 'Volume']
        """
        OverTopTotal = pd.read_csv(OvertoppingPath, delimiter = r"\s+") #, header = None
        #FIXME
        # if the flood event does not have overtopping for 1 day then continues to
        # overtop after the method considers it as two separate events however
        # it is the same event (if the gap is less than 10 days it is still
        # considered the same event)
        if not hasattr(self,"EventIndex"):
            # create the dataframe if the user did not use the CreateEventIndex method to
            # create the EventIndex dataframe
            self.EventIndex = pd.DataFrame()
            self.EventIndex['id'] = OverTopTotal['Step']
            self.IndexToDate()

            self.EventIndex.loc[:,'continue'] = 0
            # index difference maybe different than the duration as there might be
            # a gap in the middle of the event
            self.EventIndex.loc[:,'IndDiff'] = 0
            self.EventIndex.loc[:,'Duration'] = 0

            # the first day in the index file is an event beginning
            self.EventBeginning = self.EventIndex.loc[0,'date']
            for i in range(1,len(self.EventIndex)):
                # if the day is previous day+1
                if self.EventIndex.loc[i,'id'] == self.EventIndex.loc[i-1,'id']+1:
                    # then the event continues
                    self.EventIndex.loc[i,'continue'] = 1
                    # increase the duration
                    self.EventIndex.loc[i,'IndDiff'] = self.EventIndex.loc[i-1,'IndDiff'] +1

                    self.EventIndex.loc[i,'Duration'] = (self.EventIndex.loc[i,'date'] - self.EventBeginning).days + 1
                else: # if not then the day is the start of another event
                    self.EventBeginning = self.EventIndex.loc[i,'date']


        # store the overtoppiung data in the EventIndex dataframe
        self.EventIndex['Overtopping'] = OverTopTotal['overtopping(m3/s)']

        self.EventIndex.loc[0,'OvertoppingCum'] = self.EventIndex.loc[0,'Overtopping']
        for i in range(1,len(self.EventIndex)):
            if self.EventIndex.loc[i,'continue'] == 0:
                self.EventIndex.loc[i,'OvertoppingCum'] = self.EventIndex.loc[i,'Overtopping']
            else:
                self.EventIndex.loc[i,'OvertoppingCum'] = self.EventIndex.loc[i,'Overtopping'] + self.EventIndex.loc[i-1,'OvertoppingCum']
        # the volume of water is m3/s for hourly stored and acumulated values
        # volume = overtopping * 60 *60 = m3
        self.EventIndex.loc[:,'Volume'] = self.EventIndex.loc[:,'OvertoppingCum'] * 60*60


    def VolumeError(self, Path):
        """
        ===========================================
             VolumeError(Path)
        ===========================================
        VolumeError method reads the VoleError file, assign values to the
        the coresponding time step

        Parameters
        ----------
            1- Path : [String]
                a path to the folder includng the maps.

        Returns
        -------
            1-EventIndex: [dataframe attribute].
                add columns ['DEMError','StepError','TooMuchWater'] to the EventIndex
                dataframe

        """
        # read the VolError file
        VolError = pd.read_csv(Path, delimiter = r'\s+')
        self.EventIndex['DEMError'] = 0
        self.EventIndex['StepError'] = 0
        self.EventIndex['TooMuchWater'] = 0

        for i in range(len(VolError)):
            loc = np.where(VolError.loc[i,'step'] == self.EventIndex.loc[:,'id'])[0][0]
            self.EventIndex.loc[loc,['DEMError','StepError','TooMuchWater']] = VolError.loc[i,['DEM_Error', 'PreviousDepthError', 'TOOMuchWaterError']].tolist()

        self.EventIndex['VolError']  = self.EventIndex['StepError'] + self.EventIndex['DEMError'] + self.EventIndex['TooMuchWater']
        self.EventIndex['VolError2'] = self.EventIndex['VolError'] / 20


    def OverlayMaps(self, Path, BaseMapF, ExcludedValue, OccupiedCellsOnly, SavePath):
        """
        ==================================================================
          OverlayMaps(self, Path, BaseMapF, ExcludedValue, OccupiedCellsOnly, SavePath)
        ==================================================================
        OverlayMaps method reads all the maps in the folder given by Path
        input and overlay them with the basemap and for each value in the basemap
        it create a dictionary with the intersected values from all maps

        Inputs:
            1-Path
                [String] a path to the folder includng the maps.
            2-BaseMapF:
                [String] a path includng the name of the ASCII and extention like
                path="data/cropped.asc"
            3-ExcludedValue:
                [Numeric] values you want to exclude from exteacted values
            5-OccupiedCellsOnly:
                [Bool] if you want to count only cells that is not ExcludedValue.
            6-SavePath:
                [String] a path to the folder to save a text file for each
                value in the base map including all the intersected values
                from other maps.
        Outputs:
            1- ExtractedValues:
                [Dict] dictonary with a list of values in the basemap as keys
                    and for each key a list of all the intersected values in the
                    maps from the path
            2- NonZeroCells:
                [dataframe] dataframe with the first column as the "file" name
                and the second column is the number of cells in each map
        """

        self.DepthValues, NonZeroCells = Raster.OverlayMaps(Path, BaseMapF, self.DepthPrefix,
                                                    ExcludedValue, self.Compressed,OccupiedCellsOnly)

        # NonZeroCells dataframe with the first column as the "file" name and the second column
        # is the number of cells in each map

        NonZeroCells['days'] = [int(i[len(self.DepthPrefix):-4]) for i in NonZeroCells['files'].tolist()]
        # get the numbe of inundated cells in the Event index data frame
        self.EventIndex['cells'] = 0

        for i in range(len(NonZeroCells)):
            # get the location in the EventIndex dataframe
            try:
                loc = np.where(NonZeroCells.loc[i,'days'] == self.EventIndex.loc[:,"id"] )[0][0]
            except IndexError:
                # if it does not find the event in the eventindex table ignore
                continue
            # store number of cells
            self.EventIndex.loc[loc,'cells'] = NonZeroCells.loc[i,'cells']

        # save depths of each sub-basin
        inundatedSubs = list(self.DepthValues.keys())
        for i in range(len(inundatedSubs)):
            np.savetxt(SavePath +"/" + str(inundatedSubs[i]) + ".txt",
                       self.DepthValues[inundatedSubs[i]],fmt="%4.2f")



    def ReadEventIndex(self,Path):
        """
        ===============================================================
             ReadEventIndex(Path)
        ===============================================================
        ReadEventIndex method reads the EventIndex table created using the
        "CreateEventIndex" or "Overtopping" methods
        Parameters
        ----------
        Path : [str]
            Path to the EventIndex file.

        Returns
        -------
        EventIndex : [dataframe].
            dataframe of the EventIndex table
        """
        EventIndex = pd.read_csv(Path)
        self.EventIndex = EventIndex
        self.IndexToDate()


    def Histogram(self, Day, ExcludeValue, OccupiedCellsOnly, Map = 1, filter1 = 0.2,
                  filter2 = 15):
        """
        ==================================================================
           Histogram(Day, ExcludeValue, OccupiedCellsOnly, Map = 1)
        ==================================================================
        Histogram method extract values fro the event MaxDepth map and plot the histogram
        th emethod check first if you already extracted the values before then
        plot the histogram
        Parameters
        ----------
            1-Day : [Integer]
                DESCRIPTION.
            2-ExcludeValue : [Integer]
                DESCRIPTION.
            3-OccupiedCellsOnly : TYPE
                DESCRIPTION.
            4-Map : [integer], optional
                1 for the max depth maps, 2 for the duration map, 3 for the
                return period maps. The default is 1.

        Returns
        -------
        None.

        """
        # check if the object has the attribute ExtractedValues
        if hasattr(self,'ExtractedValues'):
            # get the list of event that then object has their Extractedvalues
            if Day not in list(self.ExtractedValues.keys()):
                # depth map
                if Map == 1:
                    Path = self.TwoDResultPath + self.DepthPrefix + str(Day) + ".zip"
                elif Map == 2:
                    Path = self.TwoDResultPath + self.DurationPrefix + str(Day) + ".zip"
                else:
                    Path = self.TwoDResultPath + self.ReturnPeriodPrefix + str(Day) + ".zip"

                ExtractedValues, NonZeroCells = Raster.ExtractValues(Path, ExcludeValue,
                                                                     self.Compressed,
                                                                     OccupiedCellsOnly)
                self.ExtractedValues[Day] = ExtractedValues

        ExtractedValues = self.ExtractedValues[Day]
        # filter values
        ExtractedValues = [j for j in ExtractedValues if j > filter1]
        ExtractedValues = [j for j in ExtractedValues if j < filter2]
        #plot
        # fig, ax1 = plt.subplots(figsize=(10,8))
        # ax1.hist(ExtractedValues, bins=15, alpha = 0.4) #width = 0.2,

        n, bins , patches = plt.hist(x= ExtractedValues, bins=15, color="#0504aa" , alpha=0.7,
        							 rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value',fontsize=15)
        plt.ylabel('Frequency',fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.ylabel('Frequency',fontsize=15)
        plt.tight_layout()
        # plt.title('Normal Distribution Histogram matplotlib',fontsize=15)
        plt.show()
        return n, bins , patches


    def Drop(self, DropList):
        """
        ======================================================
            Drop(self, DropList)
        ======================================================
        Drop method deletes columns from the EventIndex dataframe

        Inputs:
            1- DropList:
                [list] list of column names to delete from the EventIndex
                dataframe table
        Outputs:
            1-EventIndex:
                [datadrame] the EventIndex dataframe without the columns in the
                Droplist

        """
        dataframe = self.EventIndex.loc[:,:]
        columns = list(dataframe.columns)

        [columns.remove(i) for i in DropList]

        dataframe = dataframe.loc[:,columns]
        self.EventIndex = dataframe


    def Save(self,Path):
        """
        ==========================================================
            Save(self,Path)
        ==========================================================
        Save method saves the EventIndex table

        Parameters
        ----------
        Path : [str]
            Path to where you want to save the table.

        Returns
        -------
        None.

        """
        self.EventIndex.to_csv(Path,header=True,index = None) #index_label = "Index"


    def GetEventBeginning(self, loc):
        """
        ==========================================================
                GetEventBeginning(loc)
        ==========================================================
        EventBeginning method returns the index of the beginning of the event
        in the EventIndex dataframe

        Inputs:
            2-ind:
                [Integer] dataframe index of the day you want to trace back to get the begining
        Output:
            1- ind:
                [Integer] dataframe index of the beginning day of the event
        Example:
            1- if you want to get the beginning of the event that has the highest
            overtopping
            HighOvertoppingInd = EventIndex['Overtopping'].idxmax()
            ind = EventBeginning(HighOvertoppingInd)

        """
        # loc = np.where(self.EventIndex['id'] == day)[0][0]
        # get all the days in the same event before that day as the inundation in the maps may
        # happen due to any of the days before not in this day
        ind = self.EventIndex.index[loc - self.EventIndex.loc[loc,'IndDiff']]
        day = self.EventIndex.loc[ind, 'id']
        return ind, day

        # # filter the dataframe and get only the 'indDiff' and 'id' columns
        # FilteredEvent = self.EventIndex.loc[:,['IndDiff','id']]
        # FilteredEvent['diff'] = FilteredEvent.index - ind
        # # get only days before the day you inputed
        # FilteredEvent = FilteredEvent[FilteredEvent['diff'] <=0 ]
        # # start to search from down to up till you get the first 0 in the IndDiff
        # for i in range(self.EventIndex['Duration'].max()):

        #     if FilteredEvent.loc[len(FilteredEvent)-1-i,'IndDiff'] == 0:
        #         break

        # return FilteredEvent.index[len(FilteredEvent)-1-i]

    def GetEventEnd(self, loc):
        """
        GetEventEnd method returns the index of the beginning of the event
        in the EventIndex dataframe

        Inputs:
            2-ind:
                [Integer] dataframe index of the day you want to trace back to get the begining
        Output:
            1- ind:
                [Integer] dataframe index of the beginning day of the event
        Example:
            1- if you want to get the beginning of the event that has the highest
            overtopping
            HighOvertoppingInd = EventIndex['Overtopping'].idxmax()
            ind = EventBeginning(HighOvertoppingInd)

        """
        # loc = np.where(self.EventIndex['id'] == day)[0][0]
        # get all the days in the same event before that day as the inundation in the maps may
        # happen due to any of the days before not in this day

        # filter the dataframe and get only the 'indDiff' and 'id' columns
        FilteredEvent = self.EventIndex.loc[:,['continue','id']]
        # get only days before the day you inputed
        for i in range(loc+1, len(FilteredEvent)):
            # start search from the following day
            if FilteredEvent.loc[i,'continue'] != 1:
                break

        ind = i-1
        day = self.EventIndex.loc[ind, 'id']

        return ind, day

    def PrepareForPlotting(self,ColumnName):
        """
        ======================================================
            PrepareForPlotting(ColumnName)
        ======================================================
        PrepareForPlotting takes a time series in the EventIndex dataframe
        and fill the days that does not exist in date column and fill it with
        zero to properly plot it without letting the graph mislead the viewer of
        connecting the data over the gap period

        Parameters
        ----------
            1-ColumnName : [String]
                name of the columns you want.

        Returns
        -------
            1-NewDataFrame : [Dataframe]
                dataframe with a date column, and the required column
        """
        NewDataFrame = pd.DataFrame()
        NewDataFrame['date'] = self.ReferenceIndex['date'].tolist()
        NewDataFrame[ColumnName] = 0
        for i in range(len(self.EventIndex)):
            loc = np.where(NewDataFrame['date'] == self.EventIndex.loc[i,'date'])[0][0]
            NewDataFrame.loc[loc,ColumnName] = self.EventIndex.loc[i,ColumnName]

        return NewDataFrame


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



if __name__ == '__main__':
    x = Event()
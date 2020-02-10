# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:57:30 2020

@author: mofarrag
"""
import os
import datetime as dt
import pandas as pd 
import Hapi.Raster as Raster
import numpy as np



class Event():
    # class attributes
    

    ### constructor
    def __init__(self, name):
        # instance attribute
        self.name = name        
        start = dt.datetime(1950,1,1)
        self.start =  start
        self.end = self.start + dt.timedelta(days = 36890)
        
        
        Ref_ind = pd.date_range(self.start,self.end, freq='D')
        
        # the last day is not in the results day Ref_ind[-1]
        # write the number of days + 1 as python does not include the last number in the range
        # 19723 days so write 19724
        self.Reference_index = pd.DataFrame(index = list(range(1,36890+1)))
        self.Reference_index['date'] = Ref_ind[:-1]
    
    # method
    def IndexToDate(self):
        # convert the index into date
        dateFn = lambda x: self.Reference_index.loc[x,'date']
        # get the date of each index
        date = self.EventIndex.applymap(dateFn)
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
                dataframe with columns ['ID','continue', 'IndDiff', 'Duration']
        """
        # read the index file (containing the ID of the days where flood happens (2D 
        # algorithm works))
        EventDays = pd.read_csv(IndexPath,header = None)            
        EventIndex = EventDays.rename(columns={0:'ID'})
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
            if self.EventIndex.loc[i,'ID'] == self.EventIndex.loc[i-1,'ID']+1:
                # then the event continues 
                self.EventIndex.loc[i,'continue'] = 1
                # increase the duration
                self.EventIndex.loc[i,'IndDiff'] = self.EventIndex.loc[i-1,'IndDiff'] +1
                
                self.EventIndex.loc[i,'Duration'] = (self.EventIndex.loc[i,'date'] - self.EventBeginning).days + 1
            else: # if not then the day is the start of another event
                self.EventBeginning = self.EventIndex.loc[i,'date'] 
    
    
    
    def Overtopping(self,OvertoppingPath):
        """
        ===================================================
            Overtopping(self,OvertoppingPath)
        ===================================================
        Overtopping method reads the overtopping file and check if the EventIndex+
        dataframe has already need created by the CreateEventIndex method, it 
        will add the overtopping to it, if not it will create the EventIndex dataframe
        
        Inputs:
            1- OvertoppingPath:
                [String] path including the file name and extention of the Overtopping
                file result from the 1D model
        Outputs:
            1- EventIndex:
                [dataframe] this method creates an instance attribute of type
                dataframe with columns ['ID','continue', 'IndDiff', 'Duration', 
                'Overtopping', 'OvertoppingCum', 'Volume']
        """
        OverTopTotal = pd.read_csv(OvertoppingPath, delimiter = r"\s+", header = None)
        
        if not hasattr(self,"EventIndex"):
            # create the dataframe if the user did not use the CreateEventIndex method to 
            # create the EventIndex dataframe
            self.EventIndex = pd.DataFrame()
            self.EventIndex['ID'] = OverTopTotal[0]
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
                if self.EventIndex.loc[i,'ID'] == self.EventIndex.loc[i-1,'ID']+1:
                    # then the event continues 
                    self.EventIndex.loc[i,'continue'] = 1
                    # increase the duration
                    self.EventIndex.loc[i,'IndDiff'] = self.EventIndex.loc[i-1,'IndDiff'] +1
                    
                    self.EventIndex.loc[i,'Duration'] = (self.EventIndex.loc[i,'date'] - self.EventBeginning).days + 1
                else: # if not then the day is the start of another event
                    self.EventBeginning = self.EventIndex.loc[i,'date'] 
                    
            
        # store the overtoppiung data in the EventIndex dataframe
        self.EventIndex['Overtopping'] = OverTopTotal[1]
        
        self.EventIndex.loc[0,'OvertoppingCum'] = self.EventIndex.loc[0,'Overtopping']
        for i in range(1,len(self.EventIndex)):
            if self.EventIndex.loc[i,'continue'] == 0:
                self.EventIndex.loc[i,'OvertoppingCum'] = self.EventIndex.loc[i,'Overtopping']
            else:
                self.EventIndex.loc[i,'OvertoppingCum'] = self.EventIndex.loc[i,'Overtopping'] + self.EventIndex.loc[i-1,'OvertoppingCum']
        # the volume of water is m3/s for hourly stored and acumulated values 
        # volume = overtopping * 60 *60 = m3
        self.EventIndex.loc[:,'Volume'] = self.EventIndex.loc[:,'OvertoppingCum'] * 60*60
        
    
    
    def OverlayMaps(self, Path, BaseMapF, FilePrefix, ExcludedValue, Compressed,
                    OccupiedCellsOnly, SavePath):
        """
        ==================================================================
          OverlayMaps(self, Path, BaseMapF, FilePrefix, ExcludedValue, 
                      Compressed, OccupiedCellsOnly, SavePath)
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
            3-FilePrefix:
                [String] a string that make the files you want to filter in the folder
                uniq 
            3-ExcludedValue:
                [Numeric] values you want to exclude from exteacted values
            4-Compressed:
                [Bool] if the map you provided is compressed
            5-OccupiedCellsOnly:
                [Bool] if you want to count only cells that is not zero
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
                
        self.DepthValues, NonZeroCells = Raster.OverlayMaps(Path, BaseMapF, FilePrefix,
                                                    ExcludedValue, Compressed,OccupiedCellsOnly)
        
        # NonZeroCells dataframe with the first column as the "file" name and the second column 
        # is the number of cells in each map
        
        NonZeroCells['days'] = [int(i[len(self.DepthPrefix):-4]) for i in NonZeroCells['files'].tolist()]
        # get the numbe of inundated cells in the Event index data frame
        self.EventIndex['cells'] = 0
        for i in range(len(NonZeroCells)):
            # get the location in the EventIndex dataframe
            loc = np.where(NonZeroCells.loc[i,'days'] == self.EventIndex.loc[:,"ID"] )[0][0]
            # store number of cells
            self.EventIndex.loc[loc,'cells'] = NonZeroCells.loc[i,'cells']
    
        # save depths of each sub-basin
        inundatedSubs = list(self.DepthValues.keys())
        for i in range(len(inundatedSubs)):
            np.savetxt(SavePath +"/" + str(inundatedSubs[i]) + ".txt",
                       self.DepthValues[inundatedSubs[i]],fmt="%4.2f")
    
    
    
    def VolumeError(self, Path):
        
        VolError = pd.read_csv(Path, delimiter = r'\s+')
        self.EventIndex['DEMError'] = 0
        self.EventIndex['StepError'] = 0
        self.EventIndex['TooMuchWater'] = 0
        for i in range(len(VolError)):
            loc = np.where(VolError.loc[i,'step'] == self.EventIndex.loc[:,'ID'])[0][0]
            self.EventIndex.loc[loc,['DEMError','StepError','TooMuchWater']] = VolError.loc[i,['DEM_Error', 'PreviousDepthError', 'TOOMuchWaterError']].tolist()
        
        self.EventIndex['VolError']  = self.EventIndex['StepError'] + self.EventIndex['DEMError'] + self.EventIndex['TooMuchWater'] 
        self.EventIndex['VolError2'] = self.EventIndex['VolError'] / 20
        self.EventIndex['VolError2'] = self.EventIndex['VolError'] / 20
    
    def ReadEventIndex(self,Path):
        EventIndex = pd.read_csv(Path)
        self.EventIndex = EventIndex
        
        
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
        
    
    def Save(self):
        self.EventIndex.to_csv("EventIndex.txt",header=True,index_label = "Index")    
        
        
    def GetEventBeginning(self, loc):
        """
        EventBeginning method returns the index of the beginning of the event
        in the EventIndex dataframe
        
        Inputs:
            2-ind:
                [Integer] index of the day you want to trace back to get the begining
        Output:
            1- ind:
                [Integer] index of the beginning day of the event
        Example:
            1- if you want to get the beginning of the event that has the highest 
            overtopping 
            HighOvertoppingInd = EventIndex['Overtopping'].idxmax()
            ind = EventBeginning(HighOvertoppingInd)
        
        """
        # loc = np.where(self.EventIndex['ID'] == day)[0][0]
        # get all the days in the same event before that day as the inundation in the maps may 
        # happen due to any of the days before not in this day
        return self.EventIndex.index[loc - self.EventIndex.loc[loc,'IndDiff']]
        
        # # filter the dataframe and get only the 'indDiff' and 'ID' columns
        # FilteredEvent = self.EventIndex.loc[:,['IndDiff','ID']]
        # FilteredEvent['diff'] = FilteredEvent.index - ind
        # # get only days before the day you inputed
        # FilteredEvent = FilteredEvent[FilteredEvent['diff'] <=0 ]
        # # start to search from down to up till you get the first 0 in the IndDiff
        # for i in range(self.EventIndex['Duration'].max()):
            
        #     if FilteredEvent.loc[len(FilteredEvent)-1-i,'IndDiff'] == 0:
        #         break
            
        # return FilteredEvent.index[len(FilteredEvent)-1-i]

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
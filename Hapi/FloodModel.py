# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:57:30 2020

@author: mofarrag
"""
import datetime as dt
import pandas as pd 
import Hapi.GISpy as GIS
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
        date = self.EvendIndex.applymap(dateFn)
        self.EvendIndex['date'] = date
        
        
    
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
        EvendIndex = EventDays.rename(columns={0:'ID'})
        # convert the index into date
        self.EvendIndex = EvendIndex.loc[:,:]
        self.IndexToDate()
        
        self.EvendIndex.loc[:,'continue'] = 0
        # index difference maybe different than the duration as there might be 
        # a gap in the middle of the event
        self.EvendIndex.loc[:,'IndDiff'] = 0
        self.EvendIndex.loc[:,'Duration'] = 0
        
        # the first day in the index file is an event beginning
        self.EventBeginning = self.EvendIndex.loc[0,'date']
        for i in range(1,len(self.EvendIndex)):
            # if the day is previous day+1
            if self.EvendIndex.loc[i,'ID'] == self.EvendIndex.loc[i-1,'ID']+1:
                # then the event continues 
                self.EvendIndex.loc[i,'continue'] = 1
                # increase the duration
                self.EvendIndex.loc[i,'IndDiff'] = self.EvendIndex.loc[i-1,'IndDiff'] +1
                
                self.EvendIndex.loc[i,'Duration'] = (self.EvendIndex.loc[i,'date'] - self.EventBeginning).days + 1
            else: # if not then the day is the start of another event
                self.EventBeginning = self.EvendIndex.loc[i,'date'] 
    
    
    
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
        
        if not hasattr(self,"EvendIndex"):
            # create the dataframe if the user did not use the CreateEventIndex method to 
            # create the EventIndex dataframe
            self.EvendIndex = pd.DataFrame()
            self.EvendIndex['ID'] = OverTopTotal[0]
            self.IndexToDate()
            
            self.EvendIndex.loc[:,'continue'] = 0
            # index difference maybe different than the duration as there might be 
            # a gap in the middle of the event
            self.EvendIndex.loc[:,'IndDiff'] = 0
            self.EvendIndex.loc[:,'Duration'] = 0
            
            # the first day in the index file is an event beginning
            self.EventBeginning = self.EvendIndex.loc[0,'date']
            for i in range(1,len(self.EvendIndex)):
                # if the day is previous day+1
                if self.EvendIndex.loc[i,'ID'] == self.EvendIndex.loc[i-1,'ID']+1:
                    # then the event continues 
                    self.EvendIndex.loc[i,'continue'] = 1
                    # increase the duration
                    self.EvendIndex.loc[i,'IndDiff'] = self.EvendIndex.loc[i-1,'IndDiff'] +1
                    
                    self.EvendIndex.loc[i,'Duration'] = (self.EvendIndex.loc[i,'date'] - self.EventBeginning).days + 1
                else: # if not then the day is the start of another event
                    self.EventBeginning = self.EvendIndex.loc[i,'date'] 
                    
            
        # store the overtoppiung data in the EventIndex dataframe
        self.EvendIndex['Overtopping'] = OverTopTotal[1]
        
        self.EvendIndex.loc[0,'OvertoppingCum'] = self.EvendIndex.loc[0,'Overtopping']
        for i in range(1,len(self.EvendIndex)):
            if self.EvendIndex.loc[i,'continue'] == 0:
                self.EvendIndex.loc[i,'OvertoppingCum'] = self.EvendIndex.loc[i,'Overtopping']
            else:
                self.EvendIndex.loc[i,'OvertoppingCum'] = self.EvendIndex.loc[i,'Overtopping'] + self.EvendIndex.loc[i-1,'OvertoppingCum']
        # the volume of water is m3/s for hourly stored and acumulated values 
        # volume = overtopping * 60 *60 = m3
        self.EvendIndex.loc[:,'Volume'] = self.EvendIndex.loc[:,'OvertoppingCum'] * 60*60
        
    
    
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
                
        self.DepthValues, NonZeroCells = GIS.OverlayMaps(Path, BaseMapF, FilePrefix,
                                                    ExcludedValue, Compressed,OccupiedCellsOnly)
        
        # NonZeroCells dataframe with the first column as the "file" name and the second column 
        # is the number of cells in each map
        
        NonZeroCells['days'] = [int(i[len(self.DepthPrefix):-4]) for i in NonZeroCells['files'].tolist()]
        # get the numbe of inundated cells in the Event index data frame
        self.EvendIndex['cells'] = 0
        for i in range(len(NonZeroCells)):
            # get the location in the EventIndex dataframe
            loc = np.where(NonZeroCells.loc[i,'days'] == self.EvendIndex.loc[:,"ID"] )[0][0]
            # store number of cells
            self.EvendIndex.loc[loc,'cells'] = NonZeroCells.loc[i,'cells']
    
        # save depths of each sub-basin
        inundatedSubs = list(self.DepthValues.keys())
        for i in range(len(inundatedSubs)):
            np.savetxt(SavePath +"/" + str(inundatedSubs[i]) + ".txt",
                       self.DepthValues[inundatedSubs[i]],fmt="%4.2f")
    
    
    
    def VolumeError(self, Path):
        
        VolError = pd.read_csv(Path, delimiter = r'\s+')
        self.EvendIndex['DEMError'] = 0
        self.EvendIndex['StepError'] = 0
        self.EvendIndex['TooMuchWater'] = 0
        for i in range(len(VolError)):
            loc = np.where(VolError.loc[i,'step'] == self.EvendIndex.loc[:,'ID'])[0][0]
            self.EvendIndex.loc[loc,['DEMError','StepError','TooMuchWater']] = VolError.loc[i,['DEM_Error', 'PreviousDepthError', 'TOOMuchWaterError']].tolist()
        
        self.EventIndex['VolError']  = self.EventIndex['StepError'] + self.EventIndex['DEMError'] + self.EventIndex['TooMuchWater'] 

    
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
        self.EvendIndex.to_csv("EventIndex.txt",header=True,index_label = "Index")    
    

    
    def AddAttribute(self, AddDict):
        
        
        for i in list(AddDict.items()):
            exec("self." + i[0] + "= 0") 
            exec("self." + i[0] + "=i[1]")
        
#        dataframe = self.EventIndex.loc[:,:]
#        columns = list(dataframe.columns)
        
#        [columns.remove(i) for i in DropList]
        
#        dataframe = dataframe.loc[:,columns]
#        self.EventIndex = dataframe
        
        
        
#    def EventBeginning(EventIndex, ind):
#        """
#        ==================================================
#          EventBeginning(EventIndex, ind)
#        ==================================================
#        
#        EventBeginning is a function to return the index of the beginning of the event
#        in the EventIndex dataframe
#        
#        Inputs:
#            1-EventIndex:
#               [dataframe]  a dataframe that has to contains two columns 'indDiff'
#                   and 'ID', this dataframe was created using the PrepareEventIndex.py
#                   code
#            2-ind:
#                [Integer] index of the day you want to trace back to get the begining
#        Output:
#            1- ind:
#                [Integer] index of the beginning day of the event
#        Example:
#            1- if you want to get the beginning of the event that has the highest 
#            overtopping 
#            HighOvertoppingInd = EventIndex['Overtopping'].idxmax()        
#            ind = FM.EventBeginning(EventIndex, HighOvertoppingInd)
#        
#        """
#        # filter the dataframe and get only the 'indDiff' and 'ID' columns
#        FilteredEvent = EventIndex.loc[:,['IndDiff','ID']]
#        FilteredEvent['diff'] = FilteredEvent.index - ind
#        # get only days before the day you inputed
#        FilteredEvent = FilteredEvent[FilteredEvent['diff'] <=0 ]
#        # start to search from down to up till you get the first 0 in the IndDiff
#        for i in range(EventIndex['Duration'].max()):
#            
#            if FilteredEvent.loc[len(FilteredEvent)-1-i,'IndDiff'] == 0:
#                break
#            
#        return FilteredEvent.index[len(FilteredEvent)-1-i]

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
        
#    def DetailedOverTopping():
        
class River():
    # class attributes
#    CrossSections = pd.DataFrame()
#    RiverNetwork = pd.DataFrame()

    ### constructor
    def __init__(self, name):
        self.name = name
        
    def ReadXS(self,Path):
        self.CrossSections = pd.read_csv(Path, delimiter = ',', skiprows =1  )
        
    
    def ReturnPeriod(self,Path):
        self.RP = pd.read_csv(Path, delimiter = ",",header = None)
        self.RP.columns = ['node','HQ2','HQ10','HQ100']
    
    def Trace(self, Path):
        self.RiverNetwork = pd.read_csv(Path, delimiter = r'\s+',header = None)
        self.RiverNetwork.columns = ['subID','US','DS']
        
        
if __name__ == '__main__':
    x = Event()
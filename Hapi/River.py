# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:04:32 2020

@author: mofarrag
"""
import os 
import pandas as pd
import numpy as np
import datetime as dt
# import Hapi.GISpy as GIS

class River():
    # class attributes


    ### constructor
    def __init__(self, name, leftOvertopping_Suffix = "_left.txt", 
                 RightOvertopping_Suffix = "_right.txt",start = "1950-1-1", days = 36890):
        self.name = name
        self.start = dt.datetime.strptime(start,"%Y-%m-%d")
        self.end = self.start + dt.timedelta(days = days)
        
        self.leftOvertopping_Suffix = leftOvertopping_Suffix
        self.RightOvertopping_Suffix = RightOvertopping_Suffix
        self.oneDResultPath = ''
        
        Ref_ind = pd.date_range(self.start,self.end, freq='D')
        
        # the last day is not in the results day Ref_ind[-1]
        # write the number of days + 1 as python does not include the last number in the range
        # 19723 days so write 19724
        self.ReferenceIndex = pd.DataFrame(index = list(range(1,days+1)))
        self.ReferenceIndex['date'] = Ref_ind[:-1]
    
        
    def CrossSections(self,Path):
        """
        ===========================================
          ReadXS(self,Path)
        ===========================================
        ReadXS method reads the cross section data of the river and assign it 
        to an attribute "Crosssections" of type dataframe
        """
        self.crosssections = pd.read_csv(Path, delimiter = ',', skiprows =1  )
        
    def Slope(self,Path):
        self.slope = pd.read_csv(Path, delimiter = ",",header = None)
        self.slope.columns = ['SubID','f1','slope','f2']
        
    def ReturnPeriod(self,Path):
        self.RP = pd.read_csv(Path, delimiter = ",",header = None)
        self.RP.columns = ['node','HQ2','HQ10','HQ100']
    
    def RiverNetwork(self, Path):
        self.rivernetwork = pd.read_csv(Path, delimiter = ',',header = None)
        self.rivernetwork.columns = ['SubID','US','DS']
        
    def Trace(self,SubID):
        """
        ========================================
        	Trace(self,SubID)
        ========================================
        Trace method takes sub basin Id and trace it to get the upstream and 
        down stream computational nodes
        
        
        Parameters
        ----------
            1- SubID : TYPE
                DESCRIPTION.

        Returns
        -------
            1-SWIMUS : [Integer]
                the Upstream computational node from SWIM Configuration file.
            2-SWIMDS : [Integer]
                the Downstream computational node from SWIM Configuration file.
        
        Example:
            SubID = 42
            SWIMUS, SWIMDS = River1.Trace(SubID)
            print("sub-basin = "+str(SubID))
            print("DS node= "+str(SWIMDS))
            print("US node= "+str(SWIMUS))
        """
        
        
        SWIMDS = int(self.rivernetwork['DS'][np.where(self.rivernetwork['SubID'] == SubID)[0][0]])
        SWIMUS = int(self.rivernetwork['US'][np.where(self.rivernetwork['SubID'] == SubID)[0][0]])
        
        return  SWIMUS, SWIMDS
    
    
    def Overtopping(self):

        """ 
        =====================================================
          Overtopping(self, OvertoppingResultF)
        =====================================================
        
        Overtopping reads the over topping files and to do analysis 
        you do not need to delete empty files or anything just give the code 
        the sufix you used for the left overtopping file and the sufix you used 
        for the right overtopping file
        
        Inputs:
            1-OvertoppingResultF:
                [String] a path to the folder includng 2D results. 
        """        
        
        #sort files 
        leftOverTop = list()
        RightOverTop = list()
        # get names of files that has _left or _right at its end
        All1DFiles = os.listdir(self.oneDResultPath)
        for i in range(len(All1DFiles)) :
            if All1DFiles[i].endswith(self.leftOvertopping_Suffix):
                leftOverTop.append(All1DFiles[i])
            if All1DFiles[i].endswith(self.RightOvertopping_Suffix):
                RightOverTop.append(All1DFiles[i])
                
        # two dictionaries for overtopping left and right 
        OverToppingSubsLeft = dict()
        OverToppingSubsRight = dict()
        # the _left and _right files has all the overtopping discharge
        # but sometimes the sum of all the overtopping is less than a threshold specified 
        # and then the 2D  algorithm does not run so these cross sections you will not find 
        # any inundation beside it in the maps but you will find it in the _left or _right maps
        
        # for each sub-basin that has overtopping from the left dike
        for i in range(len(leftOverTop)):
            
            try:
                # open the file (if there is no column sthe file is empty)
                data = pd.read_csv(self.oneDResultPath + leftOverTop[i],header =None,delimiter = r'\s+')
                # add the sub basin to the overtopping dictionary of sub-basins
                OverToppingSubsLeft[leftOverTop[i][:-len(self.leftOvertopping_Suffix)]] = dict()
            except:
                continue
            # get the XS that overtopping happened from
            XSs = list(set(data.loc[:,2]))
            # for each XS get the days
            for j in range(len(XSs)):
                OverToppingSubsLeft[leftOverTop[i][:-len(self.leftOvertopping_Suffix)]][XSs[j]] = list(set(data[0][data[2] == XSs[j]].tolist()))
        
        for i in range(len(RightOverTop)):
            
            try:
                # open the file 
                data = pd.read_csv(self.oneDResultPath + RightOverTop[i],header =None,delimiter = r'\s+')
                # add the sub basin to the overtopping dictionary of sub-basins
                OverToppingSubsRight[RightOverTop[i][:-len(self.RightOvertopping_Suffix)]] = dict()
            except :
                continue
            # get the XS that overtopping happened from
            XSs = list(set(data.loc[:,2]))
            # for each XS get the days
            for j in range(len(XSs)):
                OverToppingSubsRight[RightOverTop[i][:-len(self.RightOvertopping_Suffix)]][XSs[j]] = list(set(data[0][data[2] == XSs[j]].tolist()))
        
        self.OverToppingSubsLeft = OverToppingSubsLeft
        self.OverToppingSubsRight = OverToppingSubsRight
        
    
    def GetOvertoppedXS(self, day, allEventdays = True):
        """
        ==========================================================
             GetOvertoppedXS( day, allEventdays = True)
        ==========================================================
        GetOvertoppedXS method get the cross section that was inundated in 
        a given date(you have to read the overtopping data first with the method
        Overtopping), since inudation maps gets the max depth for the whole event
        the method can also trace the event back to the beginning and get all
        the overtopped XS from the beginning of the Event till the given day
        (you have to give the River object the EventIndex attribute from the 
        Event Object)
        
        Parameters
        ----------
        
            1-day : [Integer]
                the day you want to get the overtopped cross section for.
            2-allEventdays : [Bool], optional
                if you want to get the overtopped cross section for this day only
                or for the whole event. The default is True.

        Returns
        -------
            1-XSLeft : [list]
                list of cross section IDs that has overtopping from the left bank.
            2-XSRight : [list]
                list of cross section IDs that has overtopping from the right bank.
        Example:
            1- for a given day
                RIM2River = RV.River('RIM2.0')
                RIM2River.Overtopping(wpath2 + "/results/1d/")
                day = 1122
                XSleft, XSright = RIM2River.GetOvertoppedXS(day,False)
                
            2- from the beginning of the event till the given day
                RIM2River = RV.River('RIM2.0')
                RIM2River.Overtopping(wpath2 + "/results/1d/")
                # read precreated EventIndex table
                RIM2Event.ReadEventIndex(wpath2 + "/" + "EventIndex.txt")
                # give the EventIndex table to the River Object
                RIM2River.EventIndex = RIM1.EventIndex
                day = 1122
                XSleft, XSright = RIM2River.GetOvertoppedXS(day,False)
            
        """
        if allEventdays:
            loc = np.where(self.EventIndex['ID'] == day)[0][0]
            # get all the days in the same event before that day as the inundation in the maps may 
            # happen due to any of the days before not in this day
            Eventdays = self.EventIndex.loc[loc - self.EventIndex.loc[loc,'IndDiff'] : loc,'ID'].tolist()
        else:
            Eventdays = [day,]
            
        XSLeft = list()
        XSRight = list()
        
        for k in range(len(Eventdays)):
            dayi = Eventdays[k]
            # for each sub-basin in the overtopping left dict
            for i in range(len(self.OverToppingSubsLeft.keys())):
                SubID = list(self.OverToppingSubsLeft.keys())[i]
                # get all cross section that overtopped before
                XSs = list(self.OverToppingSubsLeft[SubID].keys())
                # for each xross section check if the day is sored inside 
                for j in range(len(XSs)):
                    if dayi in self.OverToppingSubsLeft[SubID][XSs[j]]:
                        XSLeft.append(XSs[j])
            
            for i in range(len(self.OverToppingSubsRight.keys())):
                SubID = list(self.OverToppingSubsRight.keys())[i]
                XSs = list(self.OverToppingSubsRight[SubID].keys())
                for j in range(len(XSs)):
                    if dayi in self.OverToppingSubsRight[SubID][XSs[j]]:
                        XSRight.append(XSs[j])

        XSLeft = list(set(XSLeft))
        XSRight = list(set(XSRight))
        
        return XSLeft,  XSRight
    
    def GetSubBasin(self,xsid):
        loc = np.where(self.crosssections['xsid'] == xsid)[0][0]
        return self.crosssections.loc[loc,'swimid']
    
    
    def GetFloodedSubs(self, OvertoppedXS = [], day = [1,], allEventdays = True):
        """
        =============================================================================
            GetFloodedSubs(OvertoppedXS = [], day = [1,], allEventdays = True)
        =============================================================================
        GetFloodedSubs gets the inundeated sub-basins 
        
        Parameters
        ----------
            1-OvertoppedXS : [list], optional
                list of cross sections overtopped (if you already used the GetOvertoppedXS
                method to get the overtopped XSs for a specific day).The default is [].
                If entered the algorithm is not going to look at the over arguments
                of the method.
            2-day : [list], optional
                if you want to get the flooded subs for a specific list of days. The default is 1.
            3-allEventdays : [Bool], optional in case user entered OvertoppedXS
                if the user entered day the allEventdays is a must. The default is True.

        Returns
        -------
            1-Subs : TYPE
                DESCRIPTION.
        
        Examples:
            1- get the flooded subs for a specific days
                floodedSubs = RIM1River.GetFloodedSubs(day = [1122,1123], allEventdays=False)
            
            2- get the flooded subs from already obtained overtopped XSs
                day = 1122
                XSleft, XSright = RIM1River.GetOvertoppedXS(day,False)
                floodedSubs = RIM1River.GetFloodedSubs(OvertoppedXS = XSleft + XSright, allEventdays=False)
        """
        Subs = list()
        # if you already used the GetOvertoppedXS and have a list of xs overtopped
        # at specific day
        if len(OvertoppedXS) > 0:
            OvertoppedXS = list(set(OvertoppedXS))
            for i in range(len(OvertoppedXS)):
                Subs.append(self.GetSubBasin(OvertoppedXS[i]))
        else:
            for j in range(len(day)):
                XSLeft, XSRight = self.GetOvertoppedXS(day[j], allEventdays)
                OvertoppedXS = XSLeft + XSRight
                OvertoppedXS = list(set(OvertoppedXS))
                
                for i in range(len(OvertoppedXS)):
                    Subs.append(self.GetSubBasin(OvertoppedXS[i]))
        
        # to remove duplicate subs
        Subs = list(set(Subs))
        return Subs
    
    def DetailedOvertopping(self,floodedSubs,eventdays):
        """
        ========================================================
           DetailedOvertopping(floodedSubs,eventdays)
        ========================================================
        DetailedOvertopping method takes list of days ond the flooded subs-basins 
        in those days and get the left and right overtopping for each sub-basin for
        each day
        
        Parameters
        ----------
            1-floodedSubs : [list]
                list of sub-basins that are flooded during the given days.
            2-eventdays : [list]
                list od daysof an event.

        """
        columns = floodedSubs + ['sum']
        self.DetailedOvertoppingLeft  = pd.DataFrame(index = eventdays + ['sum'], columns = columns)
        self.DetailedOvertoppingRight = pd.DataFrame(index = eventdays + ['sum'], columns = columns)
        
        # Left Bank
        for i in range(len(floodedSubs)):
            try :
                # try to open and read the overtopping file 
                data = pd.read_csv(self.oneDResultPath + str(floodedSubs[i]) + self.leftOvertopping_Suffix,
                                   header =None,delimiter = r'\s+')
                # get the days in the sub
                days = list(set(data.loc[:,0]))
                
                for j in range(len(eventdays)):
                    # check whether this sub basin has flooded in this particular day
                    if eventdays[j] in days:
                        # filter the dataframe to the discharge column (3) and the days
                        self.DetailedOvertoppingLeft.loc[eventdays[j],floodedSubs[i]] = data.loc[data[0] == eventdays[j] ,3].sum()
                    else:
                        self.DetailedOvertoppingLeft.loc[eventdays[j],floodedSubs[i]] = 0
            except:
                self.DetailedOvertoppingLeft.loc[:,floodedSubs[i]] = 0
                continue
            
        # right Bank
        for i in range(len(floodedSubs)):
            try :
                # try to open and read the overtopping file 
                data = pd.read_csv(self.oneDResultPath + str(floodedSubs[i]) + self.RightOvertopping_Suffix,
                                   header =None,delimiter = r'\s+')
                # get the days in the sub
                days = list(set(data.loc[:,0]))
                
                for j in range(len(eventdays)):
                    # check whether this sub basin has flooded in this particular day
                    if eventdays[j] in days:
                        # filter the dataframe to the discharge column (3) and the days
                        self.DetailedOvertoppingRight.loc[eventdays[j],floodedSubs[i]] = data.loc[data[0] == eventdays[j] ,3].sum()
                    else:
                        self.DetailedOvertoppingRight.loc[eventdays[j],floodedSubs[i]] = 0
            except:
                self.DetailedOvertoppingRight.loc[eventdays[j],floodedSubs[i]] = 0
                continue
            
        # sum overtopping for each day
        for j in range(len(eventdays)):
            self.DetailedOvertoppingLeft.loc[eventdays[j],'sum'] = self.DetailedOvertoppingLeft.loc[eventdays[j],:].sum()
            self.DetailedOvertoppingRight.loc[eventdays[j],'sum'] = self.DetailedOvertoppingRight.loc[eventdays[j],:].sum()
        # sum overtopping for each sub basin
        for j in range(len(floodedSubs)):
            self.DetailedOvertoppingLeft.loc['sum',floodedSubs[j]] = self.DetailedOvertoppingLeft.loc[:,floodedSubs[j]].sum()
            self.DetailedOvertoppingRight.loc['sum',floodedSubs[j]] = self.DetailedOvertoppingRight.loc[:,floodedSubs[j]].sum()
        
        # self.DetailedOvertoppingLeft.loc['sum','sum'] = self.DetailedOvertoppingLeft.loc[:,'sum'].sum()
        # self.DetailedOvertoppingRight.loc['sum','sum'] = self.DetailedOvertoppingRight.loc[:,'sum'].sum()
    
    def Coordinates(self, Bankful = False):
        if Bankful == True:
            self.crosssections = self.crosssections.assign(x1 =0,y1=0,z1=0,
                                                           x2=0,y2=0,z2=0,
                                                           x3=0,y3=0,z3=0,
                                                           x4=0,y4=0,z4=0,
                                                           x5=0,y5=0,z5=0,
                                                           x6=0,y6=0,z6=0,
                                                           x7=0,y7=0,z7=0,
                                                           x8=0,y8=0,z8=0)
            
            for i in range(len(self.crosssections)):
                inputs = self.crosssections.loc[i,list(self.crosssections.columns)[3:15]].tolist()
                Dbf = self.crosssections.loc[i,list(self.crosssections.columns)[16]]
                
                outputs = self.GetCoordinates(inputs,Dbf)
                
                self.crosssections.loc[i,['x1','x2','x3','x4','x5','x6','x7','x8']] = outputs[0]
                
                self.crosssections.loc[i,['y1','y2','y3','y4','y5','y6','y7','y8']] = outputs[1]
                
                self.crosssections.loc[i,['z1','z2','z3','z4','z5','z6','z7','z8']] = outputs[2]
        else:
            self.crosssections = self.crosssections.assign(x1 =0,y1=0,z1=0,
                                                           x2=0,y2=0,z2=0,
                                                           x3=0,y3=0,z3=0,
                                                           x4=0,y4=0,z4=0,
                                                           x5=0,y5=0,z5=0,
                                                           x6=0,y6=0,z6=0)
            Dbf = False
            for i in range(len(self.crosssections)):
                inputs = self.crosssections.loc[i,list(self.crosssections.columns)[3:15]].tolist()
                
                outputs = self.GetCoordinates(inputs, Dbf)
                
                self.crosssections.loc[i,['x1','x2','x3','x4','x5','x6']] = outputs[0]
                
                self.crosssections.loc[i,['y1','y2','y3','y4','y5','y6']] = outputs[1]
                
                self.crosssections.loc[i,['z1','z2','z3','z4','z5','z6']] = outputs[2]
                
        # TODO create a method to take the created coordinates and convert each cross section 
        # into  a polygon
        # TODO another method to take the cross section coordinates of a whole sub basins
        # and convert them into one polygon
    # def CreatePolygons(self):
        
    @staticmethod
    def PolygonGeometry(Coords):
        """
        ======================================
            PolygonGeometry(Coords)
        ======================================
        PolygonGeometry method calculates the area and perimeter of some coordinates 
        
            Parameters
            ----------
                1-Coords : [array] 
                    numpy array in the shape of (n*2) where n is the number of 
                    points
    
            Returns
            -------
                1-area : [float]
                    area between the coordinates.
                2-peri : [float]
                    perimeter between the coordinates.
                
            Example:
            -------    
                coords = np.array([[0,1],[0,0],[5,0],[5,1]])
                RV.River.PolygonGeometry(coords)

        """
        area = 0.0
        peri = 0.0
        for i in range(np.shape(Coords)[0]-1):
            area = area + Coords[i,0]*Coords[i+1,1] - Coords[i+1,0]*Coords[i,1]
            peri = peri + ( (Coords[i+1,0] - Coords[i,0])**2 + (Coords[i+1,1]-Coords[i,1])**2 )**0.5
        area = area + Coords[np.shape(Coords)[0]-1,0] * Coords[0,1] - Coords[0,0] * Coords[np.shape(Coords)[0]-1,1]
        area = area*0.5
        
        return area, peri
    
    @staticmethod
    def PolyArea(Coords):
        """
        =================================
            PolyArea(Coords)
        =================================
        PolyArea method calculates the the area between given coordinates
        
            Parameters
            ----------
                1-Coords : [array] 
                    numpy array in the shape of (n*2) where n is the number of 
                    points
    
            Returns
            -------
                1-area : [float]
                    area between the coordinates.
            
            Example:
            -------    
                coords = np.array([[0,1],[0,0],[5,0],[5,1]])
                River.PolyArea(coords)
        """
        
        area = 0.0
        for i in range(np.shape(Coords)[0]-1):
            # cros multiplication
            area = area + Coords[i,0]*Coords[i+1,1] - Coords[i+1,0]*Coords[i,1]
        area = area + Coords[np.shape(Coords)[0]-1,0] * Coords[0,1] - Coords[0,0] * Coords[np.shape(Coords)[0]-1,1]
        area = area*0.5
        
        return area
    
    @staticmethod
    def PolyPerimeter(Coords):
        """
        ====================================
            PolyPerimeter
        ====================================
        PolyPerimeter method calculates the the perimeter between given coordinates
            
            Parameters
            ----------
                1-Coords : [array] 
                    numpy array in the shape of (n*2) where n is the number of 
                    points
    
            Returns
            -------
                2-peri : [float]
                    perimeter between the coordinates.
                
            Example:
            -------    
                coords = np.array([[0,1],[0,0],[5,0],[5,1]])
                RV.River.PolyPerimeter(coords)

        """
        peri = 0.0
        for i in range(np.shape(Coords)[0]-1):
            # next point coord - current point coord
            peri = peri + ( (Coords[i+1,0] - Coords[i,0])**2 + (Coords[i+1,1] - Coords[i,1])**2 )**0.5
        
        return peri
    
    @staticmethod
    def GetCoordinates(XSGeometry, Dbf):
        """
        GetCoordinates calculates the coordinates of all the points (vortices)
        of the cross section 
        
        Parameters
        ----------
            1- BedLevel : [Float]
                DESCRIPTION.
            2- BankLeftLevel : [Float]
                DESCRIPTION.
            3- BankRightLevel : [Float]
                DESCRIPTION.
            4- InterPLHeight : [Float]
                DESCRIPTION.
            5- InterPRHeight : [Float]
                DESCRIPTION.
            6- Bl : [Float]
                DESCRIPTION.
            7- Br : [Float]
                DESCRIPTION.
            8- xl : [Float]
                DESCRIPTION.
            9- yl : [Float]
                DESCRIPTION.
            10- xr : [Float]
                DESCRIPTION.
            11- yr : [Float]
                DESCRIPTION.
            12- B : [Float]
                DESCRIPTION.
            13- Dbf : [Float/Bool]
                DESCRIPTION.

        Returns
        -------
        Xcoords : [List]
            DESCRIPTION.
        Xcoords : [List]
            DESCRIPTION.
        Zcoords : [List]
            DESCRIPTION.

        """
        BedLevel = XSGeometry[0]
        BankLeftLevel = XSGeometry[1]
        BankRightLevel = XSGeometry[2]
        InterPLHeight = XSGeometry[3]
        InterPRHeight = XSGeometry[4]
        Bl = XSGeometry[5]
        Br = XSGeometry[6]
        xl = XSGeometry[7]
        yl = XSGeometry[8]
        xr = XSGeometry[9]
        yr = XSGeometry[10]
        B = XSGeometry[11]
        
        Xcoords = list()
        Ycoords = list()
        Zcoords = list()
        # point 1 
        Xcoords.append(xl)
        Ycoords.append(yl)
        Zcoords.append(BankLeftLevel)
        # 8 points cross sections
        if Dbf != False:
            # point 2
            Xcoords.append(xl)
            Ycoords.append(yl)
            Zcoords.append(BedLevel + Dbf + InterPLHeight)
            # point 3
            Xcoords.append(xl + (Bl / (Bl + Br + B)) * (xl- xr))
            Ycoords.append(yl + (Bl / (Bl + Br + B)) * (yl- yr))
            Zcoords.append(BedLevel + Dbf)
            # point 4
            Xcoords.append(xl + (Bl / (Bl + Br + B)) * (xl- xr))
            Ycoords.append(yl + (Bl / (Bl + Br + B)) * (yl- yr))
            Zcoords.append(BedLevel)
            # point 5
            Xcoords.append(xl + ((Bl+B) / (Bl + Br + B)) * (xl- xr))
            Ycoords.append(yl + ((Bl+B) / (Bl + Br + B)) * (yl- yr))
            Zcoords.append(BedLevel)
            # point 6
            Xcoords.append(xl + ((Bl+B) / (Bl + Br + B)) * (xl- xr))
            Ycoords.append(yl + ((Bl+B) / (Bl + Br + B)) * (yl- yr))
            Zcoords.append(BedLevel + Dbf)
            # point 7
            Xcoords.append(xr)
            Ycoords.append(yr)
            Zcoords.append(BedLevel + Dbf + InterPRHeight)
        else:
            # point 2
            Xcoords.append(xl)
            Ycoords.append(yl)
            Zcoords.append(BedLevel + InterPLHeight)
            # point 3
            Xcoords.append(xl + (Bl / (Bl + Br + B)) * (xl- xr))
            Ycoords.append(yl + (Bl / (Bl + Br + B)) * (yl- yr))
            Zcoords.append(BedLevel)
            # point 4
            Xcoords.append(xl + ((Bl+B) / (Bl + Br + B)) * (xl- xr))
            Ycoords.append(yl + ((Bl+B) / (Bl + Br + B)) * (yl- yr))
            Zcoords.append(BedLevel)
            # point 5
            Xcoords.append(xr)
            Ycoords.append(yr)
            Zcoords.append(BedLevel + InterPRHeight)
        
        # point 8
        Xcoords.append(xr)
        Ycoords.append(yr)
        Zcoords.append(BankRightLevel)
        
        return Xcoords, Ycoords, Zcoords
    
    def GetDays(self,FromDay,ToDay,SubID):
        """
        ========================================================
            GetDays(FromDay,ToDay)
        ========================================================
        GetDays method check if input days exist in the 1D result data
        or not since RIM1.0 simulate only days where discharge is above 
        a certain value (2 years return period), you have to enter the 
        oneDResultPath attribute of the instance first to read the results of 
        the given sub-basin
        
        Parameters
        ----------
            1-FromDay : [integer]
                the day you want to read the result from.
            2-ToDay : [integer]
                the day you want to read the result to.

        Returns
        -------
            1-Message stating whether the given days exist or not, and if not two 
            alternatives are given instead.

        """
        
        data = pd.read_csv(self.oneDResultPath + str(SubID) +'.txt', 
                               header =None,delimiter = r'\s+')
        data.columns=["day" , "hour", "xs", "q", "h", "wl"]
        days = list(set(data['day']))
        days.sort()
        
        if FromDay not in days:
            Alt1 = FromDay
            
            stop = 0    
            # search for the FromDay in the days column
            while stop == 0: 
            # for i in range(0,10):
                try:
                    loc = np.where(data['day'] == Alt1)[0][0]
                    stop = 1
                except:
                    Alt1 = Alt1 - 1
                    # print(Alt1)
                    if Alt1 <= 0 : 
                        stop = 1
                    continue
                
            Alt2 = FromDay
            # FromDay = 
            # search for closest later days
            stop = 0
            while stop == 0: 
            # for i in range(0,10):
                try:
                    loc = np.where(data['day'] == Alt2)[0][0]
                    stop = 1
                except:
                    Alt2 = Alt2 + 1
                    # print(Alt2)
                    if Alt2 >= data.loc[len(data)-1,'day']: 
                        stop = 1
                    continue
            
            text = """"
            the FromDay you entered does not exist in the data, and the closest day earlier than your input day is
            """ + str(Alt1) + """  and the closest later day is """ + str(Alt2)
            print(text)
        
        else:
            print("FromDay you entered does exist in the data ")
            
            
        if ToDay not in days:
            Alt1 = ToDay
            
            stop = 0
            # search for the FromDay in the days column
            while stop == 0: 
            # for i in range(0,10):
                try:
                    loc = np.where(data['day'] == Alt1)[0][0]
                    stop = 1
                except:
                    Alt1 = Alt1 - 1
                    # print(Alt1)
                    if Alt1 <= 0 : 
                        stop = 1
                    continue
                
            Alt2 = ToDay
            # FromDay = 
            # search for closest later days
            stop = 0
            while stop == 0: 
            # for i in range(0,10):
                try:
                    loc = np.where(data['day'] == Alt2)[0][0]
                    stop = 1
                except:
                    Alt2 = Alt2 + 1
                    # print(Alt2)
                    if Alt2 >= data.loc[len(data)-1,'day']: 
                        stop = 1
                    continue
            
            text = """" 
            the Today you entered does not exist in the data, and the closest day earlier than your input day is
            """ + str(Alt1) + """  and the closest later day is """ + str(Alt2)
            print(text)
        else:
            print("ToDay you entered does exist in the data ")
        
        
        
    def Read1DResult(self, SubID, FromDay = [], ToDay = [], Path = '', FillMissing = False):
        
        # if the inputFromDay or ToDay was integer convert it to list
        # if type(FromDay) == int:
        #     FromDay = [FromDay,]
        # if type(ToDay) == int:
        #     ToDay = [ToDay,]
        
        # if the path is not given try to read from the object predefined OneDresultPath
        if Path == '':
            Path = self.oneDResultPath
            
        data = pd.read_csv( Path + str(SubID) +'.txt', 
                               header =None,delimiter = r'\s+')
        
        data.columns=["day" , "hour", "xs", "q", "h", "wl"]
        days = list(set(data['day']))
        days.sort()
        
        if FromDay != [] and FromDay not in days:
            return print("please use the GetDays method to select FromDay that exist in the data")
        if ToDay != [] and ToDay not in days:
            return print("please use the GetDays method to select FromDay that exist in the data")
        
        # FromDay = FromDay[0]
        # ToDay = ToDay[0]
        
        if FromDay != []:
            data = data.loc[data['day'] >= FromDay,:]
        if ToDay != []:
            data = data.loc[data['day'] <= ToDay]
        
        data.index = list(range(0,len(data)))
        
        # Cross section data
        # Nxs = np.shape(data[data['day'] == data['day'][1]][data['hour'] == 1])[0]    
        XSname = data['xs'][data['day'] == data['day'][1]][data['hour'] == 1].tolist()
        
        if FillMissing == True:
            # check if there is missing days (Q was < threshold so the model didn't run)
            # fill these values with 0
            days = list(set(data['day']))
            days.sort()
            #hours = list(range(0,25))
            hours = list(range(1,25))
            
            missing_days = list()
            for i in range(days[0],days[-1]):
                if i not in days:
                    missing_days.append(i)
                    print("day = " + str(i) + " is missing")
            
            
            if len(missing_days) > 0:
                if len(missing_days) > 10000:
                    missing_1 = list()
                    missing_2 = list()
                    missing_3 = list()
                    for i in missing_days:
                        for j in hours:
                            for h in XSname:
                                missing_1.append(i)
                                missing_2.append(j)
                                missing_3.append(h)
                else :
                    missing = [(i,j,h) for i in missing_days for j in hours for h in XSname]
                    missing_1 = [i[0] for i in missing]
                    missing_2 = [i[1] for i in missing]
                    missing_3 = [i[2] for i in missing]
                    
                print("done")
                missing = pd.DataFrame(index=range(len(missing_1)) ,dtype = np.float64)
                
            #    i=0
            #    h=100000
            #    if len(missing_1) > h:
            #        while (i+1)*h-1 < len(missing_1):
            #            missing.loc[i*h:(i+1)*h-1,'day'] = missing_1[i*h:(i+1)*h]
            #            missing.loc[i*h:(i+1)*h-1,'hour'] = missing_2[i*h:(i+1)*h]
            #            missing.loc[i*h:(i+1)*h-1,'xs']  = missing_3[i*h:(i+1)*h]
            #            print(i)
            #            i=i+1
            #            
            #        missing.loc[i*h:len(missing_1),'day'] = missing_1[i*h:len(missing_1)]
            #        missing.loc[i*h:len(missing_1),'hour'] = missing_2[i*h:len(missing_1)]
            #        missing.loc[i*h:len(missing_1),'xs'] = missing_3[i*h:len(missing_1)]
            #    else:    
                missing['day'] = missing_1
                missing['hour'] = missing_2
                missing['xs'] = missing_3
                
                missing['q'] = 0
                missing['h'] = 0
                missing['wl'] = 0
            #    missing['area'] = 0
            #    missing['perimeter'] = 0
                data  = data.append(missing)
                
                del missing, missing_1, missing_2, missing_3, missing_days
                data = data.sort_values(by=['day','hour','xs'], ascending=True)
                data.index = list(range(len(data)))
        self.Result1D = data
        
    def IndexToDate(self,):
        # convert the index into date
        dateFn = lambda x: self.ReferenceIndex.loc[x,'date']
        # get the date the column 'ID' 
        date = self.EventIndex.loc[:,'ID'].to_frame().applymap(dateFn)
        self.EventIndex['date'] = date
    
    @staticmethod
    def Collect1DResults(Path, FolderNames, Left, Right, SavePath, OneD, 
                         fromf=[], tof=[]):
        
        second = "=pd.DataFrame()"
        if fromf == []:
            fromf = 0
        for i in range(len(FolderNames)):
            print(str(i) + "-" + FolderNames[i])
            if tof == []:
                tof = len(os.listdir(Path +"/" + FolderNames[i]))
            FileList = os.listdir(Path +"/" + FolderNames[i])[fromf:tof]
            
            for j in range(len(FileList)):
                
                if Left and FileList[j].split('.')[0].endswith("_left"):
                    print(str(i) + "-" + str(j) +"-" + FileList[j])
                    # create data frame for the sub-basin
                    first = "L" + FileList[j].split('.')[0]
                
                elif Right and FileList[j].split('.')[0].endswith("_right"):
                    print(str(i) + "-" + str(j) +"-" + FileList[j])
                    first = "R" + FileList[j].split('.')[0]
                    
                ## try to get the integer of the file name to make sure that it is 
                ## one of the 1D results file
                elif OneD and not FileList[j].split('.')[0].endswith("_right") and not FileList[j].split('.')[0].endswith("_left"):
                    print(str(i) + "-" + str(j) +"-" + FileList[j])
                    # create data frame for the sub-basin
                    first = "one" + FileList[j].split('.')[0]
            
                    # get the updated list of variable names
                    variables = locals()
                
                    # read the file
                    try:
                        temp_df = pd.read_csv(Path + "/" + FolderNames[i] + "/" + FileList[j],header = None, 
                                              delimiter = r'\s+')
                        
                        # check whether the variable exist or not
                        # if this is the first time this file exist
                        if not first in variables.keys():
                            # create a datafame with the name of the sub-basin
                            total = first+ second
                            exec(total)
                            
                        # concatenate the 
                        exec(first +  "= pd.concat([" + first+ ", temp_df])")
                    except:
                        continue
        
        # Save files
        variables = list(locals().keys())
        # get sub-basins variables (starts with "One")
        for i in range(len(variables)):
            var = variables[i]
            if var.endswith("_left"):
                path = SavePath + '/' + var[1:]+ '.txt'
                exec(var + ".to_csv(path ,index= None, sep = ' ', header = None)")
            elif var.endswith("_right"):
                path = SavePath + '/' + var[1:]+ '.txt'
                exec(var + ".to_csv(path ,index= None, sep = ' ', header = None)")            
            elif var.startswith("one"):
                print("Saving " + var[3:]+ '.txt')
                path = SavePath + '/' + var[3:]+ '.txt'
                exec(var + ".to_csv(path ,index= None, sep = ' ', header = None)")
        
class Sub(River):
    
    def __init__(self,ID, River):
        self.ID = ID
        self.RIM = River.name
        self.RightOvertopping_Suffix = River.RightOvertopping_Suffix
        self.leftOvertopping_Suffix = River.leftOvertopping_Suffix
        self.crosssections = River.crosssections[River.crosssections['swimid'] == ID]
        self.crosssections.index = list(range(len(self.crosssections)))
        self.LastXS = self.crosssections.loc[len(self.crosssections)-1,'xsid']
        self.FirstXS = self.crosssections.loc[0,'xsid']
        self.ReferenceIndex = River.ReferenceIndex
        self.oneDResultPath = River.oneDResultPath
        self.slope = River.slope[River.slope['SubID']==ID]['slope'].tolist()[0]
        self.USnode, self.DSnode = River.Trace(ID)
        if hasattr(River, 'RP'):
            self.RP = River.RP.loc[River.RP['node'] == self.USnode,['HQ2','HQ10','HQ100']]
        self.RRMPath = River.RRMPath
        
    def Read1DResult(self,FromDay = [], ToDay = [], FillMissing = False, addHQ2 = True):
        """
        ===================================================================
           Read1DResult(FromDay = [], ToDay = [], FillMissing = False)
        ===================================================================
        Read1DResult method reads the 1D (1D-2D coupled) result of the sub-basin the object is 
        created for and return the hydrograph of the first and last cross section
        
        Parameters
        ----------
            1-FromDay : [Integer], optional
                DESCRIPTION. The default is [].
            2-ToDay : TYPE, optional
                DESCRIPTION. The default is [].
            3-FillMissing : TYPE, optional
                DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        River.Read1DResult(self,self.ID, FromDay, ToDay, FillMissing = FillMissing)
        
        self.XSHydrographs = pd.DataFrame()
        if FromDay == []:
            FromDay = self.Result1D.loc[0,'day']
        if ToDay ==  []:
            ToDay = self.Result1D.loc[len(self.Result1D)-1,'day']
        start = self.ReferenceIndex.loc[FromDay,'date']#.tolist()[0]
        end = self.ReferenceIndex.loc[ToDay+1,'date']#.tolist()[0]
        self.XSHydrographs['ID'] = pd.date_range(start,end,freq = 'H')[:-1]
        # get the simulated hydrograph and add the cutted HQ2
        if addHQ2:
            self.XSHydrographs[self.LastXS] = self.Result1D['q'][self.Result1D['xs'] == self.LastXS ].values + self.RP['HQ2'].tolist()[0]
            self.XSHydrographs[self.FirstXS] = self.Result1D['q'][self.Result1D['xs'] == self.FirstXS ].values + self.RP['HQ2'].tolist()[0]
        else:
            self.XSHydrographs[self.LastXS] = self.Result1D['q'][self.Result1D['xs'] == self.LastXS ].values
            self.XSHydrographs[self.FirstXS] = self.Result1D['q'][self.Result1D['xs'] == self.FirstXS ].values
    
    @staticmethod
    def ReadRRMResults(Path, NodeID, FromDay, ToDay):
        
        
        Q = pd.read_csv(Path + "/" + str(NodeID) + '.txt',header = None)
        Q = Q.rename(columns = {0:NodeID})
        Q.index = list(range(1,len(Q)+1))
        
        
        if FromDay == []:
            FromDay = 1
        if ToDay == []:
            ToDay = len(Q)
        
        Q = Q.loc[Q.index >= FromDay,:]
        Q = Q.loc[Q.index <= ToDay]
        
        return Q
    
    def ReadHydrographs(self, NodeID, FromDay = [], ToDay = []):
        
        self.RRM = pd.DataFrame()
        self.RRM[NodeID] = self.ReadRRMResults(self.RRMPath, NodeID, FromDay, ToDay)[NodeID]
        # self.RRM[self.USnode] = self.ReadRRMResults(Path, self.USnode, FromDay, ToDay)[self.USnode]
        if FromDay == []:
            FromDay = 1
        if ToDay == []:
            ToDay = len(self.RRM[self.USnode])-1
        
        start = self.ReferenceIndex.loc[FromDay,'date']
        end = self.ReferenceIndex.loc[ToDay,'date']
        
        self.RRM['ID'] = pd.date_range(start,end,freq = 'D')
        # get the simulated hydrograph and add the cutted HQ2
        
    def DetailedOvertopping(self, eventdays):
        # River.DetailedOvertopping(self, [self.ID], eventdays)
        XSs = self.crosssections.loc[:,'xsid'].tolist()
        columns = [self.ID] + XSs + ['sum']
        self.DetailedOvertoppingLeft  = pd.DataFrame(index = eventdays + ['sum'], columns = columns)
        self.DetailedOvertoppingLeft.loc[:,columns] = 0
        self.DetailedOvertoppingRight = pd.DataFrame(index = eventdays + ['sum'], columns = columns)
        self.DetailedOvertoppingRight.loc[:,columns] = 0
        # Left Bank
        try :
            # try to open and read the overtopping file 
            data = pd.read_csv(self.oneDResultPath + str(self.ID) + self.leftOvertopping_Suffix,
                               header =None,delimiter = r'\s+')
            
            data.columns = ['day','hour','xsid','q','wl']
            # get the days in the sub
            days = list(set(data.loc[:,'day']))
            
            for j in range(len(eventdays)):
                # check whether this sub basin has flooded in this particular day
                if eventdays[j] in days:
                    # filter the dataframe to the discharge column (3) and the days
                    self.DetailedOvertoppingLeft.loc[eventdays[j],self.ID] = data.loc[data['day'] == eventdays[j] ,'q'].sum()
                    # get the xss that was overtopped in that particular day 
                    XSday = list(set(data.loc[data['day'] == eventdays[j] ,'xsid'].tolist()))
                    
                    for i in range(len(XSday)):
                        # dataXS = data['q'].loc[data['day'] == eventdays[j]][data['xsid'] == XSday[i]].sum()
                        self.DetailedOvertoppingLeft.loc[eventdays[j], XSday[i]] = data['q'].loc[data['day'] == eventdays[j]][data['xsid'] == XSday[i]].sum()
                else:
                    self.DetailedOvertoppingLeft.loc[eventdays[j],self.ID] = 0
        except:
            self.DetailedOvertoppingLeft.loc[:,self.ID] = 0
            
        # right Bank
        try :
            # try to open and read the overtopping file 
            data = pd.read_csv(self.oneDResultPath + str(self.ID) + self.RightOvertopping_Suffix,
                               header =None,delimiter = r'\s+')
            data.columns = ['day','hour','xsid','q','wl']
            # get the days in the sub
            days = list(set(data.loc[:,'day']))
            
            for j in range(len(eventdays)):
                # check whether this sub basin has flooded in this particular day
                if eventdays[j] in days:
                    # filter the dataframe to the discharge column (3) and the days
                    self.DetailedOvertoppingRight.loc[eventdays[j], self.ID] = data.loc[data['day'] == eventdays[j] ,'q'].sum()
                    # get the xss that was overtopped in that particular day 
                    XSday = list(set(data.loc[data['day'] == eventdays[j] ,'xsid'].tolist()))
                    
                    for i in range(len(XSday)):
                        # dataXS = data['q'].loc[data['day'] == eventdays[j]][data['xsid'] == XSday[i]].sum()
                        self.DetailedOvertoppingRight.loc[eventdays[j], XSday[i]] = data['q'].loc[data['day'] == eventdays[j]][data['xsid'] == XSday[i]].sum()
                    
                else:
                    self.DetailedOvertoppingRight.loc[eventdays[j], self.ID] = 0
        except:
            # print("file did not open")
            self.DetailedOvertoppingRight.loc[:,self.ID] = 0
            
        # sum overtopping for each day
        for j in range(len(eventdays)):
            self.DetailedOvertoppingLeft.loc[eventdays[j],'sum'] = self.DetailedOvertoppingLeft.loc[eventdays[j],XSs].sum()
            self.DetailedOvertoppingRight.loc[eventdays[j],'sum'] = self.DetailedOvertoppingRight.loc[eventdays[j],XSs].sum()
        # sum overtopping for each sub basin
        for j in range(len(XSs)):
            self.DetailedOvertoppingLeft.loc['sum',XSs[j]] = self.DetailedOvertoppingLeft.loc[:,XSs[j]].sum()
            self.DetailedOvertoppingRight.loc['sum',XSs[j]] = self.DetailedOvertoppingRight.loc[:,XSs[j]].sum()
        
        self.DetailedOvertoppingLeft.loc['sum', self.ID] = self.DetailedOvertoppingLeft.loc[:, self.ID].sum()
        self.DetailedOvertoppingRight.loc['sum', self.ID] = self.DetailedOvertoppingRight.loc[:, self.ID].sum()
        
        # self.DetailedOvertoppingLeft.loc['sum','sum'] = self.DetailedOvertoppingLeft.loc[:,'sum'].sum()
        # self.DetailedOvertoppingRight.loc['sum','sum'] = self.DetailedOvertoppingRight.loc[:,'sum'].sum()
            
    # def Read1DResult1Donly(self,Path):
        # River.Read1DResult(self,self.ID, FromDay, ToDay, FillMissing)
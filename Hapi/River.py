# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 18:04:32 2020

@author: mofarrag
"""
import os 
import pandas as pd
import numpy as np
# import Hapi.GISpy as GIS

class River():
    # class attributes


    ### constructor
    def __init__(self, name):
        self.name = name
        self.leftOvertopping_Suffix = "_left.txt"
        self.RightOvertopping_Suffix = "_right.txt"
        
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
        
        SWIMDS = int(self.rivernetwork['DS'][np.where(self.rivernetwork['SubID'] == SubID)[0][0]])
        SWIMUS = int(self.rivernetwork['US'][np.where(self.rivernetwork['SubID'] == SubID)[0][0]])
        
        return  SWIMUS, SWIMDS
    
    
    def Overtopping(self, OvertoppingResultF):

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
        All1DFiles = os.listdir(OvertoppingResultF)
        for i in range(len(All1DFiles)) :
            if All1DFiles[i].endswith("_left.txt"):
                leftOverTop.append(All1DFiles[i])
            if All1DFiles[i].endswith("_right.txt"):
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
                data = pd.read_csv(OvertoppingResultF + leftOverTop[i],header =None,delimiter = r'\s+')
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
                data = pd.read_csv(OvertoppingResultF + RightOverTop[i],header =None,delimiter = r'\s+')
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
        Parameters
        ----------
            1-OvertoppedXS : [list], optional
                list of cross sections overtopped (if you already used the GetOvertoppedXS
                method to get the overtopped XSs for a specific day).
                The default is [].
            2-day : [list], optional
                if you want to get the flooded subs for a specific list of days. The default is 1.
            3-allEventdays : TYPE, optional
                DESCRIPTION. The default is True.

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
        
    # def CreatePolygons(self):
        
    @staticmethod
    def PolygonGeometry(Coords):
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
        area = 0.0
        for i in range(np.shape(Coords)[0]-1):
            # cros multiplication
            area = area + Coords[i,0]*Coords[i+1,1] - Coords[i+1,0]*Coords[i,1]
        area = area + Coords[np.shape(Coords)[0]-1,0] * Coords[0,1] - Coords[0,0] * Coords[np.shape(Coords)[0]-1,1]
        area = area*0.5
        
        return area
    
    @staticmethod
    def PolyPerimeter(Coords):
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
    
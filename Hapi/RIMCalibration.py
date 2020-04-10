# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:27:32 2020

@author: mofarrag
"""
import pandas as pd
import datetime as dt
import numpy as np

datafn = lambda x: dt.datetime.strptime(x,"%Y-%m-%d")

class RIMCalibration():
    
    def __init__(self, name, Version):
        self.name = name
        self.Version = Version
        
    def ReadObservedWL(self, GaugesTable, Path, StartDate, EndDate, NoValue):
        """
        ============================================================================
            ReadObservedWL( WLGauges, Path, StartDate, EndDate, NoValue)
        ============================================================================

        Parameters
        ----------
            1-GaugesTable : [dataframe]
                Dataframe contains columns [swimid,xsid,datum(m)].
            2-Path : [String]
                    path to the folder containing the text files of the water level gauges
            3-StartDate : [datetime object]
                the starting date of the water level time series.
            4-EndDate : [datetime object]
                the end date of the water level time series.
            5-NoValue : [integer]
                value used to fill the missing values.

        Returns
        -------
            1-WL: [dataframe attribute].


        """

        ind = pd.date_range(StartDate, EndDate)
        columns = GaugesTable['swimid'].tolist()
        
        WLGauges = pd.DataFrame(index = ind)
        # WLGaugesf.loc[:,:] = NoValue
        WLGauges.loc[:,0] = ind
        
        for i in range(len(columns)):
            f = pd.read_csv(Path + str(int(columns[i])) + ".txt",
                           delimiter = ",", header = None)
            f[0] = f[0].map(datafn)
            # sort by date as some values are missed up
            f.sort_values(by = [0], ascending = True, inplace = True)
            # filter to the range we want
            f = f.loc[f[0] >= ind[0],:]
            f = f.loc[f[0] <= ind[-1],:]
            # reindex
            f.index = list(range(len(f)))
            # add datum and convert to meter
            f.loc[f[1]!= NoValue,1] = (f.loc[f[1]!= NoValue,1] / 100) + GaugesTable.loc[i,'datum(m)']
            f = f.rename(columns={1:columns[i]})
            
            
            # assign the values in the dateframe
            # WLGauges.loc[:,WLGauges.columns[i]].loc[f[0][0]:f[0][len(f)-1]] = f[1].tolist()
            # use merge as there are some gaps in the middle
            WLGauges = WLGauges.merge(f, on=0, how='left', sort=False)
            
        WLGauges.replace(to_replace = np.nan, value = NoValue, inplace=True)
        WLGauges.index = ind
        del WLGauges[0]
        self.WLGauges = WLGauges

    def ReadObservedQ(self, CalibratedSubs, Path, StartDate, EndDate):

        ind = pd.date_range(StartDate, EndDate)
        GRDC = pd.DataFrame(index = ind)

        for i in range(len(CalibratedSubs[0])):
            GRDC.loc[:,int(CalibratedSubs[0][i])] = np.loadtxt(Path +
                      str(int(CalibratedSubs[0][i])) + '.txt') #,skiprows = 0
        self.QGauges = GRDC
        
    def ReadRRM(self, Qgauges, Path, StartDate, EndDate):
        
        ind = pd.date_range(StartDate,EndDate)
        QSWIM = pd.DataFrame(index = ind)
        
        
        for i in range(len(Qgauges[0])):
            # read SWIM data
            # only at the begining to get the length of the time series
            QSWIM.loc[:,int(Qgauges[0][i])] = np.loadtxt(Path +
                  str(int(Qgauges[0][i])) + '.txt')#,skiprows = 0
        self.QRRM = QSWIM
        
        
    def ReadRIMQ(self, Qgauges, Path, StartDate, days, NoValue):
        
        EndDate = StartDate + dt.timedelta(days = days-1)
        ind = pd.date_range(StartDate,EndDate)
        QRIM = pd.DataFrame(index = ind, columns = Qgauges[0].tolist())
        # for RIM1.0 don't fill with -9 as the empty days will be filled with 0 so to get
        # the event days we have to filter 0 and -9
        if self.Version == 1:
            QRIM.loc[:,:] = 0
        else:
            QRIM.loc[:,:] = NoValue
        
        # fill non modelled time steps with zeros
        for i in range(len(Qgauges)):
            f = np.loadtxt( Path + str(int(QRIM.columns[i])) + ".txt",
                       delimiter = ",")
            f1 = list(range(int(f[0,0]),int(f[-1,0])+1))
            f2 = list()
            for j in range(len(f1)):
                # if the index exist in the original list
                if f1[j] in f[:,0]:
                    # put the coresponding value in f2
                    f2.append(f[np.where(f[:,0] == f1[j])[0][0],1])
                else:
                    # if it does not exist put zero
                    f2.append(0)
        
            QRIM.loc[:,QRIM.columns[i]].loc[ind[f1[0]-1]:ind[f1[-1]-1]] = f2
            
        self.QRIM = QRIM
        
    def ReadRIMWL(self, WLGaugesTable, Path, StartDate, days, NoValue):
        
        EndDate = StartDate + dt.timedelta(days = days-1)
        ind = pd.date_range(StartDate,EndDate)
        
        WLRIM = pd.DataFrame(index = ind, columns = WLGaugesTable['swimid'].tolist())
        WLRIM.loc[:,:] = NoValue
        
        for i in range(len(WLRIM.columns)):
            f = np.loadtxt(Path + str(int(WLRIM.columns[i])) + ".txt",
                       delimiter = ",")
        
            f1 = list(range(int(f[0,0]),int(f[-1,0])+1))
            f2 = list()
            for j in range(len(f1)):
                # if the index exist in the original list
                if f1[j] in f[:,0]:
                    # put the coresponding value in f2
                    f2.append(f[np.where(f[:,0] == f1[j])[0][0],1])
                else:
                    # if it does not exist put zero
                    f2.append(0)
        
            WLRIM.loc[:,WLRIM.columns[i]].loc[ind[f1[0]-1]:ind[f1[-1]-1]] = f2
            
        self.WLRIM = WLRIM
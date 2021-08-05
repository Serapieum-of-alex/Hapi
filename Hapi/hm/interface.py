# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:40:23 2021

@author: mofarrag
"""
import pandas as pd
import datetime as dt
from Hapi.hm.river import River

class Interface(River):
    """
    ==================================
        Interface(River)
    ==================================
    Interface between the Rainfall runoff model and the Hydraulic model

    Methods:
        1- ReadLateralsTable
        2- ReadLaterals
        3- ReadBoundaryConditionsTable
        4- ReadBoundaryConditions
        5- ListAttributes
    """

    def __init__(self, name, version=3, start = "1952-1-1", days =36890,):
        self.name = name
        self.version = version
        self.start = dt.datetime.strptime(start,"%Y-%m-%d")
        self.end = self.start  + dt.timedelta(days = days)
        Ref_ind = pd.date_range(self.start, self.end, freq='D')
        self.ReferenceIndex = pd.DataFrame(index = list(range(1,days+1)))
        self.ReferenceIndex['date'] = Ref_ind[:-1]
        pass

    def ReadLateralsTable(self, Path, prefix='lf_xsid', suffix='.txt'):
        """
        ===============================================================
               ReadLateralsTable(Path)
        ===============================================================
        ReadLateralsTable method reads the laterals file
            laterals file : file contains the xsid of the cross-sections that
            has laterals
        if the user has already read te cross section file, the methos is going
        to add column to the crosssection dataframe attribute and is going to add
        a value of 1 to the xs that has laterals

        Parameters
        ----------
        Path : [String], optional
            Path to read the results from.

        Returns
        -------
        None.

        """
        self.LateralsTable = pd.read_csv(Path, skiprows=[0], header=None)
        self.LateralsTable.columns = ['filename']
        l1 = len(prefix)
        l2 = len(suffix)
        self.LateralsTable['xsid'] = [int(i[l1:len(i)-l2]) for i in self.LateralsTable[self.LateralsTable.columns[0]]]
        
        
        # self.LateralsTable['xsid'] = 
        # self.LateralsTable.columns = ["xsid"]

        if hasattr(self, "crosssections"):
            self.crosssections['lateral'] = 0
            for i in range(len(self.crosssections)):
                if self.crosssections.loc[i,'xsid'] in self.LateralsTable['xsid'].tolist():
                    self.crosssections.loc[i,'lateral'] = 1
        else:
            assert False, "Please read the cross section file first using the method 'ReadCrossSections'"


    def ReadLaterals(self, FromDay = '', ToDay = '', Path = '',
                          date_format="'%Y-%m-%d'"):
        """
        =======================================================================
            ReadUSHydrograph(FromDay = '', ToDay = '', Path = '',
                                date_format="'%Y-%m-%d'")
        =======================================================================

        Parameters
        ----------
        1-FromDay : [integer], optional
                the day you want to read the result from, the first day is 1 not zero.The default is ''.
        2-ToDay : [integer], optional
                the day you want to read the result to.
        3-Path : [String], optional
            Path to read the results from. The default is ''.
        4-date_format : "TYPE, optional
            DESCRIPTION. The default is "'%Y-%m-%d'".

        Returns
        -------
        1-USHydrographs : [dataframe attribute].
            dataframe contains the hydrograph of each of the upstream segments
            with segment id as a column name and a column 'total' contains the
            sum of all the hydrographs.
        """
        assert hasattr(self, 'LateralsTable'), "Please read the lateras table first using the 'ReadLateralsTable' method"

        # if Path == '':
            # Path = self.CustomizedRunsPath

        self.Laterals = pd.DataFrame()

        for i in range(len(self.LateralsTable)):
            NodeID = self.LateralsTable.loc[i,'xsid']
            fname = "lf_xsid" + str(NodeID)
            self.Laterals[NodeID]  = self.ReadRRMResults(self.version, self.ReferenceIndex,
                                                            Path, fname, FromDay, ToDay,
                                                            date_format)[fname].tolist()


        self.Laterals['total'] = self.Laterals.sum(axis=1)
        if FromDay == '':
            FromDay = 1
        if ToDay == '':
            ToDay = len(self.Laterals[NodeID])

        start = self.ReferenceIndex.loc[FromDay,'date']
        end = self.ReferenceIndex.loc[ToDay,'date']

        self.Laterals.index = pd.date_range(start, end, freq = 'D')

    def ReadBoundaryConditionsTable(self, path, prefix='bc_xsid', suffix='.txt'):
        """
        ReadLateralsTable method reads the laterals file
            laterals file : file contains the xsid of the cross-sections that
            has laterals
        if the user has already read te cross section file, the methos is going
        to add column to the crosssection dataframe attribute and is going to add
        a value of 1 to the xs that has laterals

        Parameters
        ----------
        Path : [String], optional
            Path to read the results from.

        Returns
        -------
        None.

        """
        self.BCTable = pd.read_csv(path, skiprows=[0], header=None)
        self.BCTable.columns = ['filename']
        l1 = len(prefix)
        l2 = len(suffix)
        self.BCTable['id'] = [i[l1:len(i)-l2] for i in self.BCTable[self.BCTable.columns[0]]]
        
        # self.BCTable.columns = ["id"]
        # self.BCTable['id'] = [int(i[7:]) for i in self.BCTable['id'].tolist()]


    def ReadBoundaryConditions(self, FromDay='', ToDay='', path='',
                          date_format="'%Y-%m-%d'"):
        """
        =======================================================================
            ReadUSHydrograph(FromDay = '', ToDay = '', Path = '',
                                date_format="'%Y-%m-%d'")
        =======================================================================

        Parameters
        ----------
        1-FromDay : [integer], optional
                the day you want to read the result from, the first day is 1 not zero.The default is ''.
        2-ToDay : [integer], optional
                the day you want to read the result to.
        3-Path : [String], optional
            Path to read the results from. The default is ''.
        4-date_format : "TYPE, optional
            DESCRIPTION. The default is "'%Y-%m-%d'".

        Returns
        -------
        1-USHydrographs : [dataframe attribute].
            dataframe contains the hydrograph of each of the upstream segments
            with segment id as a column name and a column 'total' contains the
            sum of all the hydrographs.
        """
        assert hasattr(self, 'BCTable'), "Please read the lateras table first using the 'ReadLateralsTable' method"

        # if Path == '':
            # Path = self.CustomizedRunsPath

        self.BC = pd.DataFrame()

        for i in range(len(self.BCTable)):
            NodeID = self.BCTable.loc[i,'id']
            fname = "bc_xsid" + str(NodeID)
            self.BC[NodeID] = self.ReadRRMResults(self.version, self.ReferenceIndex,
                                                            path, fname, FromDay, ToDay,
                                                            date_format)[fname].tolist()


        self.BC['total'] = self.BC.sum(axis=1)
        if FromDay == '':
            FromDay = 1
        if ToDay == '':
            ToDay = len(self.BC[NodeID])

        start = self.ReferenceIndex.loc[FromDay,'date']
        end = self.ReferenceIndex.loc[ToDay,'date']

        self.BC.index = pd.date_range(start, end, freq = 'D')



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
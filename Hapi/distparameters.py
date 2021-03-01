# -*- coding: utf-8 -*-
"""
DistParameters contains functions that is responible for distributing parameters
spatially (totally distributed, totally distriby-uted with some parameters lumped,
all parameters are lumped, hydrologic response units) and also save generated parameters
into rasters

@author: Mostafa
"""
import numbers
import numpy as np
import datetime as dt
import os
import gdal
from Hapi.raster import Raster
from Hapi.giscatchment import GISCatchment as GC


class DistParameters():

    def __init__(self, raster, no_parameters, no_lumped_par=0, lumped_par_pos=[],
                 Lake = 0, Snow=0, Function=1):

        assert type(raster)==gdal.Dataset, "raster should be read using gdal (gdal dataset please read it using gdal library) "
        assert type(no_parameters)==int, " no_parameters should be integer number"
        assert type(no_lumped_par)== int, "no of lumped parameters should be integer"

        if no_lumped_par >= 1:
            if type(lumped_par_pos) == list:
                assert no_lumped_par == len(lumped_par_pos), "you have to entered"+str(no_lumped_par)+"no of lumped parameters but only"+str(len(lumped_par_pos))+" position "
            else: # if not int or list
                assert False ,"you have one or more lumped parameters so the position has to be entered as a list"

        self.Lake = Lake
        self.Snow = Snow
        self.no_lumped_par = no_lumped_par
        self.lumped_par_pos = lumped_par_pos
        # read the raster
        self.raster = raster
        self.raster_A = raster.ReadAsArray()
        # get the shape of the raster
        self.rows = raster.RasterYSize
        self.cols = raster.RasterXSize
        # get the no_value of in the raster
        self.noval = np.float32(raster.GetRasterBand(1).GetNoDataValue())
        # count the number of non-empty cells
        self.no_elem = np.size(self.raster_A[:,:])-np.count_nonzero((self.raster_A[self.raster_A == self.noval]))

        self.no_parameters = no_parameters

        # store the indeces of the non-empty cells
        self.celli=[]
        self.cellj=[]
        for i in range(self.rows):
            for j in range(self.cols):
                if self.raster_A[i,j] != self.noval:
                    self.celli.append(i)
                    self.cellj.append(j)

        # create an empty 3D array [[raster dimension], no_parameters]
        self.Par3d = np.zeros([self.rows, self.cols, self.no_parameters])*np.nan

        if no_lumped_par >= 1:
            # parameters in array
            # remove a place for the lumped parameter (k1) lower zone coefficient
            no_parameters = self.no_parameters - no_lumped_par

        # parameters in array
        # create a 2d array [no_parameters, no_cells]
        self.Par2d = np.ones((self.no_parameters,self.no_elem))

        if Function == 1:
            self.Function = self.par3dLumped
        elif Function == 2:
            self.Function = self.par3d
        elif Function == 3:
            self.Function = self.par2d_lumpedK1_lake
        elif Function == 3:
            self.Function = self.HRU
        pass


    def par3d(self,par_g, kub=1,klb=0.5,Maskingum=True):
        """
        ===========================================================
          par3d(par_g,raster, no_parameters, no_lumped_par, lumped_par_pos, kub, klb)
        ===========================================================
        this function takes a list of parameters [saved as one column or generated
        as 1D list from optimization algorithm] and distribute them horizontally on
        number of cells given by a raster

        Inputs :
        ----------
            1- par_g:
                [list] list of parameters
            2- raster:
                [gdal.dataset] raster to get the spatial information of the catchment
                (DEM, flow accumulation or flow direction raster)
            3- no_parameters
                [int] no of parameters of the cell according to the rainfall runoff model
            4-no_lumped_par:
                [int] nomber of lumped parameters, you have to enter the value of
                the lumped parameter at the end of the list, default is 0 (no lumped parameters)
            5-lumped_par_pos:
                [List] list of order or position of the lumped parameter among all
                the parameters of the lumped model (order starts from 0 to the length
                of the model parameters), default is [] (empty), the following order
                of parameters is used for the lumped HBV model used
                [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
                c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
            6- kub:
                [float] upper bound of K value (traveling time in muskingum routing method)
                default is 1 hour
            7- klb:
                [float] Lower bound of K value (traveling time in muskingum routing method)
                default is 0.5 hour (30 min)

        Output:
        ----------
            1- par_3d: 3D array of the parameters distributed horizontally on the cells

        Example:
        ----------
            EX1:totally distributed parameters
                raster=gdal.Open("dem.tif")
                [fc, beta, etf, lp, c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
                no_lumped_par=0
                lumped_par_pos=[]
                par_g=np.random.random(no_elem*(no_parameters-no_lumped_par))

                tot_dist_par=par3d(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)

            EX2: One Lumped Parameter [K1]
                raster=gdal.Open("dem.tif")
                given values of parameters are of this order
                [fc, beta, etf, lp, c_flux, k, alpha, perc, pcorr, Kmuskingum, Xmuskingum,k1]
                K1 is lumped so its value is inserted at the end and its order should
                be after K

                no_lumped_par=1
                lumped_par_pos=[6]
                par_g=np.random.random(no_elem* (no_parameters-no_lumped_par))
                # insert the value of k1 at the end
                par_g=np.append(par_g,0.005)

                dist_par=par3d(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)

            EX3:Two Lumped Parameter [K1, Perc]
                raster=gdal.Open("dem.tif")
                no_lumped_par=2
                lumped_par_pos=[6,8]
                par_g=np.random.random(no_elem* (no_parameters-no_lumped_par))
                par_g=np.append(par_g,0.005)
                par_g=np.append(par_g,0.006)

                dist_par=par3d(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)
        """
        # input data validation
        # data type
        assert type(par_g)==np.ndarray or type(par_g)==list, "par_g should be of type 1d array or list"
        assert isinstance(kub,numbers.Number) , " kub should be a number"
        assert isinstance(klb,numbers.Number) , " klb should be a number"

        # input values
        if self.no_lumped_par > 0:
            assert len(par_g) == (self.no_elem*(self.no_parameters-self.no_lumped_par))+self.no_lumped_par,"As there is "+str(self.no_lumped_par)+" lumped parameters, length of input parameters should be "+str(self.no_elem)+"*"+"("+str(self.no_parameters)+"-"+str(self.no_lumped_par)+")"+"+"+str(self.no_lumped_par)+"="+str(self.no_elem*(self.no_parameters-self.no_lumped_par)+self.no_lumped_par)+" not "+str(len(par_g))+" probably you have to add the value of the lumped parameter at the end of the list"
        else:
            # if there is no lumped parameters
            assert len(par_g) == self.no_elem*self.no_parameters,"As there is no lumped parameters length of input parameters should be "+str(self.no_elem)+"*"+str(self.no_parameters)+"="+str(self.no_elem*self.no_parameters)

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(self.no_elem):
            self.Par2d[:,i] = par_g[i*self.no_parameters:(i*self.no_parameters) + self.no_parameters]

        ### lumped parameters
        if self.no_lumped_par > 0:
            for i in range(self.no_lumped_par):
                # create a list with the value of the lumped parameter(k1)
                # (stored at the end of the list of the parameters)
                pk1 = np.ones((1,self.no_elem))*par_g[(self.no_parameters * np.shape(self.Par2d)[1])+i]
                # put the list of parameter k1 at the 6 row
                self.Par2d = np.vstack([self.Par2d[:self.lumped_par_pos[i],:],pk1,self.Par2d[self.lumped_par_pos[i]:,:]])

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d
        for i in range(self.no_elem):
            self.Par3d[self.celli[i],self.cellj[i],:] = self.Par2d[:,i]

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        if Maskingum == True:
            for i in range(self.no_elem):
                self.Par3d[self.celli[i],self.cellj[i],-2]= DistParameters.calculateK(self.Par3d[self.celli[i],self.cellj[i],-1],self.Par3d[self.celli[i],self.cellj[i],-2],kub,klb)



    def par3dLumped(self, par_g, kub=1, klb=0.5, Maskingum = True):
        """
        ===========================================================
          par3dLumped(par_g,raster, no_parameters, kub, klb)
        ===========================================================
        this function takes a list of parameters [saved as one column or generated
        as 1D list from optimization algorithm] and distribute them horizontally on
        number of cells given by a raster

        Inputs :
        ----------
            1- par_g:
                [list] list of parameters
            4- kub:
                [float] upper bound of K value (traveling time in muskingum routing method)
                default is 1 hour
            5- klb:
                [float] Lower bound of K value (traveling time in muskingum routing method)
                default is 0.5 hour (30 min)

        Output:
        ----------
            1- par_3d: 3D array of the parameters distributed horizontally on the cells

        Example:
        ----------
            EX1:Lumped parameters
                raster=gdal.Open("dem.tif")
                [fc, beta, etf, lp, c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]

                raster=gdal.Open(path+"soil_classes.tif")
                no_parameters=12
                par_g=np.random.random(no_parameters) #no_elem*(no_parameters-no_lumped_par)

                tot_dist_par=DP.par3dLumped(par_g,raster,no_parameters,lumped_par_pos,kub=1,klb=0.5)
        """
        # input data validation
        # data type
        assert type(par_g)==np.ndarray or type(par_g)==list, "par_g should be of type 1d array or list"
        assert isinstance(kub,numbers.Number) , " kub should be a number"
        assert isinstance(klb,numbers.Number) , " klb should be a number"

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(self.no_elem):
            self.Par2d[:,i] = par_g

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d
        for i in range(self.no_elem):
            self.Par3d[self.celli[i],self.cellj[i],:] = self.Par2d[:,i]

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        if Maskingum == True:
            for i in range(self.no_elem):
                self.Par3d[self.celli[i],self.cellj[i],-2] = DistParameters.calculateK(self.Par3d[self.celli[i],self.cellj[i],-1],self.Par3d[self.celli[i],self.cellj[i],-2],kub,klb)


    @staticmethod
    def calculateK(x,position,UB,LB):
        """
        ===================================================
            calculateK(x,position,UB,LB):
        ===================================================

        this function takes value of x parameter and generate 100 random value of k parameters between
        upper & lower constraint then the output will be the value coresponding to the giving position

        Inputs:
        ----------
            1- x weighting coefficient to determine the linearity of the water surface
                (one of the parameters of muskingum routing method)
            2- position
                random position between upper and lower bounds of the k parameter
            3-UB
                upper bound for k parameter
            3-LB
                Lower bound for k parameter
        """
        constraint1 = 0.5*1/(1-x) # k has to be smaller than this constraint
        constraint2 = 0.5*1/x   # k has to be greater than this constraint

        if constraint2 >= UB : #if constraint is higher than UB take UB
            constraint2 = UB

        if constraint1 <= LB : #if constraint is lower than LB take UB
            constraint1 = LB

        generatedK=np.linspace(constraint1,constraint2,101)
        k=generatedK[int(round(position,0))]
        return k


    def par2d_lumpedK1_lake(self,par_g,no_parameters_lake,kub,klb):
        """
        ===========================================================
          par2d_lumpedK1_lake(par_g,raster,no_parameters,no_par_lake,kub,klb)
        ===========================================================
        this function takes a list of parameters and distribute them horizontally on number of cells
        given by a raster

        Inputs :
        ----------
            1- par_g
                list of parameters
            2- raster
                raster of the catchment (DEM)
            3- no_parameters
                no of parameters of the cell
            4- no_parameters_lake
                no of lake parameters
            5- kub
                upper bound of K value (traveling time in muskingum routing method)
            6- klb
                Lower bound of K value (traveling time in muskingum routing method)

        Output:
        ----------
            1- Par3d: 3D array of the parameters distributed horizontally on the cells
            2- lake_par: list of the lake parameters.

        Example:
        ----------
            a list of 155 value,all parameters are distributed except lower zone coefficient
            (is written at the end of the list) each cell(14 cells) has 11 parameter plus lower zone
            (12 parameters) function will take each 11 parameter and assing them to a specific cell
            then assign the last value (lower zone parameter) to all cells
            14*11=154 + 1 = 155
        """

        # parameters in array
        # remove a place for the lumped parameter (k1) lower zone coefficient
        no_parameters = self.no_parameters-1

        # create a 2d array [no_parameters, no_cells]
        self.Par2d = np.ones((no_parameters,self.no_elem))

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(self.no_elem):
            self.Par2d[:,i] = par_g[i*no_parameters:(i*no_parameters)+no_parameters]

        # create a list with the value of the lumped parameter(k1)
        # (stored at the end of the list of the parameters)
        pk1 = np.ones((1,self.no_elem))*par_g[(np.shape(self.Par2d)[0]*np.shape(self.Par2d)[1])]

        # put the list of parameter k1 at the 6 row
        self.Par2d = np.vstack([self.Par2d[:6,:],pk1,self.Par2d[6:,:]])

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d
        for i in range(self.no_elem):
            self.Par3d[self.celli[i],self.cellj[i],:] = self.Par2d[:,i]

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        for i in range(self.no_elem):
            self.Par3d[self.celli[i],self.cellj[i],-2]= DistParameters.calculateK(self.Par3d[self.celli[i],self.cellj[i],-1],self.Par3d[self.celli[i],self.cellj[i],-2],kub,klb)

        # lake parameters
        self.lake_par = par_g[len(par_g)-no_parameters_lake:]
        self.lake_par[-2] = DistParameters.calculateK(self.lake_par[-1],self.lake_par[-2],kub,klb)

        # return self.Par3d, lake_par


    def HRU(self,par_g,kub=1,klb=0.5):
        """
        ===========================================================
           HRU(par_g, raster, no_parameters, no_lumped_par=0, lumped_par_pos=[], kub=1, klb=0.5)
        ===========================================================
        this function takes a list of parameters [saved as one column or generated
        as 1D list from optimization algorithm] and distribute them horizontally on
        number of cells given by a raster
        the input raster should be classified raster (by numbers) into class to be used
        to define the HRUs

        Inputs :
        ----------
            1- par_g:
                [list] list of parameters
            2- raster:
                [gdal.dataset] classification raster to get the spatial information
                of the catchment and the to define each cell belongs to which HRU
            3- no_parameters
                [int] no of parameters of the cell according to the rainfall runoff model
            4-no_lumped_par:
                [int] nomber of lumped parameters, you have to enter the value of
                the lumped parameter at the end of the list, default is 0 (no lumped parameters)
            5-lumped_par_pos:
                [List] list of order or position of the lumped parameter among all
                the parameters of the lumped model (order starts from 0 to the length
                of the model parameters), default is [] (empty), the following order
                of parameters is used for the lumped HBV model used
                [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
                c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
            6- kub:
                [float] upper bound of K value (traveling time in muskingum routing method)
                default is 1 hour
            7- klb:
                [float] Lower bound of K value (traveling time in muskingum routing method)
                default is 0.5 hour (30 min)

        Output:
        ----------
            1- par_3d: 3D array of the parameters distributed horizontally on the cells

        Example:
        ----------
            EX1:HRU without lumped parameters
                [fc, beta, etf, lp, c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
                raster = gdal.Open("soil_types.tif")
                no_lumped_par=0
                lumped_par_pos=[]
                par_g=np.random.random(no_elem*(no_parameters-no_lumped_par))

                par_hru=HRU(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)


            EX2: HRU with one lumped parameters
                given values of parameters are of this order
                [fc, beta, etf, lp, c_flux, k, alpha, perc, pcorr, Kmuskingum, Xmuskingum,k1]
                K1 is lumped so its value is inserted at the end and its order should
                be after K

                raster = gdal.Open("soil_types.tif")
                no_lumped_par=1
                lumped_par_pos=[6]
                par_g=np.random.random(no_elem* (no_parameters-no_lumped_par))
                # insert the value of k1 at the end
                par_g=np.append(par_g,0.005)

                par_hru=HRU(par_g,raster,no_parameters,no_lumped_par,lumped_par_pos,kub=1,klb=0.5)
        """
        # input data validation
        # data type
        assert type(par_g)==np.ndarray or type(par_g)==list, "par_g should be of type 1d array or list"
        assert isinstance(kub,numbers.Number) , " kub should be a number"
        assert isinstance(klb,numbers.Number) , " klb should be a number"

        # count the number of non-empty cells
        values = list(set([int(self.raster_A[i,j]) for i in range(self.rows) for j in range(self.cols) if self.raster_A[i,j] != self.noval]))
        no_elem=len(values)

        # input values
        if self.no_lumped_par > 0:
            assert len(par_g)==(no_elem*(self.no_parameters-self.no_lumped_par))+self.no_lumped_par,"As there is "+str(self.no_lumped_par)+" lumped parameters, length of input parameters should be "+str(self.no_elem)+"*"+"("+str(self.no_parameters)+"-"+str(self.no_lumped_par)+")"+"+"+str(self.no_lumped_par)+"="+str(self.no_elem*(self.no_parameters-self.no_lumped_par)+self.no_lumped_par)+" not "+str(len(par_g))+" probably you have to add the value of the lumped parameter at the end of the list"
        else:
            # if there is no lumped parameters
            assert len(par_g) == self.no_elem*self.no_parameters,"As there is no lumped parameters length of input parameters should be "+str(self.no_elem)+"*"+str(self.no_parameters)+"="+str(self.no_elem*self.no_parameters)

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(no_elem):
            self.Par2d[:,i] = par_g[i*self.no_parameters:(i*self.no_parameters) + self.no_parameters]

        ### lumped parameters
        if self.no_lumped_par > 0:
            for i in range(self.no_lumped_par):
                # create a list with the value of the lumped parameter(k1)
                # (stored at the end of the list of the parameters)
                pk1 = np.ones((1,self.no_elem))*par_g[(self.no_parameters*np.shape(self.Par2d)[1])+i]
                # put the list of parameter k1 at the 6 row
                self.Par2d = np.vstack([self.Par2d[:self.lumped_par_pos[i],:],pk1,self.Par2d[self.lumped_par_pos[i]:,:]])

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        for i in range(no_elem):
            self.Par2d[-2,i] = DistParameters.calculateK(self.Par2d[-1,i],self.Par2d[-2,i],kub,klb)

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d each soil type will have the same
        # generated parameters
        for i in range(no_elem):
            self.Par3d[self.raster_A == values[i]] = self.Par2d[:,i]


    @staticmethod
    def HRU_HAND(DEM,FD,FPL,River):
        """
        =============================================================
            HRU_HAND(DEM,FD,FPL,River)
        =============================================================
        this function calculates inputs for the HAND (height above nearest drainage)
        method for land use classification

        Inputs:
        ----------
            1- DEM:

            2-FD:

            3-FPL:

            4-River:



        Outputs:
        ----------
            1-HAND:
                [numpy ndarray] Height above nearest drainage

            2-DTND:
                [numpy ndarray] Distance to nearest drainage

        """

        # Use DEM raster information to run all loops
        dem_A=DEM.ReadAsArray()
        no_val=np.float32(DEM.GetRasterBand(1).GetNoDataValue())
        rows=DEM.RasterYSize
        cols=DEM.RasterXSize

        # get the indices of the flow direction path
        fd_index=GC.FlowDirectIndex(FD)

        # read the river location raster
        river_A=River.ReadAsArray()

        # read the flow path length raster
        fpl_A=FPL.ReadAsArray()

        # trace the flow direction to the nearest river reach and store the location
        # of that nearst reach
        nearest_network=np.ones((rows,cols,2))*np.nan
        try:
            for i in range(rows):
                for j in range(cols):
                    if dem_A[i,j] != no_val:
                        f=river_A[i,j]
                        old_row=i
                        old_cols=j

                        while f != 1:
                            # did not reached to the river yet then go to the next down stream cell
                            # get the down stream cell (furure position)
                            new_row=int(fd_index[old_row,old_cols,0])
                            new_cols=int(fd_index[old_row,old_cols,1])
                            # print(str(new_row)+","+str(new_cols))
                            # go to the downstream cell
                            f=river_A[new_row,new_cols]
                            # down stream cell becomes the current position (old position)
                            old_row=new_row
                            old_cols=new_cols
                            # at this moment old and new stored position are the same (current position)
                        # store the position in the array
                        nearest_network[i,j,0]=new_row
                        nearest_network[i,j,1]=new_cols

        except:
            assert 1==5, "please check the boundaries of your catchment after cropping the catchment using the a polygon it creates anomalies athe boundary "

        # calculate the elevation difference between the cell and the nearest drainage cell
        # or height avove nearst drainage
        HAND=np.ones((rows,cols))*np.nan

        for i in range(rows):
            for j in range(cols):
                if dem_A[i,j] != no_val:
                    HAND[i,j] = dem_A[i,j] - dem_A[int(nearest_network[i,j,0]),int(nearest_network[i,j,1])]

        # calculate the distance to the nearest drainage cell using flow path length
        # or distance to nearest drainage
        DTND=np.ones((rows,cols))*np.nan

        for i in range(rows):
            for j in range(cols):
                if dem_A[i,j] != no_val:
                    DTND[i,j] = fpl_A[i,j] - fpl_A[int(nearest_network[i,j,0]),int(nearest_network[i,j,1])]

        return HAND, DTND


    def SaveParameters(self, Path):
        """
        ============================================================
            SaveParameters(DistParFn, raster, Par, No_parameters, snow, kub, klb, Path=None)
        ============================================================
        this function takes generated parameters by the calibration algorithm,
        distributed them with a given function and save them as a rasters

        Inputs:
        ----------
            1-DistParFn:
                [function] function to distribute the parameters (all functions are
                in Hapi.DistParameters )
            2-raster:
                [gdal.dataset] raster to get the spatial information
            3-Par
                [list or numpy ndarray] parameters as 1D array or list
            4-no_parameters:
                [int] number of the parameters in the conceptual model
            5-snow:
                [integer] number to define whether to take parameters of
                the conceptual model with snow subroutine or without
            5-kub:
                [numeric] upper bound for k parameter in muskingum function
            6-klb:
                [numeric] lower bound for k parameter in muskingum function
             7-Path:
                 [string] path to the folder you want to save the parameters in
                 default value is None (parameters are going to be saved in the
                 current directory)

        Outputs:
        ----------
             Rasters for parameters of the distributed model

       Examples:
       ----------
            DemPath = path+"GIS/4000/dem4000.tif"
            raster=gdal.Open(DemPath)
            ParPath = "par15_7_2018.txt"
            par=np.loadtxt(ParPath)
            klb=0.5
            kub=1
            no_parameters=12
            DistParFn=DP.par3dLumped
            Path="parameters/"
            snow=0

            SaveParameters(DistParFn, raster, par, no_parameters,snow ,kub, klb,Path)
        """
        assert type(Path) == str, "path should be of type string"
        assert os.path.exists(Path), Path + " you have provided does not exist"

        # save
        if self.Snow == 0: # now snow subroutine
            pnme=["01_rfcf","02_FC", "03_BETA", "04_ETF", "05_LP", "06_K0","07_K1",
                  "08_K2","09_UZL","10_PERC", "11_Kmuskingum", "12_Xmuskingum"]
        else: # there is snow subtoutine
            pnme=["01_ltt", "02_rfcf", "03_sfcf", "04_cfmax", "05_cwh", "06_cfr", 
                  "07_fc", "08_beta","09_etf","10_lp", "11_k0", "12_k1", "13_k2",
                  "14_uzl", "18_perc"]

        if Path != None:
            pnme = [Path+i+"_"+str(dt.datetime.now())[0:10]+".tif" for i in pnme]

        for i in range(np.shape(self.Par3d)[2]):
            Raster.RasterLike(self.raster,self.Par3d[:,:,i],pnme[i])


    @staticmethod
    def ParametersNO(raster, no_parameters, no_lumped_par,
                     HRUs=0):
        """
        ==================================================================
             ParametersNO(raster,no_parameters,no_lumped_par,HRUs=0)
        ==================================================================
        this function calculates the nomber of parameters that the optimization
        algorithm is going top search for, use it only in case of totally distributed
        catchment parameters (in case of lumped parameters no of parameters are the same
        as the no of parameters of the conceptual model)

        Inputs:
        ----------
            1- raster:
                [gdal.dataset] raster to get the spatial information of the catchment
                (DEM, flow accumulation or flow direction raster)
            2- no_parameters
                [int] no of parameters of the cell according to the rainfall runoff model
            3-no_lumped_par:
                [int] nomber of lumped parameters, you have to enter the value of
                the lumped parameter at the end of the list, default is 0 (no lumped parameters)
            4-HRUs:
                [0 or 1] 0 to define that no hydrologic response units (HRUs), 1 to define that
                HRUs are used

        """
        # input data validation
        # data type
        assert type(raster)==gdal.Dataset, "raster should be read using gdal (gdal dataset please read it using gdal library) "
        assert type(no_parameters)== int, "no of lumped parameters should be integer"
        assert type(no_lumped_par)== int, "no of lumped parameters should be integer"

        # read the raster
        raster_A=raster.ReadAsArray()
        # get the shape of the raster
        rows=raster.RasterYSize
        cols=raster.RasterXSize
        # get the no_value of in the raster
        noval=np.float32(raster.GetRasterBand(1).GetNoDataValue())

        # count the number of non-empty cells
        no_elem = np.size(raster_A[:,:])-np.count_nonzero((raster_A[raster_A==noval]))

        if HRUs == 0:
            # input values
            if no_lumped_par > 0:
                ParametersNO=(no_elem*(no_parameters-no_lumped_par))+no_lumped_par #,"As there is "+str(no_lumped_par)+" lumped parameters, length of input parameters should be "+str(no_elem)+"*"+"("+str(no_parameters)+"-"+str(no_lumped_par)+")"+"+"+str(no_lumped_par)+"="+str(no_elem*(no_parameters-no_lumped_par)+no_lumped_par)+" not "+str(len(par_g))+" probably you have to add the value of the lumped parameter at the end of the list"
            else:
                # if there is no lumped parameters
                ParametersNO = no_elem*no_parameters #,"As there is no lumped parameters length of input parameters should be "+str(no_elem)+"*"+str(no_parameters)+"="+str(no_elem*no_parameters)
        else:
            # count the number of non-empty cells
            values=list(set([int(raster_A[i,j]) for i in range(rows) for j in range(cols) if raster_A[i,j] != noval]))
            no_elem=len(values)

            # input values
            if no_lumped_par > 0:
                ParametersNO = (no_elem*(no_parameters-no_lumped_par))+no_lumped_par #,"As there is "+str(no_lumped_par)+" lumped parameters, length of input parameters should be "+str(no_elem)+"*"+"("+str(no_parameters)+"-"+str(no_lumped_par)+")"+"+"+str(no_lumped_par)+"="+str(no_elem*(no_parameters-no_lumped_par)+no_lumped_par)+" not "+str(len(par_g))+" probably you have to add the value of the lumped parameter at the end of the list"
            else:
                # if there is no lumped parameters
                ParametersNO = no_elem*no_parameters #,"As there is no lumped parameters length of input parameters should be "+str(no_elem)+"*"+str(no_parameters)+"="+str(no_elem*no_parameters)

        return ParametersNO
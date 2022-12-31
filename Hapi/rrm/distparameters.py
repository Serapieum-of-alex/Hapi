"""DistParameters The module contains functions that is responible for distributing parameters spatially (totally distributed, totally distriby-uted with some parameters lumped, all parameters are lumped, hydrologic response units) and also save generated parameters into rasters.

@author: Mostafa
"""
import datetime as dt
import math

# import numbers
import os

import numpy as np
from osgeo import gdal
from pyramids.catchment import Catchment as GC
from pyramids.raster import Raster


class DistParameters:
    """Distripute parameter class is used to distribute the values of the parameter vector in the calibration process into the 3D array, considering if some of the parameters are lumped parameters, if you want to distribute the parameters in HRUs.

    the method included are
    1- par3d
    2- par3dLumped
    3- par2dLumpedK1_lake
    4- HRU
    5- HRU_HAND
    6- ParametersNumber
    7- saveParameters
    """

    def __init__(
        self,
        raster,
        no_parameters,
        no_lumped_par=0,
        lumped_par_pos=[],
        Lake=False,
        Snow=False,
        HRUs=False,
        Function=1,
        Kub=1,
        Klb=50,
        Maskingum=False,
    ):
        """DistParameters. To initiate the DistParameters class you have to provide the Flow Acc raster.

        Parameters
        ----------
        raster : [gdal.dataset]
            raster to get the spatial information of the catchment
            (DEM, flow accumulation or flow direction raster)
        no_parameters : [integer]
            Number of parameters in the HBV model.
        no_lumped_par : [integer], optional
            Number lumped parameters. The default is 0.
        lumped_par_pos : [integer], optional
            the order of the lumped parameters (order starts wti zero). The default is [].
        Lake : [integer], optional
            0 if there is no lake and 1 if there is a lake. The default is 0.
        Snow : [integer]
            0 if you dont want to run the snow related processes and 1 if there is snow.
            in case of 1 (simulate snow processes) parameters related to snow simulation
            has to be provided. The default is 0.
        HRUs : [integer], optional
            if the will consider using HRUs. The default is 0.
        Function : [integer], optional
            function you use to distribute parameters. The default is 1.
        Kub : [numeric], optional
            upper bound for the k-muskingum parameter. The default is 1.
        Klb : [numeric], optional
            lower bound for the k-muskingum parameter. The default is 0.5.
        Maskingum : [bool], optional
            if the routing function is muskingum. The default is False.

        Returns
        -------
        None.
        """
        assert isinstance(raster, gdal.Dataset), (
            "raster should be read using gdal (gdal dataset please read it "
            "using gdal library) "
        )
        assert isinstance(no_parameters, int), " no_parameters should be integer number"
        assert isinstance(
            no_lumped_par, int
        ), "no of lumped parameters should be integer"

        if no_lumped_par >= 1:
            if isinstance(lumped_par_pos, list):
                assert no_lumped_par == len(lumped_par_pos), (
                    f"you have to entered {no_lumped_par} no of lumped parameters but only {len(lumped_par_pos)} "
                    f"position "
                )
            else:  # if not int or list
                raise ValueError(
                    "you have one or more lumped parameters so the position has to be entered as a list"
                )

        self.Lake = Lake
        self.Snow = Snow
        self.no_lumped_par = no_lumped_par
        self.lumped_par_pos = lumped_par_pos
        self.HRUs = HRUs
        self.Kub = Kub
        self.Klb = Klb
        self.Maskingum = Maskingum
        # read the raster
        self.raster = raster
        self.raster_A = raster.ReadAsArray().astype(float)
        # get the shape of the raster
        self.rows = raster.RasterYSize
        self.cols = raster.RasterXSize
        # get the no_value of in the raster
        self.noval = raster.GetRasterBand(1).GetNoDataValue()

        for i in range(self.rows):
            for j in range(self.cols):
                if math.isclose(self.raster_A[i, j], self.noval, rel_tol=0.001):
                    self.raster_A[i, j] = np.nan

        # count the number of non-empty cells
        if self.HRUs:
            self.values = list(
                set(
                    [
                        int(self.raster_A[i, j])
                        for i in range(self.rows)
                        for j in range(self.cols)
                        if not np.isnan(self.raster_A[i, j])
                    ]
                )
            )
            self.no_elem = len(self.values)
        else:
            self.no_elem = np.size(self.raster_A[:, :]) - np.count_nonzero(
                (self.raster_A[np.isnan(self.raster_A)])
            )

        self.no_parameters = no_parameters

        # store the indeces of the non-empty cells
        self.celli = []
        self.cellj = []
        for i in range(self.rows):
            for j in range(self.cols):
                if not np.isnan(self.raster_A[i, j]):
                    self.celli.append(i)
                    self.cellj.append(j)

        # create an empty 3D array [[raster dimension], no_parameters]
        self.Par3d = np.zeros([self.rows, self.cols, self.no_parameters]) * np.nan

        if no_lumped_par >= 1:
            # parameters in array
            # remove a place for the lumped parameter (k1) lower zone coefficient
            self.no_parameters = self.no_parameters - no_lumped_par

        # all parameters lumped and distributed
        self.totnumberpar = self.no_parameters * self.no_elem + no_lumped_par
        # parameters in array
        # create a 2d array [no_parameters, no_cells]
        self.Par2d = np.zeros(shape=(self.no_parameters, self.no_elem), dtype=np.float)

        if Function == 1:
            self.Function = self.par3dLumped
        elif Function == 2:
            self.Function = self.par3d
        elif Function == 3:
            self.Function = self.par2d_lumpedK1_lake
        elif Function == 4:
            self.Function = self.HRU
        # to overwrite any choice user choose if the is HRUs
        if self.HRUs == 1:
            self.Function = self.HRU

        self.ParametersNumber()

        pass

    def par3d(self, par_g):  # , kub=1,klb=0.5,Maskingum=True
        """par3d method takes a list of parameters [saved as one column or generated as 1D list from optimization algorithm] and distribute them horizontally on number of cells given by a raster.

        Parameters
        ----------
        par_g : [list]
            list of parameters
        raster : [gdal.dataset]
            raster to get the spatial information of the catchment
            (DEM, flow accumulation or flow direction raster)
        no_parameters : [integer]
            no of parameters of the cell according to the rainfall runoff model
        no_lumped_par : [integer]
            nomber of lumped parameters, you have to enter the value of
            the lumped parameter at the end of the list, default is 0 (no lumped parameters)
        lumped_par_pos : [List]
            list of order or position of the lumped parameter among all
            the parameters of the lumped model (order starts from 0 to the length
            of the model parameters), default is [] (empty), the following order
            of parameters is used for the lumped HBV model used
            [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
            c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
        kub : [float]
            upper bound of K value (traveling time in muskingum routing method)
            default is 1 hour
        klb : [float]
            Lower bound of K value (traveling time in muskingum routing method)
            default is 0.5 hour (30 min)
        Maskingum : [bool], optional
            if the routing function is muskingum. The default is False.

        Returns
        -------
        par_3d : [3d array]
            3D array of the parameters distributed horizontally on the cells

        Examples
        --------
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
        # assert type(par_g)==np.ndarray or type(par_g)==list, "par_g should be of type 1d array or list"
        # assert isinstance(kub,numbers.Number) , " kub should be a number"
        # assert isinstance(klb,numbers.Number) , " klb should be a number"

        # input values
        if self.no_lumped_par > 0:
            par_no = (self.no_elem * self.no_parameters) + self.no_lumped_par
            assert len(par_g) == par_no, (
                f"As there is {self.no_lumped_par} lumped parameters, length of input parameters should be {self.no_elem}"
                + f"*({self.no_parameters + self.no_lumped_par} - {self.no_lumped_par}) + {self.no_lumped_par} = "
                + f"{self.no_elem * (self.no_parameters - self.no_lumped_par) + self.no_lumped_par} not {len(par_g)}"
                + " probably you have to add the value of the lumped parameter at the end of the list"
            )
        else:
            # if there is no lumped parameters
            par_no = self.no_elem * self.no_parameters
            assert len(par_g) == par_no, (
                f"As there is no lumped parameters length of input parameters should be {self.no_elem} * "
                + f"{self.no_parameters} = {self.no_elem * self.no_parameters}"
            )

        # parameters in array
        # create a 2d array [no_parameters, no_cells]
        self.Par2d = np.ones((self.no_parameters, self.no_elem))
        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(self.no_elem):
            self.Par2d[:, i] = par_g[
                i * self.no_parameters : (i * self.no_parameters) + self.no_parameters
            ]

        ### lumped parameters
        if self.no_lumped_par > 0:
            for i in range(self.no_lumped_par):
                # create a list with the value of the lumped parameter(k1)
                # (stored at the end of the list of the parameters)
                pk1 = (
                    np.ones((1, self.no_elem))
                    * par_g[(self.no_parameters * np.shape(self.Par2d)[1]) + i]
                )
                # put the list of parameter k1 at the 6 row
                self.Par2d = np.vstack(
                    [
                        self.Par2d[: self.lumped_par_pos[i], :],
                        pk1,
                        self.Par2d[self.lumped_par_pos[i] :, :],
                    ]
                )

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d
        for i in range(self.no_elem):
            self.Par3d[self.celli[i], self.cellj[i], :] = self.Par2d[:, i]

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value

        # if Maskingum:
        #     for i in range(self.no_elem):
        #         self.Par3d[self.celli[i],self.cellj[i],-2]= DistParameters.calculateK(self.Par3d[self.celli[i],self.cellj[i],-1],self.Par3d[self.celli[i],self.cellj[i],-2],kub,klb)

    def par3dLumped(self, par_g):  # , kub=1, klb=0.5, Maskingum = True
        """par3dLumped method takes a list of parameters [saved as one column or generated as 1D list from optimization algorithm] and distribute them horizontally on number of cells given by a raster.

        Parameters
        ----------
        par_g : [list]
            list of parameters
        kub : [float]
            upper bound of K value (traveling time in muskingum routing method)
            default is 1 hour
        klb : [float]
            Lower bound of K value (traveling time in muskingum routing method)
            default is 0.5 hour (30 min)
        Maskingum : [bool], optional
            if the routing function is muskingum. The default is False.

        Returns
        ----------
        par_3d: [3d array]
            3D array of the parameters distributed horizontally on the cells

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
        assert (
            type(par_g) == np.ndarray or type(par_g) == list
        ), "par_g should be of type 1d array or list"
        # assert isinstance(kub,numbers.Number) , " kub should be a number"
        # assert isinstance(klb,numbers.Number) , " klb should be a number"

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(self.no_elem):
            self.Par2d[:, i] = par_g

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d
        for i in range(self.no_elem):
            self.Par3d[self.celli[i], self.cellj[i], :] = self.Par2d[:, i]

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        # if Maskingum == True:
        #     for i in range(self.no_elem):
        #         self.Par3d[self.celli[i],self.cellj[i],-2] = DistParameters.calculateK(self.Par3d[self.celli[i],self.cellj[i],-1],
        #                                                                                self.Par3d[self.celli[i],self.cellj[i],-2],
        #                                                                                kub,klb)

    @staticmethod
    def calculateK(x, position, UB, LB):
        """calculateK. calculateK method takes value of x parameter and generate 100 random value of k parameters between upper & lower constraint then the output will be the value coresponding to the giving position.

        Parameters
        ----------
        x : [numeric]
            weighting coefficient to determine the linearity of the water surface
            (one of the parameters of muskingum routing method)
        position : [integer]
            random position between upper and lower bounds of the k parameter
        UB : [numeric]
            upper bound for k parameter
        LB : [numeric]
            Lower bound for k parameter
        """
        # k has to be smaller than this constraint
        constraint1 = 0.5 * 1 / (1 - x)
        # k has to be greater than this constraint
        constraint2 = 0.5 * 1 / x
        # if constraint is higher than UB take UB
        if constraint2 >= UB:
            constraint2 = UB
        # if constraint is lower than LB take UB
        if constraint1 <= LB:
            constraint1 = LB

        generatedK = np.linspace(constraint1, constraint2, 50)
        k = generatedK[int(round(position, 0))]
        return k

    def par2d_lumpedK1_lake(self, par_g, no_parameters_lake):  # ,kub,klb
        """par2d_lumpedK1_lake method takes a list of parameters and distribute them horizontally on number of cells given by a raster.

        Parameters
        ----------
        par_g : [list]
            list of parameters
        raster : [gdal.dataset]
            raster to get the spatial information of the catchment
            (DEM, flow accumulation or flow direction raster)
        no_parameters :[integer]
            no of parameters of the cell
        no_parameters_lake : [integer]
            no of lake parameters
        kub : [float]
            upper bound of K value (traveling time in muskingum routing method)
            default is 1 hour
        klb : [float]
            Lower bound of K value (traveling time in muskingum routing method)
            default is 0.5 hour (30 min)

        Returns
        -------
        Par3d: [3d array]
            3D array of the parameters distributed horizontally on the cells
        lake_par: [list]
            list of the lake parameters.

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
        no_parameters = self.no_parameters - 1

        # create a 2d array [no_parameters, no_cells]
        self.Par2d = np.ones((no_parameters, self.no_elem))

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        for i in range(self.no_elem):
            self.Par2d[:, i] = par_g[
                i * no_parameters : (i * no_parameters) + no_parameters
            ]

        # create a list with the value of the lumped parameter(k1)
        # (stored at the end of the list of the parameters)
        pk1 = (
            np.ones((1, self.no_elem))
            * par_g[(np.shape(self.Par2d)[0] * np.shape(self.Par2d)[1])]
        )

        # put the list of parameter k1 at the 6 row
        self.Par2d = np.vstack([self.Par2d[:6, :], pk1, self.Par2d[6:, :]])

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d
        for i in range(self.no_elem):
            self.Par3d[self.celli[i], self.cellj[i], :] = self.Par2d[:, i]

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        # for i in range(self.no_elem):
        #     self.Par3d[self.celli[i],self.cellj[i],-2]= DistParameters.calculateK(self.Par3d[self.celli[i],self.cellj[i],-1],self.Par3d[self.celli[i],self.cellj[i],-2],kub,klb)

        # lake parameters
        self.lake_par = par_g[len(par_g) - no_parameters_lake :]
        # self.lake_par[-2] = DistParameters.calculateK(self.lake_par[-1],self.lake_par[-2],kub,klb)

        # return self.Par3d, lake_par

    def HRU(self, par_g):  # ,kub=1,klb=0.5
        """HRU method takes a list of parameters [saved as one column or generated as 1D list from optimization algorithm] and distribute them horizontally on number of cells given by a raster the input raster should be classified raster (by numbers) into class to be used to define the HRUs.

        Parameters
        ----------
        par_g:
            [list] list of parameters
        raster:
            [gdal.dataset] classification raster to get the spatial information
            of the catchment and the to define each cell belongs to which HRU
        no_parameters
            [int] no of parameters of the cell according to the rainfall runoff model
        no_lumped_par:
            [int] nomber of lumped parameters, you have to enter the value of
            the lumped parameter at the end of the list, default is 0 (no lumped parameters)
        lumped_par_pos:
            [List] list of order or position of the lumped parameter among all
            the parameters of the lumped model (order starts from 0 to the length
            of the model parameters), default is [] (empty), the following order
            of parameters is used for the lumped HBV model used
            [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
            c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
        kub:
            [float] upper bound of K value (traveling time in muskingum routing method)
            default is 1 hour
        klb:
            [float] Lower bound of K value (traveling time in muskingum routing method)
            default is 0.5 hour (30 min)

        Returns
        ----------
        par_3d:
            3D array of the parameters distributed horizontally on the cells

        Examples
        --------
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
        assert (
            type(par_g) == np.ndarray or type(par_g) == list
        ), "par_g should be of type 1d array or list"
        # assert isinstance(kub,numbers.Number) , " kub should be a number"
        # assert isinstance(klb,numbers.Number) , " klb should be a number"

        # input values
        if self.no_lumped_par > 0:
            par_no = (self.no_elem * self.no_parameters) + self.no_lumped_par
            assert len(par_g) == par_no, (
                f"As there is {self.no_lumped_par} lumped parameters, length of input parameters should be "
                f"{self.no_elem}*({self.no_parameters}-{self.no_lumped_par})+{self.no_lumped_par}="
                + str(
                    self.no_elem * (self.no_parameters - self.no_lumped_par)
                    + self.no_lumped_par
                )
                + f" not {len(par_g)} probably you have to add the value of the lumped parameter at the end of the list"
            )
        else:
            # if there is no lumped parameters
            assert len(par_g) == self.no_elem * self.no_parameters, (
                f"As there is no lumped parameters length of input parameters should be {self.no_elem}*{self.no_parameters}"
                + f"={self.no_elem * self.no_parameters}"
            )

        # take the parameters from the generated parameters or the 1D list and
        # assign them to each cell
        self.Par2d = np.zeros(shape=(self.no_parameters, self.no_elem), dtype=np.float)
        for i in range(self.no_elem):
            self.Par2d[:, i] = par_g[
                i * self.no_parameters : (i * self.no_parameters) + self.no_parameters
            ]

        ### lumped parameters
        if self.no_lumped_par > 0:
            for i in range(self.no_lumped_par):
                # create a list with the value of the lumped parameter(k1)
                # (stored at the end of the list of the parameters)
                pk1 = (
                    np.ones((1, self.no_elem))
                    * par_g[(self.no_parameters * np.shape(self.Par2d)[1]) + i]
                )
                # put the list of parameter k1 at the 6 row
                self.Par2d = np.vstack(
                    [
                        self.Par2d[: self.lumped_par_pos[i], :],
                        pk1,
                        self.Par2d[self.lumped_par_pos[i] :, :],
                    ]
                )

        # calculate the value of k(travelling time in muskingum based on value of
        # x and the position and upper, lower bound of k value
        # for i in range(self.no_elem):
        #     self.Par2d[-2,i] = DistParameters.calculateK(self.Par2d[-1,i],self.Par2d[-2,i],kub,klb)

        # assign the parameters from the array (no_parameters, no_cells) to
        # the spatially corrected location in par2d each soil type will have the same
        # generated parameters
        for i in range(self.no_elem):
            self.Par3d[self.raster_A == self.values[i]] = self.Par2d[:, i]

    @staticmethod
    def HRU_HAND(DEM, FD, FPL, River):
        """HRU_HAND this function calculates inputs for the HAND (height above nearest drainage) method for land use classification.

        Parameters
        ----------
        DEM :
            raster to get the spatial information of the catchment
            (DEM raster)
        FD : [gdal.dataset]
            flow direction  raster to get the spatial information of the catchment
        FPL : [gdal.dataset]
            raster to get the spatial information of the catchment
        River : [gdal.dataset]
            raster to get the spatial information of the catchment


        Returns
        -------
        HAND: [numpy ndarray]
            Height above nearest drainage

        DTND: [numpy ndarray]
            Distance to nearest drainage
        """
        # Use DEM raster information to run all loops
        dem_A = DEM.ReadAsArray()
        no_val = np.float32(DEM.GetRasterBand(1).GetNoDataValue())
        rows = DEM.RasterYSize
        cols = DEM.RasterXSize

        # get the indices of the flow direction path
        fd_index = GC.flowDirectionIndex(FD)

        # read the river location raster
        river_A = River.ReadAsArray()

        # read the flow path length raster
        fpl_A = FPL.ReadAsArray()

        # trace the flow direction to the nearest river reach and store the location
        # of that nearst reach
        nearest_network = np.ones((rows, cols, 2)) * np.nan
        try:
            for i in range(rows):
                for j in range(cols):
                    if dem_A[i, j] != no_val:
                        f = river_A[i, j]
                        old_row = i
                        old_cols = j

                        while f != 1:
                            # did not reached to the river yet then go to the next down stream cell
                            # get the down stream cell (furure position)
                            new_row = int(fd_index[old_row, old_cols, 0])
                            new_cols = int(fd_index[old_row, old_cols, 1])
                            # print(str(new_row)+","+str(new_cols))
                            # go to the downstream cell
                            f = river_A[new_row, new_cols]
                            # down stream cell becomes the current position (old position)
                            old_row = new_row
                            old_cols = new_cols
                            # at this moment old and new stored position are the same (current position)
                        # store the position in the array
                        nearest_network[i, j, 0] = new_row
                        nearest_network[i, j, 1] = new_cols

        except:
            assert (
                False
            ), "please check the boundaries of your catchment after cropping the catchment using the a polygon it creates anomalies athe boundary "

        # calculate the elevation difference between the cell and the nearest drainage cell
        # or height avove nearst drainage
        HAND = np.ones((rows, cols)) * np.nan

        for i in range(rows):
            for j in range(cols):
                if dem_A[i, j] != no_val:
                    HAND[i, j] = (
                        dem_A[i, j]
                        - dem_A[
                            int(nearest_network[i, j, 0]), int(nearest_network[i, j, 1])
                        ]
                    )

        # calculate the distance to the nearest drainage cell using flow path length
        # or distance to nearest drainage
        DTND = np.ones((rows, cols)) * np.nan

        for i in range(rows):
            for j in range(cols):
                if dem_A[i, j] != no_val:
                    DTND[i, j] = (
                        fpl_A[i, j]
                        - fpl_A[
                            int(nearest_network[i, j, 0]), int(nearest_network[i, j, 1])
                        ]
                    )

        return HAND, DTND

    def ParametersNumber(self):
        """ParametersNumber.

            ParametersNO method calculates the nomber of parameters that the optimization algorithm is going top search
            for, use it only in case of totally distributed catchment parameters (in case of lumped parameters no of
            parameters are the same as the no of parameters of the conceptual model)

        Parameters
        ----------
        The DistParameters object should have the following attributes before tyring so use the saveParameters function
        self
            raster : [gdal.dataset]
                raster to get the spatial information of the catchment
                (DEM, flow accumulation or flow direction raster)
            no_parameters : [integer]
                no of parameters of the cell according to the rainfall runoff model
            no_lumped_par : [integer]
                nomber of lumped parameters, you have to enter the value of
                the lumped parameter at the end of the list, default is 0 (no lumped parameters)
            HRUs : [0 or 1]
                0 to define that no hydrologic response units (HRUs), 1 to define that
                HRUs are used
        """
        if not self.HRUs:
            if self.no_lumped_par > 0:
                # self.ParametersNO = (self.no_elem *( self.no_parameters - self.no_lumped_par)) + self.no_lumped_par
                self.ParametersNO = (
                    self.no_elem * self.no_parameters
                ) + self.no_lumped_par
            else:
                # if there is no lumped parameters
                self.ParametersNO = self.no_elem * self.no_parameters
        else:
            if self.no_lumped_par > 0:
                # self.ParametersNO = (self.no_elem * (self.no_parameters - self.no_lumped_par)) + self.no_lumped_par
                self.ParametersNO = (
                    self.no_elem * self.no_parameters
                ) + self.no_lumped_par
            else:
                # if there is no lumped parameters
                self.ParametersNO = self.no_elem * self.no_parameters

    def saveParameters(self, Path):
        """saveParameters. saveParameters method takes generated parameters by the calibration algorithm, distributed them with a given function and save them as a rasters.

        Parameters
        ----------
        The DistParameters object should have the following attributes before tyring so use the saveParameters function
        self:
             DistParFn:
                 [function] function to distribute the parameters (all functions are
                 in Hapi.DistParameters )
             raster:
                 [gdal.dataset] raster to get the spatial information
             Par
                 [list or numpy ndarray] parameters as 1D array or list
             no_parameters:
                 [int] number of the parameters in the conceptual model
             snow:
                 [integer] number to define whether to take parameters of
                 the conceptual model with snow subroutine or without
             kub:
                 [numeric] upper bound for k parameter in muskingum function
             klb:
                 [numeric] lower bound for k parameter in muskingum function
        Path:
              [string] path to the folder you want to save the parameters in
              default value is None (parameters are going to be saved in the
              current directory)

        Returns
        -------
        Rasters for parameters of the distributed model

        Examples
        ----------
        >>> from Hapi.rrm.distparameters import DistParameters as DP
        >>> DemPath = "GIS/4000/dem4000.tif"
        >>> raster = gdal.Open(DemPath)
        >>> ParPath = "par15_7_2018.txt"
        >>> par = np.loadtxt(ParPath)
        >>> klb = 0.5
        >>> kub = 1
        >>> no_parameters = 12
        >>> DistParFn = DP.par3dLumped
        >>> Path = "parameters/"
        >>> snow = 0
        >>> DP.saveParameters(DistParFn, raster, par, no_parameters, snow, kub, klb, Path)
        """
        assert type(Path) == str, "path should be of type string"
        assert os.path.exists(Path), Path + " you have provided does not exist"

        # save
        if self.Snow == 0:  # now snow subroutine
            pnme = [
                "01_rfcf",
                "02_FC",
                "03_BETA",
                "04_ETF",
                "05_LP",
                "06_K0",
                "07_K1",
                "08_K2",
                "09_UZL",
                "10_PERC",
                "11_Kmuskingum",
                "12_Xmuskingum",
            ]
        else:  # there is snow subtoutine
            pnme = [
                "01_ltt",
                "02_rfcf",
                "03_sfcf",
                "04_cfmax",
                "05_cwh",
                "06_cfr",
                "07_fc",
                "08_beta",
                "09_etf",
                "10_lp",
                "11_k0",
                "12_k1",
                "13_k2",
                "14_uzl",
                "18_perc",
            ]

        if Path is not None:
            pnme = [
                Path + i + "_" + str(dt.datetime.now())[0:10] + ".tif" for i in pnme
            ]

        for i in range(np.shape(self.Par3d)[2]):
            Raster.rasterLike(self.raster, self.Par3d[:, :, i], pnme[i])

    def ListAttributes(self):
        """Print Attributes List."""

        print("\n")
        print(
            "Attributes List of: "
            + repr(self.__dict__["name"])
            + " - "
            + self.__class__.__name__
            + " Instance\n"
        )
        self_keys = list(self.__dict__.keys())
        self_keys.sort()
        for key in self_keys:
            if key != "name":
                print(str(key) + " : " + repr(self.__dict__[key]))

        print("\n")

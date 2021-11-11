"""
Created on Thu May 17 04:26:42 2018

@author: Mostafa
"""

import numpy as np


class StatisticalTools:
    """StatisticalTools.

    StatisticalTools different statistical and interpolation tools

    """

    def __init__(self):
        pass

    @staticmethod
    def IDW(raster, coordinates, data, No_data_cells=False):
        """IDW.

        this function generates distributred values from reading at stations
        using inverse distance weighting method

        inputs:
            1-raster:
                GDAL calss represent the GIS raster
            2-coordinates:
                dict {'x':[],'y':[]} with two columns contains x coordinates and y
                coordinates of the stations
            3-data:
                numpy array contains values of the timeseries at each gauge in the
                same order as the coordinates in the coordinates lists (x,y)
            4- No_data_cells:
                boolen value (True or False) if the user want to calculate the
                values in the cells that has no data value (cropped) No_data_cells
                equal True if not No_data_cells equals False (default is false )
        outputs:
            1-sp_dist:
                numpy array with the same dimension of the raster

        """
        # get the shaoe of the raster
        shape_base_dem = raster.ReadAsArray().shape
        # get the coordinates of the top left corner and cell size [x,dx,y,dy]
        geo_trans = raster.GetGeoTransform()
        # get the no_value
        no_val = np.float32(
            raster.GetRasterBand(1).GetNoDataValue()
        )  # get the value stores in novalue cells
        # read the raster
        raster_array = raster.ReadAsArray()
        # calculate the coordinates of the center of each cell
        # X_coordinate= upperleft corner x+ index* cell size+celsize/2
        coox = np.ones(shape_base_dem)
        cooy = np.ones(shape_base_dem)
        # calculate the coordinates of the cells
        if (
            not No_data_cells
        ):  # if user decide not to calculate values in the o data cells
            for i in range(shape_base_dem[0]):  # iteration by row
                for j in range(shape_base_dem[1]):  # iteration by column
                    if raster_array[i, j] != no_val:
                        coox[i, j] = (
                            geo_trans[0] + geo_trans[1] / 2 + j * geo_trans[1]
                        )  # calculate x
                        cooy[i, j] = (
                            geo_trans[3] + geo_trans[5] / 2 + i * geo_trans[5]
                        )  # calculate y
        else:
            for i in range(shape_base_dem[0]):  # iteration by row
                for j in range(shape_base_dem[1]):  # iteration by column
                    coox[i, j] = (
                        geo_trans[0] + geo_trans[1] / 2 + j * geo_trans[1]
                    )  # calculate x
                    cooy[i, j] = (
                        geo_trans[3] + geo_trans[5] / 2 + i * geo_trans[5]
                    )  # calculate y

        # inverse the distance from the cell to each station
        inverseDist = np.ones(
            (shape_base_dem[0], shape_base_dem[1], len(coordinates["x"]))
        )
        # denominator of the equation (sum(1/d))
        denominator = np.ones((shape_base_dem[0], shape_base_dem[1]))

        for i in range(shape_base_dem[0]):  # iteration by row
            for j in range(shape_base_dem[1]):  # iteration by column
                if not np.isnan(coox[i, j]):  # calculate only if the coox is not nan
                    for st in range(
                        len(coordinates["x"])
                    ):  # iteration by station [0]: #
                        # distance= sqrt((xstation-xcell)**2+ystation-ycell)**2)
                        inverseDist[i, j, st] = 1 / (
                            np.sqrt(
                                np.power((coordinates["x"][st] - coox[i, j]), 2)
                                + np.power((coordinates["y"][st] - cooy[i, j]), 2)
                            )
                        )
        denominator = np.sum(inverseDist, axis=2)

        sp_dist = (
            np.ones((len(raster[:, 0]), shape_base_dem[0], shape_base_dem[1])) * np.nan
        )

        for t in range(len(raster[:, 0])):  # iteration by time step
            for i in range(shape_base_dem[0]):  # iteration by row
                for j in range(shape_base_dem[1]):  # iteration by column
                    if not np.isnan(
                        coox[i, j]
                    ):  # calculate only if the coox is not nan
                        sp_dist[i, j, t] = (
                            np.sum(inverseDist[i, j, :] * raster[t, :])
                            / denominator[i, j]
                        )
        # change the type to float 32
        sp_dist = sp_dist.astype(np.float32)

        return sp_dist

    @staticmethod
    def ISDW(raster, coordinates, data, No_data_cells=False):
        """ISDW.

        this function generates distributred values from reading at stations using
        inverse squared distance weighting method

        inputs:
            1-raster:
                GDAL calss represent the GIS raster
            2-coordinates:
                dict {'x':[],'y':[]} with two columns contains x coordinates and y
                coordinates of the stations
            3-data:
                numpy array contains values of the timeseries at each gauge in the
                same order as the coordinates in the coordinates lists (x,y)
            4- No_data_cells:
                boolen value (True or False) if the user want to calculate the
                values in the cells that has no data value (cropped) No_data_cells
                equal True if not No_data_cells equals False
                (default is false )
        outputs:
            1-sp_dist:
                numpy array with the same dimension of the raster
        """
        shape_base_dem = raster.ReadAsArray().shape
        # get the coordinates of the top left corner and cell size [x,dx,y,dy]
        geo_trans = raster.GetGeoTransform()
        # get the no_value
        no_val = np.float32(
            raster.GetRasterBand(1).GetNoDataValue()
        )  # get the value stores in novalue cells
        # read the raster
        raster_array = raster.ReadAsArray()
        # calculate the coordinates of the center of each cell
        # X_coordinate= upperleft corner x+ index* cell size+celsize/2
        coox = np.ones(shape_base_dem) * np.nan
        cooy = np.ones(shape_base_dem) * np.nan
        # calculate the coordinates of the cells
        if (
            not No_data_cells
        ):  # if user decide not to calculate values in the o data cells
            for i in range(shape_base_dem[0]):  # iteration by row
                for j in range(shape_base_dem[1]):  # iteration by column
                    if raster_array[i, j] != no_val:
                        coox[i, j] = (
                            geo_trans[0] + geo_trans[1] / 2 + j * geo_trans[1]
                        )  # calculate x
                        cooy[i, j] = (
                            geo_trans[3] + geo_trans[5] / 2 + i * geo_trans[5]
                        )  # calculate y
        else:
            for i in range(shape_base_dem[0]):  # iteration by row
                for j in range(shape_base_dem[1]):  # iteration by column
                    coox[i, j] = (
                        geo_trans[0] + geo_trans[1] / 2 + j * geo_trans[1]
                    )  # calculate x
                    cooy[i, j] = (
                        geo_trans[3] + geo_trans[5] / 2 + i * geo_trans[5]
                    )  # calculate y

        print("step 1 finished")
        # inverse the distance from the cell to each station
        inverseDist = np.ones(
            (shape_base_dem[0], shape_base_dem[1], len(coordinates["x"]))
        )
        # denominator of the equation (sum(1/d))
        denominator = np.ones((shape_base_dem[0], shape_base_dem[1]))

        for i in range(shape_base_dem[0]):  # iteration by row
            for j in range(shape_base_dem[1]):  # iteration by column
                if not np.isnan(coox[i, j]):  # calculate only if the coox is not nan
                    for st in range(
                        len(coordinates["x"])
                    ):  # iteration by station [0]: #
                        inverseDist[i, j, st] = (
                            1
                            / (
                                np.sqrt(
                                    np.power((coordinates["x"][st] - coox[i, j]), 2)
                                    + np.power((coordinates["y"][st] - cooy[i, j]), 2)
                                )
                            )
                            ** 2
                        )
        print("step 2 finished")
        denominator = np.sum(inverseDist, axis=2)
        sp_dist = (
            np.ones((shape_base_dem[0], shape_base_dem[1], len(data[:, 0]))) * np.nan
        )
        for t in range(len(data[:, 0])):  # iteration by time step
            for i in range(shape_base_dem[0]):  # iteration by row
                for j in range(shape_base_dem[1]):  # iteration by column
                    if not np.isnan(
                        coox[i, j]
                    ):  # calculate only if the coox is not nan
                        sp_dist[i, j, t] = (
                            np.sum(inverseDist[i, j, :] * data[t, :])
                            / denominator[i, j]
                        )
        # change the type to float 32
        sp_dist = sp_dist.astype(np.float32)

        return sp_dist

    @staticmethod
    def Normalize(x):
        """Normalizer

        to normalize values between 0 and 1

        Inputs:
        -------
            1-x : [List]
                list of values
        Outputs:
        -------
            1- normalized numbers : [List]
                list of normalized values
        """
        x = np.array(x)
        DataMax = max(x)
        DataMin = min(x)
        N = (x - DataMin) / (DataMax - DataMin)
        # [i - DataMin / (DataMax - DataMin) for i in x]
        return N

    @staticmethod
    def Standardize(x):
        """Standardize.

        to standardize (make the average equals 1 and the standard deviation
        equals 0)

        Inputs :
            1-x:
                [List] list of values
        Outputs:
            1-
            [List] list of normalized values
        """
        x = np.array(x)

        mean = np.mean(x)
        std = np.std(x)
        s = (x - mean) / std
        # [i - mean / (std) for i in x]
        return s

    @staticmethod
    def Rescale(OldValue, OldMin, OldMax, NewMin, NewMax):
        """Rescale.

        Rescale nethod rescales a value between two boundaries to a new value
        bewteen two other boundaries
        inputs:
            1-OldValue:
                [float] value need to transformed
            2-OldMin:
                [float] min old value
            3-OldMax:
                [float] max old value
            4-NewMin:
                [float] min new value
            5-NewMax:
                [float] max new value
        output:
            1-NewValue:
                [float] transformed new value

        """
        OldRange = OldMax - OldMin
        NewRange = NewMax - NewMin
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

        return NewValue

    @staticmethod
    def LogarithmicRescale(x, min_old, max_old, min_new, max_new):
        """LogarithmicRescale.

        this function transform the value between two normal values to a logarithmic scale
        between logarithmic value of both boundaries
            np.log(base)(number) = power
            the inverse of logarithmic is base**power = number
        inputs:
            1-x:
                [float] new value needed to be transformed to a logarithmic scale
            2-min_old:
                [float] min old value in normal scale
            3-max_old:
                [float] max old value in normal scale
            4-min_new:
                [float] min new value in normal scale
            5-max_new:
                [float] max_new max new value
        output:
            1-Y:
                [int] integer number between new max_new and min_new boundaries
        """
        # get the boundaries of the logarithmic scale
        if min_old == 0.0:
            min_old_log = -7
        else:
            min_old_log = np.log(min_old)

        max_old_log = np.log(max_old)

        if x == 0:
            x_log = -7
        else:
            x_log = np.log(x)

        y = int(
            np.round(
                StatisticalTools.Rescale(
                    x_log, min_old_log, max_old_log, min_new, max_new
                )
            )
        )

        return y

    @staticmethod
    def InvLogarithmicRescale(x, min_old, max_old, min_new, max_new, base=np.e):
        """LogarithmicRescale.

        this function transform the value between two normal values to a logarithmic scale
        between logarithmic value of both boundaries
            np.log(base)(number) = power
            the inverse of logarithmic is base**power = number
        inputs:
            1-x:
                [float] new value needed to be transformed to a logarithmic scale
            2-min_old:
                [float] min old value in normal scale
            3-max_old:
                [float] max old value in normal scale
            4-min_new:
                [float] min new value in normal scale
            5-max_new:
                [float] max_new max new value
        output:
            1-Y:
                [int] integer number between new max_new and min_new boundaries
        """
        # get the boundaries of the logarithmic scale

        min_old_power = np.power(base, min_old)
        max_old_power = np.power(base, max_old)
        x_power = np.power(base, x)

        y = int(
            np.round(
                StatisticalTools.Rescale(
                    x_power, min_old_power, max_old_power, min_new, max_new
                )
            )
        )
        return y

    @staticmethod
    def Round(number, roundto):
        return round(number / roundto) * roundto

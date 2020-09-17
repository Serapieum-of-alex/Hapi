# -*- coding: utf-8 -*-
"""
Created on Thu May 17 04:26:42 2018

@author: Mostafa
"""


#%library
import numpy as np
# import pandas as pd
import scipy.optimize as so
from scipy.stats import gumbel_r, norm, genextreme

class StatisticalTools():

    def __init__(self):
        pass

    @staticmethod
    def IDW(raster,coordinates,data,No_data_cells=False):
        """
        # =============================================================================
        #  IDW(flp,coordinates,prec_data):
        # =============================================================================
        this function generates distributred values from reading at stations using
        inverse distance weighting method

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
        # get the shaoe of the raster
        shape_base_dem = raster.ReadAsArray().shape
        # get the coordinates of the top left corner and cell size [x,dx,y,dy]
        geo_trans = raster.GetGeoTransform()
        # get the no_value
        no_val = np.float32(raster.GetRasterBand(1).GetNoDataValue()) # get the value stores in novalue cells
        # read the raster
        raster_array=raster.ReadAsArray()
        # calculate the coordinates of the center of each cell
        # X_coordinate= upperleft corner x+ index* cell size+celsize/2
        coox=np.ones(shape_base_dem)
        cooy=np.ones(shape_base_dem)
        # calculate the coordinates of the cells
        if not No_data_cells: # if user decide not to calculate values in the o data cells
            for i in range(shape_base_dem[0]): # iteration by row
                for j in range(shape_base_dem[1]):# iteration by column
                    if raster_array[i,j] != no_val:
                        coox[i,j]=geo_trans[0]+geo_trans[1]/2+j*geo_trans[1] # calculate x
                        cooy[i,j]=geo_trans[3]+geo_trans[5]/2+i*geo_trans[5] # calculate y
        else:
            for i in range(shape_base_dem[0]): # iteration by row
                for j in range(shape_base_dem[1]):# iteration by column
                    coox[i,j]=geo_trans[0]+geo_trans[1]/2+j*geo_trans[1] # calculate x
                    cooy[i,j]=geo_trans[3]+geo_trans[5]/2+i*geo_trans[5] # calculate y

        # inverse the distance from the cell to each station
        inverseDist=np.ones((shape_base_dem[0],shape_base_dem[1],len(coordinates['x'])))
        # denominator of the equation (sum(1/d))
        denominator=np.ones((shape_base_dem[0],shape_base_dem[1]))

        for i in range(shape_base_dem[0]): # iteration by row
            for j in range(shape_base_dem[1]):# iteration by column
                if not np.isnan(coox[i,j]): # calculate only if the coox is not nan
                    for st in range(len(coordinates['x'])): #iteration by station [0]: #
                    # distance= sqrt((xstation-xcell)**2+ystation-ycell)**2)
                        inverseDist[i,j,st]=1/(np.sqrt(np.power((coordinates['x'][st]-coox[i,j]),2)
                                             +np.power((coordinates['y'][st]-cooy[i,j]),2)))
        denominator=np.sum(inverseDist,axis=2)

        sp_dist=np.ones((len(raster[:,0]),shape_base_dem[0],shape_base_dem[1]))*np.nan

        for t in range(len(raster[:,0])): # iteration by time step
            for i in range(shape_base_dem[0]): # iteration by row
                for j in range(shape_base_dem[1]):# iteration by column
                    if not np.isnan(coox[i,j]): # calculate only if the coox is not nan
                        sp_dist[i,j,t]=np.sum(inverseDist[i,j,:]*raster[t,:])/denominator[i,j]
        # change the type to float 32
        sp_dist=sp_dist.astype(np.float32)

        return sp_dist

    @staticmethod
    def ISDW(raster,coordinates,data,No_data_cells=False):
        """
        # =============================================================================
        #  ISDW(flp,coordinates,prec_data):
        # =============================================================================
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
        no_val = np.float32(raster.GetRasterBand(1).GetNoDataValue()) # get the value stores in novalue cells
        # read the raster
        raster_array=raster.ReadAsArray()
        # calculate the coordinates of the center of each cell
        # X_coordinate= upperleft corner x+ index* cell size+celsize/2
        coox=np.ones(shape_base_dem)*np.nan
        cooy=np.ones(shape_base_dem)*np.nan
        # calculate the coordinates of the cells
        if not No_data_cells: # if user decide not to calculate values in the o data cells
            for i in range(shape_base_dem[0]): # iteration by row
                for j in range(shape_base_dem[1]):# iteration by column
                    if raster_array[i,j] != no_val:
                        coox[i,j]=geo_trans[0]+geo_trans[1]/2+j*geo_trans[1] # calculate x
                        cooy[i,j]=geo_trans[3]+geo_trans[5]/2+i*geo_trans[5] # calculate y
        else:
            for i in range(shape_base_dem[0]): # iteration by row
                for j in range(shape_base_dem[1]):# iteration by column
                    coox[i,j]=geo_trans[0]+geo_trans[1]/2+j*geo_trans[1] # calculate x
                    cooy[i,j]=geo_trans[3]+geo_trans[5]/2+i*geo_trans[5] # calculate y

        print("step 1 finished")
        # inverse the distance from the cell to each station
        inverseDist=np.ones((shape_base_dem[0],shape_base_dem[1],len(coordinates['x'])))
        # denominator of the equation (sum(1/d))
        denominator=np.ones((shape_base_dem[0],shape_base_dem[1]))

        for i in range(shape_base_dem[0]): # iteration by row
            for j in range(shape_base_dem[1]):# iteration by column
                if not np.isnan(coox[i,j]): # calculate only if the coox is not nan
                    for st in range(len(coordinates['x'])): #iteration by station [0]: #
                        inverseDist[i,j,st]=1/(np.sqrt(np.power((coordinates['x'][st]-coox[i,j]),2)
                                             +np.power((coordinates['y'][st]-cooy[i,j]),2)))**2
        print("step 2 finished")
        denominator=np.sum(inverseDist,axis=2)
        sp_dist=np.ones((shape_base_dem[0],shape_base_dem[1],len(data[:,0])))*np.nan
        for t in range(len(data[:,0])): # iteration by time step
            for i in range(shape_base_dem[0]): # iteration by row
                for j in range(shape_base_dem[1]):# iteration by column
                    if not np.isnan(coox[i,j]): # calculate only if the coox is not nan
                        sp_dist[i,j,t]=np.sum(inverseDist[i,j,:]*data[t,:])/denominator[i,j]
        # change the type to float 32
        sp_dist=sp_dist.astype(np.float32)

        return sp_dist

    @staticmethod
    def Normalizer(x):
        """
        ================================================
           Normalizer(x, maxx, minn)
        ================================================
        to normalize values between 0 and 1

        Inputs :
            1-x:
                [List] list of values
        Outputs:
            1-
            [List] list of normalized values
        """
        DataMax = max(x)
        DataMin = min(x)

        return [i - DataMin/(DataMax-DataMin) for i in x]
    @staticmethod
    def Standardize(x):
        """
        ================================================
           Standardize(x)
        ================================================
        to standardize (make the average equals 1 and the standard deviation
        equals1)

        Inputs :
            1-x:
                [List] list of values
        Outputs:
            1-
            [List] list of normalized values
        """
        mean = np.mean(x)
        std = np.std(x)

        return [i - mean/(std) for i in x]

    @staticmethod
    def SensitivityAnalysis(Parameter, LB, UB, Function, *args,**kwargs):
        """
        ======================================================================
           SensitivityAnalysis(Parameter, LB, UB, Function,*args,**kwargs)
        ======================================================================

        Parameters
        ----------
        Parameter : [dataframe]
            parameters dataframe including the parameters values in a column with
            name 'value' and the parameters name as index.
        LB : [List]
            parameters upper bounds.
        UB : [List]
            parameters lower bounds.
        Function : [function]
            DESCRIPTION.
        *args : TYPE
            arguments of the function with the same exact names inside the function.
        **kwargs : TYPE
            keyword arguments of the function with the same exact names inside the function.

        Returns
        -------
        sen : [Dictionary]
            DESCRIPTION.

        """

        sen={}
        for i in range(len(Parameter)):
            sen[Parameter.index[i]]=[[],[]]
            # generate 5 random values between the high and low parameter bounds
            rand_value = np.linspace(LB[i],UB[i],5)
            # add the value of the calibrated parameter and sort the values
            rand_value = np.sort(np.append(rand_value,Parameter['value'][i]))
            # relative values of the parameters
            sen[Parameter.index[i]][0] = [((h)/Parameter['value'][i]) for h in rand_value]

            Randpar = Parameter['value'].tolist()
            for j in range(len(rand_value)):
                Randpar[i]=rand_value[j]
                # args = list(args)
                # args.insert(Position,Randpar)
                metric = Function(Randpar,*args,**kwargs)

                sen[Parameter.index[i]][1].append(metric)
                print(round(metric,4))
                print( str(i)+'-'+Parameter.index[i]+' -'+ str(j))

        return sen


    @staticmethod
    def Weibul(data,option=1):
        """
        =========================================
          Weibul(data,option)
        =========================================
        Weibul method to calculate the cumulative distribution function or
        return period.

        Parameters
        ----------
        data : [list/array]
            list/array of the data.
        option : [1/2]
            1 to calculate the cumulative distribution function cdf or
            2 to calculate the return period.default=1

        Returns
        -------
        CDF/T: [list]
            list of cumulative distribution function or return period.
        """
        data.sort()

        if option==1:
            CDF = [j/(len(data)+1) for j in range(1,len(data)+1)]
            return CDF
        else:
            CDF = [j/(len(data)+1) for j in range(1,len(data)+1)]
            T = [1/(1-j) for j in CDF]
            return T


class Gumbel():

    def __init__(self):
        pass

    @staticmethod
    def Pdf(x, loc, scale):
        """
        ========================================================================
          Returns the value of Gumbel's pdf with parameters loc and scale at x .
        ========================================================================

        Parameters
        ----------
        x : TYPE
            data.
        loc : TYPE
            DESCRIPTION.
        scale : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        z = (x - loc)/scale

        return (1./scale) * (np.exp(-(z + (np.exp(-z)))))

    @staticmethod
    def Cdf(x, loc, scale):
        """
        Returns the value of Gumbel's cdf with parameters loc and scale at x.
        """
        return np.exp(-np.exp(-(x-loc)/scale))

    @staticmethod
    def ObjectiveFn(p, x): #, threshold

        threshold=p[0]
        loc=p[1]
        scale=p[2]

        x1=x[x<threshold]
        nx2=len(x[x>=threshold])
        # pdf with a scaled pdf
        # L1 is pdf based
        L1=(-np.log((Gumbel.Pdf(x1, loc, scale)/scale))).sum()
        # L2 is cdf based
        L2=(-np.log(1-Gumbel.Cdf(threshold, loc, scale)))*nx2
        #print x1, nx2, L1, L2
        return L1+L2

    @staticmethod
    def EstimateParameter(data, ObjFunc, threshold):
        """
        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for values below (x < C), now the likeliest parameters
        are those at the max value of mutiplication between two functions max(L1*L2).

        In this case the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        threshold : TYPE
            DESCRIPTION.

        Returns
        -------
        Param : TYPE
            DESCRIPTION.

        Example:
            from Hapi.statisticaltools import StatisticalTools as ST
            Param_dist = Gumbel.EstimateParameter(data, threshold)

        """
        # obj_func = lambda p, x: (-np.log(Gumbel.Pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
        Par1 = gumbel_r.fit(data)
        #then we use the result as starting value for your truncated Gumbel fit
        Param = so.fmin(ObjFunc, [threshold, Par1[0], Par1[1]],  args=(np.array(data),),
                        maxiter=500, maxfun=500)
        # Param_dist = [Param[1], Param[2]]

        return Param

    @staticmethod
    def ProbapilityPlot(param, cdf, data, SignificanceLevel):
        """
        ===================================================================
            ProbapilityPlot(param, cdf, data, SignificanceLevel)
        ===================================================================
        this method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculate the confidence interval.

        Parameters
        ----------
        param : [list]
            list of the distribution parameters [loc, scale].
        cdf : [list]
            theoretical cdf calculated using weibul or using the distribution cdf function.
        data : [list/array]
            list of the values.
        SignificanceLevel : [float]
            value between 0 and 1.

        Returns
        -------
        Qth : [list]
            theoretical generated values based on the theoretical cdf calculated from
            weibul or the distribution parameters.
        Qupper : [list]
            upper bound coresponding to the confidence interval.
        Qlower : [list]
            lower bound coresponding to the confidence interval.
        """

        Qth = [param[0] - param[1]*(np.log(-np.log(j))) for j in cdf]
        Y = [-np.log(-np.log(j)) for j in cdf]
        StdError = [(param[1]/np.sqrt(len(data))) * np.sqrt(1.1087+0.5140*j+0.6079*j**2) for j in Y]
        v = norm.ppf(1-SignificanceLevel/2)
        Qupper = [Qth[j] + v * StdError[j] for j in range(len(data))]
        Qlower = [Qth[j] - v * StdError[j] for j in range(len(data))]

        return Qth, Qupper, Qlower

class GEV():

    def __init__():
        pass

    @staticmethod
    def ProbapilityPlot(param, cdf, data, SignificanceLevel):
        """
        still not finished
        the equations are the same of the gumbel dist and have to be changed to
        GEV equations
        ===================================================================
            ProbapilityPlot(param, cdf, data, SignificanceLevel)
        ===================================================================
        this method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculate the confidence interval.

        Parameters
        ----------
        param : [list]
            list of the distribution parameters [loc, scale].
        cdf : [list]
            theoretical cdf calculated using weibul or using the distribution cdf function.
        data : [list/array]
            list of the values.
        SignificanceLevel : [float]
            value between 0 and 1.

        Returns
        -------
        Qth : [list]
            theoretical generated values based on the theoretical cdf calculated from
            weibul or the distribution parameters.
        Qupper : [list]
            upper bound coresponding to the confidence interval.
        Qlower : [list]
            lower bound coresponding to the confidence interval.
        """

        # Qth = [param[0] - param[1]*(np.log(-np.log(j))) for j in cdf]
        Qth = genextreme.ppf(cdf, c=param[0], loc=param[1], scale=param[2])
        Y = [-np.log(-np.log(j)) for j in cdf]
        StdError = [(param[1]/np.sqrt(len(data))) * np.sqrt(1.1087+0.5140*j+0.6079*j**2) for j in Y]
        v = norm.ppf(1-SignificanceLevel/2)
        Qupper = [Qth[j] + v * StdError[j] for j in range(len(data))]
        Qlower = [Qth[j] - v * StdError[j] for j in range(len(data))]

        return Qth, Qupper, Qlower

# -*- coding: utf-8 -*-
"""
Calibration

calibration contains functions to to connect the parameter spatial distribution
function with the with both component of the spatial representation of the hydrological
process (conceptual model & spatial routing) to calculate the performance of predicted
runoff at known locations based on given performance function

@author: Mostafa


"""
import numpy as np
import datetime as dt
from Oasis.optimization import Optimization
from Oasis.hsapi import HSapi
from Hapi.run import Model
from Hapi.wrapper import Wrapper


class Calibration(Model):

    def __init__(self, name, start, end, fmt="%Y-%m-%d", SpatialResolution = 'Lumped',
                 TemporalResolution = "Daily"):
        """
        SpatialR : TYPE, optional
            Spatial Resolution "Distributed" or "Lumped". The default is 'Lumped'.

        Returns
        -------
        None.

        """
        self.name = name
        self.start = dt.datetime.strptime(start,fmt)
        self.end = dt.datetime.strptime(end,fmt)
        self.SpatialResolution = SpatialResolution
        self.TemporalResolution = TemporalResolution
        if TemporalResolution == "Daily":
            self.Timef = 24
        else:
            #TODO calculate the temporal resolution factor
            self.Tfactor = 24
        pass


    def ReadObjectiveFn(self,OF,args):
        # check objective_function
        assert callable(OF) , "The Objective function should be a function"
        self.OF = OF

        if args == None :
            args = []

        self.OFArgs = args

    def RunCalibration(self, SpatialVarFun, OptimizationArgs, printError=None):
        """
        =======================================================================
            RunCalibration(ConceptualModel, Paths, p2, Q_obs, UB, LB, SpatialVarFun, lumpedParNo, lumpedParPos, objective_function, printError=None, *args):
        =======================================================================
        this function runs the calibration algorithm for the conceptual distributed
        hydrological model

        Inputs:
        ----------
            1-ConceptualModel:
                [function] conceptual model and it should contain a function called simulate

            2-Basic_inputs:
                1-p2:
                    [List] list of unoptimized parameters
                    p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                    p2[1] = catchment area in km2
                2-init_st:
                    [list] initial values for the state variables [sp,sm,uz,lz,wc] in mm
                3-UB:
                    [Numeric] upper bound of the values of the parameters
                4-LB:
                    [Numeric] Lower bound of the values of the parameters
            3-Q_obs:
                [Numeric] Observed values of discharge

            6-lumpedParNo:
                [int] nomber of lumped parameters, you have to enter the value of
                the lumped parameter at the end of the list, default is 0 (no lumped parameters)
            7-lumpedParPos:
                [List] list of order or position of the lumped parameter among all
                the parameters of the lumped model (order starts from 0 to the length
                of the model parameters), default is [] (empty), the following order
                of parameters is used for the lumped HBV model used
                [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
                c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
            8-objective_function:
                [function] objective function to calculate the performance of the model
                and to be used in the calibration
            9-*args:
                other arguments needed on the objective function

        Outputs:
        ----------
            1- st:
                [4D array] state variables
            2- q_out:
                [1D array] calculated Discharge at the outlet of the catchment
            3- q_uz:
                [3D array] Distributed discharge for each cell

        Example:
        ----------
            PrecPath = prec_path="meteodata/4000/calib/prec"
            Evap_Path = evap_path="meteodata/4000/calib/evap"
            TempPath = temp_path="meteodata/4000/calib/temp"
            FlowAccPath = "GIS/4000/acc4000.tif"
            FlowDPath = "GIS/4000/fd4000.tif"
            ParPath = "meteodata/4000/"+"parameters.txt"
            p2=[1, 227.31]
            st, q_out, q_uz_routed = RunModel(PrecPath,Evap_Path,TempPath,DemPath,
                                              FlowAccPath,FlowDPath,ParPath,p2)
        """
        # input dimensions
        [rows,cols] = self.FlowAcc.ReadAsArray().shape
        [fd_rows,fd_cols] = self.FlowDir.ReadAsArray().shape
        assert fd_rows == rows and fd_cols == cols, "all input data should have the same number of rows"

        # input dimensions
        assert np.shape(self.Prec)[0] == rows and np.shape(self.ET)[0] == rows and np.shape(self.Temp)[0] == rows, "all input data should have the same number of rows"
        assert np.shape(self.Prec)[1] == cols and np.shape(self.ET)[1] == cols and np.shape(self.Temp)[1] == cols, "all input data should have the same number of columns"
        assert np.shape(self.Prec)[2] == np.shape(self.ET)[2] and np.shape(self.Temp)[2], "all meteorological input data should have the same length"

        # basic inputs
        # check if all inputs are included
        # assert all(["p2","init_st","UB","LB","snow "][i] in Basic_inputs.keys() for i in range(4)), "Basic_inputs should contain ['p2','init_st','UB','LB'] "

        ### optimization

        # get arguments
        ApiObjArgs = OptimizationArgs[0]
        pll_type = OptimizationArgs[1]
        ApiSolveArgs = OptimizationArgs[2]
        # check optimization arguement
        assert type(ApiObjArgs) == dict, "store_history should be 0 or 1"
        assert type(ApiSolveArgs) == dict, "history_fname should be of type string "

        print('Calibration starts')
        ### calculate the objective function
        def opt_fun(par):
            try:
                # parameters
                klb=float(par[-2])
                kub=float(par[-1])
                par=par[:-2]
                SpatialVarFun.Function(par, kub=kub, klb=klb)
                self.Parameters = SpatialVarFun.Par3d
                #run the model
                q_out, q_uz_routed, q_lz_trans = Wrapper.HapiModel(self)
                # calculate performance of the model
                try:
                    error = self.OF(self.QGauges, q_out, q_uz_routed, q_lz_trans,*[self.GaugesTable])
                except TypeError: # if no of inputs less than what the function needs
                    assert False, "the objective function you have entered needs more inputs please enter then in a list as *args"

                # print error
                if printError != 0:
                    print(error)
                    print(par)

                fail = 0
            except:
                error = np.nan
                fail = 1

            return error, [], fail

        ### define the optimization components
        opt_prob = Optimization('HBV Calibration', opt_fun)
        for i in range(len(self.LB)):
            opt_prob.addVar('x{0}'.format(i), type='c', lower=self.LB[i], upper=self.UB[i])

        print(opt_prob)

        opt_engine = HSapi(pll_type=pll_type , options=ApiObjArgs)


        store_sol = ApiSolveArgs['store_sol']
        display_opts = ApiSolveArgs['display_opts']
        store_hst = ApiSolveArgs['store_hst']
        hot_start = ApiSolveArgs['hot_start']


        res = opt_engine(opt_prob, store_sol=store_sol, display_opts=display_opts,
                         store_hst=store_hst, hot_start=hot_start)

        self.Parameters = res[1]
        self.OFvalue = res[0]

        return res


    def LumpedCalibration(self, Basic_inputs, OptimizationArgs, printError=None):
        """
        =======================================================================
            RunCalibration(ConceptualModel, data,parameters, p2, init_st, snow, Routing=0, RoutingFn=[], objective_function, printError=None, *args):
        =======================================================================
        this function runs the calibration algorithm for the Lumped conceptual hydrological model

        Inputs:
        ----------
            1-ConceptualModel:
                [function] conceptual model and it should contain a function called simulate
            2-data:
                [numpy array] meteorological data as array with the first column as precipitation
                second as evapotranspiration, third as temperature and forth column as
                long term average temperature
            2-Basic_inputs:
                1-p2:
                    [List] list of unoptimized parameters
                    p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
                    p2[1] = catchment area in km2
                2-init_st:
                    [list] initial values for the state variables [sp,sm,uz,lz,wc] in mm
                3-UB:
                    [Numeric] upper bound of the values of the parameters
                4-LB:
                    [Numeric] Lower bound of the values of the parameters
            3-Q_obs:
                [Numeric] Observed values of discharge

            6-lumpedParNo:
                [int] nomber of lumped parameters, you have to enter the value of
                the lumped parameter at the end of the list, default is 0 (no lumped parameters)
            7-lumpedParPos:
                [List] list of order or position of the lumped parameter among all
                the parameters of the lumped model (order starts from 0 to the length
                of the model parameters), default is [] (empty), the following order
                of parameters is used for the lumped HBV model used
                [ltt, utt, rfcf, sfcf, ttm, cfmax, cwh, cfr, fc, beta, e_corr, etf, lp,
                c_flux, k, k1, alpha, perc, pcorr, Kmuskingum, Xmuskingum]
            8-objective_function:
                [function] objective function to calculate the performance of the model
                and to be used in the calibration
            9-*args:
                other arguments needed on the objective function

        Outputs:
        ----------
            1- st:
                [4D array] state variables
            2- q_out:
                [1D array] calculated Discharge at the outlet of the catchment
            3- q_uz:
                [3D array] Distributed discharge for each cell

        Example:
        ----------
            PrecPath = prec_path="meteodata/4000/calib/prec"
            Evap_Path = evap_path="meteodata/4000/calib/evap"
            TempPath = temp_path="meteodata/4000/calib/temp"
            FlowAccPath = "GIS/4000/acc4000.tif"
            FlowDPath = "GIS/4000/fd4000.tif"
            ParPath = "meteodata/4000/"+"parameters.txt"
            p2=[1, 227.31]
            st, q_out, q_uz_routed = RunModel(PrecPath,Evap_Path,TempPath,DemPath,
                                              FlowAccPath,FlowDPath,ParPath,p2)
        """
        # basic inputs
        # check if all inputs are included
        assert all(["Route","RoutingFn"][i] in Basic_inputs.keys() for i in range(2)), "Basic_inputs should contain ['p2','init_st','UB','LB'] "

        Route = Basic_inputs["Route"]
        RoutingFn = Basic_inputs["RoutingFn"]
        if 'InitialValues' in Basic_inputs.keys():
            InitialValues = Basic_inputs['InitialValues']
        else:
            InitialValues = []

        ### optimization

        # get arguments
        ApiObjArgs = OptimizationArgs[0]
        pll_type = OptimizationArgs[1]
        ApiSolveArgs = OptimizationArgs[2]
        # check optimization arguement
        assert type(ApiObjArgs) == dict, "store_history should be 0 or 1"
        assert type(ApiSolveArgs) == dict, "history_fname should be of type string "

        # assert history_fname[-4:] == ".txt", "history_fname should be txt file please change extension or add .txt ad the end of the history_fname"

        print('Calibration starts')
        ### calculate the objective function
        def opt_fun(par):
            try:
                # parameters
                self.Parameters = par
                #run the model
                _, q_out = Wrapper.Lumped(self, Route, RoutingFn)

                # calculate performance of the model
                try:
                    error = self.OF(self.QGauges[self.QGauges.columns[0]],q_out,*self.OFArgs)
                except TypeError: # if no of inputs less than what the function needs
                    assert 1==5, "the objective function you have entered needs more inputs please enter then in a list as *args"

                # print error
                if printError != 0:
                    print(error)
                    # print(par)

                fail = 0
            except:
                error = np.nan
                fail = 1

            return error, [], fail

        ### define the optimization components
        opt_prob = Optimization('HBV Calibration', opt_fun)
        
        if InitialValues != []:
            for i in range(len(self.LB)):
                opt_prob.addVar('x{0}'.format(i), type='c', lower=self.LB[i], upper=self.UB[i], value=InitialValues[i])
        else:
            for i in range(len(self.LB)):
                opt_prob.addVar('x{0}'.format(i), type='c', lower=self.LB[i], upper=self.UB[i])

        print(opt_prob)


        opt_engine = HSapi(pll_type=pll_type , options=ApiObjArgs)

        # parse the ApiSolveArgs inputs
        # availablekeys = ['store_sol',"display_opts","store_hst","hot_start"]

        store_sol = ApiSolveArgs['store_sol']
        display_opts = ApiSolveArgs['display_opts']
        store_hst = ApiSolveArgs['store_hst']
        hot_start = ApiSolveArgs['hot_start']

        # for i in range(len(availablekeys)):
            # if availablekeys[i] in ApiSolveArgs.keys():
                # exec(availablekeys[i] + "=" + str(ApiSolveArgs[availablekeys[i]]))
            # print(availablekeys[i] + " = " + str(ApiSolveArgs[availablekeys[i]]))

        res = opt_engine(opt_prob, store_sol=store_sol, display_opts=display_opts,
                         store_hst=store_hst, hot_start=hot_start)

        self.Parameters = res[1]
        self.OFvalue = res[0]

        return res
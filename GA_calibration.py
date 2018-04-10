# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:18:46 2018

@author: Mostafa
"""
#%% paths & links 
from IPython import get_ipython   # to reset the variable explorer each time
get_ipython().magic('reset -f')
import os
os.chdir("C:\\Users\\Mostafa\\Desktop\\My Files\\thesis\\My Thesis\\Data_and_Models\\Model\\Code\\06Fully_distributed\\")
import sys
sys.path.append("C:\\Users\\Mostafa\\Desktop\\My Files\\thesis\\My Thesis\\Data_and_Models\\Model\\Code\\python_functions")
# precipitation data
datapath="C:\\Users\\Mostafa\\Desktop\\My Files\\thesis\\My Thesis\\Data_and_Models\\Data\\05new_model\\outputs\\4km\\"
#datapath="C:\\Users\\Mostafa\\Desktop\\My Files\\thesis\\My Thesis\\Data_and_Models\\Data\\05new_model\\3km\\"
# DEM
#dempath="C:\\Users\\Mostafa\\Desktop\\My Files\\thesis\\My Thesis\\Data_and_Models\\Data\\05new_model\\inputs\\DEM\\dem_500.tif"
dempath="C:\\Users\\Mostafa\\Desktop\\My Files\\thesis\\My Thesis\\Data_and_Models\\Data\\05new_model\\new\\dem_4km.tif"
#dempath="C:\\Users\\Mostafa\\Desktop\\My Files\\thesis\\My Thesis\\Data_and_Models\\Data\\05new_model\\3km\\dem3km.tif"
# calibration data
calibopath="C:\\Users\\Mostafa\\Desktop\\My Files\\thesis\\My Thesis\\Data_and_Models\\Data\\03semi_distributed(2)\\matlab\\calibration\\"
#%%library
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as spio
import gdal

from pyOpt import Optimization, ALHSO,Optimizer
from pyevolve import GSimpleGA, Consts, GAllele,Crossovers #,Interaction
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import DBAdapters
from pyevolve import Initializators, Mutators

#import pandas as pd
import datetime as dt
#from time import ctime

# functions
from alldataold import alldata
from wrapper import calib_tot_distributed_new_model_structure2
from HBV96d_edited import _get_mask
from save_dict import load_obj
#from par3d import par3d_lumpedK1_newmodel2
#import HBV_explicit
#import HBV96d_edited_trials as HBV96d
#from muskingum import muskingum_routing
#from flow_direction import flow_direction
#%% lake subcatchment (load data)
totaldata,_, p2,curve=alldata(typee='hourly')
data=spio.loadmat(calibopath+'vlake.mat')
totaldata['plake']=data['vlake'][:,0]
del totaldata['p'], data

#s='2012-06-14 19:00:00'
#e='2013-12-23 00:00:00'
#e2='2014-11-17 00:00:00'

s=dt.datetime(2012,06,14,19,00,00)
e=dt.datetime(2013,12,23,00,00,00)
e2=dt.datetime(2014,11,17,00,00,00)

calib=totaldata.loc[s:e]
#valid=totaldata.loc[e:e2]

#curve=np.loadtxt('01txt\\curve.txt',dtype=np.float64)
# read initial conditions
calibration_array=calib.as_matrix()
#calibration_array=calibration_array.astype(np.float32)
#validation_array=valid.as_matrix()
#validation_array=validation_array.astype(np.float32)
# distributed data
# calibration
sp_prec_c=np.load(datapath+'sp_prec_c.npy')
#sp_prec_c=sp_prec_c.astype(np.float32)
sp_et_c=np.load(datapath+'sp_et_c.npy')
#sp_et_c=sp_et_c.astype(np.float32)
sp_temp_c=np.load(datapath+'sp_temp_c.npy')
#sp_temp_c=sp_temp_c.astype(np.float32)


flow_acc_table=load_obj(datapath+"flow_acc_table")
flow_acc=np.load(datapath+'flow_acc.npy')
#flow_acc=flow_acc.astype(np.float16)
#lakecell=[3,1]

#lakecell=[10,4] # 1km
#lakecell=[19,10] # 500m
lakecell=[2,1] # 4km

#no_cells=np.size(flow_direct[:,:,0])-np.count_nonzero(np.isnan(flow_direct[:,:,0]))

# validation
#sp_prec_v=np.load(datapath+'sp_prec_v.npy')
#sp_et_v=np.load(datapath+'sp_et_v.npy')
#sp_temp_v=np.load(datapath+'sp_temp_v.npy')

DEM = gdal.Open(dempath)
shape_base_dem = DEM.ReadAsArray().shape
elev, no_val=_get_mask(DEM)
elev[elev==no_val]=np.nan
no_cells=np.size(elev[:,:])-np.count_nonzero(np.isnan(elev[:,:]))
#elev=np.array(elev,dtype='float32')
#no_val=np.float32(no_val)

#%%
#path="02optimization_results\\GA\\2km\\"

jiboa_initial=np.loadtxt('01txt\\Initia-jiboa.txt',usecols=0).tolist()
lake_initial=np.loadtxt('01txt\\Initia-lake.txt',usecols=0).tolist()
LB=np.loadtxt('01txt\\constrained_muskingum\\LB-4km.txt',usecols=0).tolist()#[:9]
UB=np.loadtxt('01txt\\constrained_muskingum\\UB-4km.txt',usecols=0).tolist()#[:9]
# 
klb=0.5
kub=1.5

#%% Genatic Algorithm
run_GA=False

# Find negative element
def eval_func(genome):
#    print(genome)
    _,_, RMSEE ,_, _, _=calib_tot_distributed_new_model_structure2(calibration_array,
                             p2,curve,lakecell,DEM,flow_acc_table,flow_acc,sp_prec_c,sp_et_c,
                             sp_temp_c, genome,kub,klb,jiboa_initial=jiboa_initial,
                             lake_initial=lake_initial,ll_temp=None, q_0=None)
#    _,_, RMSEE ,_, _, _=calib_tot_distributed_new_model(calibration_array,
#                             p2,curve,lakecell,DEM,flow_acc_table,flow_acc,sp_prec_c,sp_et_c,
#                             sp_temp_c, genome,jiboa_initial=jiboa_initial,
#                             lake_initial=lake_initial,ll_temp=None, q_0=None)
    print(RMSEE)
    return RMSEE

def Grid_constructor(LB=LB,UB=UB):
    alleles = GAllele.GAlleles()
    for i in range(len(UB)):
        alleles.add(GAllele.GAlleleRange(LB[i],UB[i],real=True))
#        alleles.add(round(uniform(LB[i],UB[i]),int(decimals[i])))
#        GAllele.GAlleleRange.getRandomAllele.
#    a=GAllele.GAlleleList([round(uniform(LB[i],UB[i]),int(decimals[i])) for i in range(len(decimals))])
#    a=[round(uniform(LB[i],UB[i]),int(decimals[i])) for i in range(len(decimals))]
#    print(a)
#    alleles.add(a)
    return alleles

def run_main():
    # Genome instance
    genome = G1DList.G1DList(len(LB))
    genome.setParams(allele=Grid_constructor() ,bestrawscore=0.0000, rounddecimal=4) #rangemin=-6.0, rangemax=6.0
#    Consts.CDefRangeMax
#    genome.setParams.
    # Change the initializator to Real values
    genome.initializator.set(Initializators.G1DListInitializatorAllele)
#    genome.initializator.set(Initializators.G1DBinaryStringInitializator)

    # Change the mutator to Gaussian Mutator
    genome.mutator.set(Mutators.G1DListMutatorAllele)
#    genome.mutator.set(Mutators.G1DBinaryStringMutatorSwap)
    
#    genome.crossover.set(Crossovers.G1DListCrossoverSinglePoint)
#    genome.crossover.set(Crossovers.G1DBinaryStringXTwoPoint)
#    genome.crossover.set(Crossovers.G1DListCrossoverCutCrossfill)
    genome.crossover.set(Crossovers.G1DListCrossoverSinglePoint)

    # The evaluator function (objective function)
    genome.evaluator.set(eval_func)

    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setMinimax(Consts.minimaxType["minimize"])
    ga.selector.set(Selectors.GRouletteWheel)
    
    ga.setGenerations(200)
    ga.setPopulationSize(200)
    ga.setMutationRate(0.6)
    ga.setCrossoverRate(0.85)
#    ga.setMultiProcessing(True)
    # stopping criteria
    ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
    ga.terminationCriteria.set(GSimpleGA.FitnessStatsCriteria)
    
    sqlite_adapter = DBAdapters.DBSQLite(identify="ex1", resetDB=True)
    ga.setDBAdapter(sqlite_adapter)
    
    # Do the evolution
    ga.evolve(freq_stats=1)

    # Best individual
    print ga.bestIndividual()
    print ga.currentGeneration
#if __name__ == "__main__":
if run_GA== True:
    run_main()
#%% harmony_search
harmony_search=1

if harmony_search==1:
    par=np.random.uniform(LB, UB)
    print('Calibration starts')
    def opt_fun(par):
        try:
#            print("start")
            _,_, RMSEE ,_, _, _=calib_tot_distributed_new_model_structure2(calibration_array,
                             p2,curve,lakecell,DEM,flow_acc_table,flow_acc,sp_prec_c,sp_et_c,
                             sp_temp_c, par,kub,klb,jiboa_initial=jiboa_initial,
                             lake_initial=lake_initial,ll_temp=None, q_0=None)
#            f1=open("allparameters.txt","a+")
#            f1.write(str(par)+'\n')
#            f1.close()
#            RMSEE=1
#            print("end")
            print(RMSEE)
            print(par)
            fail = 0
        except:
            RMSEE = np.nan
            fail = 1
        return RMSEE, [], fail
    
    opt_prob = Optimization('HBV Calibration', opt_fun)
    for i in xrange(len(LB)):# [:10]
        opt_prob.addVar('x{0}'.format(i), type='c', lower=LB[i], upper=UB[i])
    
    print(opt_prob)
    
    opt_engine = ALHSO(etol=0.0001,atol=0.0001,rtol=0.0001, stopiters=10, hmcr=0.5,par=0.5,filename='mostafa.out')
#    opt_engine.__solve__(opt_engine,store_sol=True,disp_opts=False,store_hst=True,hot_start=True,filename="parameters.txt") #'hotstart.txt'
    
    Optimizer.__init__(opt_engine,def_options={
                    'hms':[int,9],					# Memory Size [1,50]
                		'hmcr':[float,0.95],			# Probability rate of choosing from memory [0.7,0.99]
                		'par':[float,0.99],				# Pitch adjustment rate [0.1,0.99]
                		'dbw':[int,2000],				# Variable Bandwidth Quantization
                		'maxoutiter':[int,2e3],			# Maximum Number of Outer Loop Iterations (Major Iterations)
                		'maxinniter':[int,2e2],			# Maximum Number of Inner Loop Iterations (Minor Iterations)
                		'stopcriteria':[int,1],			# Stopping Criteria Flag
                		'stopiters':[int,20],			# Consecutively Number of Outer Iterations for which the Stopping Criteria must be Satisfied
                		'etol':[float,0.0001],			# Absolute Tolerance for Equality constraints
                		'itol':[float,0.0001],			# Absolute Tolerance for Inequality constraints 
                		'atol':[float,0.0001],			# Absolute Tolerance for Objective Function 1e-6
                		'rtol':[float,0.0001],			# Relative Tolerance for Objective Function
                		'prtoutiter':[int,0],			# Number of Iterations Before Print Outer Loop Information
                		'prtinniter':[int,0],			# Number of Iterations Before Print Inner Loop Information
                		'xinit':[int,0],				# Initial Position Flag (0 - no position, 1 - position given)
                		'rinit':[float,1.0],			# Initial Penalty Factor
                		'fileout':[int,1],				# Flag to Turn On Output to filename
                		'filename':[str,'parameters.txt'],	# We could probably remove fileout flag if filename or fileinstance is given
                		'seed':[float,0.5],				# Random Number Seed (0 - Auto-Seed based on time clock)
                		'scaling':[int,1],				# Design Variables Scaling Flag (0 - no scaling, 1 - scaling between [-1,1]) 
                		})
    
#    opt_engine.__solve__(opt_prob,store_sol=True,display_opts=True ,store_hst=True,hot_start=True,filename="parameters.txt") #'hotstart.txt'
    res = opt_engine(opt_prob)
#    opt_engine.__init__(etol=0.001,atol=0.001,rtol=0.001)    

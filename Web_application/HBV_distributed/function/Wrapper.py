# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:17:54 2018

@author: Mostafa
"""
#%links


#%library
import numpy as np


# functions
import DHBV_functions
import Conceptual_HBV
import Dist_HBV
import Routing
import Performance_criteria


def Dist_model(data,p2,curve,lakecell,DEM,flow_acc,flow_acc_plan,sp_prec,sp_et,sp_temp, sp_pars,
                    kub,klb,jiboa_initial,lake_initial,ll_temp=None, q_0=None):
#    p=data[:,0]
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
#    Qobs=data[:,4]
    
    
    jiboa_par,lake_par=DHBV_functions.par2d_lumpedK1(sp_pars,DEM,12,13,kub,klb)
    
    
    # lake simulation
    q_lake, _ = Conceptual_HBV.simulate(plake,t,et,lake_par , p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    qlake_r=Routing.muskingum(q_lake,q_lake[0],lake_par[11],lake_par[12],p2[0])
    
    # Jiboa subcatchment
    q_tot, st , q_uz_routed, q_lz,_= Dist_HBV.Dist_HBV2(lakecell,qlake_r,DEM ,flow_acc,flow_acc_plan,sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=jiboa_par, p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_tot=q_tot[:-1]
    
#    RMSEE=Performance_criteria.rmse(Qobs,q_tot)
    
    return q_tot,q_lake,q_uz_routed, q_lz
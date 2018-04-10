#libraries
import numpy as np

#functions
from par3d import par3d_lumpedK1_newmodel
import HBV_explicit
import HBV96_distributed as HBV96d

#jiboa_initial=np.loadtxt('Initia-jiboa.txt',usecols=0).tolist()
#lake_initial=np.loadtxt('Initia-lake.txt',usecols=0).tolist()

def calib_tot_distributed_new_model_structure2(data,p2,curve,lakecell,DEM,flow_acc,flow_acc_plan,sp_prec,sp_et,sp_temp, sp_pars,
                    kub,klb,jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
#    p=data[:,0]
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
    Qobs=data[:,4]
    
    # parameters
    jiboa_par,lake_par=par3d_lumpedK1_newmodel(sp_pars,DEM,12,13,kub,klb)
        
    # lake simulation
    q_lake, _ = HBV_explicit.simulate(plake,t,et,lake_par , p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    qlake_r=HBV96d.muskingum_routing(q_lake,q_lake[0],lake_par[11],lake_par[12],p2[0])
    
    # Jiboa subcatchment
    q_tot, st , q_uz_routed, q_lz, q_uz= HBV96d.distributed_new_model_struture2(lakecell,qlake_r,DEM ,flow_acc,flow_acc_plan,sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=jiboa_par, p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_tot=q_tot[:-1]
    
    RMSEE=HBV_explicit.rmse(Qobs,q_tot)
    
    return q_tot,q_lake, RMSEE ,q_uz_routed, q_lz, q_uz

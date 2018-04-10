import HBV_explicit
import HBV_explicit_newstructure
import numpy as np
import HBV96d_edited as HBV96d
from wrmse import rmseHF,rmseLF
#from muskingum import muskingum_routing
from par3d import par3d_lumpedK1
from par3d import par3d_lumpedK1_newmodel

jiboa_initial=np.loadtxt('Initia-jiboa.txt',usecols=0).tolist()
lake_initial=np.loadtxt('Initia-lake.txt',usecols=0).tolist()


def run_semidistributed_oldstructure(data,p2,parameters,curve,q_0,warmup,jiboa_initial=jiboa_initial,
                    lake_initial=lake_initial,nsec=False):
    """
    ====
    run(data,p2,parameters,curve)
    ====
    to run the the semi-distributed model
    
    inputs:
        1- data : array with 6 columns
            1- jiboa subcatchment precipitation p
            2- Evapotranspiration et
            3- Temperature t
            4- long term average temperature tm
            5- lake subcatchment precipitation plake
            6- observed discharge Q
        2- p2 : list of 
            1- tfac : time conversion factor
            2- area of jiboa subcatchment
            3- area of lake subcatchment
            4- area of the lake
        3- parameters:
            list of 22 parameter
        4- curve:
            array of the lake storage-discharge curve
        5- initial discharge:
            the discharge at the begining of the simulation 
        6- nse : (optional)
            if you want to calculate the Nash sutcliff error make it True
    Output:
        1- q_tr : 
            - calculated and routed flow
        2- rmse : 
            - root mean square error
        3- nse  : 
            - nash sutcliff error
        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
            1- q_lake : 
               - lake subcatchment flow
            2- st_lake: 
                - lake subcatchment state variables
            3- q_jiboa: 
                - jiboa subcatchment flow
            4- st_jiboa :
                - jiboa subcatchment state variables
    """
    import HBV_explicitold

    p=data[:,0]
    et=data[:,1]
    t=data[:,2]
    tm=data[:,3]
    plake=data[:,4]
    Qobs=data[:,5]
    
    # read the initial state variables 
    
    
    # lake simulation
    q_lake, st_lake = HBV_explicitold.simulate(plake,t,et, parameters[9:19], p2,
                                           curve,q_0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # jiboa catchment simulation
    q_jiboa, st_jiboa = HBV_explicitold.simulate(p,t,et, parameters[:9],p2,
                                            curve,q_0,init_st=jiboa_initial,ll_temp=tm,lake_sim=False)
    # sum both discharges
    q_sim=[q_jiboa[i]+q_lake[i] for i in range(len(q_jiboa))]
    # routing 
    q_tr = HBV_explicitold.RoutingMAXBAS(np.array(q_sim),parameters[-1])
#    q_tr1=q_tr[:-1]
    q_tr1=q_tr
    rmse = HBV_explicitold.rmse(Qobs,q_tr1[warmup:])
    if nsec:
        nse=HBV_explicitold.nse(Qobs,q_tr1[warmup:])
    else:
        nse='please enter True on the nse input'
    
    return q_tr1[warmup:],rmse,nse,[q_lake,st_lake,q_jiboa,st_jiboa]


def run_semidistributed_newstructure(data,p2,parameters,curve,q_0,warmup,jiboa_initial=jiboa_initial,
                    lake_initial=lake_initial,nsec=False):
    """
    ====
    run(data,p2,parameters,curve)
    ====
    to run the the semi-distributed model
    
    inputs:
        1- data : array with 6 columns
            1- jiboa subcatchment precipitation p
            2- Evapotranspiration et
            3- Temperature t
            4- long term average temperature tm
            5- lake subcatchment precipitation plake
            6- observed discharge Q
        2- p2 : list of 
            1- tfac : time conversion factor
            2- area of jiboa subcatchment
            3- area of lake subcatchment
            4- area of the lake
        3- parameters:
            list of 22 parameter
        4- curve:
            array of the lake storage-discharge curve
        5- initial discharge:
            the discharge at the begining of the simulation 
        6- nse : (optional)
            if you want to calculate the Nash sutcliff error make it True
    Output:
        1- q_tr : 
            - calculated and routed flow
        2- rmse : 
            - root mean square error
        3- nse  : 
            - nash sutcliff error
        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
            1- q_lake : 
               - lake subcatchment flow
            2- st_lake: 
                - lake subcatchment state variables
            3- q_jiboa: 
                - jiboa subcatchment flow
            4- st_jiboa :
                - jiboa subcatchment state variables
    """
    import HBV_explicit_newstructure
    
    
    p=data[:,0]
    et=data[:,1]
    t=data[:,2]
    tm=data[:,3]
    plake=data[:,4]
    Qobs=data[:,5]
    
    # read the initial state variables 
    
    
    # lake simulation
    q_lake, st_lake = HBV_explicit.simulate(plake,t,et, parameters[11:22], p2,
                                           curve,q_0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    
    q_lake =HBV_explicit._routing(np.array(q_lake), parameters[22])
    
    # jiboa catchment simulation
    q_jiboa, st_jiboa = HBV_explicit_newstructure.simulate(p,t,et, parameters[:10],p2,
                                            curve,q_0,init_st=jiboa_initial,ll_temp=tm,lake_sim=False)
    
    q_jiboa =HBV_explicit_newstructure._routing(np.array(q_jiboa), parameters[10])
        
    # sum both discharges
    q_sim=[q_jiboa[i]+q_lake[i] for i in range(len(q_jiboa))]
    
    rmse = HBV_explicit_newstructure.rmse(Qobs,q_sim[warmup:])
    if nsec:
        nse=HBV_explicit_newstructure.nse(Qobs,q_sim[warmup:])
    else:
        nse='please enter True on the nse input'
    
    return q_sim[warmup:],rmse,nse,[q_lake,st_lake,q_jiboa,st_jiboa]

"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""

def calib_distributed(data,p2,curve,flp,sp_prec,sp_et,sp_temp, sp_pars,
                    jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
    """
    ====
    run(data,p2,parameters,curve)
    ====
    to run the the semi-distributed model
    
    inputs:
        1- data : array with 6 columns
            1- jiboa subcatchment precipitation p
            2- Evapotranspiration et
            3- Temperature t
            4- long term average temperature tm
            5- lake subcatchment precipitation plake
            6- observed discharge Q
        2- p2 : list of 
            1- tfac : time conversion factor
            2- area of jiboa subcatchment
            3- area of lake subcatchment
            4- area of the lake
        3- parameters:
            list of 22 parameter
        4- curve:
            array of the lake storage-discharge curve
        5- initial discharge:
            the discharge at the begining of the simulation 
        6- nse : (optional)
            if you want to calculate the Nash sutcliff error make it True
    Output:
        1- q_tr : 
            - calculated and routed flow
        2- rmse : 
            - root mean square error
        3- nse  : 
            - nash sutcliff error
        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
            1- q_lake : 
               - lake subcatchment flow
            2- st_lake: 
                - lake subcatchment state variables
            3- q_jiboa: 
                - jiboa subcatchment flow
            4- st_jiboa :
                - jiboa subcatchment state variables
    """
#    p=data[:,0]
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
    Qobs=data[:,4]
    
    # lake simulation 
    q_lake, _ = HBV_explicit.simulate(plake,t,et, sp_pars[11:-1], p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    q_lake =HBV_explicit._routing(np.array(q_lake), sp_pars[-1])
    # Jiboa subcatchment
    q_jiboa, st, q_uz,q_lz1 = HBV96d.distributed(flp=flp, sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=sp_pars[:11], p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_jiboa=q_jiboa[:-1]
    q_sim=q_jiboa+np.array(q_lake)
    
    RMSEE=HBV_explicit.rmse(Qobs,q_sim)
    
    return RMSEE #,rmse,nse,[q_lake,st_lake,q_jiboa,st_jiboa]

"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def run_distributed(data,p2,curve,flp,sp_prec,sp_et,sp_temp, sp_pars,
                    jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
    """
    ====
    run(data,p2,parameters,curve)
    ====
    to run the the semi-distributed model
    
    inputs:
        1- data : array with 6 columns
            1- jiboa subcatchment precipitation p
            2- Evapotranspiration et
            3- Temperature t
            4- long term average temperature tm
            5- lake subcatchment precipitation plake
            6- observed discharge Q
        2- p2 : list of 
            1- tfac : time conversion factor
            2- area of jiboa subcatchment
            3- area of lake subcatchment
            4- area of the lake
        3- parameters:
            list of 22 parameter
        4- curve:
            array of the lake storage-discharge curve
        5- initial discharge:
            the discharge at the begining of the simulation 
        6- nse : (optional)
            if you want to calculate the Nash sutcliff error make it True
    Output:
        1- q_tr : 
            - calculated and routed flow
        2- rmse : 
            - root mean square error
        3- nse  : 
            - nash sutcliff error
        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
            1- q_lake : 
               - lake subcatchment flow
            2- st_lake: 
                - lake subcatchment state variables
            3- q_jiboa: 
                - jiboa subcatchment flow
            4- st_jiboa :
                - jiboa subcatchment state variables
    """
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
    Qobs=data[:,4]
    
    # lake simulation 
    q_lake, _ = HBV_explicit.simulate(plake,t,et, sp_pars[11:-1], p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    q_lake =HBV_explicit._routing(np.array(q_lake), sp_pars[-1])
    # Jiboa subcatchment
    
    q_jiboa, st, q_uz, q_lz1 = HBV96d.distributed(flp=flp, sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=sp_pars[:11], p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_jiboa=q_jiboa[:-1]
    q_sim=q_jiboa+np.array(q_lake)
    
#    RMSEE=HBV_explicit.rmse(Qobs,q_sim)
#    NSEE=HBV_explicit.nse(Qobs,q_sim)
    
    return q_sim,q_lake,q_uz,q_lz1 #,rmse,nse,[q_lake,st_lake,q_jiboa,st_jiboa]
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""

#def run_distributed_detailed(data,p2,curve,flp,sp_prec,sp_et,sp_temp, sp_pars,
#                    jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
#    """
#    ====
#    run(data,p2,parameters,curve)
#    ====
#    to run the the semi-distributed model
#    
#    inputs:
#        1- data : array with 6 columns
#            1- jiboa subcatchment precipitation p
#            2- Evapotranspiration et
#            3- Temperature t
#            4- long term average temperature tm
#            5- lake subcatchment precipitation plake
#            6- observed discharge Q
#        2- p2 : list of 
#            1- tfac : time conversion factor
#            2- area of jiboa subcatchment
#            3- area of lake subcatchment
#            4- area of the lake
#        3- parameters:
#            list of 22 parameter
#        4- curve:
#            array of the lake storage-discharge curve
#        5- initial discharge:
#            the discharge at the begining of the simulation 
#        6- nse : (optional)
#            if you want to calculate the Nash sutcliff error make it True
#    Output:
#        1- q_tr : 
#            - calculated and routed flow
#        2- rmse : 
#            - root mean square error
#        3- nse  : 
#            - nash sutcliff error
#        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
#            1- q_lake : 
#               - lake subcatchment flow
#            2- st_lake: 
#                - lake subcatchment state variables
#            3- q_jiboa: 
#                - jiboa subcatchment flow
#            4- st_jiboa :
#                - jiboa subcatchment state variables
#    """
#    et=data[:,0]
#    t=data[:,1]
#    tm=data[:,2]
#    plake=data[:,3]
#    Qobs=data[:,4]
#    
#    # lake simulation 
#    q_lake, _ = HBV_explicit.simulate(plake,t,et, sp_pars[11:-1], p2,
#                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
#    # lake routing
#    q_lake =HBV_explicit._routing(np.array(q_lake), sp_pars[-1])
#    # Jiboa subcatchment
#    
#    q_jiboa, st = HBV96d.distributed(flp=flp, sp_prec=sp_prec, sp_et=sp_et, 
#                           sp_temp=sp_temp, sp_pars=sp_pars[:11], p2=p2, 
#                           init_st=jiboa_initial, ll_temp=None, q_0=None)
#    
#    q_jiboa=q_jiboa[:-1]
#    q_sim=q_jiboa+np.array(q_lake)
#    
#    RMSEE=HBV_explicit.rmse(Qobs,q_sim)
#    NSEE=HBV_explicit.nse(Qobs,q_sim)
#    
#    return q_sim,q_lake,RMSEE,NSEE #,rmse,nse,[q_lake,st_lake,q_jiboa,st_jiboa]
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""

def calib_distributed_Multi_lumped(WStype,WSN,data,p2,curve,flp,sp_prec,sp_et,sp_temp, sp_pars,
                    jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
    """
    ====
    run(data,p2,parameters,curve)
    ====
    to run the the semi-distributed model
    
    inputs:
        1- data : array with 6 columns
            1- jiboa subcatchment precipitation p
            2- Evapotranspiration et
            3- Temperature t
            4- long term average temperature tm
            5- lake subcatchment precipitation plake
            6- observed discharge Q
        2- p2 : list of 
            1- tfac : time conversion factor
            2- area of jiboa subcatchment
            3- area of lake subcatchment
            4- area of the lake
        3- parameters:
            list of 22 parameter
        4- curve:
            array of the lake storage-discharge curve
        5- initial discharge:
            the discharge at the begining of the simulation 
        6- nse : (optional)
            if you want to calculate the Nash sutcliff error make it True
    Output:
        1- q_tr : 
            - calculated and routed flow
        2- rmse : 
            - root mean square error
        3- nse  : 
            - nash sutcliff error
        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
            1- q_lake : 
               - lake subcatchment flow
            2- st_lake: 
                - lake subcatchment state variables
            3- q_jiboa: 
                - jiboa subcatchment flow
            4- st_jiboa :
                - jiboa subcatchment state variables
    """

    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
    Qobs=data[:,4]
    
    # lake simulation 
    q_lake, _ = HBV_explicit.simulate(plake,t,et, sp_pars[11:-1], p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    q_lake =HBV_explicit._routing(np.array(q_lake), sp_pars[-1])
    # Jiboa subcatchment


    q_jiboa, st = HBV96d.distributed(flp=flp, sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=sp_pars[:11], p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_jiboa=q_jiboa[:-1]
    q_sim=q_jiboa+np.array(q_lake)
    

    rmsehf=rmseHF(Qobs,q_sim,WStype,WSN,0.75)
    rmself=rmseLF(Qobs,q_sim,WStype,WSN,0.75)
    
    return rmself,rmsehf #,rmse,nse,[q_lake,st_lake,q_jiboa,st_jiboa]
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
#totally distributed parameters
#def calib_tot_distributed(data,p2,curve,flp,sp_prec,sp_et,sp_temp, sp_pars,
#                    jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
#    """
#    ====
#    run(data,p2,parameters,curve)
#    ====
#    to run the the semi-distributed model
#    
#    inputs:
#        1- data : array with 6 columns
#            1- jiboa subcatchment precipitation p
#            2- Evapotranspiration et
#            3- Temperature t
#            4- long term average temperature tm
#            5- lake subcatchment precipitation plake
#            6- observed discharge Q
#        2- p2 : list of 
#            1- tfac : time conversion factor
#            2- area of jiboa subcatchment
#            3- area of lake subcatchment
#            4- area of the lake
#        3- parameters:
#            list of 22 parameter
#        4- curve:
#            array of the lake storage-discharge curve
#        5- initial discharge:
#            the discharge at the begining of the simulation 
#        6- nse : (optional)
#            if you want to calculate the Nash sutcliff error make it True
#    Output:
#        1- q_tr : 
#            - calculated and routed flow
#        2- rmse : 
#            - root mean square error
#        3- nse  : 
#            - nash sutcliff error
#        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
#            1- q_lake : 
#               - lake subcatchment flow
#            2- st_lake: 
#                - lake subcatchment state variables
#            3- q_jiboa: 
#                - jiboa subcatchment flow
#            4- st_jiboa :
#                - jiboa subcatchment state variables
#    """
##    p=data[:,0]
#    et=data[:,0]
#    t=data[:,1]
#    tm=data[:,2]
#    plake=data[:,3]
#    Qobs=data[:,4]
#    
#    # parameters
#    par_g=sp_pars
#    shape_base_dem = flp.ReadAsArray().shape
#    f=flp.ReadAsArray()
#    no_val = flp.GetRasterBand(1).GetNoDataValue() 
#    no_elem = np.sum(np.sum([[1 for elem in mask_i if elem != no_val] for mask_i in f]))
#    no_parameters=11
#    
#    # generated parameters
#    # determin which cells are not empty
#    celli=[]#np.ones((no_elem,2))
#    cellj=[]
#    for i in range(shape_base_dem[0]): # rows
#        for j in range(shape_base_dem[1]): # columns
#    #        print(f[i,j])
#            if f[i,j]!= no_val:
#                celli.append(i)
#                cellj.append(j)
#    
#    
#    par_3d=np.zeros([shape_base_dem[0], shape_base_dem[1], no_parameters])*np.nan
#    # parameters in array
#    par_arr=np.ones((no_parameters,no_elem))
#    for i in range(no_elem):
#        par_arr[:,i] = par_g[i*no_parameters:(i*no_parameters)+no_parameters]
#    
#    for i in range(no_elem):
#        par_3d[celli[i],cellj[i],:]=par_arr[:,i]
#    
#    lake_par=par_g[len(par_g)-12:]
#    jiboa_par=par_3d
#    
#    
#    # lake simulation 
#    q_lake, _ = HBV_explicit.simulate(plake,t,et,lake_par , p2,
#                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
#    # lake routing
#    q_lake =HBV_explicit._routing(np.array(q_lake), lake_par[-1])
#    # Jiboa subcatchment
#    q_jiboa, st = HBV96d.distributed_totally(flp=flp, sp_prec=sp_prec, sp_et=sp_et, 
#                           sp_temp=sp_temp, sp_pars=jiboa_par, p2=p2, 
#                           init_st=jiboa_initial, ll_temp=None, q_0=None)
#    
#    q_jiboa=q_jiboa[:-1]
#    q_sim=q_jiboa+np.array(q_lake)
#    
#    RMSEE=HBV_explicit.rmse(Qobs,q_sim)
#    
#    return RMSEE 
"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""
def run_tot_distributed(data,p2,curve,flp,sp_prec,sp_et,sp_temp, sp_pars,
                    jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
    """
    ====
    run(data,p2,parameters,curve)
    ====
    to run the the semi-distributed model
    
    inputs:
        1- data : array with 6 columns
            1- jiboa subcatchment precipitation p
            2- Evapotranspiration et
            3- Temperature t
            4- long term average temperature tm
            5- lake subcatchment precipitation plake
            6- observed discharge Q
        2- p2 : list of 
            1- tfac : time conversion factor
            2- area of jiboa subcatchment
            3- area of lake subcatchment
            4- area of the lake
        3- parameters:
            list of 22 parameter
        4- curve:
            array of the lake storage-discharge curve
        5- initial discharge:
            the discharge at the begining of the simulation 
        6- nse : (optional)
            if you want to calculate the Nash sutcliff error make it True
    Output:
        1- q_tr : 
            - calculated and routed flow
        2- rmse : 
            - root mean square error
        3- nse  : 
            - nash sutcliff error
        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
            1- q_lake : 
               - lake subcatchment flow
            2- st_lake: 
                - lake subcatchment state variables
            3- q_jiboa: 
                - jiboa subcatchment flow
            4- st_jiboa :
                - jiboa subcatchment state variables
    """
#    p=data[:,0]
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
    Qobs=data[:,4]
    
    # parameters
    jiboa_par,lake_par=par3d_lumpedK1(sp_pars,flp,11,12)
    
    
    # lake simulation 
    q_lake, _ = HBV_explicit.simulate(plake,t,et,lake_par , p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    q_lake =HBV_explicit._routing(np.array(q_lake), lake_par[-1])
    # Jiboa subcatchment
    q_jiboa, st, q_uz, q_lz1 = HBV96d.distributed_totally(flp=flp, sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=jiboa_par, p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_jiboa=q_jiboa[:-1]
    q_sim=q_jiboa+np.array(q_lake)
    
    RMSEE=HBV_explicit.rmse(Qobs,q_sim)
#    NSEE=HBV_explicit.nse(Qobs,q_sim)
    
    return q_sim,q_lake,RMSEE, q_uz, q_lz1

"""
#_____________________________________________________________________________________________
#_____________________________________________________________________________________________
"""

def calib_tot_distributed_new_model_unconstraintK(data,p2,curve,lakecell,DEM,flow_acc,flow_acc_plan,sp_prec,sp_et,sp_temp, sp_pars,
                    jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
#    p=data[:,0]
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
    Qobs=data[:,4]
    
    # parameters
#    par_g=sp_pars
    jiboa_par,lake_par=par3d_lumpedK1(sp_pars,DEM,12,13)
    
    # lake simulation
    q_lake, _ = HBV_explicit.simulate(plake,t,et,lake_par , p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    qlake_r=HBV96d.muskingum_routing(q_lake,q_lake[0],lake_par[11],lake_par[12],p2[0])
    
    # Jiboa subcatchment
    q_tot, st , q_uz_routed, q_lz, q_uz= HBV96d.distributed_new_model(lakecell,qlake_r,DEM ,flow_acc,flow_acc_plan,sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=jiboa_par, p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_tot=q_tot[:-1]
    
    RMSEE=HBV_explicit.rmse(Qobs,q_tot)
    
    return q_tot,q_lake, RMSEE ,q_uz_routed, q_lz, q_uz

#def calib_tot_distributed_new_model1(data,p2,curve,lakecell,DEM,flow_acc,flow_acc_plan,sp_prec,sp_et,sp_temp, sp_pars,
#                    jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
##    p=data[:,0]
#    et=data[:,0]
#    t=data[:,1]
#    tm=data[:,2]
#    plake=data[:,3]
#    Qobs=data[:,4]
#    
#    # parameters
##    par_g=sp_pars
#    jiboa_par,lake_par=par3d_lumpedK1(sp_pars,DEM,12,13)
#    
#    shape_base_dem = DEM.ReadAsArray().shape
#    cond=np.ones((shape_base_dem[0],shape_base_dem[1]))*np.nan
#    
#    dem=DEM.ReadAsArray()
#    no_val = np.float32(DEM.GetRasterBand(1).GetNoDataValue())
#    
#    for x in range(shape_base_dem[0]): # no of rows
#        for y in range(shape_base_dem[1]): # no of columns  
#            if dem [x, y] != no_val :
#                if (0.5*1/jiboa_par[x,y,10]) >=jiboa_par[x,y,11] and (0.5*1/jiboa_par[x,y,10]) <=1-jiboa_par[x,y,11]:
#                    cond[x,y]=0    #"no dispersion"
#                else:
#                    cond[x,y]=1 #" there is dispersion"
#                    
#    if np.nansum(cond) == 0 : #don't calculate just put RMSE high number
#        # lake simulation
#        q_lake, _ = HBV_explicit.simulate(plake,t,et,lake_par , p2,
#                                       curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
#        # lake routing
#        qlake_r=HBV96d.muskingum_routing(q_lake,q_lake[0],lake_par[11],lake_par[12],p2[0])
#        
#        # Jiboa subcatchment
#        q_tot, st , q_uz_routed, q_lz, q_uz= HBV96d.distributed_new_model(lakecell,qlake_r,DEM ,flow_acc,flow_acc_plan,sp_prec=sp_prec, sp_et=sp_et, 
#                               sp_temp=sp_temp, sp_pars=jiboa_par, p2=p2, 
#                               init_st=jiboa_initial, ll_temp=None, q_0=None)
#        
#        q_tot=q_tot[:-1]
#        
#        RMSEE=HBV_explicit.rmse(Qobs,q_tot)
#    else :
#        RMSEE=np.nansum(cond)*3
#        q_tot=0
#        q_lake=0
#        q_uz_routed=0
#        q_lz=0
#        q_uz=0
#        
#    return q_tot,q_lake, RMSEE ,q_uz_routed, q_lz, q_uz
#%%
def calib_tot_distributed_new_model_constraintk(data,p2,curve,lakecell,DEM,flow_acc,flow_acc_plan,sp_prec,sp_et,sp_temp, sp_pars,
                    kub,klb,jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
#    p=data[:,0]
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
    Qobs=data[:,4]
    
    # parameters
#    par_g=sp_pars
    
    jiboa_par,lake_par=par3d_lumpedK1_newmodel(sp_pars,DEM,12,13,kub,klb)
    
    
    # lake simulation
    q_lake, _ = HBV_explicit.simulate(plake,t,et,lake_par , p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    qlake_r=HBV96d.muskingum_routing(q_lake,q_lake[0],lake_par[11],lake_par[12],p2[0])
    
    # Jiboa subcatchment
    q_tot, st , q_uz_routed, q_lz, q_uz= HBV96d.distributed_new_model(lakecell,qlake_r,DEM ,flow_acc,flow_acc_plan,sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=jiboa_par, p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_tot=q_tot[:-1]
    
    RMSEE=HBV_explicit.rmse(Qobs,q_tot)
    
    return q_tot,q_lake, RMSEE ,q_uz_routed, q_lz, q_uz

def calib_tot_distributed_new_model_structure2(data,p2,curve,lakecell,DEM,flow_acc,flow_acc_plan,sp_prec,sp_et,sp_temp, sp_pars,
                    kub,klb,jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
#    p=data[:,0]
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
    Qobs=data[:,4]
    
    # parameters
#    par_g=sp_pars
    
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

def calib_tot_distributed_new_model_structure2_withoutlake(data,p2,curve,lakecell,DEM,flow_acc,flow_acc_plan,sp_prec,sp_et,sp_temp, sp_pars,
                    kub,klb,jiboa_initial=jiboa_initial,lake_initial=lake_initial,ll_temp=None, q_0=None):
#    p=data[:,0]
    et=data[:,0]
    t=data[:,1]
    tm=data[:,2]
    plake=data[:,3]
    Qobs=data[:,4]
    
    # parameters
#    par_g=sp_pars
    
    jiboa_par,lake_par=par3d_lumpedK1_newmodel(sp_pars,DEM,12,13,kub,klb)
    
    
    # lake simulation
    q_lake, _ = HBV_explicit.simulate(plake,t,et,lake_par , p2,
                                   curve,0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # lake routing
    qlake_r=HBV96d.muskingum_routing(q_lake,q_lake[0],lake_par[11],lake_par[12],p2[0])
    
    # Jiboa subcatchment
    q_tot, st , q_uz_routed, q_lz, q_uz= HBV96d.distributed_new_model_struture2_withoutlake(lakecell,qlake_r,DEM ,flow_acc,flow_acc_plan,sp_prec=sp_prec, sp_et=sp_et, 
                           sp_temp=sp_temp, sp_pars=jiboa_par, p2=p2, 
                           init_st=jiboa_initial, ll_temp=None, q_0=None)
    
    q_tot=q_tot[:-1]
    
    RMSEE=HBV_explicit.rmse(Qobs,q_tot)
    
    return q_tot,q_lake, RMSEE ,q_uz_routed, q_lz, q_uz
#%%
"""
_______________________________________________________________________________
"""
def soil_moisture_run_oldstructure(data,jiboa_index,lake_index,p2,parameters,curve,q_0,warmup,jiboa_initial=jiboa_initial,
                    lake_initial=lake_initial,nsec=False):
    """
    ====
    run(data,p2,parameters,curve)
    ====
    to run the the semi-distributed model
    
    inputs:
        1- data : array with 6 columns
            1- jiboa subcatchment precipitation p
            2- Evapotranspiration et
            3- Temperature t
            4- long term average temperature tm
            5- lake subcatchment precipitation plake
            6- observed discharge Q
        2- p2 : list of 
            1- tfac : time conversion factor
            2- area of jiboa subcatchment
            3- area of lake subcatchment
            4- area of the lake
        3- parameters:
            list of 22 parameter
        4- curve:
            array of the lake storage-discharge curve
        5- initial discharge:
            the discharge at the begining of the simulation 
        6- nse : (optional)
            if you want to calculate the Nash sutcliff error make it True
    Output:
        1- q_tr : 
            - calculated and routed flow
        2- rmse : 
            - root mean square error
        3- nse  : 
            - nash sutcliff error
        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
            1- q_lake : 
               - lake subcatchment flow
            2- st_lake: 
                - lake subcatchment state variables
            3- q_jiboa: 
                - jiboa subcatchment flow
            4- st_jiboa :
                - jiboa subcatchment state variables
    """
    import HBV_explicitold

    p=data[:,0]
    et=data[:,1]
    t=data[:,2]
    tm=data[:,3]
    plake=data[:,4]
#    Qobs=data[:,5]
    
    # read the initial state variables 
    
    
    # lake simulation
    q_lake, st_lake = HBV_explicitold.simulate(plake,t,et, parameters[9:19], p2,
                                           curve,q_0,init_st=lake_initial,ll_temp=tm,lake_sim=True)
    # jiboa catchment simulation
    q_jiboa, st_jiboa = HBV_explicitold.simulate(p,t,et, parameters[:9],p2,
                                            curve,q_0,init_st=jiboa_initial,ll_temp=tm,lake_sim=False)
    # multiply by soil moisture index
    q_lake=np.array(q_lake)*lake_index
    q_jiboa=np.array(q_jiboa)*jiboa_index
    # sum both discharges
    q_sim=[q_jiboa[i]+q_lake[i] for i in range(len(q_jiboa))]
    # routing 
    q_tr = HBV_explicitold.RoutingMAXBAS(np.array(q_sim),parameters[-1])
#    q_tr1=q_tr[:-1]
    q_tr1=q_tr
#    rmse = HBV_explicit.rmse(Qobs,q_tr1[warmup:])
#    if nsec:
#        nse=HBV_explicit.nse(Qobs,q_tr1[warmup:])
#    else:
#        nse='please enter True on the nse input'
    
#    return q_tr1[warmup:],rmse,nse,[q_lake,st_lake,q_jiboa,st_jiboa]
    return q_tr1[warmup:], q_lake, q_jiboa



def soil_moisture_run_newstructure(data,jiboa_index,lake_index,p2,parameters,curve,q_0,warmup,jiboa_initial=jiboa_initial,
                    lake_initial=lake_initial,nsec=False):
    """
    ====
    run(data,p2,parameters,curve)
    ====
    to run the the semi-distributed model
    
    inputs:
        1- data : array with 6 columns
            1- jiboa subcatchment precipitation p
            2- Evapotranspiration et
            3- Temperature t
            4- long term average temperature tm
            5- lake subcatchment precipitation plake
            6- observed discharge Q
        2- p2 : list of 
            1- tfac : time conversion factor
            2- area of jiboa subcatchment
            3- area of lake subcatchment
            4- area of the lake
        3- parameters:
            list of 22 parameter
        4- curve:
            array of the lake storage-discharge curve
        5- initial discharge:
            the discharge at the begining of the simulation 
        6- nse : (optional)
            if you want to calculate the Nash sutcliff error make it True
    Output:
        1- q_tr : 
            - calculated and routed flow
        2- rmse : 
            - root mean square error
        3- nse  : 
            - nash sutcliff error
        4- list of [q_lake,st_lake,q_jiboa,st_jiboa] 
            1- q_lake : 
               - lake subcatchment flow
            2- st_lake: 
                - lake subcatchment state variables
            3- q_jiboa: 
                - jiboa subcatchment flow
            4- st_jiboa :
                - jiboa subcatchment state variables
    """
    p=data[:,0]
    et=data[:,1]
    t=data[:,2]
    tm=data[:,3]
    plake=data[:,4]
#    Qobs=data[:,5]
    
    # read the initial state variables 
    
    
    # lake simulation
    q_lake, st_lake = HBV_explicit.simulate(plake,t,et, parameters[11:22], p2,
                                           curve,q_0,init_st=lake_initial,ll_temp=tm,lake_sim=True)

    # jiboa catchment simulation
    q_jiboa, st_jiboa = HBV_explicit_newstructure.simulate(p,t,et, parameters[:10],p2,
                                            curve,q_0,init_st=jiboa_initial,ll_temp=tm,lake_sim=False)
    # multiply by soil moisture index
    q_lake=np.array(q_lake)*lake_index
    q_jiboa=np.array(q_jiboa)*jiboa_index

    # routing 
    q_lake =HBV_explicit._routing(np.array(q_lake), parameters[22])
    q_jiboa =HBV_explicit_newstructure._routing(np.array(q_jiboa), parameters[10])
    
    # sum both discharges
    q_sim=[q_jiboa[i]+q_lake[i] for i in range(len(q_jiboa))]
        
    return q_sim, q_lake, q_jiboa
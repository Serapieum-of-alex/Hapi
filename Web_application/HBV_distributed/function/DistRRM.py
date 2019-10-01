# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 19:17:15 2018

@author: Mostafa
"""
#library
import numpy as np
import gdal
from types import ModuleType

# functions
import GISpy
import HBV
import Routing
import GISCatchment as GC

def Dist_HBV2(lakecell,q_lake,DEM,flow_acc,flow_acc_plan, sp_prec, sp_et, sp_temp, sp_pars, p2, init_st=None, 
                ll_temp=None, q_0=None):
    '''
    Make spatially distributed HBV in the SM and UZ
    interacting cells 
    '''
    
    n_steps = sp_prec.shape[2] + 1 # no of time steps =length of time series +1
    # intiialise vector of nans to fill states
    dummy_states = np.empty([n_steps, 5]) # [sp,sm,uz,lz,wc]
    dummy_states[:] = np.nan
    
    # Get the mask
    mask, no_val = GISpy.get_mask(DEM)
    x_ext, y_ext = mask.shape # shape of the fpl raster (rows, columns)-------------- rows are x and columns are y
    #    y_ext, x_ext = mask.shape # shape of the fpl raster (rows, columns)------------ should change rows are y and columns are x
    
    # Get deltas of pixel
    geo_trans = DEM.GetGeoTransform() # get the coordinates of the top left corner and cell size [x,dx,y,dy]
    dx = np.abs(geo_trans[1])/1000.0  # dx in Km
    dy = np.abs(geo_trans[-1])/1000.0  # dy in Km
    px_area = dx*dy  # area of the cell
    
    # Enumerate the total number of pixels in the catchment
    tot_elem = np.sum(np.sum([[1 for elem in mask_i if elem != no_val] for mask_i in mask])) # get row by row and search [mask_i for mask_i in mask]
    
    # total pixel area
    px_tot_area = tot_elem*px_area # total area of pixels 
    
    # Get number of non-value data
    
    st = []  # Spatially distributed states
    q_lz = []
    q_uz = []
    #------------------------------------------------------------------------------
    for x in range(x_ext): # no of rows
        st_i = []
        q_lzi = []
        q_uzi = []
    #        q_out_i = []
    # run all cells in one row ----------------------------------------------------
        for y in range(y_ext): # no of columns
            if mask [x, y] != no_val:  # only for cells in the domain
                # Calculate the states per cell
                # TODO optimise for multiprocessing these loops   
                _, _st, _uzg, _lzg = HBV.simulate_new_model(avg_prec = sp_prec[x, y,:], 
                                              temp = sp_temp[x, y,:], 
                                              et = sp_et[x, y,:], 
                                              par = sp_pars[x, y, :], 
                                              p2 = p2, 
                                              init_st = init_st, 
                                              ll_temp = None, 
                                              q_0 = q_0,
                                              extra_out = True)
    #               # append column after column in the same row -----------------
                st_i.append(np.array(_st))
                #calculate upper zone Q = K1*(LZ_int_1)
                q_lz_temp=np.array(sp_pars[x, y, 6])*_lzg
                q_lzi.append(q_lz_temp)
                # calculate lower zone Q = k*(UZ_int_3)**(1+alpha)
                q_uz_temp = np.array(sp_pars[x, y, 5])*(np.power(_uzg, (1.0 + sp_pars[x, y, 7])))
                q_uzi.append(q_uz_temp)
                
    #                print("total = "+str(fff)+"/"+str(tot_elem)+" cell, row= "+str(x+1)+" column= "+str(y+1) )
            else: # if the cell is novalue-------------------------------------
                # Fill the empty cells with a nan vector
                st_i.append(dummy_states) # fill all states(5 states) for all time steps = nan
                q_lzi.append(dummy_states[:,0]) # q lower zone =nan  for all time steps = nan
                q_uzi.append(dummy_states[:,0]) # q upper zone =nan  for all time steps = nan
    # store row by row-------- ---------------------------------------------------- 
    #        st.append(st_i) # state variables 
        st.append(st_i) # state variables 
        q_lz.append(np.array(q_lzi)) # lower zone discharge mm/timestep
        q_uz.append(np.array(q_uzi)) # upper zone routed discharge mm/timestep
    #------------------------------------------------------------------------------            
    # convert to arrays 
    st = np.array(st)
    q_lz = np.array(q_lz)
    q_uz = np.array(q_uz)
    #%% convert quz from mm/time step to m3/sec
    area_coef=p2[1]/px_tot_area
    q_uz=q_uz*px_area*area_coef/(p2[0]*3.6)
    
    no_cells=list(set([flow_acc_plan[i,j] for i in range(x_ext) for j in range(y_ext) if not np.isnan(flow_acc_plan[i,j])]))
#    no_cells=list(set([int(flow_acc_plan[i,j]) for i in range(x_ext) for j in range(y_ext) if flow_acc_plan[i,j] != no_val]))
    no_cells.sort()

    #%% routing lake discharge with DS cell k & x and adding to cell Q
    q_lake=Routing.muskingum(q_lake,q_lake[0],sp_pars[lakecell[0],lakecell[1],10],sp_pars[lakecell[0],lakecell[1],11],p2[0])
    q_lake=np.append(q_lake,q_lake[-1])
    # both lake & Quz are in m3/s
    #new
    q_uz[lakecell[0],lakecell[1],:]=q_uz[lakecell[0],lakecell[1],:]+q_lake
    #%% cells at the divider
    q_uz_routed=np.zeros_like(q_uz)*np.nan
    # for all cell with 0 flow acc put the q_uz
    for x in range(x_ext): # no of rows
        for y in range(y_ext): # no of columns
            if mask [x, y] != no_val and flow_acc_plan[x, y]==0: 
                q_uz_routed[x,y,:]=q_uz[x,y,:]        
    #%% new
    for j in range(1,len(no_cells)): #2):#
        for x in range(x_ext): # no of rows
            for y in range(y_ext): # no of columns
                    # check from total flow accumulation 
                    if mask [x, y] != no_val and flow_acc_plan[x, y]==no_cells[j]:
#                        print(no_cells[j])
                        q_r=np.zeros(n_steps)
                        for i in range(len(flow_acc[str(x)+","+str(y)])): #  no_cells[j]
                            # bring the indexes of the us cell
                            x_ind=flow_acc[str(x)+","+str(y)][i][0]
                            y_ind=flow_acc[str(x)+","+str(y)][i][1]
                            # sum the Q of the US cells (already routed for its cell)
                             # route first with there own k & xthen sum
                            q_r=q_r+Routing.muskingum(q_uz_routed[x_ind,y_ind,:],q_uz_routed[x_ind,y_ind,0],sp_pars[x_ind,y_ind,10],sp_pars[x_ind,y_ind,11],p2[0]) 
#                        q=q_r
                         # add the routed upstream flows to the current Quz in the cell
                        q_uz_routed[x,y,:]=q_uz[x,y,:]+q_r
    #%% check if the max flow _acc is at the outlet
#    if tot_elem != np.nanmax(flow_acc_plan):
#        raise ("flow accumulation plan is not correct")
    # outlet is the cell that has the max flow_acc
    outlet=np.where(flow_acc_plan==np.nanmax(flow_acc_plan)) #np.nanmax(flow_acc_plan)
    outletx=outlet[0][0]
    outlety=outlet[1][0]              
    #%%
    q_lz = np.array([np.nanmean(q_lz[:,:,i]) for i in range(n_steps)]) # average of all cells (not routed mm/timestep)
    # convert Qlz to m3/sec 
    q_lz = q_lz* p2[1]/ (p2[0]*3.6) # generation
    
    q_out = q_lz + q_uz_routed[outletx,outlety,:]    

    return q_out, st, q_uz_routed, q_lz, q_uz


def RunLumpedRRP(ConceptualModel,Raster, sp_prec, sp_et, sp_temp, sp_pars, p2, snow, init_st=None, 
                ll_temp=None, q_0=None):
    """
    ========================================================================
      RunLumpedRRP(Raster,sp_prec,sp_et,sp_temp,sp_pars,p2,init_st,ll_temp,q_0)
    ========================================================================
    
    this function runs the rainfall runoff lumped model (HBV, GR4,...) separately 
    for each cell and return a time series of arrays 
    
    Inputs:
    ----------
        1-ConceptualModel:
            [function] conceptual model function 
        2-Raster:
            [gdal.dataset] raster to get the spatial information (nodata cells)
            raster input could be dem, flow accumulation or flow direction raster of the catchment
            but the nodata value stored in the raster should be far from the
            range of values that could result from the calculation
        3-sp_prec:
            [numpy array] 3d array of the precipitation data, sp_prec should
            have the same 2d dimension of raster input
        4-sp_et:
            [numpy array] 3d array of the evapotranspiration data, sp_et should
            have the same 2d dimension of raster input
        5-sp_temp:
            [numpy array] 3d array of the temperature data, sp_temp should
            have the same 2d dimension of raster input
        6-sp_pars:
            [numpy array] number of 2d arrays of the catchment properties spatially 
            distributed in 2d and the third dimension is the number of parameters,
            sp_pars should have the same 2d dimension of raster input
        7-p2:
            [List] list of unoptimized parameters
            p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
            p2[1] = catchment area in km2
        8-init_st:
            [list] initial state variables values [sp, sm, uz, lz, wc]. default=None
        9-ll_temp:
            [numpy array] 3d array of the long term average temperature data
        10-q_0:
            [float] initial discharge m3/s
    Outputs:
    ----------
        1-st:
            [numpy ndarray] 4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
        2-q_lz:
            [numpy ndarray] 3D array of the lower zone discharge  
        3-q_uz:
            [numpy ndarray] 3D array of the upper zone discharge
    Example:
    ----------
        ### meteorological data
        prec=GIS.ReadRastersFolder(PrecPath)
        evap=GIS.ReadRastersFolder(Evap_Path)
        temp=GIS.ReadRastersFolder(TempPath)
        sp_pars=GIS.ReadRastersFolder(parPath)
    
        #### GIS data
        dem= gdal.Open(DemPath) 
        
        p2=[1, 227.31]
        init_st=[0,5,5,5,0]
        
        st, q_lz, q_uz = DistRRM.RunLumpedRRP(DEM,sp_prec=sp_prec, sp_et=sp_et, 
                               sp_temp=sp_temp, sp_pars=sp_par, p2=p2, 
                               init_st=init_st)
    """
    ### input data validation
    # data type
    assert isinstance(ConceptualModel,ModuleType) , "ConceptualModel should be a module or a python file contains functions "
    assert type(Raster)==gdal.Dataset, "Raster should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(sp_prec)==np.ndarray, "array should be of type numpy array"
    assert type(sp_et)==np.ndarray, "array should be of type numpy array"
    assert type(sp_temp)==np.ndarray, "array should be of type numpy array"
    assert type(sp_pars)==np.ndarray, "array should be of type numpy array"
    assert type(p2)==list, "p2 should be of type list"
    
    if init_st != None:
        assert type(init_st)==list, "init_st should be of type list"
    if ll_temp != None:
        assert type(ll_temp)==np.ndarray, "init_st should be of type list"
    if q_0 != None:
        assert type(q_0)==float, "init_st should be of type list"
    
    # input dimensions
    [rows,cols]=Raster.ReadAsArray().shape
    assert np.shape(sp_prec)[0] == rows and np.shape(sp_et)[0] == rows and np.shape(sp_temp)[0] == rows and np.shape(sp_pars)[0] == rows, "all input data should have the same number of rows"
    assert np.shape(sp_prec)[1] == cols and np.shape(sp_et)[1] == cols and np.shape(sp_temp)[1] == cols and np.shape(sp_pars)[1] == cols, "all input data should have the same number of columns"
    assert np.shape(sp_prec)[2] == np.shape(sp_et)[2] and np.shape(sp_temp)[2] == np.shape(sp_prec)[2], "all meteorological input data should have the same length"
        
    
    n_steps = sp_prec.shape[2] + 1 # no of time steps =length of time series +1
    # intiialise vector of nans to fill states
    dummy_states = np.empty([n_steps, 5]) # [sp,sm,uz,lz,wc]
    dummy_states[:] = np.nan
    dummy_states=np.float32(dummy_states)
    
    # Get the mask
    no_val = np.float32(Raster.GetRasterBand(1).GetNoDataValue())
    raster = Raster.ReadAsArray()
    
    # calculate area covered by cells
    geo_trans = Raster.GetGeoTransform() # get the coordinates of the top left corner and cell size [x,dx,y,dy]
    dx = np.abs(geo_trans[1])/1000.0  # dx in Km
    dy = np.abs(geo_trans[-1])/1000.0  # dy in Km
    px_area = dx*dy  # area of the cell
    no_cells=np.size(raster[:,:])-np.count_nonzero(raster[raster==no_val])
    px_tot_area = no_cells*px_area # total area of pixels 
    
    st = []  # Spatially distributed states
    q_lz = []
    q_uz = []
    
    for x in range(rows): # no of rows
        st_row = []
        q_lz_row = []
        q_uz_row = []
        
        for y in range(cols): # no of columns
            if raster [x, y] != no_val:  # only for cells in the domain
                # Calculate the states per cell
                # TODO optimise for multiprocessing these loops
                try:
                    
#                    _st, _uzg, _lzg = ConceptualModel.Simulate(prec = sp_prec[x, y,:], 
                    uzg, lzg,  stvar  = ConceptualModel.Simulate(prec = sp_prec[x, y,:], 
                                                                 temp = sp_temp[x, y,:], 
                                                                 et = sp_et[x, y,:], 
                                                                 par = sp_pars[x, y, :], 
                                                                 p2 = p2, 
                                                                 init_st = init_st, 
                                                                 ll_temp = None, 
                                                                 q_0 = q_0, 
                                                                 snow=0) #extra_out = True
                except:
                    print("conceptual model argument are not correct")
                
                # append column after column in the same row
#                st_i.append(np.array(_st))
                st_row.append(stvar)
                
                # calculate upper zone Q = k*(UZ_int_3)**(1+alpha)
#                q_uz_temp = np.array(sp_pars[x, y, 5])*(np.power(_uzg, (1.0 + sp_pars[x, y, 7])))
#                q_uzi.append(q_uz_temp)
                q_uz_row.append(uzg)
                
                #calculate lower zone Q = K1*(LZ_int_1)
#                q_lz_temp=np.array(sp_pars[x, y, 6])*_lzg
#                q_lzi.append(q_lz_temp)
                q_lz_row.append(lzg)
                
    #                print("total = "+str(fff)+"/"+str(tot_elem)+" cell, row= "+str(x+1)+" column= "+str(y+1) )
            else: # if the cell is novalue-------------------------------------
                # Fill the empty cells with a nan vector
                st_row.append(dummy_states) # fill all states(5 states) for all time steps = nan
                q_lz_row.append(dummy_states[:,0]) # q lower zone =nan  for all time steps = nan
                q_uz_row.append(dummy_states[:,0]) # q upper zone =nan  for all time steps = nan
    # store row by row-------- ---------------------------------------------------- 
    #        st.append(st_i) # state variables 
        st.append(st_row) # state variables 
        q_lz.append(q_lz_row) # lower zone discharge mm/timestep
        q_uz.append(q_uz_row) # upper zone routed discharge mm/timestep
    #------------------------------------------------------------------------------            
    # convert to arrays 
    st = np.array(st)
    q_lz = np.array(q_lz)
    q_uz = np.array(q_uz)
    # convert quz from mm/time step to m3/sec
    area_coef=p2[1]/px_tot_area
    q_uz=q_uz*px_area*area_coef/(p2[0]*3.6)
    
#    # convert QLZ to 1D time series 
#    q_lz = np.array([np.nanmean(q_lz[:,:,i]) for i in range(n_steps)]) # average of all cells (not routed mm/timestep)
#    # convert Qlz to m3/sec 
#    q_lz = q_lz* p2[1]/ (p2[0]*3.6) # generation
    
    q_lz=q_lz*px_area*area_coef/(p2[0]*3.6)
    
    # convert all to float32 to save storage
    q_lz = np.float32(q_lz)
    q_uz = np.float32(q_uz)
    st = np.float32(st)
    return st, q_lz, q_uz

def SpatialRouting(q_lz, q_uz,flow_acc,flow_direct,sp_pars,p2):
    """
    ====================================================================
      SpatialRouting(q_lz,q_uz,flow_acc,flow_direct,sp_pars,p2)
    ====================================================================
    
    
    Inputs:
    ----------
        1-q_lz:
            [numpy ndarray] 3D array of the lower zone discharge  
        2-q_uz:
            [numpy ndarray] 3D array of the upper zone discharge  
        3-flow_acc:
            [gdal.dataset] flow accumulation raster file of the catchment (clipped to the catchment only)
        4-flow_direct:
            [gdal.dataset] flow Direction raster file of the catchment (clipped to the catchment only)
        5-sp_pars:
            [numpy ndarray] 3D array of the parameters
        6-p2:
            [List] list of unoptimized parameters
            p2[0] = tfac, 1 for hourly, 0.25 for 15 min time step and 24 for daily time step
            p2[1] = catchment area in km2
    
    Outputs:
    ----------
        1-q_out:
            [numpy array] 1D timeseries of discharge at the outlet og the catchment
            of unit m3/sec
        2-q_uz_routed:
            [numpy ndarray] 3D array of the upper zone discharge  accumulated and 
            routed at each time step
        3-q_lz:
            [numpy ndarray] 3D array of the lower zone discharge translated at each time step
    
    Example:
    ----------
        #### GIS data
        dem= gdal.Open(DemPath) 
        flow_acc=gdal.Open(FlowAccPath)
        flow_direct=gdal.Open(FlowDPath)
        sp_pars=GIS.ReadRastersFolder(parPath)
        p2=[1, 227.31]
        prec=GIS.ReadRastersFolder(PrecPath)
        evap=GIS.ReadRastersFolder(Evap_Path)
        temp=GIS.ReadRastersFolder(TempPath)
    
        init_st=[0,5,5,5,0]
        
        st, q_lz, q_uz = DistRRM.RunLumpedRRP(DEM,sp_prec=sp_prec, sp_et=sp_et, 
                               sp_temp=sp_temp, sp_pars=sp_par, p2=p2, 
                               init_st=init_st)
        q_out, q_uz_routed, q_lz_trans = DistRRM.SpatialRouting(q_lz, q_uz,
                                                                flow_acc,flow_direct,sp_par,p2)
    """
    
    n_steps = np.shape(q_lz)[2] #len(q_lz)
    
    rows=flow_acc.RasterYSize
    cols=flow_acc.RasterXSize
    FAA=flow_acc.ReadAsArray()
    no_val=np.float32(flow_acc.GetRasterBand(1).GetNoDataValue())
#    no_cells=list(set([flow_acc[i,j] for i in range(rows) for j in range(cols) if not np.isnan(flow_acc[i,j])]))
    no_cells=list(set([int(FAA[i,j]) for i in range(rows) for j in range(cols) if FAA[i,j] != no_val]))
    no_cells.sort()
    
    # create the flow direction table 
    FDT=GC.FlowDirecTable(flow_direct)

#    #%% routing lake discharge with DS cell k & x and adding to cell Q
#    q_lake=Routing.muskingum(q_lake,q_lake[0],sp_pars[lakecell[0],lakecell[1],10],sp_pars[lakecell[0],lakecell[1],11],p2[0])
#    q_lake=np.append(q_lake,q_lake[-1])
#    # both lake & Quz are in m3/s
#    #new
#    q_uz[lakecell[0],lakecell[1],:]=q_uz[lakecell[0],lakecell[1],:]+q_lake
    
    ### cells at the divider
    q_uz_routed=np.zeros_like(q_uz)*np.nan
    
    """
    lower zone discharge is going to be just translated without any attenuation
    in order to be able to calculate total discharge (uz+lz) at internal points 
    in the catchment
    """
    q_lz_translated=np.zeros_like(q_uz)*np.nan
    
    # for all cell with 0 flow acc put the q_uz
    for x in range(rows): # no of rows
        for y in range(cols): # no of columns
            if FAA [x, y] != no_val and FAA [x, y]==0: 
                q_uz_routed[x,y,:]=q_uz[x,y,:]
                q_lz_translated[x,y,:]=q_lz[x,y,:]
    
    ### remaining cells
    for j in range(1,len(no_cells)): #2):#
        for x in range(rows): # no of rows
            for y in range(cols): # no of columns
                    # check from total flow accumulation 
                    if FAA [x, y] != no_val and FAA[x, y]==no_cells[j]:
                        # for UZ
                        q_r=np.zeros(n_steps)
                        # for lz
                        q=np.zeros(n_steps)
                        # iterate to route uz and translate lz
                        for i in range(len(FDT[str(x)+","+str(y)])): #  no_cells[j]
                            # bring the indexes of the us cell
                            x_ind=FDT[str(x)+","+str(y)][i][0]
                            y_ind=FDT[str(x)+","+str(y)][i][1]
                            # sum the Q of the US cells (already routed for its cell)
                            # route first with there own k & xthen sum
                            q_r=q_r+Routing.muskingum(q_uz_routed[x_ind,y_ind,:],q_uz_routed[x_ind,y_ind,0],sp_pars[x_ind,y_ind,10],sp_pars[x_ind,y_ind,11],p2[0]) 
                            q=q+q_lz_translated[x_ind,y_ind,:]
                            
                        # add the routed upstream flows to the current Quz in the cell
                        q_uz_routed[x,y,:]=q_uz[x,y,:]+q_r
                        q_lz_translated[x,y,:]=q_lz[x,y,:]+q

    # outlet is the cell that has the max flow_acc
    outlet=np.where(FAA==np.nanmax(FAA[FAA != no_val])) 
    outletx=outlet[0][0]
    outlety=outlet[1][0]              
    
    q_out = q_lz_translated[outletx,outlety,:] + q_uz_routed[outletx,outlety,:]

    return q_out, q_uz_routed, q_lz_translated
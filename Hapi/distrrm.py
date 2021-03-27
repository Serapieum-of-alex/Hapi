"""
Created on Wed Jun 27 19:17:15 2018

@author: Mostafa
"""

import numpy as np
from Hapi.raster import Raster as raster
from Hapi.routing import Routing as routing

class DistributedRRM():

    def __init__(self):
        pass

    @staticmethod
    def RunLumpedRRM(Model, ll_temp=None, q_init=None):
        """
        ========================================================================
          RunLumpedRRM(Raster,sp_prec,sp_et,sp_temp,sp_pars,p2,init_st,ll_temp,q_init)
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
            10-q_init:
                [float] initial discharge m3/s
        Outputs:
        ----------
            1-statevariables:
                [numpy ndarray] 4D array (rows,cols,time,states) states are [sp,wc,sm,uz,lv]
            2-qlz:
                [numpy ndarray] 3D array of the lower zone discharge
            3-quz:
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

            st, qlz, quz = DistRRM.RunLumpedRRP(DEM,sp_prec=sp_prec, sp_et=sp_et,
                                   sp_temp=sp_temp, sp_pars=sp_par, p2=p2,
                                   init_st=init_st)
        """
        ### input data validation
        if ll_temp != None:
            assert type(ll_temp)==np.ndarray, "init_st should be of type list"
        if q_init != None:
            assert type(q_init)==float, "init_st should be of type list"

        Model.statevariables = np.zeros([Model.rows,Model.cols,Model.TS,5], dtype=np.float32)
        # Model.statevariables[:] = np.nan
        Model.quz = np.zeros([Model.rows,Model.cols, Model.TS], dtype=np.float32)
        # Model.q_uz[:] = np.nan
        Model.qlz = np.zeros([Model.rows,Model.cols, Model.TS], dtype=np.float32)
        # Model.qlz[:] = np.nan

        for x in range(Model.rows):
            for y in range(Model.cols):
                if Model.FlowAccArr [x, y] != Model.NoDataValue:  # only for cells in the domain
                        Model.quz[x,y,:], Model.qlz[x,y,:], Model.statevariables[x,y,:,:] = Model.LumpedModel.Simulate(prec = Model.Prec[x, y,:],
                                                                     temp = Model.Temp[x, y,:],
                                                                     et = Model.ET[x, y,:],
                                                                     par = Model.Parameters[x, y, :],
                                                                     init_st = Model.InitialCond,
                                                                     ll_temp = None,
                                                                     q_init = q_init,
                                                                     snow=Model.Snow)

        area_coef = Model.CatArea/Model.px_tot_area
        # convert quz from mm/time step to m3/sec
        Model.quz = Model.quz * Model.px_area * area_coef / (Model.Timef*3.6)
        # convert Qlz to m3/sec
        Model.qlz = Model.qlz * Model.px_area * area_coef / (Model.Timef*3.6)


    @staticmethod
    def SpatialRouting(Model):
        """
        ====================================================================
          SpatialRouting(qlz,quz,flow_acc,flow_direct,sp_pars,p2)
        ====================================================================


        Inputs:
        ----------
            1-qlz:
                [numpy ndarray] 3D array of the lower zone discharge
            2-quz:
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
            1-qout:
                [numpy array] 1D timeseries of discharge at the outlet og the catchment
                of unit m3/sec
            2-quz_routed:
                [numpy ndarray] 3D array of the upper zone discharge  accumulated and
                routed at each time step
            3-qlz:
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

            st, qlz, quz = DistRRM.RunLumpedRRP(DEM,sp_prec=sp_prec, sp_et=sp_et,
                                   sp_temp=sp_temp, sp_pars=sp_par, p2=p2,
                                   init_st=init_st)
            qout, quz_routed, q_lz_trans = DistRRM.SpatialRouting(qlz, quz,
                                                                    flow_acc,flow_direct,sp_par,p2)
        """
    #    # routing lake discharge with DS cell k & x and adding to cell Q
    #    q_lake=Routing.muskingum(q_lake,q_lake[0],sp_pars[lakecell[0],lakecell[1],10],sp_pars[lakecell[0],lakecell[1],11],p2[0])
    #    q_lake=np.append(q_lake,q_lake[-1])
    #    # both lake & Quz are in m3/s
    #    #new
    #    quz[lakecell[0],lakecell[1],:]=quz[lakecell[0],lakecell[1],:]+q_lake

        ### cells at the divider
        Model.quz_routed = np.zeros_like(Model.quz)#*np.nan
        """
        lower zone discharge is going to be just translated without any attenuation
        in order to be able to calculate total discharge (uz+lz) at internal points
        in the catchment
        """
        Model.qlz_translated = np.zeros_like(Model.quz)#*np.nan

        # for all cell with 0 flow acc put the quz
        for x in range(Model.rows): # no of rows
            for y in range(Model.cols): # no of columns
                if Model.FlowAccArr [x, y] != Model.NoDataValue and Model.FlowAccArr [x, y]==0:
                    Model.quz_routed[x,y,:]= Model.quz[x,y,:]
                    Model.qlz_translated[x,y,:]= Model.qlz[x,y,:]

        ### remaining cells
        for j in range(1,len(Model.acc_val)):
            #TODO parallelize 
            # all cells with the same acc_val can run at the same time
            for x in range(Model.rows): # no of rows
                for y in range(Model.cols): # no of columns
                        # check from total flow accumulation
                        if Model.FlowAccArr[x, y] != Model.NoDataValue and Model.FlowAccArr[x, y] == Model.acc_val[j]:
                            # for UZ
                            q_r = np.zeros(Model.TS)
                            # for lz
                            q = np.zeros(Model.TS)
                            # iterate to route uz and translate lz
                            for i in range(len(Model.FDT[str(x)+","+str(y)])): #  Model.acc_val[j]
                                # bring the indexes of the us cell
                                x_ind = Model.FDT[str(x)+","+str(y)][i][0]
                                y_ind = Model.FDT[str(x)+","+str(y)][i][1]
                                # sum the Q of the US cells (already routed for its cell)
                                # route first with there own k & xthen sum
                                q_r = q_r + routing.muskingum(Model.quz_routed[x_ind,y_ind,:],Model.quz_routed[x_ind,y_ind,0],Model.Parameters[x_ind,y_ind,10],Model.Parameters[x_ind,y_ind,11],Model.Timef)
                                q = q + Model.qlz_translated[x_ind,y_ind,:]

                            # add the routed upstream flows to the current Quz in the cell
                            Model.quz_routed[x,y,:] = Model.quz[x,y,:] + q_r
                            Model.qlz_translated[x,y,:] = Model.qlz[x,y,:] + q


        outletx = Model.Outlet[0][0]
        outlety = Model.Outlet[1][0]

        Model.qout = Model.qlz_translated[outletx,outlety,:] + Model.quz_routed[outletx,outlety,:]
        
    @staticmethod
    def DistMaxbas1(Model):
        
        Maxbas = Model.Parameters[:,:,-1]
        # Model.FPLArr[Model.FPLArr == Model.NoDataValue] = np.nan # replace novalue cells by nan

        # MaxFPL = np.nanmax(Model.FPLArr)
        # MinFPL = np.nanmin(Model.FPLArr)
    #    resize_fun = lambda x: np.round(((((x - min_dist)/(max_dist - min_dist))*(1*maxbas - 1)) + 1), 0)
        # resize_fun = lambda x: ((((x - MinFPL)/(MaxFPL - MinFPL))*(1*MAXBAS - 1)) + 1)

        # NormalizedFPL = resize_fun(Model.FPLArr)

        for x in range(Model.rows):
            for y in range(Model.cols):
                if Model.FlowAccArr [x, y] != Model.NoDataValue:
                    Model.quz[x,y,:] = routing.TriangularRouting2(Model.quz[x,y,:], Maxbas[x,y])

    @staticmethod
    def DistMaxbas2(Model):
        
        MAXBAS = np.nanmax(Model.Parameters[:,:,-1])
        Model.FPLArr[Model.FPLArr == Model.NoDataValue] = np.nan # replace novalue cells by nan

        MaxFPL = np.nanmax(Model.FPLArr)
        MinFPL = np.nanmin(Model.FPLArr)
    #    resize_fun = lambda x: np.round(((((x - min_dist)/(max_dist - min_dist))*(1*maxbas - 1)) + 1), 0)
        resize_fun = lambda x: ((((x - MinFPL)/(MaxFPL - MinFPL))*(1*MAXBAS - 1)) + 1)

        NormalizedFPL = resize_fun(Model.FPLArr)

        for x in range(Model.rows):
            for y in range(Model.cols):
                if not np.isnan(Model.FPLArr[x,y]):# FPLArray[x,y] != np.nan: #NoDataValue:
                    Model.quz[x,y,:] = routing.TriangularRouting(Model.quz[x,y,:], NormalizedFPL[x,y])



    @staticmethod
    def Dist_HBV2(ConceptualModel,lakecell,q_lake,DEM,flow_acc,flow_acc_plan, sp_prec, sp_et, sp_temp, sp_pars, p2, init_st=None,
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
        mask, no_val = raster.get_mask(DEM)
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
        qlz = []
        quz = []
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
    #                _, _st, _uzg, _lzg = ConceptualModel.simulate_new_model(avg_prec = sp_prec[x, y,:],
                    _, _st, _uzg, _lzg = ConceptualModel.Simulate(prec = sp_prec[x, y,:],
                                                  temp = sp_temp[x, y,:],
                                                  et = sp_et[x, y,:],
                                                  par = sp_pars[x, y, :],
                                                  p2 = p2,
                                                  init_st = init_st,
                                                  ll_temp = None,
                                                  q_0 = q_0,
                                                  snow=0) #extra_out = True
                    # append column after column in the same row -----------------
                    st_i.append(np.array(_st))
                    #calculate upper zone Q = K1*(LZ_int_1)
                    q_lz_temp=np.array(sp_pars[x, y, 6])*_lzg
                    q_lzi.append(q_lz_temp)
                    # calculate lower zone Q = k*(UZ_int_3)**(1+alpha)
                    q_uz_temp = np.array(sp_pars[x, y, 5])*(np.power(_uzg, (1.0 + sp_pars[x, y, 7])))
                    q_uzi.append(q_uz_temp)

                    #print("total = "+str(fff)+"/"+str(tot_elem)+" cell, row= "+str(x+1)+" column= "+str(y+1) )
                else: # if the cell is novalue-------------------------------------
                    # Fill the empty cells with a nan vector
                    st_i.append(dummy_states) # fill all states(5 states) for all time steps = nan
                    q_lzi.append(dummy_states[:,0]) # q lower zone =nan  for all time steps = nan
                    q_uzi.append(dummy_states[:,0]) # q upper zone =nan  for all time steps = nan

            # store row by row-------- ----------------------------------------------------
            #st.append(st_i) # state variables
            st.append(st_i) # state variables
            qlz.append(np.array(q_lzi)) # lower zone discharge mm/timestep
            quz.append(np.array(q_uzi)) # upper zone routed discharge mm/timestep
            #------------------------------------------------------------------------------
            # convert to arrays
        st = np.array(st)
        qlz = np.array(qlz)
        quz = np.array(quz)
        #%% convert quz from mm/time step to m3/sec
        area_coef=p2[1]/px_tot_area
        quz=quz*px_area*area_coef/(p2[0]*3.6)

        no_cells=list(set([flow_acc_plan[i,j] for i in range(x_ext) for j in range(y_ext) if not np.isnan(flow_acc_plan[i,j])]))
    #    no_cells=list(set([int(flow_acc_plan[i,j]) for i in range(x_ext) for j in range(y_ext) if flow_acc_plan[i,j] != no_val]))
        no_cells.sort()

        #%% routing lake discharge with DS cell k & x and adding to cell Q
        q_lake = routing.muskingum(q_lake,q_lake[0],sp_pars[lakecell[0],lakecell[1],10],sp_pars[lakecell[0],lakecell[1],11],p2[0])
        q_lake=np.append(q_lake,q_lake[-1])
        # both lake & Quz are in m3/s
        #new
        quz[lakecell[0],lakecell[1],:]=quz[lakecell[0],lakecell[1],:]+q_lake
        #%% cells at the divider
        quz_routed=np.zeros_like(quz)*np.nan
        # for all cell with 0 flow acc put the quz
        for x in range(x_ext): # no of rows
            for y in range(y_ext): # no of columns
                if mask [x, y] != no_val and flow_acc_plan[x, y]==0:
                    quz_routed[x,y,:]=quz[x,y,:]
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
                                q_r = q_r + routing.muskingum(quz_routed[x_ind,y_ind,:],quz_routed[x_ind,y_ind,0],sp_pars[x_ind,y_ind,10],sp_pars[x_ind,y_ind,11],p2[0])
    #                        q=q_r
                             # add the routed upstream flows to the current Quz in the cell
                            quz_routed[x,y,:]=quz[x,y,:]+q_r
        #%% check if the max flow _acc is at the outlet
    #    if tot_elem != np.nanmax(flow_acc_plan):
    #        raise ("flow accumulation plan is not correct")
        # outlet is the cell that has the max flow_acc
        outlet=np.where(flow_acc_plan==np.nanmax(flow_acc_plan)) #np.nanmax(flow_acc_plan)
        outletx=outlet[0][0]
        outlety=outlet[1][0]
        #%%
        qlz = np.array([np.nanmean(qlz[:,:,i]) for i in range(n_steps)]) # average of all cells (not routed mm/timestep)
        # convert Qlz to m3/sec
        qlz = qlz* p2[1]/ (p2[0]*3.6) # generation

        qout = qlz + quz_routed[outletx,outlety,:]

        return qout, st, quz_routed, qlz, quz

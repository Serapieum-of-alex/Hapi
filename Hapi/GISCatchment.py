# -*- coding: utf-8 -*-
"""
Created on Sat May 26 04:52:15 2018

@author: Mostafa
"""
# library
import numpy as np
import gdal
#from osgeo import gdalconst
#import osr

def FD_from_DEM(DEM):
    """
    ===========================================================
       FD_from_DEM(Raster)
    ===========================================================
    this function generate flow direction raster from DEM and fill sinks
    
    inputs:
        1-Raster:
            [Gdal object] DEM 
    Outputs:
        1- flow_direction_cell:
            [numpy array] with the same dimensions of the raster and 2 layers
            first layer for row index and second row for column index
        2-elev_sinkless:
            [numpy array] DEM after filling sinks
    """
#        DEM=self.DEM
    
    gt=DEM.GetGeoTransform()
    cellsize=gt[1]
    dist2=cellsize*np.sqrt(2)
    no_columns=DEM.RasterXSize
    no_rows=DEM.RasterYSize
    
    
    elev=DEM.ReadAsArray()
    # get the value stores in novalue cells
    dem_no_val = np.float32(DEM.GetRasterBand(1).GetNoDataValue())
    elev[elev==dem_no_val]=np.nan
    
    slopes=np.ones((no_rows,no_columns,9))*np.nan
    distances=[cellsize,dist2,cellsize,dist2,cellsize,dist2,cellsize,dist2]
    
    # filling sinks 
    elev_sinkless=elev
    for i in range(1,no_rows-1):
        for j in range(1,no_columns-1):
            # get elevation of surrounding cells
            f=[elev[i-1,j],elev[i-1,j-1],elev[i,j-1],elev[i+1,j-1],elev[i+1,j],elev[i+1,j+1],elev[i,j+1],elev[i-1,j+1]]
            if elev[i,j]< min(f):
                elev_sinkless[i,j]=min(f)+0.1
            
    
    flow_direction=np.ones((no_rows,no_columns))*np.nan
    
    for i in range(1,no_rows-1):
        for j in range(1,no_columns-1):
            # calculate only if cell in elev is not nan
            if not np.isnan(elev[i,j]) :
                # calculate slope
                slopes[i,j,0]=(elev_sinkless[i,j]-elev_sinkless[i,j+1])/distances[0] # slope with cell to the right
                slopes[i,j,1]=(elev_sinkless[i,j]-elev_sinkless[i-1,j+1])/distances[1]# slope with cell to the top right
                slopes[i,j,2]=(elev_sinkless[i,j]-elev_sinkless[i-1,j])/distances[2] # slope with cell to the top
                slopes[i,j,3]=(elev_sinkless[i,j]-elev_sinkless[i-1,j-1])/distances[3] # slope with cell to the top left
                slopes[i,j,4]=(elev_sinkless[i,j]-elev_sinkless[i,j-1])/distances[4] # slope with cell to the left
                slopes[i,j,5]=(elev_sinkless[i,j]-elev_sinkless[i+1,j-1])/distances[5] # slope with cell to the bottom left
                slopes[i,j,6]=(elev_sinkless[i,j]-elev_sinkless[i+1,j])/distances[6] # slope with cell to the bottom
                slopes[i,j,7]=(elev_sinkless[i,j]-elev_sinkless[i+1,j+1])/distances[7] # slope with cell to the bottom right 
                # get the flow direction index
                flow_direction[i,j]=np.where(slopes[i,j,:]==np.nanmax(slopes[i,j,:]))[0][0]
                slopes[i,j,8]=np.nanmax(slopes[i,j,:])
    
    # first row without corners
    for i in [0]: 
        for j in range(1,no_columns-1): # all columns
            if not np.isnan(elev[i,j]) :
                slopes[i,j,0]=(elev_sinkless[i,j]-elev_sinkless[i,j+1])/distances[0] # slope with cell to the right
                slopes[i,j,4]=(elev_sinkless[i,j]-elev_sinkless[i,j-1])/distances[4] # slope with cell to the left
                slopes[i,j,5]=(elev_sinkless[i,j]-elev_sinkless[i+1,j-1])/distances[5] # slope with cell to the bottom left
                slopes[i,j,6]=(elev_sinkless[i,j]-elev_sinkless[i+1,j])/distances[6] # slope with cell to the bottom
                slopes[i,j,7]=(elev_sinkless[i,j]-elev_sinkless[i+1,j+1])/distances[7] # slope with cell to the bottom right
                
                flow_direction[i,j]=np.where(slopes[i,j,:]==np.nanmax(slopes[i,j,:]))[0][0]
                slopes[i,j,8]=np.nanmax(slopes[i,j,:])
    
    # last row without corners
    for i in [no_rows-1]:
        for j in range(1,no_columns-1): # all columns
            if not np.isnan(elev[i,j]) :
                slopes[i,j,0]=(elev_sinkless[i,j]-elev_sinkless[i,j+1])/distances[0] # slope with cell to the right
                slopes[i,j,1]=(elev_sinkless[i,j]-elev_sinkless[i-1,j+1])/distances[1]# slope with cell to the top right
                slopes[i,j,2]=(elev_sinkless[i,j]-elev_sinkless[i-1,j])/distances[2] # slope with cell to the top
                slopes[i,j,3]=(elev_sinkless[i,j]-elev_sinkless[i-1,j-1])/distances[3] # slope with cell to the top left
                slopes[i,j,4]=(elev_sinkless[i,j]-elev_sinkless[i,j-1])/distances[4] # slope with cell to the left
                
                flow_direction[i,j]=np.where(slopes[i,j,:]==np.nanmax(slopes[i,j,:]))[0][0]
                slopes[i,j,8]=np.nanmax(slopes[i,j,:])
    
    # top left corner
    i=0
    j=0
    if not np.isnan(elev[i,j]) :
        slopes[i,j,0]=(elev_sinkless[i,j]-elev_sinkless[i,j+1])/distances[0] # slope with cell to the left 
        slopes[i,j,6]=(elev_sinkless[i,j]-elev_sinkless[i+1,j])/distances[6] # slope with cell to the bottom
        slopes[i,j,7]=(elev_sinkless[i,j]-elev_sinkless[i+1,j+1])/distances[7] # slope with cell to the bottom right
        
        flow_direction[i,j]=np.where(slopes[i,j,:]==np.nanmax(slopes[i,j,:]))[0][0]
        slopes[i,j,8]=np.nanmax(slopes[i,j,:])
    
    # top right corner
    i=0
    j=no_columns-1
    if not np.isnan(elev[i,j]) :
        slopes[i,j,4]=(elev_sinkless[i,j]-elev_sinkless[i,j-1])/distances[4] # slope with cell to the left
        slopes[i,j,5]=(elev_sinkless[i,j]-elev_sinkless[i+1,j-1])/distances[5] # slope with cell to the bottom left
        slopes[i,j,6]=(elev_sinkless[i,j]-elev_sinkless[i+1,j])/distances[6] # slope with cell to the bott
        
        flow_direction[i,j]=np.where(slopes[i,j,:]==np.nanmax(slopes[i,j,:]))[0][0]
        slopes[i,j,8]=np.nanmax(slopes[i,j,:])
    
    # bottom left corner 
    i=no_rows-1
    j=0
    if not np.isnan(elev[i,j]) :
        slopes[i,j,0]=(elev_sinkless[i,j]-elev_sinkless[i,j+1])/distances[0] # slope with cell to the right
        slopes[i,j,1]=(elev_sinkless[i,j]-elev_sinkless[i-1,j+1])/distances[1]# slope with cell to the top right
        slopes[i,j,2]=(elev_sinkless[i,j]-elev_sinkless[i-1,j])/distances[2] # slope with cell to the top
        
        flow_direction[i,j]=np.where(slopes[i,j,:]==np.nanmax(slopes[i,j,:]))[0][0]
        slopes[i,j,8]=np.nanmax(slopes[i,j,:])
    
    # bottom right
    i=no_rows-1
    j=no_columns-1
    if not np.isnan(elev[i,j]) :
        slopes[i,j,2]=(elev_sinkless[i,j]-elev_sinkless[i-1,j])/distances[2] # slope with cell to the top
        slopes[i,j,3]=(elev_sinkless[i,j]-elev_sinkless[i-1,j-1])/distances[3] # slope with cell to the top left
        slopes[i,j,4]=(elev_sinkless[i,j]-elev_sinkless[i,j-1])/distances[4] # slope with cell to the left
        
        flow_direction[i,j]=np.where(slopes[i,j,:]==np.nanmax(slopes[i,j,:]))[0][0]
        slopes[i,j,8]=np.nanmax(slopes[i,j,:])
    
    # first column
    for i in range(1,no_rows-1):
        for j in [0]:
            if not np.isnan(elev[i,j]) :
                slopes[i,j,0]=(elev_sinkless[i,j]-elev_sinkless[i,j+1])/distances[0] # slope with cell to the right
                slopes[i,j,1]=(elev_sinkless[i,j]-elev_sinkless[i-1,j+1])/distances[1]# slope with cell to the top right
                slopes[i,j,2]=(elev_sinkless[i,j]-elev_sinkless[i-1,j])/distances[2] # slope with cell to the top
                slopes[i,j,6]=(elev_sinkless[i,j]-elev_sinkless[i+1,j])/distances[6] # slope with cell to the bottom
                slopes[i,j,7]=(elev_sinkless[i,j]-elev_sinkless[i+1,j+1])/distances[7] # slope with cell to the bottom right 
                # get the flow direction index
                
                flow_direction[i,j]=np.where(slopes[i,j,:]==np.nanmax(slopes[i,j,:]))[0][0]
                slopes[i,j,8]=np.nanmax(slopes[i,j,:])
    
    # last column
    for i in range(1,no_rows-1):
        for j in [no_columns-1]:
            if not np.isnan(elev[i,j]) :
                slopes[i,j,2]=(elev_sinkless[i,j]-elev_sinkless[i-1,j])/distances[2] # slope with cell to the top
                slopes[i,j,3]=(elev_sinkless[i,j]-elev_sinkless[i-1,j-1])/distances[3] # slope with cell to the top left
                slopes[i,j,4]=(elev_sinkless[i,j]-elev_sinkless[i,j-1])/distances[4] # slope with cell to the left
                slopes[i,j,5]=(elev_sinkless[i,j]-elev_sinkless[i+1,j-1])/distances[5] # slope with cell to the bottom left
                slopes[i,j,6]=(elev_sinkless[i,j]-elev_sinkless[i+1,j])/distances[6] # slope with cell to the bottom
                # get the flow direction index
                
                flow_direction[i,j]=np.where(slopes[i,j,:]==np.nanmax(slopes[i,j,:]))[0][0]
                slopes[i,j,8]=np.nanmax(slopes[i,j,:])        
    #        print(str(i)+","+str(j))
    
    flow_direction_cell=np.ones((no_rows,no_columns,2))*np.nan
    #for i in range(1,no_rows-1):
    #    for j in range(1,no_columns-1):
    for i in range(no_rows):
        for j in range(no_columns):
            if flow_direction[i,j]==0:
                flow_direction_cell[i,j,0]=i  # index of the row 
                flow_direction_cell[i,j,1]=j+1 # index of the column
            elif flow_direction[i,j]==1:
                flow_direction_cell[i,j,0]=i-1
                flow_direction_cell[i,j,1]=j+1
            elif flow_direction[i,j]==2:
                flow_direction_cell[i,j,0]=i-1
                flow_direction_cell[i,j,1]=j   
            elif flow_direction[i,j]==3:
                flow_direction_cell[i,j,0]=i-1
                flow_direction_cell[i,j,1]=j-1
            elif flow_direction[i,j]==4:
                flow_direction_cell[i,j,0]=i
                flow_direction_cell[i,j,1]=j-1
            elif flow_direction[i,j]==5:
                flow_direction_cell[i,j,0]=i+1
                flow_direction_cell[i,j,1]=j-1
            elif flow_direction[i,j]==6:
                flow_direction_cell[i,j,0]=i+1
                flow_direction_cell[i,j,1]=j
            elif flow_direction[i,j]==7:
                flow_direction_cell[i,j,0]=i+1
                flow_direction_cell[i,j,1]=j+1
    
    return flow_direction_cell,elev_sinkless




#    def FD_array(self):
def FlowDirectIndex(flow_direct): 
    """
    =============================================================
      FlowDirectِِIndex(flow_direct)
    =============================================================
    this function takes flow firection raster and convert codes for the 8 directions
    (1,2,4,8,16,32,64,128) into indices of the Downstream cell
    
    inputs:
        1- flow_direct:
            [gdal.dataset] flow direction raster obtained from catchment delineation
            it only contains values [1,2,4,8,16,32,64,128]
    output:
        1-fd_indices:
            [numpy array] with the same dimensions of the raster and 2 layers
            first layer for row index and second row for column index
    Example:
        fd=gdal.Open("Flowdir.tif")
        fd_indices=FlowDirectِِIndex(fd)
    """
    # input data validation
    # data type
    assert type(flow_direct)==gdal.Dataset, "src should be read using gdal (gdal dataset please read it using gdal library) "
    
    # check flow direction input raster
    no_val=np.float32(flow_direct.GetRasterBand(1).GetNoDataValue())
    cols=flow_direct.RasterXSize
    rows=flow_direct.RasterYSize
    
    fd=flow_direct.ReadAsArray()
    fd_val=[int(fd[i,j]) for i in range(rows) for j in range(cols) if fd[i,j] != no_val]
    fd_val=list(set(fd_val))
    fd_should=[1,2,4,8,16,32,64,128]
    assert all(fd_val[i] in fd_should for i in range(len(fd_val))), "flow direction raster should contain values 1,2,4,8,16,32,64,128 only "
    
    
#    fd=np.float32(flow_direct.ReadAsArray())
    fd[fd==no_val]=np.nan
    fd_cell=np.ones((rows,cols,2))*np.nan

    for i in range(rows):
        for j in range(cols):
            if fd[i,j]==1:
                fd_cell[i,j,0]=i  # index of the row 
                fd_cell[i,j,1]=j+1 # index of the column
            elif fd[i,j]==128:
                fd_cell[i,j,0]=i-1
                fd_cell[i,j,1]=j+1
            elif fd[i,j]==64:
                fd_cell[i,j,0]=i-1
                fd_cell[i,j,1]=j   
            elif fd[i,j]==32:
                fd_cell[i,j,0]=i-1
                fd_cell[i,j,1]=j-1
            elif fd[i,j]==16:
                fd_cell[i,j,0]=i
                fd_cell[i,j,1]=j-1
            elif fd[i,j]==8:
                fd_cell[i,j,0]=i+1
                fd_cell[i,j,1]=j-1
            elif fd[i,j]==4:
                fd_cell[i,j,0]=i+1
                fd_cell[i,j,1]=j
            elif fd[i,j]==2:
                fd_cell[i,j,0]=i+1
                fd_cell[i,j,1]=j+1
    
    return fd_cell

def FlowDirecTable(flow_direct):
    """
    ====================================================================
         FlowDirecTable(flow_direct)
    ====================================================================
    this function takes flow direction indices created by FlowDirectِِIndex function 
    and create a dictionary with the cells indices as a key and  indices of directly 
    upstream cells as values (list of tuples)
    Inputs:
        1- flow_direct:
            [gdal.dataset] flow direction raster obtained from catchment delineation
            it only contains values [1,2,4,8,16,32,64,128]
    Outputs:
        1-flowAccTable:
            [Dict] dictionary with the cells indices as a key and indices of directly 
            upstream cells as values (list of tuples)
    Example:
        
    """
    # input data validation
    # validation is inside FlowDirectِِIndex 
    FDI=FlowDirectIndex(flow_direct)
    
    rows=flow_direct.RasterYSize
    cols=flow_direct.RasterXSize
    
    celli=[]
    cellj=[]
    celli_content=[]
    cellj_content=[]
    for i in range(rows): # rows
        for j in range(cols): # columns
            if not np.isnan(FDI[i,j,0]):
                # store the indexes of not empty cells and the indexes stored inside these cells
                celli.append(i)
                cellj.append(j)
                # store the index of the receiving cells
                celli_content.append(FDI[i,j,0])
                cellj_content.append(FDI[i,j,1])
                
    flow_acc_table={}
    # for each cell store the directly giving cells 
    for i in range(rows): # rows
        for j in range(cols): # columns
            if not np.isnan(FDI[i,j,0]):
                # get the indexes of the cell and use it as a key in a dictionary
                name=str(i)+','+str(j)
                flow_acc_table[name]=[]
                for k in range(len(celli_content)):
                    # search if any cell are giving this cell
                    if i==celli_content[k] and j==cellj_content[k]:
                        flow_acc_table[name].append((celli[k],cellj[k]))
    
    return flow_acc_table
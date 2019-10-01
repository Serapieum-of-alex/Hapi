# -*- coding: utf-8 -*-
"""
Created on Sat May 26 04:52:15 2018

@author: Mostafa
"""
# library
import numpy as np
#import gdal
#from osgeo import gdalconst
#import osr

class Inputs:
    
    def __init__(self,raster):
        self.raster=raster

    def FD_from_DEM(self,DEM):
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
    
    
    
    
    def FD_array(self):
        """
        =============================================================
          FD_array(FD)
        =============================================================
        this function takes flow firection codes for the 8 directions (1,2,4,8,16,32,64,128)
        and return the indeces of the upstream cells
        
        inputs:
            1- FD:
                flow direction raster
        output:
            1-flow_direction_cell:
                [numpy array] with the same dimensions of the raster and 2 layers
                first layer for row index and second row for column index
        """
        FD=self.raster
        no_val = FD.GetRasterBand(1).GetNoDataValue()
        no_columns=FD.RasterXSize
        no_rows=FD.RasterYSize
    
        flow_direction=np.float32(FD.ReadAsArray())
        flow_direction[flow_direction==no_val]=np.nan
    
        flow_direction_cell=np.ones((no_rows,no_columns,2))*np.nan
    
        for i in range(no_rows):
            for j in range(no_columns):
                if flow_direction[i,j]==1:
                    flow_direction_cell[i,j,0]=i  # index of the row 
                    flow_direction_cell[i,j,1]=j+1 # index of the column
                elif flow_direction[i,j]==128:
                    flow_direction_cell[i,j,0]=i-1
                    flow_direction_cell[i,j,1]=j+1
                elif flow_direction[i,j]==64:
                    flow_direction_cell[i,j,0]=i-1
                    flow_direction_cell[i,j,1]=j   
                elif flow_direction[i,j]==32:
                    flow_direction_cell[i,j,0]=i-1
                    flow_direction_cell[i,j,1]=j-1
                elif flow_direction[i,j]==16:
                    flow_direction_cell[i,j,0]=i
                    flow_direction_cell[i,j,1]=j-1
                elif flow_direction[i,j]==8:
                    flow_direction_cell[i,j,0]=i+1
                    flow_direction_cell[i,j,1]=j-1
                elif flow_direction[i,j]==4:
                    flow_direction_cell[i,j,0]=i+1
                    flow_direction_cell[i,j,1]=j
                elif flow_direction[i,j]==2:
                    flow_direction_cell[i,j,0]=i+1
                    flow_direction_cell[i,j,1]=j+1
        
        return flow_direction_cell

#    def FlowAccTable()
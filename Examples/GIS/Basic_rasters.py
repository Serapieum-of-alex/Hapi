import os, sys
os.chdir("F:/01Algorithms/Hydrology/HAPI/Examples")
rootpath = os.path.abspath(os.getcwd())
# sys.path.append(rootpath + "/src")
datapath = os.path.join(rootpath, "data/GIS/Hapi_GIS_Data")
datapath2 = os.path.join(rootpath, "data/GIS")
os.chdir(rootpath)

from Hapi.gis.raster import Raster
from Hapi.gis.giscatchment import GISCatchment as GC
from Hapi.visualizer import Visualize as vis
import gdal
import numpy as np
import pandas as pd
#%% Paths
RasterAPath = datapath + "/acc4000.tif"
RasterBPath = datapath + "/dem_100_f.tif"
pointsPath = datapath + "/points.csv"
#%% read the raster
src = gdal.Open(RasterAPath)
vis.PlotArray(src, Title="Flow Accumulation")
#%% GetRasterData
"""
get the basic data inside a raster (the array and the nodatavalue)

Inputs:
----------
    Input: [gdal.Dataset]
        a gdal.Dataset is a raster already been read using gdal
    band : [integer]
        the band you want to get its data. Default is 1
Outputs:
----------
    1- mask:[array]
        array with all the values in the flow path length raster
    2- no_val: [numeric]
        value stored in novalue cells
"""
arr, nodataval = Raster.GetRasterData(src)
#%%
"""GetProjectionData.

GetProjectionData returns the projection details of a given gdal.Dataset

Inputs:
-------
    1- src : [gdal.Dataset]
        raster read by gdal

Returns:
    1- epsg : [integer]
         integer reference number to the new projection (https://epsg.io/)
        (default 3857 the reference no of WGS84 web mercator )
    2- geo : [tuple]
        geotransform data (minimum lon/x, pixelsize, rotation, maximum lat/y, rotation,
                            pixelsize). The default is ''.
"""
epsg, geo = Raster.GetProjectionData(src)
print("EPSG = " + str(epsg))
print(geo)
#%% GetCoords
"""GetCoords.

Returns the coordinates of the cell centres (only the cells that
does not have nodata value)

Parameters
----------

dem : [gdal_Dataset]
    Get the data from the gdal datasetof the DEM

Returns
-------
coords : array
    Array with a list of the coordinates to be interpolated, without the Nan
mat_range : array
    Array with all the centres of cells in the domain of the DEM

"""
coords, centerscoords = Raster.GetCellCoords(src)
#%% SaveRaster
"""SaveRaster.

SaveRaster saves a raster to a path

inputs:
----------
    1- raster:
        [gdal object]
    2- path:
        [string] a path includng the name of the raster and extention like
        path="data/cropped.tif"

Outputs:
----------
    the function does not return and data but only save the raster to the hard drive

EX:
----------
    SaveRaster(raster,output_path)
"""
path = datapath + "/rasterexample.tif"
Raster.SaveRaster(src,path)
#%% CreateRaster
"""
We will recreate the raster that we have already read using the 'GetRasterData' method at the 
top from the array and the projection data we obtained using the 'GetProjectionData' method 
"""

"""CreateRaster.

CreateRaster method creates a raster from a given array and geotransform data
and save the tif file if a Path is given or it will return the gdal.Dataset

Parameters
----------
Path : [str], optional
    Path to save the Raster, if '' is given a memory raster will be returned. The default is ''.
arr : [array], optional
    numpy array. The default is ''.
geo : [list], optional
    geotransform list [minimum lon, pixelsize, rotation, maximum lat, rotation,
        pixelsize]. The default is ''.
NoDataValue : TYPE, optional
    DESCRIPTION. The default is -9999.
EPSG: [integer]
    integer reference number to the new projection (https://epsg.io/)
        (default 3857 the reference no of WGS84 web mercator )

Returns
-------
1- dst : [gdal.Dataset/save raster to drive].
            if a path is given the created raster will be saved to drive, if not
            a gdal.Dataset will be returned.
"""

src = Raster.CreateRaster(arr=arr, geo=geo, EPSG=str(epsg), NoDataValue=nodataval)
vis.PlotArray(src, Title="Flow Accumulation")
#%%
"""MapAlgebra.

MapAlgebra executes a mathematical operation on raster array and returns
the result

inputs:
----------
    1-src : [gdal.dataset]
        source raster to that you want to make some calculation on its values
    3-function:
        defined function that takes one input which is the cell value

Example :
----------
    A=gdal.Open(evap.tif)
    func=np.abs
    new_raster=MapAlgebra(A,func)
"""
def func1(val):
    if val < 20 :
        val =  1
    elif val < 40 :
        val = 2
    elif val <  60 :
        val = 3
    elif val < 80 :
        val = 4
    elif val < 100 :
        val = 5
    else:
        val = 0
    return val
dst = Raster.MapAlgebra(src, func1)
vis.PlotArray(dst, Title="Classes", ColorScale=4, TicksSpacing=1)
#%%
"""RasterFill.

RasterFill takes a raster and fill it with one value


inputs:
----------
    1- src : [gdal.dataset]
        source raster
    2- Val: [numeric]
        numeric value
    3- SaveTo : [str]
        path including the extension (.tif)

Returns:
--------
    1- raster : [saved on disk]
        the raster will be saved directly to the path you provided.
"""
path = datapath + "/fillrasterexample.tif"
Raster.RasterFill(src, 1, SaveTo=path)

"now the resulted raster is saved to disk"
dst = gdal.Open(path)
vis.PlotArray(dst, Title="Flow Accumulation")
#%%
"""ResampleRaster.

this function reproject a raster to any projection
(default the WGS84 web mercator projection, without resampling)
The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

inputs:
----------
    1- raster : [gdal.Dataset]
         gdal raster (src=gdal.Open("dem.tif"))
    3-cell_size : [integer]
         new cell size to resample the raster.
        (default empty so raster will not be resampled)
    4- resample_technique : [String]
        resampling technique default is "Nearest"
        https://gisgeography.com/raster-resampling/
        "Nearest" for nearest neighbour,"cubic" for cubic convolution,
        "bilinear" for bilinear

Outputs:
----------
    1-raster : [gdal.Dataset]
         gdal object (you can read it by ReadAsArray)
"""
print("Original Cell Size =" + str(geo[1]))
cell_size = 100
dst = Raster.ResampleRaster(src, cell_size, resample_technique="Nearest")

dst_arr,_ = Raster.GetRasterData(dst)
_, newgeo = Raster.GetProjectionData(dst)
print("New cell size is " + str(newgeo[1]))
vis.PlotArray(dst, Title="Flow Accumulation")
#%%
"""ProjectRaster.

ProjectRaster reprojects a raster to any projection
(default the WGS84 web mercator projection, without resampling)
The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

inputs:
----------
    1- raster: [gdal object]
        gdal dataset (src=gdal.Open("dem.tif"))
    2-to_epsg: [integer]
        reference number to the new projection (https://epsg.io/)
        (default 3857 the reference no of WGS84 web mercator )
    3- resample_technique: [String]
        resampling technique default is "Nearest"
        https://gisgeography.com/raster-resampling/
        "Nearest" for nearest neighbour,"cubic" for cubic convolution,
        "bilinear" for bilinear
    4- Option : [1 or 2]


Outputs:
----------
    1-raster:
        gdal dataset (you can read it by ReadAsArray)

Example :
----------
    projected_raster=project_dataset(src, to_epsg=3857)
"""
print("current EPSG - " + str(epsg))
to_epsg = 4326
dst = Raster.ProjectRaster(src, to_epsg=to_epsg, Option=1)
newepsg, newgeo = Raster.GetProjectionData(dst)
print("New EPSG - " + str(newepsg))
print("New Geotransform - " + str(newgeo))
"""Option 2"""
dst = Raster.ProjectRaster(src, to_epsg=to_epsg, Option=2)
newepsg, newgeo = Raster.GetProjectionData(dst)
print("New EPSG - " + str(newepsg))
print("New Geotransform - " + str(newgeo))
#%%
"""ReprojectDataset.

ReprojectDataset reprojects and resamples a folder of rasters to any projection
(default the WGS84 web mercator projection, without resampling)
The function returns a GDAL in-memory file object, where you can ReadAsArray etc.

inputs:
----------
    1- raster:
        gdal dataset (src=gdal.Open("dem.tif"))
    2-to_epsg:
        integer reference number to the new projection (https://epsg.io/)
        (default 3857 the reference no of WGS84 web mercator )
    3-cell_size:
        integer number to resample the raster cell size to a new cell size
        (default empty so raster will not be resampled)
    4- resample_technique:
        [String] resampling technique default is "Nearest"
        https://gisgeography.com/raster-resampling/
        "Nearest" for nearest neighbour,"cubic" for cubic convolution,
        "bilinear" for bilinear

Outputs:
----------
    1-raster:
        gdal dataset (you can read it by ReadAsArray)
"""
# to_epsg = 4326
# cell_size = 0.05
# dst = Raster.ReprojectDataset(src, to_epsg=to_epsg, cell_size=cell_size, resample_technique="Nearest")
# arr , noval = Raster.GetRasterData(dst)
# newepsg, newgeo = Raster.GetProjectionData(dst)
# print("New EPSG - " + str(newepsg))
# print("New Geotransform - " + str(newgeo))
# vis.PlotArray(dst, Title="Flow Accumulation")
#%%
"""RasterLike.

RasterLike method creates a Geotiff raster like another input raster, new raster
will have the same projection, coordinates or the top left corner of the original
raster, cell size, nodata velue, and number of rows and columns
the raster and the dem should have the same number of columns and rows

inputs:
-------
    1- src : [gdal.dataset]
        source raster to get the spatial information
    2- array:
        [numpy array]to store in the new raster
    3- path : [String]
        path to save the new raster including new raster name and extension (.tif)
    4- pixel_type : [integer]
        type of the data to be stored in the pixels,default is 1 (float32)
        for example pixel type of flow direction raster is unsigned integer
        1 for float32
        2 for float64
        3 for Unsigned integer 16
        4 for Unsigned integer 32
        5 for integer 16
        6 for integer 32

outputs:
--------
    1- save the new raster to the given path

Ex:
----------
    data=np.load("RAIN_5k.npy")
    src=gdal.Open("DEM.tif")
    name="rain.tif"
    RasterLike(src,data,name)
"""
# src2 = gdal.Open(RasterBPath)
# vis.PlotArray(src2, Title="Elevation", ColorScale=2, TicksSpacing=300)
# epsg, geo = Raster.GetProjectionData(src2)
# arr2, _ = Raster.GetRasterData(src2)
# """
# the previous rasater has a cell size of 100 m, if we want to create a raster from the previous raster
# DEM similar to the properties of the flow accumulation raster in the previous example
# """
# path = datapath + "/rasterlike.tif"
# Raster.RasterLike(src,arr2,path, )
#%%

#%% read the points
points = pd.read_csv(pointsPath)
points['row'] = np.nan
points['col'] = np.nan
points.loc[:,['row', 'col']] = GC.NearestCell(src,points[['x','y']][:]).values
#%%


# -*- coding: utf-8 -*-
"""
GISpy contains python functions to handle raster data align them together 
based on a source raster, perform any algebric operation on cell's values

@author: Mostafa
"""

#%library
import os
import numpy as np
import geopandas as gpd
#from shapely.geometry.polygon import Polygon
from shapely.geometry import Point, Polygon
from shapely.geometry.multipolygon import MultiPolygon
from fiona.crs import from_epsg
import ogr
import gdal
import osr
import pandas as pd
#import rasterio
from osgeo import gdalconst

def GetMask(raster):
    """
    =======================================================================
       get_mask(dem)
    =======================================================================
    
    to create a mask by knowing the stored value inside novalue cells 
    
    Inputs:
    ----------
        1- flow path lenth raster
    
    Outputs:
    ----------
        1- mask:array with all the values in the flow path length raster
        2- no_val: value stored in novalue cells
    """
    no_val = np.float32(raster.GetRasterBand(1).GetNoDataValue()) # get the value stores in novalue cells
    mask = raster.ReadAsArray() # read all values
    return mask, no_val


def AddMask(var, dem=None, mask=None, no_val=None):
    """
    ===================================================================
         add_mask(var, dem=None, mask=None, no_val=None)
    ===================================================================
    Put a mask in the spatially distributed values
    
    Inputs
    ----------
    var : nd_array
        Matrix with values to be masked
    cut_dem : gdal_dataset
        Instance of the gdal raster of the catchment to be cutted with. DEM 
        overrides the mask_vals and no_val
    mask_vals : nd_array
        Mask with the no_val data
    no_val : float
        value to be defined as no_val. Will mask anything is not this value
    
    Outputs
    -------
    var : nd_array
        Array with masked values 
    """
    
    if dem is not None:
        mask, no_val = GetMask(dem)
    
    # Replace the no_data value
    assert var.shape == mask.shape, 'Mask and data do not have the same shape'
    var[mask == no_val] = no_val
    
    return var


def GetTargets(dem):
    """
    ===================================================================
        get_targets(dem)
    ===================================================================
    Returns the centres of the interpolation targets
    
    Parameters
    ----------
    dem : gdal_Dataset
        Get the data from the gdal datasetof the DEM
    
    Returns
    -------
    
    coords : nd_array [nxm - nan, 2]
        Array with a list of the coordinates to be interpolated, without the Nan
    
    mat_range : nd_array [n, m]
        Array with all the centres of cells in the domain of the DEM (rectangular)
    
    """
    # Getting data for the whole grid
    x_init, xx_span, xy_span, y_init, yy_span, yx_span = dem.GetGeoTransform()
    shape_dem = dem.ReadAsArray().shape
    
    # Getting data of the mask
    no_val = dem.GetRasterBand(1).GetNoDataValue()
    mask = dem.ReadAsArray()
    
    # Adding 0.5 to get the centre
    x = np.array([x_init + xx_span*(i+0.5) for i in range(shape_dem[0])])
    y = np.array([y_init + yy_span*(i+0.5) for i in range(shape_dem[1])])
    #mat_range = np.array([[(xi, yi) for xi in x] for yi in y])
    mat_range = [[(xi, yi) for xi in x] for yi in y]
    
    # applying the mask
    coords = []
    for i in range(len(x)):
        for j in range(len(y)):
            if mask[j, i] != no_val:
                coords.append(mat_range[j][i])
                #mat_range[j, i, :] = [np.nan, np.nan]

    return np.array(coords), np.array(mat_range)



def SaveRaster(raster,path):
    """
    ===================================================================
      SaveRaster(raster,path)
    ===================================================================
    this function saves a raster to a path
    
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
    #### input data validation
    # data type
    assert type(raster)==gdal.Dataset, "src should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(path)== str, "Raster_path input should be string type"
    # input values
    ext=path[-4:]
    assert ext == ".tif", "please add the extension at the end of the path input"
    
    driver = gdal.GetDriverByName ( "GTiff" )
    dst_ds = driver.CreateCopy( path, raster, 0 )
    dst_ds = None # Flush the dataset to disk


def GetRasterData(Raster):
    """
    =====================================================
        GetRasterData(Raster)
    =====================================================
    to create a mask by knowing the stored value inside novalue cells 
    
    Inputs:
    ----------
        1- flow path lenth raster
    
    Outputs:
    ----------
        1- mask:array with all the values in the flow path length raster
        2- no_val: value stored in novalue cells
    """
    
    no_val = np.float32(Raster.GetRasterBand(1).GetNoDataValue()) # get the value stores in novalue cells
    mask = Raster.ReadAsArray() # read all values
    return mask, no_val

def MapAlgebra(src, fun):
    """
    ==============================================================
      MapAlgebra(src, dst, function)
    ==============================================================
    this function executes a mathematical operation on raster array and returns 
    the result
    
    inputs:
    ----------
        1-src:
            [gdal.dataset] source raster to get the location of the NoDataValue and
            where it is in the array
        2-dst:
            [gdal.dataset] source raster to get the location of the NoDataValue and
            where it is in the array
        3-function:
            numpy function 
    
    Example :
    ----------
        A=gdal.Open(evap.tif)
        func=np.abs
        new_raster=MapAlgebra(A,func)
    """
    # input data validation
    # data type
    assert type(src)==gdal.Dataset, "src should be read using gdal (gdal dataset please read it using gdal library) "
    assert callable(fun) , "second argument should be a function"
    
    src_gt=src.GetGeoTransform()
    src_proj=src.GetProjection()
    src_row=src.RasterYSize
    src_col=src.RasterXSize
    noval=np.float32(src.GetRasterBand(1).GetNoDataValue())
    src_sref=osr.SpatialReference(wkt=src_proj)
    src_array=src.ReadAsArray()
    
    # fill the new array with the nodata value
    new_array=np.ones((src_row,src_col))*noval
    # execute the function on each cell
    for i in range(src_row):
        for j in range(src_col):
            if src_array[i,j] != noval:
                new_array[i,j]=fun(src_array[i,j])
    
    # create the output raster
    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",src_col,src_row,1,gdalconst.GDT_Float32) #,['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation
    
    # set the geotransform
    dst.SetGeoTransform(src_gt)
    # set the projection
    dst.SetProjection(src_sref.ExportToWkt())
    # set the no data value
    dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
    # initialize the band with the nodata value instead of 0
    dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
    dst.GetRasterBand(1).WriteArray(new_array)
    
    return dst


def ResampleRaster(src,cell_size,resample_technique="Nearest"):
    """
    ======================================================================
      project_dataset(src, to_epsg):
    ======================================================================
    this function reproject a raster to any projection 
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
    
    Ex:
    ----------
    projected_raster=project_dataset(src, to_epsg=3857)
    
    """
    # input data validation
    # data type
    assert type(src)==gdal.Dataset, "src should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(resample_technique)== str ," please enter correct resample_technique more information see docmentation " 
    
    if resample_technique=="Nearest":
        resample_technique=gdal.GRA_NearestNeighbour
    elif resample_technique=="cubic":
        resample_technique=gdal.GRA_Cubic
    elif resample_technique=="bilinear":
        resample_technique=gdal.GRA_Bilinear
        
#    # READ THE RASTER
#    src = gdal.Open(inputspath+"dem_4km.tif")
    # GET PROJECTION
    src_proj=src.GetProjection()
    # GET THE GEOTRANSFORM
    src_gt=src.GetGeoTransform()
    # GET NUMBER OF columns
    src_x=src.RasterXSize
    # get number of rows
    src_y=src.RasterYSize
    # number of bands
#    src_bands=src.RasterCount
    # spatial ref
    src_epsg=osr.SpatialReference(wkt=src_proj)

    ulx = src_gt[0] 
    uly = src_gt[3]
    # transform the right lower corner point
    lrx = src_gt[0]+src_gt[1]*src_x
    lry = src_gt[3]+src_gt[5]*src_y

    pixel_spacing=cell_size
    # create a new raster
    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",int(np.round((lrx-ulx)/pixel_spacing)),int(np.round((uly-lry)/pixel_spacing)),
                       1,gdalconst.GDT_Float32,['COMPRESS=LZW']) # LZW is a lossless compression method achieve the highst compression but with lot of computation
    
    # set the geotransform
    dst.SetGeoTransform(src_gt)
    # set the projection
    dst.SetProjection(src_epsg.ExportToWkt())
    # set the no data value
    dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
    # initialize the band with the nodata value instead of 0
    dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
    # perform the projection & resampling    
    gdal.ReprojectImage(src,dst,src_epsg.ExportToWkt(),src_epsg.ExportToWkt(),resample_technique)

    return dst

def ProjectRaster(src, to_epsg,resample_technique="Nearest"):
    """
    =====================================================================
       project_dataset(src, to_epsg):
    =====================================================================
    this function reproject a raster to any projection 
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
    
    Ex:
    ----------
        projected_raster=project_dataset(src, to_epsg=3857)
    """
    #### input data validation
    # data type
    assert type(src)==gdal.Dataset, "src should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(to_epsg)==int,"please enter correct integer number for to_epsg more information https://epsg.io/"
    assert type(resample_technique)== str ," please enter correct resample_technique more information see docmentation " 
    
    if resample_technique=="Nearest":
        resample_technique=gdal.GRA_NearestNeighbour
    elif resample_technique=="cubic":
        resample_technique=gdal.GRA_Cubic
    elif resample_technique=="bilinear":
        resample_technique=gdal.GRA_Bilinear
        
    ### Source raster
    # GET PROJECTION
    src_proj=src.GetProjection()
    # GET THE GEOTRANSFORM
    src_gt=src.GetGeoTransform()
    # GET NUMBER OF columns
    src_x=src.RasterXSize
    # get number of rows
    src_y=src.RasterYSize
    # number of bands
#    src_bands=src.RasterCount
    # spatial ref
    src_epsg=osr.SpatialReference(wkt=src_proj)
    
    ### distination raster
    # spatial ref
    dst_epsg=osr.SpatialReference()
    dst_epsg.ImportFromEPSG(to_epsg)
    # transformation factors
    tx = osr.CoordinateTransformation(src_epsg,dst_epsg)
    
    # in case the source crs is GCS and longitude is in the west hemisphere gdal 
    # reads longitude fron 0 to 360 and transformation factor wont work with valeus
    # greater than 180
    if src_epsg.GetAttrValue('AUTHORITY',1) != str(to_epsg) :
        if src_epsg.GetAttrValue('AUTHORITY',1)=="4326" and src_gt[0] > 180:
            lng_new=src_gt[0]-360
            # transform the right upper corner point    
            (ulx,uly,ulz) = tx.TransformPoint(lng_new, src_gt[3])
            # transform the right lower corner point
            (lrx,lry,lrz)=tx.TransformPoint(lng_new+src_gt[1]*src_x,
                                            src_gt[3]+src_gt[5]*src_y)
        else: 
            # transform the right upper corner point    
            (ulx,uly,ulz) = tx.TransformPoint(src_gt[0], src_gt[3])
            # transform the right lower corner point
            (lrx,lry,lrz)=tx.TransformPoint(src_gt[0]+src_gt[1]*src_x,
                                            src_gt[3]+src_gt[5]*src_y) 
    else:
        ulx = src_gt[0]
        uly = src_gt[3]
#        ulz = 0
        lrx = src_gt[0]+src_gt[1]*src_x
        lry = src_gt[3]+src_gt[5]*src_y
#        lrz = 0
        
    
    # get the cell size in the source raster and convert it to the new crs
    # x coordinates or longitudes
    xs=[src_gt[0],src_gt[0]+src_gt[1]]
    # y coordinates or latitudes
    ys=[src_gt[3],src_gt[3]]
    
    if src_epsg.GetAttrValue('AUTHORITY',1) != str(to_epsg):
        # transform the two points coordinates to the new crs to calculate the new cell size
        new_xs, new_ys= ReprojectPoints(ys,xs,from_epsg=int(src_epsg.GetAttrValue('AUTHORITY',1)),
                                         to_epsg=int(dst_epsg.GetAttrValue('AUTHORITY',1)))
    else:
        new_xs = xs
        new_ys = ys
        
    pixel_spacing=np.abs(new_xs[0]-new_xs[1])

    # create a new raster
    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",int(np.round((lrx-ulx)/pixel_spacing)),int(np.round((uly-lry)/pixel_spacing)),
                       1,gdalconst.GDT_Float32) #['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation
    
    # new geotransform 
    new_geo=(ulx,pixel_spacing,src_gt[2],uly,src_gt[4],-pixel_spacing)
    # set the geotransform
    dst.SetGeoTransform(new_geo)
    # set the projection
    dst.SetProjection(dst_epsg.ExportToWkt())
    # set the no data value
    dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
    # initialize the band with the nodata value instead of 0
    dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
    # perform the projection & resampling    
    gdal.ReprojectImage(src,dst,src_epsg.ExportToWkt(),dst_epsg.ExportToWkt(),resample_technique)

    return dst

def ReprojectDataset(src, to_epsg=3857, cell_size=[], resample_technique="Nearest"):
    """
    =====================================================================
     reproject_dataset(src, to_epsg=3857, pixel_spacing=[]):
    =====================================================================
    this function reproject and resample a raster to any projection 
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
    # input data validation
    # type of inputs
    assert type(src)==gdal.Dataset, "src should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(to_epsg)==int,"please enter correct integer number for to_epsg more information https://epsg.io/"
    assert type(resample_technique)== str ," please enter correct resample_technique more information see docmentation " 
    if cell_size != []:
        assert type(cell_size)== int or type(cell_size)== float , "please enter an integer or float cell size"
    
    if resample_technique=="Nearest":
        resample_technique=gdal.GRA_NearestNeighbour
    elif resample_technique=="cubic":
        resample_technique=gdal.GRA_Cubic
    elif resample_technique=="bilinear":
        resample_technique=gdal.GRA_Bilinear
    
#    # READ THE RASTER
#    src = gdal.Open(inputspath+"dem_4km.tif")
    # GET PROJECTION
    src_proj=src.GetProjection()
    # GET THE GEOTRANSFORM
    src_gt=src.GetGeoTransform()
    # GET NUMBER OF columns
    src_x=src.RasterXSize
    # get number of rows
    src_y=src.RasterYSize
    # number of bands
#    src_bands=src.RasterCount
    # spatial ref
    src_epsg=osr.SpatialReference(wkt=src_proj)

    # distination
    # spatial ref
    dst_epsg=osr.SpatialReference()
    dst_epsg.ImportFromEPSG(to_epsg)
    # transformation factors
    tx = osr.CoordinateTransformation(src_epsg,dst_epsg)
    
    # incase the source crs is GCS and longitude is in the west hemisphere gdal 
    # reads longitude fron 0 to 360 and transformation factor wont work with valeus
    # greater than 180
    if src_epsg.GetAttrValue('AUTHORITY',1)=="4326" and src_gt[0] > 180:
        lng_new=src_gt[0]-360
        # transform the right upper corner point    
        (ulx,uly,ulz) = tx.TransformPoint(lng_new, src_gt[3])
        # transform the right lower corner point
        (lrx,lry,lrz)=tx.TransformPoint(lng_new+src_gt[1]*src_x,
                                        src_gt[3]+src_gt[5]*src_y)
    else: 
        # transform the right upper corner point    
        (ulx,uly,ulz) = tx.TransformPoint(src_gt[0], src_gt[3])
        # transform the right lower corner point
        (lrx,lry,lrz)=tx.TransformPoint(src_gt[0]+src_gt[1]*src_x,
                                        src_gt[3]+src_gt[5]*src_y)

    if cell_size==[]:
    # the result raster has the same pixcel size as the source 
        # check if the coordinate system is GCS convert the distance from angular to metric
        if src_epsg.GetAttrValue('AUTHORITY',1)=="4326":
            coords_1 = (src_gt[3], src_gt[0])
            coords_2 = (src_gt[3], src_gt[0]+src_gt[1])
#            pixel_spacing=geopy.distance.vincenty(coords_1, coords_2).m
            pixel_spacing=GCSDistance(coords_1, coords_2)
        else: 
            pixel_spacing=src_gt[1]
    else:
        assert (cell_size > 1),"please enter cell size greater than 1"
    # if the user input a cell size resample the raster
        pixel_spacing=cell_size 
        
    # create a new raster
    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",int(np.round((lrx-ulx)/pixel_spacing)),int(np.round((uly-lry)/pixel_spacing)),
                       1,gdalconst.GDT_Float32) # ['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation
    
    # new geotransform 
    new_geo=(ulx,pixel_spacing,src_gt[2],uly,src_gt[4],-pixel_spacing)
    # set the geotransform
    dst.SetGeoTransform(new_geo)
    # set the projection
    dst.SetProjection(dst_epsg.ExportToWkt())
    # set the no data value
    dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
    # initialize the band with the nodata value instead of 0
    dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
    # perform the projection & resampling 
    gdal.ReprojectImage(src,dst,src_epsg.ExportToWkt(),dst_epsg.ExportToWkt(),resample_technique)

    return dst


def RasterLike(src,array,path,pixel_type=1):
    """
    ====================================================================
      RasterLike(src,array,path)
    ====================================================================
    this function creates a Geotiff raster like another input raster, new raster 
    will have the same projection, coordinates or the top left corner of the original
    raster, cell size, nodata velue, and number of rows and columns
    the raster and the dem should have the same number of columns and rows
    
    inputs:
    ----------
        1- src:
            [gdal.dataset] source raster to get the spatial information
        2- array:
            [numpy array]to store in the new raster
        3- path:
            [String] path to save the new raster including new raster name and extension (.tif)
        4-pixel_type:
            [integer] type of the data to be stored in the pixels,default is 1 (float32)
            for example pixel type of flow direction raster is unsigned integer
            1 for float32
            2 for float64
            3 for Unsigned integer 16
            4 for Unsigned integer 32
            5 for integer 16
            6 for integer 32
    
    outputs:
    ----------
        1- save the new raster to the given path
    
    Ex:
    ----------
        data=np.load("RAIN_5k.npy")
        src=gdal.Open("DEM.tif")
        name="rain.tif"
        RasterLike(src,data,name)
    """
    # input data validation
    # data type
    assert type(src)==gdal.Dataset, "src should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(array)==np.ndarray, "array should be of type numpy array"
    assert type(path)== str, "Raster_path input should be string type"
    assert type(pixel_type)== int, "pixel type input should be integer type please check documentations"
    # input values
#    assert os.path.exists(path), path+ " you have provided does not exist"
    ext=path[-4:]
    assert ext == ".tif", "please add the extension at the end of the path input"
#    assert os.path.exists(path), "source raster you have provided does not exist"
    
    prj=src.GetProjection()
    cols=src.RasterXSize
    rows=src.RasterYSize
    gt=src.GetGeoTransform()
    noval=src.GetRasterBand(1).GetNoDataValue()
    if pixel_type==1:
        outputraster=gdal.GetDriverByName('GTiff').Create(path,cols,rows,1,gdal.GDT_Float32)
    elif pixel_type==2: 
        outputraster=gdal.GetDriverByName('GTiff').Create(path,cols,rows,1,gdal.GDT_Float64)
    elif pixel_type==3: 
        outputraster=gdal.GetDriverByName('GTiff').Create(path,cols,rows,1,gdal.GDT_UInt16)
    elif pixel_type==4: 
        outputraster=gdal.GetDriverByName('GTiff').Create(path,cols,rows,1,gdal.GDT_UInt32)
    elif pixel_type==5: 
        outputraster=gdal.GetDriverByName('GTiff').Create(path,cols,rows,1,gdal.GDT_Int16)
    elif pixel_type==6: 
        outputraster=gdal.GetDriverByName('GTiff').Create(path,cols,rows,1,gdal.GDT_Int32)
        
    outputraster.SetGeoTransform(gt)
    outputraster.SetProjection(prj)
    outputraster.GetRasterBand(1).SetNoDataValue(noval)
    outputraster.GetRasterBand(1).Fill(noval)
    outputraster.GetRasterBand(1).WriteArray(array)
    outputraster.FlushCache()
    outputraster = None
    
def MatchNoDataValue(src,dst):
    """
    ==================================================================
      MatchNoDataValue(src,dst)
    ==================================================================
    this function matches the location of nodata value from src raster to dst 
    raster, Both rasters have to have the same dimensions (no of rows & columns)
    so MatchRasterAlignment should be used prior to this function to align both
    rasters
    
    
    inputs:
    ----------
        1-src:
            [gdal.dataset] source raster to get the location of the NoDataValue and
            where it is in the array
        1-dst:
            [gdal.dataset] raster you want to store NoDataValue in its cells
            exactly the same like src raster
    
    Outputs:
    ----------
        1- dst:
            [gdal.dataset] the second raster with NoDataValue stored in its cells
            exactly the same like src raster
    """
    # input data validation
    # data type
    assert type(src)==gdal.Dataset, "src should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(dst)==gdal.Dataset, "dst should be read using gdal (gdal dataset please read it using gdal library) "
    
    src_gt=src.GetGeoTransform()
    src_proj=src.GetProjection()
    src_row=src.RasterYSize
    src_col=src.RasterXSize
    src_noval=np.float32(src.GetRasterBand(1).GetNoDataValue())
    src_sref=osr.SpatialReference(wkt=src_proj)
    src_epsg=int(src_sref.GetAttrValue('AUTHORITY',1))
    
    src_array=src.ReadAsArray()
    
    dst_gt=dst.GetGeoTransform()
    dst_proj=dst.GetProjection()
    dst_row=dst.RasterYSize
    dst_col=dst.RasterXSize
    
    dst_sref=osr.SpatialReference(wkt=dst_proj)
    dst_epsg=int(dst_sref.GetAttrValue('AUTHORITY',1))
    
    #check proj 
    assert src_row==dst_row and src_col==dst_col, "two rasters has different no of columns or rows please resample or match both rasters"
    assert dst_gt==src_gt, "location of upper left corner of both rasters are not the same or cell size is different please match both rasters first "
    assert src_epsg == dst_epsg, "Raster A & B are using different coordinate system please reproject one of them to the other raster coordinate system"
    
    dst_array = np.float32(dst.ReadAsArray())
    dst_array[src_array==src_noval] = src_noval
    
    # align function only equate the no of rows and columns only
    # match nodatavalue inserts nodatavalue in dst raster to all places like src
    # still places that has nodatavalue in the dst raster but it is not nodatavalue in the src 
    # and now has to be filled with values
    # compare no of element that is not nodata value in both rasters to make sure they are matched
    elem_src = np.size(src_array[:,:])-np.count_nonzero((src_array[src_array==src_noval]))
    elem_dst = np.size(dst_array[:,:])-np.count_nonzero((dst_array[dst_array==src_noval])) 
    # if not equal then store indices of those cells that doesn't matchs
    if elem_src > elem_dst :
        rows=[i for i in range(src_row) for j in range(src_col) if dst_array[i,j]==src_noval and src_array[i,j] != src_noval]
        cols=[j for i in range(src_row) for j in range(src_col) if dst_array[i,j]==src_noval and src_array[i,j] != src_noval]
    # interpolate those missing cells by nearest neighbour
    if elem_src > elem_dst :
        dst_array = NearestNeighbour(dst_array, src_noval, rows, cols)
    
    
    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",src_col,src_row,1,gdalconst.GDT_Float32) #,['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation
    
    # set the geotransform
    dst.SetGeoTransform(src_gt)
    # set the projection
    dst.SetProjection(src_sref.ExportToWkt())
    # set the no data value
    dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
    # initialize the band with the nodata value instead of 0
    dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
    dst.GetRasterBand(1).WriteArray(dst_array)
    
    return dst

def ChangeNoDataValue(src,dst):
    """
    ==================================================================
      ChangeNoDataValue(src,dst)
    ==================================================================
    this function changes the value of nodata value in a dst raster to be like 
    a src raster.
    
    inputs:
    ----------
        1-src:
            [gdal.dataset] source raster to get the location of the NoDataValue and
            where it is in the array
        1-dst:
            [gdal.dataset] raster you want to store NoDataValue in its cells
            exactly the same like src raster
    
    Outputs:
    ----------
        1- dst:
            [gdal.dataset] the second raster with NoDataValue stored in its cells
            exactly the same like src raster
    """
    # input data validation
    # data type
    assert type(src)==gdal.Dataset, "src should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(dst)==gdal.Dataset, "dst should be read using gdal (gdal dataset please read it using gdal library) "
    
    src_noval=np.float32(src.GetRasterBand(1).GetNoDataValue())    
    
    dst_gt = dst.GetGeoTransform()
    dst_proj = dst.GetProjection()
    dst_row = dst.RasterYSize
    dst_col = dst.RasterXSize
    dst_noval = np.float32(dst.GetRasterBand(1).GetNoDataValue())
    dst_sref = osr.SpatialReference(wkt=dst_proj)
#    dst_epsg = int(dst_sref.GetAttrValue('AUTHORITY',1))
    
    dst_array = dst.ReadAsArray()
    dst_array[dst_array==dst_noval]=src_noval

    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",dst_col,dst_row,1,gdalconst.GDT_Float32) #,['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation
    
    # set the geotransform
    dst.SetGeoTransform(dst_gt)
    # set the projection
    dst.SetProjection(dst_sref.ExportToWkt())
    # set the no data value
    dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
    # initialize the band with the nodata value instead of 0
    dst.GetRasterBand(1).Fill(src.GetRasterBand(1).GetNoDataValue())
    dst.GetRasterBand(1).WriteArray(dst_array)
    
    return dst


def MatchRasterAlignment(RasterA,RasterB):
    """
    =========================================================================
      MatchRasterAlignment(RasterA,RasterB)
    =========================================================================
    this function matches the coordinate system and the number of of rows & columns
    between two rasters
    Raster A is the source of the coordinate system, no of rows and no of columns & cell size
    Raster B is the source of data values in cells 
    the result will be a raster with the same structure like RasterA but with 
    values from RasterB using Nearest Neighbour interpolation algorithm
    
    Inputs:
    ----------
        1- RasterA:
            [gdal.dataset] spatial information source raster to get the spatial information 
            (coordinate system, no of rows & columns)
        2- RasterB:
            [gdal.dataset] data values source raster to get the data (values of each cell)
    
    Outputs:
    ----------
        1- dst:
            [gdal.dataset] result raster in memory
    
    Example:
    ----------
        A=gdal.Open("dem4km.tif")
        B=gdal.Open("P_CHIRPS.v2.0_mm-day-1_daily_2009.01.06.tif")
        matched_raster=MatchRasters(A,B)
    """
    # input data validation
    # data type
    assert type(RasterA)==gdal.Dataset, "RasterA should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(RasterB)==gdal.Dataset, "RasterB should be read using gdal (gdal dataset please read it using gdal library) "
    
    gt_src=RasterA
    # we need number of rows and cols from src A and data from src B to store both in dst
    gt_src_proj=gt_src.GetProjection()
    # GET THE GEOTRANSFORM
    gt_src_gt=gt_src.GetGeoTransform()
    # GET NUMBER OF columns
    gt_src_x=gt_src.RasterXSize
    # get number of rows
    gt_src_y=gt_src.RasterYSize
    
    gt_src_epsg=osr.SpatialReference(wkt=gt_src_proj)
#    gt_src_epsg.GetAttrValue('AUTHORITY',1)
    
    # unite the crs
    # TODO still doesn't work with all projections better to use UTM zones for the moment 
    data_src=ProjectRaster(RasterB,int(gt_src_epsg.GetAttrValue('AUTHORITY',1)))
        
    # create a new raster
    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",gt_src_x,gt_src_y,1,gdalconst.GDT_Float32) #,['COMPRESS=LZW'] LZW is a lossless compression method achieve the highst compression but with lot of computation
    # set the geotransform
    dst.SetGeoTransform(gt_src_gt)
    # set the projection
    dst.SetProjection(gt_src_epsg.ExportToWkt())
    # set the no data value
    dst.GetRasterBand(1).SetNoDataValue(gt_src.GetRasterBand(1).GetNoDataValue())
    # initialize the band with the nodata value instead of 0
    dst.GetRasterBand(1).Fill(gt_src.GetRasterBand(1).GetNoDataValue())
    # perform the projection & resampling
    resample_technique=gdal.GRA_NearestNeighbour #gdal.GRA_NearestNeighbour
    
    gdal.ReprojectImage(data_src,dst,gt_src_epsg.ExportToWkt(),gt_src_epsg.ExportToWkt(),resample_technique)
    
#    SaveRaster(dst,"colombia/newraster.tif")
    return dst


def NearestNeighbour(array, Noval, rows, cols):
    """
    ===============================================================
        NearestNeighbour(array, Noval, rows, cols)
    ===============================================================
    this function filles cells of a given indices in rows and cols with
    the value of the nearest neighbour.
    as the raster grid is square so the 4 perpendicular direction are of the same 
    close so the function give priority to the right then left then bottom then top
    and the same for 45 degree inclined direction right bottom then left bottom
    then left Top then right Top
    
    Inputs:
    ----------
        1-array:
            [numpy.array] Array to fill some of its cells with Nearest value.
        2-Noval:
            [float32] value stored in cells that is out of the domain
        3-rows:
            [List] list of the row index of the cells you want to fill it with
            nearest neighbour.
        4-cols:
            [List] list of the column index of the cells you want to fill it with
            nearest neighbour.
    
    Output:
    ----------
        - array:
            [numpy array] Cells of given indices will be filled with value of the Nearest neighbour
    
    Example:
    ----------
        - raster=gdal.opne("dem.tif")
          rows=[3,12]
          cols=[9,2]
          new_array=NearestNeighbour(rasters, rows, cols)
    """
    #### input data validation
    # data type
    assert type(array)==np.ndarray , "src should be read using gdal (gdal dataset please read it using gdal library) "
    assert type(rows) == list,"rows input has to be of type list"
    assert type(cols) == list,"cols input has to be of type list"
    
    
#    array=raster.ReadAsArray()
#    Noval=np.float32(raster.GetRasterBand(1).GetNoDataValue())
#    no_rows=raster.RasterYSize
    no_rows=np.shape(array)[0]
#    no_cols=raster.RasterXSize
    no_cols=np.shape(array)[1]
    
    for i in range(len(rows)):
        # give the cell the value of the cell that is at the right
        if array[rows[i],cols[i]+1] != Noval and cols[i]+1 <= no_cols:
            array[rows[i],cols[i]] = array[rows[i],cols[i]+1]
        
        elif array[rows[i],cols[i]-1] != Noval and cols[i]-1 > 0 :
            # give the cell the value of the cell that is at the left
            array[rows[i],cols[i]] = array[rows[i],cols[i]-1]

        elif array[rows[i]-1,cols[i]] != Noval and rows[i]-1 > 0:
        # give the cell the value of the cell that is at the bottom
            array[rows[i],cols[i]] = array[rows[i]-1,cols[i]]
            
        elif array[rows[i]+1,cols[i]] != Noval and rows[i]+1 <= no_rows:
        # give the cell the value of the cell that is at the Top
            array[rows[i],cols[i]] = array[rows[i]+1,cols[i]]

        elif array[rows[i]-1,cols[i]+1] != Noval and rows[i]-1 > 0 and cols[i]+1 <=no_cols :
        # give the cell the value of the cell that is at the right bottom
            array[rows[i],cols[i]] = array[rows[i]-1,cols[i]+1]
                    
        elif array[rows[i]-1,cols[i]-1] != Noval and rows[i]-1 >0 and cols[i]-1 > 0:
        # give the cell the value of the cell that is at the left bottom
            array[rows[i],cols[i]] = array[rows[i]-1,cols[i]-1]
                        
        elif array[rows[i]+1,cols[i]-1] != Noval and rows[i]+1 <= no_rows and cols[i]-1 > 0:
        # give the cell the value of the cell that is at the left Top
            array[rows[i],cols[i]] = array[rows[i]+1,cols[i]-1]
                            
        elif array[rows[i]+1,cols[i]+1] != Noval and rows[i]+1 <= no_rows and cols[i]+1 <= no_cols:
        # give the cell the value of the cell that is at the right Top
            array[rows[i],cols[i]] = array[rows[i]+1,cols[i]+1]
        else:
            print("the cell is isolated (No surrounding cells exist)")
    return array


def ReadASCII(ASCIIFile,pixel_type=1):
    """  
    =========================================================================
        ReadASCII(ASCIIFile,pixel_type)
    =========================================================================
     
    This function reads an ASCII file the spatial information
    
    Inputs:
        1-ASCIIFileName:
            [String] name of the ASCII file you want to convert and the name 
            should include the extension ".asc"
            
        2-pixel_type:
            [Integer] type of the data to be stored in the pixels,default is 1 (float32)
            for example pixel type of flow direction raster is unsigned integer
            1 for float32
            2 for float64
            3 for Unsigned integer 16
            4 for Unsigned integer 32
            5 for integer 16
            6 for integer 32
    Outputs:
        1-ASCIIValues:
            [numpy array] 2D arrays containing the values stored in the ASCII
            file
            
        2-ASCIIDetails:
            [List] list of the six spatial information of the ASCII file 
            [ASCIIRows, ASCIIColumns, XLowLeftCorner, YLowLeftCorner, 
            CellSize, NoValue]
    Example:
        Elevation_values,DEMSpatialDetails = ReadASCII("dem.asc",1)
    """
    # input data validation
    # data type
    assert type(ASCIIFile) == str, "ASCIIFile input should be string type"
    assert type(pixel_type)== int, "pixel type input should be integer type please check documentations"
    
    # input values
    ASCIIExt=ASCIIFile[-4:]
    assert ASCIIExt == ".asc", "please add the extension at the end of the path input"
    assert os.path.exists(ASCIIFile), "ASCII file path you have provided does not exist"
    
    ### read the ASCII file 
    
    File  = open (ASCIIFile)
    Wholefile = File.readlines()
    File.close()
    
    ASCIIColumns = int(Wholefile[0].split()[1])
    ASCIIRows = int(Wholefile[1].split()[1])
    
    XLeftSide = int(float(Wholefile[2].split()[1]))
    YLowerSide = int(float(Wholefile[3].split()[1]))
    CellSize = int(Wholefile[4].split()[1])
    NoValue = int(Wholefile[5].split()[1])
    
    ASCIIValues = np.ones((ASCIIRows,ASCIIColumns), dtype = np.float32)
    try:
        for i in range(ASCIIRows):
            x = Wholefile[6+i].split()
            ASCIIValues[i,:] = list(map(float, x ))
    except:
        try:
            for j in range(len(x)):
                float(x[j])
        except:
            print("Error reading the ARCII file please check row " + str(i+6+1) +", column " + str(j))
            print("A value of " + x[j] + " , is stored in the ASCII file ")
        
    ASCIIDetails = [ASCIIRows, ASCIIColumns, XLeftSide , YLowerSide, 
                    CellSize, NoValue]
    
    return ASCIIValues, ASCIIDetails

def WriteASCII(ASCIIFile, ASCIIDetails, ASCIIValues):
    """  
    =========================================================================
        WriteASCII(ASCIIFile, ASCIIDetails, ASCIIValues, pixel_type=1)
    =========================================================================
     
    This function reads an ASCII file the spatial information
    
    Inputs:
        1-ASCIIFile:
            [String] name of the ASCII file you want to convert and the name 
            should include the extension ".asc"

        2-ASCIIDetails:
            [List] list of the six spatial information of the ASCII file 
            [ASCIIRows, ASCIIColumns, XLowLeftCorner, YLowLeftCorner, 
            CellSize, NoValue]
        
        3-ASCIIValues:
            [numpy array] 2D arrays containing the values stored in the ASCII
            file
            
    Outputs:
            
    Example:
        Elevation_values,DEMSpatialDetails = ReadASCII("dem.asc",1)
    """
    # input data validation
    # data type
    assert type(ASCIIFile) == str, "ASCIIFile input should be string type"
    
    # input values
    ASCIIExt=ASCIIFile[-4:]
    assert ASCIIExt == ".asc", "please add the extension at the end of the path input"
#    assert os.path.exists(ASCIIFile), "ASCII file path you have provided does not exist"
    
    ## used function 
    
    def StringSpace(Inp):
        return str(Inp) + "  " 

    ### read the ASCII file 
    try:
        File  = open (ASCIIFile,'w')
    except:
        print("path you have provided does not exist")
        print("please check" + ASCIIFile)
        
    # write the the ASCII file details
    File.write('ncols         ' + str(ASCIIDetails[1])+ "\n")
    File.write('nrows         ' + str(ASCIIDetails[0])+ "\n") 
    File.write('xllcorner     ' + str(ASCIIDetails[2])+ "\n")
    File.write('yllcorner     ' + str(ASCIIDetails[3])+ "\n")
    File.write('cellsize      ' + str(ASCIIDetails[4])+ "\n")
    File.write('NODATA_value  ' + str(ASCIIDetails[5])+ "\n")
    
    # write the array
    for i in range(np.shape(ASCIIValues)[0]):
        File.writelines(list(map(StringSpace,ASCIIValues[i,:])))
        File.write("\n")
    
    File.close()


def ASCIItoRaster(ASCIIFile,savePath,pixel_type=1,RasterFile = None,epsg = None):
    """  
    =========================================================================
        ASCIItoRaster(ASCIIFile,savePath,pixel_type=1,RasterFile = None,epsg = None)
    =========================================================================
     
    This function convert an ASCII file into a raster format and in takes  all
    the spatial information (projection, coordinates of the corner point), and
    number of rows and columns from raster file or you have to define the epsg corresponding 
    to the you coordinate system and projection
    
    Inputs:
        1-ASCIIFileName:
            [String] name of the ASCII file you want to convert and the name 
            should include the extension ".asc"
            
        2-savePath:
            [String] path to save the new raster including new raster name and extension (.tif)
            
        3-pixel_type:
            [Integer] type of the data to be stored in the pixels,default is 1 (float32)
            for example pixel type of flow direction raster is unsigned integer
            1 for float32
            2 for float64
            3 for Unsigned integer 16
            4 for Unsigned integer 32
            5 for integer 16
            6 for integer 32
            
        4-RasterFile:
            [String] source raster to get the spatial information, both ASCII
            file and source raster should have the same number of rows, and 
            same number of columns default value is [None].
         
        5-epsg: 
            EPSG stands for European Petroleum Survey Group and is an organization 
            that maintains a geodetic parameter database with standard codes,
            the EPSG codes, for coordinate systems, datums, spheroids, units 
            and such alike (https://epsg.io/) default value is [None].
            
    Outputs:
        1- a New Raster will be saved in the savePath containing the values 
        of the ASCII file
    
    Example:
        1- ASCII to raster given a raster file:
            ASCIIFile = "soiltype.asc"
            RasterFile = "DEM.tif"
            savePath = "Soil_raster.tif"
            pixel_type = 1
            ASCIItoRaster(ASCIIFile,  savePath, pixel_type, RasterFile)
        2- ASCII to Raster given an EPSG number
            ASCIIFile = "soiltype.asc"
            savePath = "Soil_raster.tif"
            pixel_type = 1
            epsg = 4647
        ASCIItoRaster(ASCIIFile, savePath,pixel_type, epsg = epsg)
    """
    # input data validation
    # data type
    assert type(ASCIIFile) == str, "ASCIIFile input should be string type"
    assert type(savePath) == str, "savePath input should be string type"
    assert type(pixel_type)== int, "pixel type input should be integer type please check documentations"
    
    # input values
    ASCIIExt=ASCIIFile[-4:]
    assert ASCIIExt == ".asc", "please add the extension at the end of the path input"
    
    # assert os.path.exists(path), "source raster you have provided does not exist"
    
    # check what does the user enter
#    try: RasterFile
#    except NameError : RasterFile = None
    
#    try: epsg
#    except NameError : epsg = None
    
    message = """ you have to enter one of the following inputs 
    - RasterFile : if you have a raster with the same spatial information 
        (projection, coordinate system), and have the same number of rows, 
        and columns
    - epsg : if you have the EPSG number (https://epsg.io/) refering to 
        the spatial information of the ASCII file 
    """
    assert RasterFile != None or epsg != None, message
    
    ### read the ASCII file 
    ASCIIValues, ASCIIDetails = ReadASCII(ASCIIFile,pixel_type)
    ASCIIRows = ASCIIDetails[0]
    ASCIIColumns = ASCIIDetails[1]
    
    # check the optional inputs 
    if RasterFile != None :
        assert type(RasterFile) == str, "RasterFile input should be string type"    
        
        RasterExt=RasterFile[-4:]
        assert RasterExt == ".tif", "please add the extension at the end of the path input"
        # read the raster file 
        src = gdal.Open(RasterFile)
        RasterColumns = src.RasterXSize
        RasterRows = src.RasterYSize
        
        assert ASCIIRows == RasterRows and ASCIIColumns == RasterColumns, " Data in both ASCII file and Raster file should have the same number of row and columns"
        
        RasterLike(src,ASCIIValues, savePath, pixel_type)
    elif epsg != None :
        assert type(epsg)== int, "epsg input should be integer type please check documentations"
        # coordinates of the lower left corner
        XLeftSide  = ASCIIDetails[2]
#        YLowSide = ASCIIDetails[3]
        
        CellSize = ASCIIDetails[4]
        NoValue = ASCIIDetails[5]        
        # calculate Geotransform coordinates for the raster
        YUpperSide = ASCIIDetails[3] + ASCIIRows * CellSize
        
        dst_gt = (XLeftSide, CellSize, 0.0, YUpperSide, 0.0, -1*CellSize)
        dst_epsg=osr.SpatialReference()
        dst_epsg.ImportFromEPSG(epsg)
        
        if pixel_type==1:
            dst=gdal.GetDriverByName('GTiff').Create(savePath,ASCIIColumns,ASCIIRows,1,gdal.GDT_Float32)
        elif pixel_type==2: 
            dst=gdal.GetDriverByName('GTiff').Create(savePath,ASCIIColumns,ASCIIRows,1,gdal.GDT_Float64)
        elif pixel_type==3: 
            dst=gdal.GetDriverByName('GTiff').Create(savePath,ASCIIColumns,ASCIIRows,1,gdal.GDT_UInt16)
        elif pixel_type==4: 
            dst=gdal.GetDriverByName('GTiff').Create(savePath,ASCIIColumns,ASCIIRows,1,gdal.GDT_UInt32)
        elif pixel_type==5: 
            dst=gdal.GetDriverByName('GTiff').Create(savePath,ASCIIColumns,ASCIIRows,1,gdal.GDT_Int16)
        elif pixel_type==6: 
            dst=gdal.GetDriverByName('GTiff').Create(savePath,ASCIIColumns,ASCIIRows,1,gdal.GDT_Int32)
            
        dst.SetGeoTransform(dst_gt)
        dst.SetProjection(dst_epsg.ExportToWkt())
        dst.GetRasterBand(1).SetNoDataValue(NoValue)
        dst.GetRasterBand(1).Fill(NoValue)
        dst.GetRasterBand(1).WriteArray(ASCIIValues)
        dst.FlushCache()
        dst = None


def GCSDistance(coords_1,coords_2):
    """
    =====================================================================
      GCS_distance(coords_1,coords_2)
    =====================================================================
    this function calculates the distance between two points that have 
    geographic coordinate system
    
    inputs:
    ----------
        1-coord_1:
            tuple of (long, lat) of the first point
        2- coord_2:
            tuple of (long, lat) of the second point
    
    Output:
    ----------
        1-distance between the two points 
    """
    import geopy.distance
    
#    coords_1 = (52.2296756, 21.0122287)
#    coords_2 = (52.406374, 16.9251681)
    
    dist=geopy.distance.vincenty(coords_1, coords_2).m
    
    return dist


def ReprojectPoints(lat,lng,from_epsg=4326,to_epsg=3857):
    """
    =====================================================================
      reproject_points(lat, lng, from_epsg=4326,to_epsg=3857)
    =====================================================================
    this function change the projection of the coordinates from a coordinate system
    to another (default from GCS to web mercator used by google maps)
    
    Inputs:
    ----------
        1- lat: 
            list of latitudes of the points 
        2- lng:
            list of longitude of the points
        3- from_epsg:
            integer reference number to the projection of the points (https://epsg.io/)
        4- to_epsg:
            integer reference number to the new projection of the points (https://epsg.io/)
    
    outputs:
    ----------
        1-x:
            list of x coordinates of the points 
        2-y:
            list of y coordinates of the points 
    
    Ex:
    ----------
        # from web mercator to GCS WGS64:
        x=[-8418583.96378159, -8404716.499972705], y=[529374.3212213353, 529374.3212213353]
        from_epsg = 3857, to_epsg = 4326
        longs, lats=reproject_points(y,x,from_epsg="3857", to_epsg="4326")
    """
    from pyproj import Proj, transform
    from_epsg="epsg:"+str(from_epsg)
    inproj = Proj(init=from_epsg) # GCS geographic coordinate system
    to_epsg="epsg:"+str(to_epsg)
    outproj=Proj(init=to_epsg) # WGS84 web mercator 
    
    x=np.ones(len(lat))*np.nan
    y=np.ones(len(lat))*np.nan
    
    for i in range(len(lat)):
        x[i],y[i]=transform(inproj,outproj,lng[i],lat[i])

    return x,y

def ReprojectPoints_2(lat,lng,from_epsg=4326,to_epsg=3857):
    """
    ======================================================================
     reproject_points(lat,lng, from_epsg=4326,to_epsg=3857):
    ======================================================================
    this function change the projection of the coordinates from a coordinate system
    to another (default from GCS to web mercator used by google maps)
    
    Inputs:
    ----------
        1- lat: 
            list of latitudes of the points 
        2- lng:
            list of longitude of the points
        3- from_epsg:
            integer reference number to the projection of the points (https://epsg.io/)
        4- to_epsg:
            integer reference number to the new projection of the points (https://epsg.io/)
    
    outputs:
    ----------
        1-x:
            list of x coordinates of the points 
        2-y:
            list of y coordinates of the points 
    
    Ex:
    ----------
        # from web mercator to GCS WGS64:
        x=[-8418583.96378159, -8404716.499972705], y=[529374.3212213353, 529374.3212213353]
        from_epsg = 3857, to_epsg = 4326
        longs, lats=reproject_points(y,x,from_epsg="3857", to_epsg="4326")
    """
    from osgeo import ogr
    from osgeo import osr
    
    source = osr.SpatialReference()
    source.ImportFromEPSG(from_epsg)
    
    target = osr.SpatialReference()
    target.ImportFromEPSG(to_epsg)
    
    transform = osr.CoordinateTransformation(source, target)
    x=[]
    y=[]
    for i in range(len(lat)):
        point = ogr.CreateGeometryFromWkt("POINT ("+str(lng[i])+" "+str(lat[i])+")")
        point.Transform(transform)
        x.append(point.GetPoints()[0][0])
        y.append(point.GetPoints()[0][1])
    return x,y


def CreatePolygon(coords, Type=1):
    """
    ======================================================================
        create_polygon(coords)
    ======================================================================
    this function creates a polygon from coordinates
    
    inputs:
    ----------
        coords : 
            [List] list of tuples [(x1,y1),(x2,y2)]
        Type :
            [Integer] 1 to return a polygon in the form of WellKnownText, 2 to return a 
            polygon as an object
    
    outputs:
    ----------
        Type 1 returns a string of the polygon and its coordinates as 
        a WellKnownText, Type 2 returns Shapely Polygon object you can assign it
        to a GeoPandas GeoDataFrame directly
    
    
    Example:
    ----------
        coords = [(-106.6472953, 24.0370137), (-106.4933356, 24.05293569), (-106.4941789, 24.01969175), (-106.4927777, 23.98804445)]
        GIS.CreatePolygon(coords,1)
        it will give
        'POLYGON ((24.950899 60.169158 0,24.953492 60.169158 0,24.95351 60.170104 0,24.950958 60.16999 0))'
        while 
        NewGeometry = gpd.GeoDataFrame()
        NewGeometry.loc[0,'geometry'] = GIS.CreatePolygon(coordinates,2)
        then 
        NewGeometry.loc[0,'geometry'] 
        will draw an object 
    """
    if Type == 1:
        # create a ring          
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for coord in coords:
            ring.AddPoint(np.double(coord[0]), np.double(coord[1]))
        
        # Create polygon
        poly = ogr.Geometry(ogr.wkbPolygon)
        
        poly.AddGeometry(ring)
        return poly.ExportToWkt()
    else:
        poly = Polygon(coords)
        return poly



def CreatePoint(coords):
    """
    =============================================
        CreatePoint(coords)
    =============================================
    CreatePoint takes a list of tuples of coordinates and convert it into
    a list of Shapely point object
    
    Inputs:
        1-coords:
        [List] list of tuples [(x1,y1),(x2,y2)] or [(long1,lat1),(long2,lat1)]
        
    Outputs:
        1-points:
        [List] list of Shaply point objects [Point,Point]
        
    Examples:
        coordinates = [(24.950899, 60.169158), (24.953492, 60.169158), (24.953510, 60.170104), (24.950958, 60.169990)]
        PointList = GIS.CreatePoint(coordinates)
        # to assign these objects to a geopandas dataframe 
        # NopreviousGeoms is the number of geometries already exists in the 
        # geopandas dataframe
        NopreviousGeoms = 5
        for i in range(NopreviousGeoms,NopreviousGeoms+len(PointList)):
            NewGeometry.loc[i,'geometry'] = PointList[i-NopreviousGeoms]
    """
    points = list()
    for i in range(len(coords)):
        points.append(Point(coords[i]))
    
    return points


def CombineGeometrics(Path1,Path2, Save=False, SavePath= None):
    """
    ============================================
        CombineGeometrics(Path1,Path2)
    ============================================
    CombineGeometrics reads two shapefiles and combine them into one 
    shapefile
    
    Inputs:
        1-Path1:
            [String] a path includng the name of the shapefile and extention like 
            path="data/subbasins.shp"
            
        2-Path2:
            [String] a path includng the name of the shapefile and extention like 
            path="data/subbasins.shp"
        3-Save:
            [Boolen] True if you want to save the result shapefile in a certain
            path "SavePath"
        3-SavePath:
            [String] a path includng the name of the shapefile and extention like 
            path="data/subbasins.shp"
            
    Output:
        1-SaveIng the shapefile or NewGeoDataFrame :
            If you choose True in the "Save" input the function will save the 
            shapefile in the given "SavePath" 
            If you choose False in the "Save" input the function will return a 
            [geodataframe] dataframe containing both input shapefiles 
            you can save it as a shapefile using 
            NewDataFrame.to_file("Anyname.shp")
        
    Example:
        1- Return a geodata frame
            RIMSubPath = "Inputs/RIM_sub.shp"
            AddSubsPath = "Inputs/addSubs.shp"
            NewDataFrame = GIS.CombineGeometrics(RIMSubPath,AddSubsPath, Save=False)
        2- Save a shapefile
            RIMSubPath = "Inputs/RIM_sub.shp"
            AddSubsPath = "Inputs/addSubs.shp"
            GIS.CombineGeometrics(RIMSubPath,AddSubsPath, Save=True, SavePath = "AllBasins.shp")
    """
    assert type(Path1) == str, "Path1 input should be string type"
    assert type(Path2) == str, "Path2 input should be string type"
    assert type(Save) == bool, "SavePath input should be string type"
    
    # input values
    ext = Path1[-4:]
    assert ext == ".shp", "please add the extension at the end of the Path1"
    ext = Path2[-4:]
    assert ext == ".shp", "please add the extension at the end of the Path2"
    if Save == True:
        assert type(SavePath) == str, "SavePath input should be string type"
        ext = SavePath[-4:]
        assert ext == ".shp", "please add the extension at the end of the SavePath"
    
    # read shapefiles
    GeoDataFrame1 = gpd.read_file(Path1)
    GeoDataFrame2 = gpd.read_file(Path2)
    
    # concatenate the second shapefile into the first shapefile
    NewGeoDataFrame = gpd.GeoDataFrame(pd.concat([GeoDataFrame1,GeoDataFrame2]))
    # re-index the data frame
    NewGeoDataFrame.index = [i for i in range(len(NewGeoDataFrame))]
    # take the spatial reference of the first geodataframe
    NewGeoDataFrame.crs = GeoDataFrame1.crs
    if Save == True:
        NewGeoDataFrame.to_file(SavePath)
    else:
        return NewGeoDataFrame
    
    
    
def AddSpatialReference(GpdDF, epsg):
    """
    =======================================================
        AddSpatialReference(GpdDF, epsg)
    =======================================================
    AddSpatialReference takes GeoPandas DataFrame and set the coordinate system
    based on the given epsg input
    
    Inputs:
        1-GpdDF:
            [geopandas.geodataframe.GeoDataFrame] geopandas dataframe 
        2-epsg:
            [integer] EPSG stands for European Petroleum Survey Group and is an organization 
            that maintains a geodetic parameter database with standard codes,
            the EPSG codes, for coordinate systems, datums, spheroids, units 
            and such alike (https://epsg.io/) default value is [None].
    
    Outputs:
        1-GpdDF: 
            [geopandas.geodataframe.GeoDataFrame] the same input geopandas 
            dataframe but with spatial reference
            
    Examples:
        NewGeometry = gpd.GeoDataFrame()
        coordinates = [(24.950899, 60.169158), (24.953492, 60.169158), (24.953510, 60.170104), (24.950958, 60.169990)]
        NewGeometry.loc[0,'geometry'] = GIS.CreatePolygon(coordinates,2)
        # adding spatial reference system 
        NewGeometry.crs = from_epsg(4326)        
        # to check the spatial reference 
        NewGeometry.crs
        the you will get 
        {'init': 'epsg:4326', 'no_defs': True}
    """
    
    GpdDF.crs = from_epsg(epsg)
    
    return GpdDF
    
def PolygonCenterPoint(PolygonDataFrame, Save=False, SavePath=None):
    """
    ======================================================================
        PolygonCenterPoint(PolygonDataFrame, Save=False, SavePath)
    ======================================================================
    PolygonCenterPoint function takes the a geodata frame of polygons and and 
    returns the center of each polygon
    
    Inputs:
        1-PolygonDataFrame:
            [geopandas.geodataframe.GeoDataFrame] GeoDataframe containing 
            all the polygons you want to get the center point
        3-Save:
            [Boolen] True if you want to save the result shapefile in a certain
            path "SavePath"
        3-SavePath:
            [String] a path includng the name of the shapefile and extention like 
            path="data/subbasins.shp"
    Outputs:
        1-SaveIng the shapefile or CenterPointDataFrame :
            If you choose True in the "Save" input the function will save the 
            shapefile in the given "SavePath" 
            If you choose False in the "Save" input the function will return a 
            [geodataframe] dataframe containing CenterPoint DataFrame 
            you can save it as a shapefile using 
            CenterPointDataFrame.to_file("Anyname.shp")        
        
        
    Example:
        1- Return a geodata frame
            RIMSubPath = "Inputs/RIM_sub.shp"
            RIMSub = gpd.read_file(RIMSubPath)
            CenterPointDataFrame = GIS.PolygonCenterPoint(RIMSub, Save=False)
        2- Save a shapefile
            RIMSubPath = "Inputs/RIM_sub.shp"
            RIMSub = gpd.read_file(RIMSubPath)
            GIS.PolygonCenterPoint(RIMSub, Save=True, SavePath = "centerpoint.shp")
            
    """
    assert type(PolygonDataFrame) == gpd.geopandas.geodataframe.GeoDataFrame, "PolygonDataFrame input should be GeoDataFrame type"
    assert type(Save) == bool, "SavePath input should be string type"
    
    # input values
    if Save == True:
        assert type(SavePath) == str, "SavePath input should be string type"
        ext = SavePath[-4:]
        assert ext == ".shp", "please add the extension at the end of the SavePath"
    
    # get the X, Y coordinates of the points of the polygons and the multipolygons
    PolygonDataFrame = XY(PolygonDataFrame)
    
    # re-index the data frame
    PolygonDataFrame.index = [i for i in range(len(PolygonDataFrame))]
    # calculate the average X & Y coordinate for each geometry in the shapefile
    for i in range(len(PolygonDataFrame)):
        PolygonDataFrame.loc[i,'AvgX'] = np.mean(PolygonDataFrame.loc[i,'x'])
        PolygonDataFrame.loc[i,'AvgY'] = np.mean(PolygonDataFrame.loc[i,'y'])
    
    # create a new geopandas dataframe of points that is in the middle of each 
    # sub-basin
    PolygonDataFrame = PolygonDataFrame.drop(['geometry','x','y'],axis=1)
    
    MiddlePointdf = gpd.GeoDataFrame()
#    MiddlePointdf = PolygonDataFrame
    
    MiddlePointdf['geometry'] = None
    # create a list of tuples of the coordinates (x,y) or (long, lat)
    # of the points 
    CoordinatesList = zip(PolygonDataFrame['AvgX'].tolist(),PolygonDataFrame['AvgY'].tolist())
    PointsList = CreatePoint(CoordinatesList)
    # set the spatial reference
    MiddlePointdf['geometry'] = PointsList
    MiddlePointdf.crs = PolygonDataFrame.crs    
    MiddlePointdf[PolygonDataFrame.columns.tolist()] = PolygonDataFrame[PolygonDataFrame.columns.tolist()]
    
    if Save == True:
        MiddlePointdf.to_file(SavePath)
    else:
        return MiddlePointdf


    

def WriteShapefile(poly, out_shp):
    """
    =====================================================================
       write_shapefile(poly, out_shp):
    =====================================================================
    this function takes a polygon geometry and creates a ashapefile and save it 
    (https://gis.stackexchange.com/a/52708/8104)
    
    inputs:
    ----------
        1-geometry:
            polygon, point, or lines or multi
        2-path:
            string, of the path and name of the shapefile
    
    outputs:
    ----------
        1-saving the shapefile to the path
    """
    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(out_shp)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    
    ## If there are multiple geometries, put the "for" loop here
    
    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)
    
    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(poly)
    #geom = ogr.CreateGeometryFromWkb(poly.wkb)
    feat.SetGeometry(geom)
    
    layer.CreateFeature(feat)
    feat = geom = None  # destroy these
    
    # Save and close everything
    ds = layer = feat = geom = None


#def ClipRasterWithPolygon(Raster_path,shapefile_path,save=False,output_path=None):
#    """
#    =========================================================================
#      ClipRasterWithPolygon(Raster_path, shapefile_path, output_path)
#    =========================================================================
#    this function clip a raster using polygon shapefile
#    
#    inputs:
#    ----------
#        1- Raster_path:
#            [String] path to the input raster including the raster extension (.tif)
#        2- shapefile_path:
#            [String] path to the input shapefile including the shapefile extension (.shp)
#        3-save:
#            [Boolen] True or False to decide whether to save the clipped raster or not
#            default is False
#        3- output_path:
#            [String] path to the place in your drive you want to save the clipped raster 
#            including the raster name & extension (.tif), default is None 
#    
#    Outputs:
#    ----------
#        1- projected_raster:
#            [gdal object] clipped raster 
#        2- if save is True function is going to save the clipped raster to the output_path
#    
#    EX:
#    ----------
#        import plotting_function as plf
#        Raster_path = r"data/Evaporation_ERA-Interim_2009.01.01.tif"
#        shapefile_path ="data/"+"Outline.shp"
#        clipped_raster=plf.ClipRasterWithPolygon(Raster_path,shapefile_path)
#        or 
#        output_path = r"data/cropped.tif"
#        clipped_raster=ClipRasterWithPolygon(Raster_path,shapefile_path,True,output_path)
#    """
#    # input data validation
#    # type of inputs
#    assert type(Raster_path)== str, "Raster_path input should be string type"
#    assert type(shapefile_path)== str, "shapefile_path input should be string type"
#    assert type(save)== bool , "save input should be bool type (True or False)"
#    if save == True:
#        assert type(output_path)== str," pleaase enter a path to save the clipped raster"
#    # inputs value
#    if save == True:
#        ext=output_path[-4:]
#        assert ext == ".tif", "please add the extention at the end of the output_path input"
#    
#    raster=gdal.Open(Raster_path)
#    proj=raster.GetProjection()
#    src_epsg=osr.SpatialReference(wkt=proj)
#    gt=raster.GetGeoTransform()
#    
#    # first check if the crs is GCS if yes check whether the long is greater than 180
#    # geopandas read -ve longitude values if location is west of the prime meridian 
#    # while rasterio and gdal not 
#    if src_epsg.GetAttrValue('AUTHORITY',1) == "4326" and gt[0] > 180:
#        # reproject the raster to web mercator crs 
#        raster=reproject_dataset(raster)
#        out_transformed = os.environ['Temp']+"/transformed.tif"    
#        # save the raster with the new crs
#        SaveRaster(raster,out_transformed)
#        raster = rasterio.open(out_transformed)
#    else:
#        # crs of the raster was not GCS or longitudes are less than 180
#        raster = rasterio.open(Raster_path)
#    
#    ### Cropping the raster with the shapefile
#    # read the shapefile
#    shpfile=gpd.read_file(shapefile_path)
#    # Re-project into the same coordinate system as the raster data
#    shpfile= shpfile.to_crs(crs=raster.crs.data)
#    
#    def getFeatures(gdf):
#        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
#        import json
#        return [json.loads(gdf.to_json())['features'][0]['geometry']]
#    
#    # Get the geometry coordinates by using the function.
#    coords=getFeatures(shpfile)
#    
#    out_img, out_transform = rasterio.mask.mask(raster=raster, shapes=coords, crop=True)
#    
#    # copy the metadata from the original data file.
#    out_meta = raster.meta.copy()
#    
#    # Next we need to parse the EPSG value from the CRS so that we can create
#    #a Proj4 string using PyCRS library (to ensure that the projection information is saved correctly).
#    #epsg_code=int(raster.crs.data['init'][5:])
#    
#    # Now we need to update the metadata with new dimensions, transform (affine) and CRS (as Proj4 text)
#    out_meta.update({"driver": "GTiff",
#                         "height": out_img.shape[1],
#                         "width": out_img.shape[2],
#                         "transform": out_transform,
#    #                     "crs": pycrs.parser.from_epsg_code(epsg_code).to_proj4()
#                    }
#                   )
#
#    # save the clipped raster.
#    temp_path = os.environ['Temp'] + "/cropped.tif"
#    with rasterio.open(temp_path,"w", **out_meta) as dest:
#        dest.write(out_img)
#
#    # close the transformed raster
#    raster.close()
#    # delete the transformed raster
#    os.remove(out_transformed)
#    # read the clipped raster
#    raster=gdal.Open(temp_path)
#    # reproject the clipped raster back to its original crs
#    projected_raster=project_raster(raster,int(src_epsg.GetAttrValue('AUTHORITY',1)))
#    # close the clipped raster
#    raster=None
#    
#    # delete the clipped raster
#    os.remove(temp_path)
#    # write the raster to the file
#    if save:
#        SaveRaster(projected_raster,output_path)
#    
#    return projected_raster



def ReadASCIIsFolder(path, pixel_type):
    """
    ===========================================================
       ReadASCIIsFolder(path, pixel_type)
    ===========================================================
    this function reads rasters from a folder and creates a 3d arraywith the same
    2d dimensions of the first raster in the folder and len as the number of files
    inside the folder.
    - all rasters should have the same dimensions
    - folder should only contain raster files
    
    Inputs:
    ----------
        1- path:
            [String] path of the folder that contains all the rasters.
    
    Outputs:
    ----------
        1- arr_3d:
            [numpy.ndarray] 3d array contains arrays read from all rasters in the folder.
            
        2-ASCIIDetails:
            [List] list of the six spatial information of the ASCII file 
            [ASCIIRows, ASCIIColumns, XLowLeftCorner, YLowLeftCorner, 
            CellSize, NoValue]
        3- files:
            [list] list of names of all files inside the folder
    
    Example:
    ----------
        path = "ASCII folder/"
        pixel_type = 1
        ASCIIArray, ASCIIDetails, NameList = ReadASCIIsFolder(path, pixel_type)
        
    """
    # input data validation
    # data type
    assert type(path)== str, "A_path input should be string type"
    # input values
    # check wether the path exist or not 
    assert os.path.exists(path), "the path you have provided does not exist"
    # check whether there are files or not inside the folder
    assert os.listdir(path)!= "","the path you have provided is empty"
    # get list of all files 
    files=os.listdir(path)
    if "desktop.ini" in files: files.remove("desktop.ini")
    # check that folder only contains rasters
    assert all(f.endswith(".asc") for f in files), "all files in the given folder should have .tif extension"
    # create a 3d array with the 2d dimension of the first raster and the len 
    # of the number of rasters in the folder
    ASCIIValues, ASCIIDetails = ReadASCII(path+"/"+files[0], pixel_type)
    noval = ASCIIDetails[5]
    # fill the array with noval data
    arr_3d=np.ones((ASCIIDetails[0],ASCIIDetails[1],len(files)))*noval
    
    for i in range(len(files)):
        # read the tif file
        f,_ = ReadASCII(path+"/"+files[0], pixel_type)
        arr_3d[:,:,i]=f
    
    return arr_3d, ASCIIDetails, files

def ASCIIFoldertoRaster(path,savePath,pixel_type=1,RasterFile = None,epsg = None):
    """  
    =========================================================================
    ASCIItoRaster(path,savePath,pixel_type)
    =========================================================================
     
    This function takes the path of a folder contains ASCII files and convert 
    them into a raster format and in takes  all the spatial information 
    (projection, coordinates of the corner point), and number of rows 
    and columns from raster file or you have to define the epsg corresponding 
    to the you coordinate system and projection
    
    Inputs:
    =========
    
        1-path:
            [String] path to the folder containing the ASCII files
    
        2-savePath:
            [String] path to save the new raster including new raster name and extension (.tif)
            
        3-pixel_type:
            [Integer] type of the data to be stored in the pixels,default is 1 (float32)
            for example pixel type of flow direction raster is unsigned integer
            1 for float32
            2 for float64
            3 for Unsigned integer 16
            4 for Unsigned integer 32
            5 for integer 16
            6 for integer 32
    
        4-RasterFile:
            [String] source raster to get the spatial information, both ASCII
            file and source raster should have the same number of rows, and 
            same number of columns default value is [None].
         
        5-epsg: 
            EPSG stands for European Petroleum Survey Group and is an organization 
            that maintains a geodetic parameter database with standard codes,
            the EPSG codes, for coordinate systems, datums, spheroids, units 
            and such alike (https://epsg.io/) default value is [None].
                
    Outputs:
    =========
        1- a New Raster will be saved in the savePath containing the values 
        of the ASCII file
        
    Example:
    =========
        1- ASCII to raster given a raster file:
            ASCIIFile = "soiltype.asc"
            RasterFile = "DEM.tif"
            savePath = "Soil_raster.tif"
            pixel_type = 1
            ASCIItoRaster(ASCIIFile,  savePath, pixel_type, RasterFile)
        2- ASCII to Raster given an EPSG number
            ASCIIFile = "soiltype.asc"
            savePath = "Soil_raster.tif"
            pixel_type = 1
            epsg = 4647
        ASCIIFoldertoRaster(path,savePath,pixel_type=5,epsg = epsg)
    """
    
    # input data validation
    # data type
    assert type(path)== str, "A_path input should be string type"
    # input values
    # check wether the path exist or not 
    assert os.path.exists(path), "the path you have provided does not exist"
    # check whether there are files or not inside the folder
    assert os.listdir(path)!= "","the path you have provided is empty"
    # get list of all files 
    files=os.listdir(path)
    if "desktop.ini" in files: files.remove("desktop.ini")
    # check that folder only contains rasters
    assert all(f.endswith(".asc") for f in files), "all files in the given folder should have .tif extension"
    # create a 3d array with the 2d dimension of the first raster and the len 
    # of the number of rasters in the folder
    
    for i in range(len(files)):
            ASCIIFile = path + "/" + files[i]
            name = savePath + "/" + files[i].split(".")[0] + ".tif"
            ASCIItoRaster(ASCIIFile,name,pixel_type,RasterFile = None,epsg = epsg)


def RastersLike(src,array,path=None):
    """
    ====================================================================
      RasterLike(src,array,path)
    ====================================================================
    this function creates a Geotiff raster like another input raster, new raster 
    will have the same projection, coordinates or the top left corner of the original
    raster, cell size, nodata velue, and number of rows and columns
    the raster and the dem should have the same number of columns and rows
    
    inputs:
    ----------
        1- src:
            [gdal.dataset] source raster to get the spatial information
        2- array:
            [numpy array] 3D array to be stores as a rasters, the dimensions should be 
            [rows, columns, timeseries length]
        3- path:
            [String] list of names to save the new rasters
            like ["results/surfaceDischarge_2012_08_13_23.tif","results/surfaceDischarge_2012_08_14_00.tif"]
            Default value is None
    
    outputs:
    ----------
        1- save the new raster to the given path
    
    Ex:
    ----------
        data
        src=gdal.Open("DEM.tif")
        name=["Q_2012_01_01_01.tif","Q_2012_01_01_02.tif","Q_2012_01_01_03.tif","Q_2012_01_01_04.tif"]
        RastersLike(src,data,name)
    
    """
    # input data validation
    # length of the 3rd dimension of the array
    try:
        l=np.shape(array)[2]
    except IndexError:
        assert 5==1, "the array you have entered is 2D you have to use RasterLike function not RastersLike"
        
    # check length of the list of names to be equal to 3rd dimension of the array
    if path != None: # paths are given
        assert len(path)==np.shape(array)[2], "length of list of names should equal the 3d dimension of the array"
    else: # paths are not given
        # try to create a folder called results at the current working directory to store resulted rasters
        try:
            os.makedirs(os.path.join(os.getcwd(),"result_rasters"))
        except WindowsError:
            assert 5==1 ,"please either to provide your own paths including folder name and rasternames.tif in a list or rename the folder called result_rasters"
        # careate list of names
        path=["result_rasters/"+str(i)+".tif" for i in range(l)]
        
    for i in range(l):
        RasterLike(src,array[:,:,i],path[i])
        
    
def MatchDataAlignment(A_path,B_input_path,new_B_path):
    """
    =========================================================================
      MatchData(A_path,B_input_path,new_B_path)
    =========================================================================
    this function matches the coordinate system and the number of of rows & columns
    between two rasters
    Raster A is the source of the coordinate system, no of rows and no of columns & cell size
    B_input_path is path to the folder where Raster B exist where  Raster B is 
    the source of data values in cells 
    the result will be a raster with the same structure like RasterA but with 
    values from RasterB using Nearest Neighbour interpolation algorithm
    
    Inputs:
    ----------
        1- A_path:
            [String] path to the spatial information source raster to get the spatial information 
            (coordinate system, no of rows & columns) A_path should include the name of the raster 
            and the extension like "data/dem.tif"
        2- B_input_path:
            [String] path of the folder of the rasters (Raster B) you want to adjust their 
            no of rows, columns and resolution (alignment) like raster A 
            the folder should not have any other files except the rasters
        3- new_B_path:
            [String] [String] path where new rasters are going to be saved with exact 
            same old names
    
    Outputs:
    ----------
        1- new rasters:
            ٌRasters have the values from rasters in B_input_path with the same 
        cell size, no of rows & columns, coordinate system and alignment like raster A
    
    Example:
    ----------
        dem_path = "01GIS/inputs/4000/acc4000.tif"
        prec_in_path = "02Precipitation/CHIRPS/Daily/"
        prec_out_path = "02Precipitation/4km/"
        MatchData(dem_path,prec_in_path,prec_out_path)
    """
    # input data validation
    # data type
    assert type(A_path)== str, "A_path input should be string type"
    assert type(B_input_path)== str, "B_input_path input should be string type"
    assert type(new_B_path)== str, "new_B_path input should be string type"
    # input values
    ext=A_path[-4:]
    assert ext == ".tif", "please add the extension at the end of the path input"
    
    A=gdal.Open(A_path)
    files_list=os.listdir(B_input_path)
    if "desktop.ini" in files_list: files_list.remove("desktop.ini")
    
    for i in range(len(files_list)):
        B=gdal.Open(B_input_path+files_list[i])
        new_B=MatchRasterAlignment(A,B)
        SaveRaster(new_B,new_B_path+files_list[i])


def MatchDataNoValuecells(A_path,B_input_path,new_B_path):
    """
    ==============================================================
      MatchData(A_path,B_input_path,new_B_path)
    ==============================================================
    this function matches the location of nodata value from src raster to dst 
    raster
    Raster A is where the NoDatavalue will be taken and the location of this value 
    B_input_path is path to the folder where Raster B exist where  we need to put 
    the NoDataValue of RasterA in RasterB at the same locations
    
    Inputs:
    ----------
        1- A_path:
            [String] path to the source raster to get the NoData value and it location in the array
            A_path should include the name of the raster and the extension like "data/dem.tif"
        2- B_input_path:
            [String] path of the folder of the rasters (Raster B) you want to set Nodata Value 
            on the same location of NodataValue of Raster A, the folder should 
            not have any other files except the rasters
        3- new_B_path:
            [String] [String] path where new rasters are going to be saved with exact 
            same old names
    
    Outputs:
    ----------
        1- new rasters have the values from rasters in B_input_path with the NoDataValue in the same 
        locations like raster A
    
    Example:
    ----------
        dem_path="01GIS/inputs/4000/acc4000.tif"
        temp_in_path="03Weather_Data/new/4km/temp/"
        temp_out_path="03Weather_Data/new/4km_f/temp/"
        MatchDataNoValuecells(dem_path,temp_in_path,temp_out_path)

    """
    # input data validation
    # data type
    assert type(A_path)== str, "A_path input should be string type"
    assert type(B_input_path)== str, "B_input_path input should be string type"
    assert type(new_B_path)== str, "new_B_path input should be string type"
    # input values
    ext=A_path[-4:]
    assert ext == ".tif", "please add the extension at the end of the path input"
    # check wether the path exists or not
    assert os.path.exists(A_path), "source raster you have provided does not exist"
    assert os.path.exists(B_input_path), B_input_path+" path you have provided does not exist"
    assert os.path.exists(new_B_path), new_B_path+" path you have provided does not exist"
    # check wether the folder has the rasters or not 
    assert len(os.listdir(B_input_path)) > 0, B_input_path+" folder you have provided is empty"
    
    A=gdal.Open(A_path)
    files_list=os.listdir(B_input_path)
    if "desktop.ini" in files_list:  files_list.remove("desktop.ini")
    
    for i in range(len(files_list)):
        B=gdal.Open(B_input_path+files_list[i])
        new_B=MatchNoDataValue(A,B) 
        SaveRaster(new_B,new_B_path+files_list[i])
        
def FolderCalculator(folder_path,new_folder_path,function):
    """
    =========================================================================
      FolderCalculator(folder_path, new_folder_path, function)
    =========================================================================
    this function matches the location of nodata value from src raster to dst 
    raster
    Raster A is where the NoDatavalue will be taken and the location of this value 
    B_input_path is path to the folder where Raster B exist where  we need to put 
    the NoDataValue of RasterA in RasterB at the same locations
    
    Inputs:
    ----------
        1- folder_path:
            [String] path of the folder of rasters you want to execute a certain function on all
            of them
        2- new_folder_path:
            [String] path of the folder where resulted raster will be saved
        3- function:
            [function] callable function (builtin or user defined)
    
    Outputs:
    ----------
        1- new rasters will be saved to the new_folder_path
    
    Example:
    ----------
        def function(args):
            A = args[0]
            func=np.abs
            path = args[1]
            B=MapAlgebra(A,func)
            SaveRaster(B,path)
        
        folder_path = "03Weather_Data/new/4km_f/evap/"
        new_folder_path="03Weather_Data/new/4km_f/new_evap/"
        FolderCalculator(folder_path,new_folder_path,function)
    """
    # input data validation
    # data type
    assert type(folder_path)== str, "A_path input should be string type"
    assert type(new_folder_path)== str, "B_input_path input should be string type"
    assert callable(function) , "second argument should be a function"
    
    # get names of rasters
    files_list=os.listdir(folder_path)
    if "desktop.ini" in files_list: files_list.remove("desktop.ini")
    
    # execute the function on each raster
    for i in range(len(files_list)):
        B=gdal.Open(folder_path+files_list[i])
        args=[B,new_folder_path+files_list[i]]
        function(args)

def ReadRastersFolder(path):
    """
    ===========================================================
       ReadRastersFolder(path)
    ===========================================================
    this function reads rasters from a folder and creates a 3d arraywith the same
    2d dimensions of the first raster in the folder and len as the number of files
    inside the folder.
    - all rasters should have the same dimensions
    - folder should only contain raster files
    
    Inputs:
    ----------
        1- path:
            [String] path of the folder that contains all the rasters.
    
    Outputs:
    ----------
        1- arr_3d:
            [numpy.ndarray] 3d array contains arrays read from all rasters in the folder.
    
    Example:
    ----------
        prec_path="00inputs/meteodata/4000/prec"
        prec=ReadRastersFolder(prec_path)
        
    """
    # input data validation
    # data type
    assert type(path)== str, "A_path input should be string type"
    # input values
    # check wether the path exist or not 
    assert os.path.exists(path), "the path you have provided does not exist"
    # check whether there are files or not inside the folder
    assert os.listdir(path)!= "","the path you have provided is empty"
    # get list of all files 
    files=os.listdir(path)
    if "desktop.ini" in files: files.remove("desktop.ini")
    
    # to sort the files in the same order as the first number in the name
    filesNo = [int(files[i].split("_")[0]) for i in range(len(files))]
    filetuple = sorted(zip(filesNo, files))
    files = [x for _,x in filetuple]
    
    # check that folder only contains rasters
    assert all(f.endswith(".tif") for f in files), "all files in the given folder should have .tif extension"
    # create a 3d array with the 2d dimension of the first raster and the len 
    # of the number of rasters in the folder
    sample=gdal.Open(path+"/"+files[0])
    dim=sample.ReadAsArray().shape
    naval=sample.GetRasterBand(1).GetNoDataValue()
    # fill the array with noval data
    arr_3d=np.ones((dim[0],dim[1],len(files)))*naval
    
    for i in range(len(files)):
        # read the tif file
        f=gdal.Open(path+"/"+files[i])
        arr_3d[:,:,i]=f.ReadAsArray()
    
    return arr_3d




def GetXYCoords(geometry, coord_type):
    """
    ====================================================
        getXYCoords(geometry, coord_type)
    ====================================================
    Returns either x or y coordinates from  geometry coordinate sequence.
     Used with LineString and Polygon geometries.
     inputs:
         1- geometry:
              [Geometry object of type LineString] the geometry of a shpefile
         2- coord_type:
             [string] either "x" or "y"
     outpus:
         1-array: 
             contains x coordinates or y coordinates of all edges of the shapefile
    """
    if coord_type=="x":
        return geometry.coords.xy[0]
    elif coord_type=="y":
        return geometry.coords.xy[1]


def GetPointCoords(geometry,coord_type):
    """
    ========================================================
        GetPointCoords(geometry,coord_type)
    ========================================================
    Returns Coordinates of Point object.
    inputs:
        1- geometry:
            [Geometry object of type point] the geometry of a shpefile
        2- coord_type:
            [string] either "x" or "y"
    outpus:
        1-array: 
            contains x coordinates or y coordinates of all edges of the shapefile
    """
    if coord_type=="x":
        return geometry.x
    if coord_type=="y":
        return geometry.y


def GetLineCoords(geometry,coord_type):
    """
    ====================================================
        getLineCoords(geometry)
    ====================================================
    Returns Coordinates of Linestring object.
    inputs:
        1- geometry:
             [Geometry object of type type Linestring] the geometry of a shpefile
        2- coord_type:
            [string] either "x" or "y"
    outpus:
        1-array: 
            contains x coordinates or y coordinates of all edges of the shapefile
    """
    return GetXYCoords(geometry,coord_type)

def GetPolyCoords(geometry,coord_type):
    """
    =====================================================
         getPolyCoords(geometry,coord_type)
    =====================================================
    Returns Coordinates of Polygon using the Exterior of the Polygon.
    inputs:
        1- geometry:
         [Geometry object of type type polygon] the geometry of a shpefile
        2- coord_type:
             [string] either "x" or "y"
    outpus:
        1-array: 
            contains x coordinates or y coordinates of all edges of the shapefile
    """
    # convert the polygon into lines
    ext=geometry.exterior # type = LinearRing

    return GetXYCoords(ext,coord_type)

    
#def multiGeomHandler(multi_geometry, coord_type, geom_type):
#    """
#    # =============================================================================
#    #     multiGeomHandler(multi_geometry, coord_type, geom_type)
#    # =============================================================================
#    Function for handling multi-geometries. Can be MultiPoint, MultiLineString or MultiPolygon.
#    Returns a list of coordinates where all parts of Multi-geometries are merged into a single list.
#    Individual geometries are separated with np.nan which is how Bokeh wants them.
#    # Bokeh documentation regarding the Multi-geometry issues can be found here (it is an open issue)
#    # https://github.com/bokeh/bokeh/issues/2321
#    
#    inputs:
#        1- multi_geometry (geometry)
#         the geometry of a shpefile
#        2- coord_type (string)
#         "string" either "x" or "y"
#        3- geom_type (string)
#            "MultiPoint" or "MultiLineString" or "MultiPolygon"
#    outpus:
#        1-array: 
#         contains x coordinates or y coordinates of all edges of the shapefile    
#    """
#    for i,part in enumerate(multi_geometry):
#        # On the first part of the Multi-geometry initialize the coord_array (np.array)
#        if i ==0:
#            if geom_type=="MultiPoint":
#                coord_arrays= getPointCoords(part, coord_type)#,np.nan)
#            elif geom_type=="MultiLineString":
##                coord_arrays= np.append(getLineCoords(part,coord_type))#,np.nan)
#                coord_arrays= getLineCoords(part,coord_type)
#            elif geom_type=="MultiPolygon":
#                coord_arrays= 999 #getPolyCoords(part,coord_type)#,np.nan)
#        else:
#            if geom_type=="MultiPoint":
#                coord_arrays= np.concatenate([coord_arrays,getPointCoords(part, coord_type)]) #,np.nan
#            elif geom_type=="MultiLineString":
#                coord_arrays= np.concatenate([coord_arrays,getLineCoords(part,coord_type)]) #,np.nan
#            elif geom_type=="MultiPolygon":
#                coord_arrays= 999 #np.concatenate([coord_arrays,getPolyCoords(part,coord_type)]) #,np.nan
#        # return the coordinates 
#        return coord_arrays

def Explode(dataframe_row):
    """
    ==============================================
        explode(indata)
    ==============================================
    explode function converts the multipolygon into a polygons
    Inputs:
        1- dataframe_row: (data frame series)
            the dataframe row that its geometry type is Multipolygon
    outputs:
        1- outdf
            the dataframe of the created polygons 
    """
    row = dataframe_row
    outdf = gpd.GeoDataFrame() #columns=dataframe_row.columns
#    for idx, row in enumerate(dataframe_row):#dataframe_row.iterrows():
#            if type(row.geometry) == Polygon:
#                outdf = outdf.append(row,ignore_index=True)
#            if type(row.geometry) == MultiPolygon:
    multdf = gpd.GeoDataFrame() #columns=dataframe_row.columns
#    recs = len(row.geometry)
    recs = len(row)
    multdf = multdf.append([row]*recs,ignore_index=True)
    for geom in range(recs):
        multdf.loc[geom,'geometry'] = row.geometry[geom]
    outdf = outdf.append(multdf,ignore_index=True)


def multiGeomHandler(multi_geometry, coord_type, geom_type):
    """
    # =============================================================================
    #     multiGeomHandler(multi_geometry, coord_type, geom_type)
    # =============================================================================
    Function for handling multi-geometries. Can be MultiPoint, MultiLineString or MultiPolygon.
    Returns a list of coordinates where all parts of Multi-geometries are merged into a single list.
    Individual geometries are separated with np.nan which is how Bokeh wants them.
    # Bokeh documentation regarding the Multi-geometry issues can be found here (it is an open issue)
    # https://github.com/bokeh/bokeh/issues/2321
    
    inputs:
        1- multi_geometry (geometry)
         the geometry of a shpefile
        2- coord_type (string)
         "string" either "x" or "y"
        3- geom_type (string)
            "MultiPoint" or "MultiLineString" or "MultiPolygon"
    outpus:
        1-array: 
         contains x coordinates or y coordinates of all edges of the shapefile    
    """
    if geom_type=="MultiPoint" or geom_type== "MultiLineString":
        for i,part in enumerate(multi_geometry):
            # On the first part of the Multi-geometry initialize the coord_array (np.array)
            if i ==0:
                if geom_type=="MultiPoint":
                    coord_arrays= GetPointCoords(part, coord_type)#,np.nan)
                elif geom_type=="MultiLineString":
    #                coord_arrays= np.append(getLineCoords(part,coord_type))#,np.nan)
                    coord_arrays= GetLineCoords(part,coord_type)
            else:
                if geom_type=="MultiPoint":
                    coord_arrays= np.concatenate([coord_arrays,GetPointCoords(part, coord_type)]) #,np.nan
                elif geom_type=="MultiLineString":
                    coord_arrays= np.concatenate([coord_arrays,GetLineCoords(part,coord_type)]) #,np.nan
            
    elif geom_type=="MultiPolygon":
        if i ==0:
#            coord_arrays= getPolyCoords(part,coord_type)#,np.nan)
            multi_2_single=Explode(multi_geometry)
            for j in range(len(multi_2_single)):
                if j ==0:
                    coord_arrays= GetPolyCoords(multi_2_single[j],coord_type)#,np.nan)
                else: 
                    coord_arrays= np.concatenate([coord_arrays,GetPolyCoords(multi_2_single[j],coord_type)]) #,np.nan
        else:
            # explode the multipolygon into polygons 
            multi_2_single=Explode(part)
            for j in range(len(multi_2_single)):
                coord_arrays= np.concatenate([coord_arrays,GetPolyCoords(multi_2_single[j],coord_type)]) #,np.nan
        # return the coordinates 
        return coord_arrays


def GetCoords(row, geom_col, coord_type):
    """
    ======================================================
        getCoords(row, geom_col, coord_type)
    ======================================================
    Returns coordinates ('x' or 'y') of a geometry (Point, LineString or Polygon)
    as a list (if geometry is Points, LineString or Polygon). Can handle also 
    MultiGeometries but not MultiPolygon.
    
    inputs:
        1- row:
            [dataframe] a whole rwo of the dataframe
        2- geom_col"
            [string] name of the column where the geometry is stored in the dataframe
        3- coord_type:
            [string] "X" or "Y" choose which coordinate toy want to get from 
            the function
            
    outpus:
        1-array: 
         contains x coordinates or y coordinates of all edges of the shapefile    
    """
    # get geometry object 
    geom=row[geom_col]
    # check the geometry type
    gtype=geom.geom_type
    # "Normal" geometries 
    if gtype=="Point":
        return GetPointCoords(geom,coord_type)  
    elif gtype=="LineString":
        return list(GetLineCoords(geom,coord_type))
    elif gtype=="Polygon":    
        return list(GetPolyCoords(geom,coord_type))
    elif gtype=="MultiPolygon":
        return 999
    # Multi geometries
    else:
        return list(multiGeomHandler(geom,coord_type,gtype))



def XY(input_dataframe):
    """
    ===================================================
      XY(input_dataframe)
    ===================================================
    
    """
    # get the x & y coordinates for all types of geometries except multi_polygon
    input_dataframe['x']=input_dataframe.apply(GetCoords,geom_col="geometry", coord_type="x", axis=1)
    input_dataframe['y']=input_dataframe.apply(GetCoords,geom_col="geometry", coord_type="y", axis=1)
    
    # if the Geometry of type MultiPolygon
    # explode the multi_polygon into polygon
    for idx, row in input_dataframe.iterrows():
    #        if type(row.geometry) == Polygon:
    #            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            # create a new geodataframe
            multdf = gpd.GeoDataFrame() #columns=indf.columns
            # get number of the polygons inside the multipolygon class
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            # for each row assign each polygon
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            input_dataframe= input_dataframe.append(multdf,ignore_index=True)
    
    # get the x & y coordinates of the exploded multi_polygons
    input_dataframe['x']=input_dataframe.apply(GetCoords,geom_col="geometry", coord_type="x", axis=1)
    input_dataframe['y']=input_dataframe.apply(GetCoords,geom_col="geometry", coord_type="y", axis=1)
    
    to_delete=np.where(input_dataframe['x']==999)[0]
    input_dataframe=input_dataframe.drop(to_delete)
    
    return input_dataframe
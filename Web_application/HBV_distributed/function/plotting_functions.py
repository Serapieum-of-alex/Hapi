# -*- coding: utf-8 -*-
"""
Created on Sat May 05 19:47:52 2018

@author: Mostafa
"""

#%library
import numpy as np
import geopandas as gpd
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
# functions
def getXYCoords(geometry, coord_type):
    """
    # =============================================================================
    #     getXYCoords(geometry, coord_type)
    # =============================================================================
    Returns either x or y coordinates from  geometry coordinate sequence.
     Used with LineString and Polygon geometries.
     inputs:
         1- geometry (type LineString)
          the geometry of a shpefile
         2- coord_type
         "string" either "x" or "y"
     outpus:
         1-array: 
             contains x coordinates or y coordinates of all edges of the shapefile
    """
    if coord_type=="x":
        return geometry.coords.xy[0]
    elif coord_type=="y":
        return geometry.coords.xy[1]

def getPointCoords(geometry,coord_type):
    """
    # =============================================================================
    #     getPointCoords(geometry,coord_type)
    # =============================================================================
    Returns Coordinates of Point object.
    inputs:
        1- geometry (type point)
         the geometry of a shpefile
        2- coord_type
         "string" either "x" or "y"
    outpus:
        1-array: 
         contains x coordinates or y coordinates of all edges of the shapefile
    """
    if coord_type=="x":
        return geometry.x
    if coord_type=="y":
        return geometry.y

def getLineCoords(geometry,coord_type):
    """
    # =============================================================================
    #     getLineCoords(geometry)
    # =============================================================================
    Returns Coordinates of Linestring object.
    inputs:
        1- geometry (type Linestring )
         the geometry of a shpefile
        2- coord_type
         "string" either "x" or "y"
    outpus:
        1-array: 
         contains x coordinates or y coordinates of all edges of the shapefile
    """
    return getXYCoords(geometry,coord_type)

def getPolyCoords(geometry,coord_type):
    """
    # =============================================================================
    #  getPolyCoords(geometry,coord_type)
    # =============================================================================
    Returns Coordinates of Polygon using the Exterior of the Polygon.
    inputs:
        1- geometry (type polygon)
         the geometry of a shpefile
        2- coord_type
         "string" either "x" or "y"
    outpus:
        1-array: 
         contains x coordinates or y coordinates of all edges of the shapefile
    """
    # convert the polygon into lines
    ext=geometry.exterior # type = LinearRing

    return getXYCoords(ext,coord_type)

    
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
    for i,part in enumerate(multi_geometry):
        # On the first part of the Multi-geometry initialize the coord_array (np.array)
        if i ==0:
            if geom_type=="MultiPoint":
                coord_arrays= getPointCoords(part, coord_type)#,np.nan)
            elif geom_type=="MultiLineString":
#                coord_arrays= np.append(getLineCoords(part,coord_type))#,np.nan)
                coord_arrays= getLineCoords(part,coord_type)
            elif geom_type=="MultiPolygon":
                coord_arrays= 999 #getPolyCoords(part,coord_type)#,np.nan)
        else:
            if geom_type=="MultiPoint":
                coord_arrays= np.concatenate([coord_arrays,getPointCoords(part, coord_type)]) #,np.nan
            elif geom_type=="MultiLineString":
                coord_arrays= np.concatenate([coord_arrays,getLineCoords(part,coord_type)]) #,np.nan
            elif geom_type=="MultiPolygon":
                coord_arrays= 999 #np.concatenate([coord_arrays,getPolyCoords(part,coord_type)]) #,np.nan
        # return the coordinates 
        return coord_arrays

def getCoords(row, geom_col, coord_type):
    """
    # =============================================================================
    #     getCoords(row, geom_col, coord_type)
    # =============================================================================
    Returns coordinates ('x' or 'y') of a geometry (Point, LineString or Polygon)
    as a list (if geometry is LineString or Polygon). Can handle also MultiGeometries.
    
    inputs:
        1- row (dataframe)
         a whole rwo of the dataframe
        2- geom_col (string)
         name of the column where the geometry is stored in the dataframe
        3- coord_type (string)
            "MultiPoint" or "MultiLineString" or "MultiPolygon"
    outpus:
        1-array: 
         contains x coordinates or y coordinates of all edges of the shapefile    
    """
    # get geometry 
    geom=row[geom_col]
    # check the geometry type
    gtype=geom.geom_type
    # "Normal" geometries 
    if gtype=="Point":
        return getPointCoords(geom,coord_type)  
    elif gtype=="LineString":
        return list(getLineCoords(geom,coord_type))
    elif gtype=="Polygon":    
        return list(getPolyCoords(geom,coord_type))
    elif gtype=="MultiPolygon":
        return 999
    # Multi geometries
    else:
        return list(multiGeomHandler(geom,coord_type,gtype))
    
def XY(input_dataframe):
    # get the x & y coordinates for all types of geometries except multi_polygon
    input_dataframe['x']=input_dataframe.apply(getCoords,geom_col="geometry", coord_type="x", axis=1)
    input_dataframe['y']=input_dataframe.apply(getCoords,geom_col="geometry", coord_type="y", axis=1)
    # explode the multi_polygon into polygon
    for idx, row in input_dataframe.iterrows():
    #        if type(row.geometry) == Polygon:
    #            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame() #columns=indf.columns
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            input_dataframe= input_dataframe.append(multdf,ignore_index=True)
    # get the x & y coordinates of the exploded multi_polygons
    input_dataframe['x']=input_dataframe.apply(getCoords,geom_col="geometry", coord_type="x", axis=1)
    input_dataframe['y']=input_dataframe.apply(getCoords,geom_col="geometry", coord_type="y", axis=1)
    
    to_delete=np.where(input_dataframe['x']==999)[0]
    input_dataframe=input_dataframe.drop(to_delete)
    
    return input_dataframe


def get_mask(raster):
    """
    =============
    get_mask(dem)
    =============
    to create a mask by knowing the stored value inside novalue cells 
    
    Inputs:
        1- flow path lenth raster
    Outputs:
        1- mask:array with all the values in the flow path length raster
        2- no_val: value stored in novalue cells
    """
    no_val = np.float32(raster.GetRasterBand(1).GetNoDataValue()) # get the value stores in novalue cells
    mask = raster.ReadAsArray() # read all values
    return mask, no_val


def reproject_points(lat,lng,from_epsg=4326,to_epsg=3857):
    """
    # =============================================================================
    #  reproject(lat,lng,from_epsg=4326,to_epsg=3857):
    # =============================================================================
    this function change the projection of the coordinates from a coordinate system
    to another (default from GCS to web mercator used by google maps)
    Inputs:
        1- lat: 
            list of latitudes of the points 
        2- lng:
            list of longitude of the points
        3- from_epsg:
            integer reference number to the projection of the points (https://epsg.io/)
        4- to_epsg:
            integer reference number to the new projection of the points (https://epsg.io/)
    outputs:
        1-x:
            list of x coordinates of the points 
        2-y:
            list of y coordinates of the points 
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


def reproject_dataset(src, to_epsg=3857, cell_size=[]):
    """
    # =============================================================================
     reproject_dataset(src, to_epsg=3857, pixel_spacing=[]):
    # =============================================================================
    this function reproject and resample a raster to any projection 
    (default the WGS84 web mercator projection, without resampling)
    The function returns a GDAL in-memory file object, where you can ReadAsArray etc.
    
    inputs:
        1- raster:
            gdal dataset (src=gdal.Open("dem.tif"))
        2-to_epsg:
            integer reference number to the new projection (https://epsg.io/)
            (default 3857 the reference no of WGS84 web mercator )
        3-cell_size:
            integer number to resample the raster cell size to a new cell size
            (default empty so raster will not be resampled)
    Outputs:
        1-raster:
            gdal dataset (you can read it by ReadAsArray)
    """
    from osgeo import gdalconst
    import gdal, osr
    
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
    #src_epsg.GetAttrValue('AUTHORITY',1)
    
    # distination
    # spatial ref
    dst_epsg=osr.SpatialReference()
    dst_epsg.ImportFromEPSG(to_epsg)
    # transformation factors
    tx = osr.CoordinateTransformation(src_epsg,dst_epsg)
    # transform the right upper corner point
    (ulx,uly,ulz) = tx.TransformPoint(src_gt[0], src_gt[3])
    # transform the right lower corner point
    (lrx,lry,lrz)=tx.TransformPoint(src_gt[0]+src_gt[1]*src_x,
                                    src_gt[3]+src_gt[5]*src_y)
    # the result raster has the same pixcel size as the source 
    if cell_size==[]:
        pixel_spacing=src_gt[1]
    else:
        pixel_spacing=cell_size 
        
    # create a new raster
    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",int((lrx-ulx)/pixel_spacing),int((uly-lry)/pixel_spacing),
                       1,gdalconst.GDT_Float32)
    
    # new geotransform 
    new_geo=(ulx,pixel_spacing,src_gt[2],uly,src_gt[4],-pixel_spacing)
    # set the geotransform
    dst.SetGeoTransform(new_geo)
    # set the projection
    dst.SetProjection(dst_epsg.ExportToWkt())
    # set the no data value
    dst.GetRasterBand(1).SetNoDataValue(src.GetRasterBand(1).GetNoDataValue())
    # perform the projection & resampling 
    gdal.ReprojectImage(src,dst,src_epsg.ExportToWkt(),dst_epsg.ExportToWkt(),gdal.GRA_Bilinear)

    return dst


def GCS_distance(coords_1,coords_2):
    """
    =========================================================================
     GCS_distance(coords_1,coords_2):
    =========================================================================
    this function calculates the distance between two points that have 
    geographic coordinate system
    
    inputs:
        1-coord_1:
            tuple of (long, lat) of the first point
        2- coord_2:
            tuple of (long, lat) of the second point
    Output:
        1-distance between the two points 
    """
    import geopy.distance
    
#    coords_1 = (52.2296756, 21.0122287)
#    coords_2 = (52.406374, 16.9251681)
    
    dist=geopy.distance.vincenty(coords_1, coords_2).km
    
    return dist
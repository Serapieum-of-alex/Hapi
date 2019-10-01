# -*- coding: utf-8 -*-
"""
Created on Sun Jul 01 17:07:40 2018

@author: Mostafa
"""
#%links


#%library
import numpy as np
import geopandas as gpd
from shapely.geometry.multipolygon import MultiPolygon

# functions


def getXYCoords(geometry, coord_type):
    """
    ========================================================================
        getXYCoords(geometry, coord_type)
    ========================================================================
    Returns either x or y coordinates from  geometry coordinate sequence.
     Used with LineString and Polygon geometries.
     
     inputs:
     ----------
         1- geometry (type LineString)
          the geometry of a shpefile
         2- coord_type
         "string" either "x" or "y"
     
        outpus:
     ----------
         1-array: 
             contains x coordinates or y coordinates of all edges of the shapefile
    """
    if coord_type=="x":
        return geometry.coords.xy[0]
    elif coord_type=="y":
        return geometry.coords.xy[1]

def getPointCoords(geometry,coord_type):
    """
    =======================================================================
        getPointCoords(geometry,coord_type)
    =======================================================================
    Returns Coordinates of Point object.
    
    inputs:
    ----------
        1- geometry (type point)
         the geometry of a shpefile
        2- coord_type
         "string" either "x" or "y"
    
    outpus:
    ----------
        1-array: 
         contains x coordinates or y coordinates of all edges of the shapefile
    """
    if coord_type=="x":
        return geometry.x
    if coord_type=="y":
        return geometry.y

def getLineCoords(geometry,coord_type):
    """
    ====================================================================
        getLineCoords(geometry)
    ====================================================================
    Returns Coordinates of Linestring object.
    
    inputs:
    ----------
        1- geometry (type Linestring )
         the geometry of a shpefile
        2- coord_type
         "string" either "x" or "y"
    
    outpus:
    ----------
        1-array: 
         contains x coordinates or y coordinates of all edges of the shapefile
    """
    return getXYCoords(geometry,coord_type)

def getPolyCoords(geometry,coord_type):
    """
    =================================================================
      getPolyCoords(geometry,coord_type)
    =================================================================
    Returns Coordinates of Polygon using the Exterior of the Polygon.
    
    inputs:
    ----------
        1- geometry (type polygon)
         the geometry of a shpefile
        2- coord_type
         "string" either "x" or "y"
    
    outpus:
    ----------
        1-array: 
         contains x coordinates or y coordinates of all edges of the shapefile
    """
    # convert the polygon into lines
    ext=geometry.exterior # type = LinearRing

    return getXYCoords(ext,coord_type)

    
def multiGeomHandler(multi_geometry, coord_type, geom_type):
    """
    ===================================================================
       multiGeomHandler(multi_geometry, coord_type, geom_type)
    ===================================================================
    Function for handling multi-geometries. Can be MultiPoint, MultiLineString or MultiPolygon.
    Returns a list of coordinates where all parts of Multi-geometries are merged into a single list.
    Individual geometries are separated with np.nan which is how Bokeh wants them.
    # Bokeh documentation regarding the Multi-geometry issues can be found here (it is an open issue)
    # https://github.com/bokeh/bokeh/issues/2321
    
    inputs:
    ----------
        1- multi_geometry (geometry)
         the geometry of a shpefile
        2- coord_type (string)
         "string" either "x" or "y"
        3- geom_type (string)
            "MultiPoint" or "MultiLineString" or "MultiPolygon"
    outpus:
    ----------
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
    =====================================================================
         getCoords(row, geom_col, coord_type)
    =====================================================================
    Returns coordinates ('x' or 'y') of a geometry (Point, LineString or Polygon)
    as a list (if geometry is LineString or Polygon). Can handle also MultiGeometries.
    
    inputs:
    ----------
        1- row (dataframe)
         a whole rwo of the dataframe
        2- geom_col (string)
         name of the column where the geometry is stored in the dataframe
        3- coord_type (string)
            "MultiPoint" or "MultiLineString" or "MultiPolygon"
    outpus:
    ----------
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
    """
    =====================================================================
        XY(input_dataframe)
    =====================================================================
    
    """
    # get the x & y coordinates for all types of geometries except multi_polygon
#    input_dataframe['x']=input_dataframe.apply(getCoords,geom_col="geometry", coord_type="x", axis=1)
#    input_dataframe['y']=input_dataframe.apply(getCoords,geom_col="geometry", coord_type="y", axis=1)
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
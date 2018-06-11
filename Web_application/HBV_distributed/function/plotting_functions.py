# -*- coding: utf-8 -*-
"""
Created on Sat May 05 19:47:52 2018

@author: Mostafa
"""

#%library
import os
import numpy as np
import geopandas as gpd
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import ogr
import gdal
import osr
import rasterio
from osgeo import gdalconst




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


def SaveRaster(raster,path):
    """
    # =============================================================================
    SaveRaster(raster,path)
    # =============================================================================
    this function saves a raster to a path
    inputs:
        1- raster:
            [gdal object]
        2- path:
            [string] a path includng the name of the raster and extention like 
            path="data/cropped.tif"
    Outputs:
        the function does not return and data but only save the raster to the hard drive
    EX:
        SaveRaster(raster,output_path)
    """
    driver = gdal.GetDriverByName ( "GTiff" )
    dst_ds = driver.CreateCopy( path, raster, 0 )
    dst_ds = None # Flush the dataset to disk

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
    
    dist=geopy.distance.vincenty(coords_1, coords_2).m
    
    return dist

def reproject_points(lat,lng,from_epsg=4326,to_epsg=3857):
    """
    # =============================================================================
    #  reproject_points(lat,lng,from_epsg=4326,to_epsg=3857):
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
    Ex
    1- from web mercator to GCS WGS64:
        x=[-8418583.96378159, -8404716.499972705], y=[529374.3212213353, 529374.3212213353]
        from_epsg="3857", to_epsg="4326"
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


def project_raster(src, to_epsg,resample_technique="Nearest"):
    """
    # =============================================================================
    project_dataset(src, to_epsg):
    # =============================================================================
    this function reproject a raster to any projection 
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
        4- resample_technique:
            [String] resampling technique default is "Nearest"
            https://gisgeography.com/raster-resampling/
            "Nearest" for nearest neighbour,"cubic" for cubic convolution,
            "bilinear" for bilinear
    Outputs:
        1-raster:
            gdal dataset (you can read it by ReadAsArray)
    Ex
    projected_raster=project_dataset(src, to_epsg=3857)
    
    """
    # input data validation
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
    
    # get the cell size in the source raster and convert it to the new crs
    # x coordinates or longitudes
    xs=[src_gt[0],src_gt[0]+src_gt[1]]
    # y coordinates or latitudes
    ys=[src_gt[3],src_gt[3]]
    # transform the two points coordinates to the new crs
    
    new_xs, new_ys= reproject_points(ys,xs,from_epsg=src_epsg.GetAttrValue('AUTHORITY',1),
                                     to_epsg=dst_epsg.GetAttrValue('AUTHORITY',1))
    
    pixel_spacing=np.abs(new_xs[0]-new_xs[1])
    # create a new raster
    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",int(np.round((lrx-ulx)/pixel_spacing)),int(np.round((uly-lry)/pixel_spacing)),
                       1,gdalconst.GDT_Float32,['COMPRESS=LZW']) # LZW is a lossless compression method achieve the highst compression but with lot of computation
    
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


def reproject_dataset(src, to_epsg=3857, cell_size=[], resample_technique="Nearest"):
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
        4- resample_technique:
            [String] resampling technique default is "Nearest"
            https://gisgeography.com/raster-resampling/
            "Nearest" for nearest neighbour,"cubic" for cubic convolution,
            "bilinear" for bilinear 
    Outputs:
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
            pixel_spacing=GCS_distance(coords_1, coords_2)
        else: 
            pixel_spacing=src_gt[1]
    else:
        assert (cell_size > 1),"please enter cell size greater than 1"
    # if the user input a cell size resample the raster
        pixel_spacing=cell_size 
        
    # create a new raster
    mem_drv=gdal.GetDriverByName("MEM") 
    dst=mem_drv.Create("",int(np.round((lrx-ulx)/pixel_spacing)),int(np.round((uly-lry)/pixel_spacing)),
                       1,gdalconst.GDT_Float32,['COMPRESS=LZW']) # LZW is a lossless compression method achieve the highst compression but with lot of computation
    
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


def create_polygon(coords):
    """
    # =========================================================================
    #     create_polygon(coords)
    # =========================================================================
    this function creates a ring from coordinates
    inputs:
        coords : list of tuples [(x1,y1),(x2,y2)]
    outputs:
        string of the polygon and its coordinates 
    
    Example:
        coords = [(-106.6472953, 24.0370137), (-106.4933356, 24.05293569), (-106.4941789, 24.01969175), (-106.4927777, 23.98804445)]
        polygon = create_polygon(coords)
    """
    # create a ring          
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for coord in coords:
        ring.AddPoint(np.double(coord[0]), np.double(coord[1]))
    
    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    
    poly.AddGeometry(ring)
    return poly.ExportToWkt()


def write_shapefile(poly, out_shp):
    """
    # =========================================================================
    write_shapefile(poly, out_shp):
    # =========================================================================
    this function takes a polygon geometry and creates a ashapefile and save it 
    (https://gis.stackexchange.com/a/52708/8104)
    inputs:
        1-geometry:
            polygon, point, or lines or multi
        2-path:
            string, of the path and name of the shapefile
    outputs:
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


def ClipRasterWithPolygon(Raster_path,shapefile_path,save=False,output_path=None):
    """
    # =========================================================================
     ClipRasterWithPolygon(Raster_path,shapefile_path,output_path)
    # =========================================================================
    this function clip a raster using polygon shapefile
    inputs:
        1- Raster_path:
            [String] path to the input raster including the raster extension (.tif)
        2- shapefile_path:
            [String] path to the input shapefile including the shapefile extension (.shp)
        3-save:
            [Boolen] True or False to decide whether to save the clipped raster or not
            default is False
        3- output_path:
            [String] path to the place in your drive you want to save the clipped raster 
            including the raster name & extension (.tif), default is None 
    Outputs:
        1- projected_raster:
            [gdal object] clipped raster 
        2- if save is True function is going to save the clipped raster to the output_path
    EX:
        import plotting_function as plf
        Raster_path = r"data/Evaporation_ERA-Interim_2009.01.01.tif"
        shapefile_path ="data/"+"Outline.shp"
        clipped_raster=plf.ClipRasterWithPolygon(Raster_path,shapefile_path)
        or 
        output_path = r"data/cropped.tif"
        clipped_raster=ClipRasterWithPolygon(Raster_path,shapefile_path,True,output_path)
    """
    # input data validation
    # type of inputs
    assert type(Raster_path)== str, "Raster_path input should be string type"
    assert type(shapefile_path)== str, "shapefile_path input should be string type"
    assert type(save)== bool , "save input should be bool type (True or False)"
    if save == True:
        assert type(output_path)== str," pleaase enter a path to save the clipped raster"
    # inputs value 
    ext=output_path[-4:]
    assert ext == ".tif", "please add the extention at the end of the output_path input"
    
    raster=gdal.Open(Raster_path)
    proj=raster.GetProjection()
    src_epsg=osr.SpatialReference(wkt=proj)
    gt=raster.GetGeoTransform()
    
    # first check if the crs is GCS if yes check whether the long is greater than 180
    # geopandas read -ve longitude values if location is west of the prime meridian 
    # while rasterio and gdal not 
    if src_epsg.GetAttrValue('AUTHORITY',1) == "4326" and gt[0] > 180:
        # reproject the raster to web mercator crs 
        raster=reproject_dataset(raster)
        out_transformed = os.environ['Temp']+"/transformed.tif"    
        # save the raster with the new crs
        SaveRaster(raster,out_transformed)
        raster = rasterio.open(out_transformed)
    else:
        # crs of the raster was not GCS or longitudes are less than 180
        raster = rasterio.open(Raster_path)
    
    ### Cropping the raster with the shapefile
    # read the shapefile
    shpfile=gpd.read_file(shapefile_path)
    # Re-project into the same coordinate system as the raster data
    shpfile= shpfile.to_crs(crs=raster.crs.data)
    
    def getFeatures(gdf):
        """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
        import json
        return [json.loads(gdf.to_json())['features'][0]['geometry']]
    
    # Get the geometry coordinates by using the function.
    coords=getFeatures(shpfile)
    
    out_img, out_transform = rasterio.mask.mask(raster=raster, shapes=coords, crop=True)
    
    # copy the metadata from the original data file.
    out_meta = raster.meta.copy()
    
    # Next we need to parse the EPSG value from the CRS so that we can create
    #a Proj4 string using PyCRS library (to ensure that the projection information is saved correctly).
    #epsg_code=int(raster.crs.data['init'][5:])
    
    # Now we need to update the metadata with new dimensions, transform (affine) and CRS (as Proj4 text)
    out_meta.update({"driver": "GTiff",
                         "height": out_img.shape[1],
                         "width": out_img.shape[2],
                         "transform": out_transform,
    #                     "crs": pycrs.parser.from_epsg_code(epsg_code).to_proj4()
                    }
                   )

    # save the clipped raster.
    temp_path = os.environ['Temp'] + "/cropped.tif"
    with rasterio.open(temp_path,"w", **out_meta) as dest:
        dest.write(out_img)

    # close the transformed raster
    raster.close()
    # delete the transformed raster
    os.remove(out_transformed)
    # read the clipped raster
    raster=gdal.Open(temp_path)
    # reproject the clipped raster back to its original crs
    projected_raster=project_raster(raster,int(src_epsg.GetAttrValue('AUTHORITY',1)))
    # close the clipped raster
    raster=None
    
    # delete the clipped raster
    os.remove(temp_path)
    # write the raster to the file
    if save:
        SaveRaster(projected_raster,output_path)
    
    return projected_raster

def RasterLike(src,array,path):
    """
    # =========================================================================
     RasterLike(src,array,path)
    # =========================================================================
    this function creates a Geotiff raster like another input raster, new raster 
    will have the same projection, coordinates or the top left corner of the original
    raster, cell size, and nodata velue, and number of rows and columns
    the raster and the dem should have the same number of columns and rows
    inputs:
        1- src:
            [gdal.dataset] source raster to get the spatial information
        2- array:
            [numpy array]to store in the new raster
        3- path:
            [String] path to save the new raster including new raster name and extension (.tif)
    outputs:
        1- save the new raster to the given path
    Ex:
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
    # input values
    ext=path[-4:]
    assert ext == ".tif", "please add the extension at the end of the path input"
    
    prj=src.GetProjection()
    cols=src.RasterXSize
    rows=src.RasterYSize
    gt=src.GetGeoTransform()
    noval=src.GetRasterBand(1).GetNoDataValue()
    outputraster=gdal.GetDriverByName('GTiff').Create(path,cols,rows,1,gdal.GDT_Float32)
    outputraster.SetGeoTransform(gt)
    outputraster.SetProjection(prj)
    outputraster.GetRasterBand(1).SetNoDataValue(noval)
    outputraster.GetRasterBand(1).Fill(noval)
    outputraster.GetRasterBand(1).WriteArray(array)
    outputraster.FlushCache()
    outputraster = None


    
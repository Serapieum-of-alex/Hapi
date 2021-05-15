# import numpy as np
from Hapi.gis.giscatchment import GISCatchment as GC
import gdal
Path = "F:/04Private/990110182315.csv" #Book1.csv

Path = "F:/01Algorithms/Hydrology/HAPI/Examples/data/GIS/DEM5km_Rhine_burned_acc.tif"
# data = np.loadtxt(Path, delimiter=',')

Data = gdal.Open(Path)
DataArr = Data.ReadAsArray()
NoDataValue = Data.GetRasterBand(1).GetNoDataValue()

import sys
print(sys.getrecursionlimit())
# sys.setrecursionlimit(6000)
# print(sys.getrecursionlimit())
#%%
lowervalue = 10 #DataArr[DataArr != NoDataValue].min()
uppervalue = 5000# DataArr[DataArr != NoDataValue].max()

ClusterArray, count, Position, Values = GC.Cluster(DataArr, lowervalue, uppervalue)

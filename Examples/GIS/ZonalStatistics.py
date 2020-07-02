# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:06:03 2020

@author: mofarrag
"""
from IPython import get_ipython
get_ipython().magic("reset -f")
import os
import numpy as np
import Hapi.raster as R

Path = "F:/02Case studies/Hapi Examples/"
SavePath  = Path + "results/ZonalStatistics"
BaseMapF = Path + "data/Polygons.tif"


ExcludedValue = 0
Compressed = True
OccupiedCellsOnly = False

# one map
ExtractedValues, NonZeroCells = R.OverlayMap(Path+"data/Map1.zip", BaseMapF,
                                             ExcludedValue, Compressed,OccupiedCellsOnly)

MapPrefix = "Map"
# several maps
ExtractedValues, NonZeroCells = R.OverlayMaps(Path+"data", BaseMapF, MapPrefix,
                                              ExcludedValue, Compressed,OccupiedCellsOnly)

# save extracted values in different files
Polygons = list(ExtractedValues.keys())
for i in range(len(Polygons)):
    np.savetxt(SavePath +"/" + str(Polygons[i]) + ".txt",
               ExtractedValues[Polygons[i]],fmt="%4.2f")
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:33:09 2021

@author: mofarrag
"""

# import numpy as np
# import pandas as pd
# import datetime as dt
from Hapi.hm.river import River
Path = "F:/01Algorithms/Hydrology/HAPI/Examples/"
#%%
start = "2010-1-1"
end = "2010-1-2"
# dx in meter
dx = 1000
# dt in sec
dto = 3*60

Coello = River("Coello", Version=4, start=start, end=end, dto=dto, dx=dx)
Coello.OneMinResultPath = Path + "/data/hydrodynamic model/"


Coello.ReadCrossSections(Path + "/data/hydrodynamic model/xs.csv")
Coello.ReadBoundaryConditions(Path = Path + "/data/hydrodynamic model/USBC.txt",
                              fmt="%Y-%m-%d %H:%M:%S")
Coello.Laterals = False
Coello.LateralsQ = 0
Coello.InihQ = 50
#%% run the kinematic wave model
Coello.KinematicWave()
#%% Visualization

PlotStart = "2010-01-01"
PlotEnd = "2010-1-2"
# ffmpegPath = "F:/Users/mofarrag/.matplotlib/ffmpeg-4.4-full_build/bin/ffmpeg.exe"


Coello.AnimateFloodWave(PlotStart=PlotStart, PlotEnd=PlotEnd)
#%% save results
Coello.SaveResult(Path+"/data/hydrodynamic model/")
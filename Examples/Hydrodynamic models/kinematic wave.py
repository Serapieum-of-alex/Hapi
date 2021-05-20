# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:33:09 2021

@author: mofarrag
"""
# import numpy as np
# import pandas as pd
# import datetime as dt
from Hapi.hm.river import River
path = "F:/01Algorithms/Hydrology/HAPI/Examples/"
#%% create the River object
start = "2010-1-1 00:00:00"
end = "2010-1-1 01:48:00"
# dx in meter
dx = 1000
# dt in sec
dto = 1*60

Test = River("Test", version=4, start=start, end=end, dto=dto, dx=dx, fmt="%Y-%m-%d %H:%M:%S")
Test.oneminresultpath = path + "/data/hydrodynamic model/"
#%% Read Input Data
Test.ReadCrossSections(path + "/data/hydrodynamic model/xs.csv")
Test.ReadBoundaryConditions(path=path + "/data/hydrodynamic model/USBC2.txt",
                            fmt="%Y-%m-%d %H:%M:%S")
Test.Laterals = False
Test.LateralsQ = 0
Test.InihQ = 50
#%% Run the kinematic wave model
start = "2010-1-1 00:00:00"
end = "2010-1-1 01:48:00"
Test.kinematicwave(start, end, fmt="%Y-%m-%d %H:%M:%S")
#%% Visualization
plotstart = "2010-01-01 00:00:00"
plotend = "2010-1-1 01:30:00"

# ffmpegPath = "F:/Users/mofarrag/.matplotlib/ffmpeg-4.4-full_build/bin/ffmpeg.exe"
anim = Test.animatefloodwave(start=plotstart, end=plotend, interval=1)
#%% save results
# Test.SaveResult(path + "/data/hydrodynamic model/" )
#%% read results
plotstart = "2010-01-01"
plotend = "2010-1-2"

Test.ReadSubDailyResults(plotstart, plotend, fmt="%Y-%m-%d")
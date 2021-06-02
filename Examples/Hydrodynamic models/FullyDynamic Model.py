import numpy as np
import pandas as pd

from Hapi.hm.river import River
path = "F:/01Algorithms/Hydrology/HAPI/Examples/"
#%% create the River object
start = "2010-1-1 00:00:00"
end = "2010-1-3 00:00:00"
# dx in meter
dx = 20

maxiteration = 10
# dt in sec
dto = 50 # sec
Time = 5*60*60 #(hrs to seconds)

b = 100 #m
c = 100000000
s = 0 # slope
L = 500

Test = River("Test", version=4, start=start, end=end, dto=dto, dx=dx, fmt="%Y-%m-%d %H:%M:%S")
Test.oneminresultpath = path + "/data/hydrodynamic model/"
#%%
ICQ = 10
ICH = 15

LBC = dict()
LBC['type'] = 'q'
LBC['interpolatedvalues'] = pd.read_csv(path + "/data/hydrodynamic model/"+"LBC.txt").values#.reshape((361))
LBC['interpolatedvalues'] = LBC['interpolatedvalues'].reshape(len(LBC['interpolatedvalues']))
RBC = dict()
RBC['type'] = 'q'
RBC['interpolatedvalues'] = pd.read_csv(path + "/data/hydrodynamic model/"+"RBC.txt").values
RBC['interpolatedvalues'] = RBC['interpolatedvalues'].reshape(len(RBC['interpolatedvalues']))
#%%
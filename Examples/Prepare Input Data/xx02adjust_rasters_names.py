# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:42:31 2020

@author: mofarrag
"""
#links
from IPython import get_ipython   # to reset the variable explorer each time
get_ipython().magic('reset -f')
import os
os.chdir("F:/02Case studies/Coello/HAPI/Data")

from datetime import datetime
import pandas as pd

import Hapi.inputs as IN
# Path = "F:/02Case studies/Coello/HAPI/Data/02Precipitation/CHIRPS/Daily"
Path = "F:/02Case studies/Coello/HAPI/Data/00inputs/meteodata/4000/valid/prec"
#%%

IN.RenameFiles(Path, fmt = '%Y.%m.%d')

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 06:07:08 2021

@author: mofarrag
"""

from Hapi.inputs import Inputs as IN
import numpy as np

Path = "F:/02Case studies/Coello/Hapi/Data/00inputs/meteodata/4000/calib/prec-MSWEP"
SaveTo ="F:/02Case studies/Coello/Hapi/Data/00inputs/Lumped/Prec-MSWEP.txt"

data = IN.CreateLumpedInputs(Path)
np.savetxt(SaveTo,data,fmt="%7.2f")

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:57:45 2021

@author: mofarrag
"""
# __name__ = 'Rainfall'
# from .calibration import *
# from .distparameters import *
# from .distrrm import *
# # from .hbv import *
# # from .hbv_bergestrom92 import *
# # from .hbv_lake import *
# # from .hbvlumped import *
# # from .hbvold import *
# from .inputs import *
# from .routing import *
# from .run import *
# from .wrapper import *

import Hapi.rrm.calibration as calibration
import Hapi.rrm.distparameters as distparameters
import Hapi.rrm.distrrm as distrrm
import Hapi.rrm.hbv as hbv
import Hapi.rrm.hbv_bergestrom92 as hbv_bergestrom92
import Hapi.rrm.hbv_lake as hbv_lake
import Hapi.rrm.hbvlumped as hbvlumped
import Hapi.rrm.hbvold as hbvold
import Hapi.rrm.inputs as inputs
import Hapi.rrm.routing as routing
# import Hapi.rrm.run as run
import Hapi.rrm.wrapper as wrapper

if __name__ == "__main__":
    print("rrm")

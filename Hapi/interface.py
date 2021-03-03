# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:40:23 2021

@author: mofarrag
"""
import pandas as pd
from Hapi.river import River

class Interface(River):
    """
    Interface between the Rainfall runoff model and the Hydraulic model
    """
    
    def __init__(self, name):
        self.name = name
        pass
    
    def ReadLateralsTable(self, Path):
        self.LateralsTable = pd.read_csv(Path, skiprows=[0], header=None)
        self.LateralsTable.columns = ["xsid"]
        
        
            
        
    
        
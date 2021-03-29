# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:32:29 2021

@author: mofarrag
"""
import numpy as np
import matplotlib.pyplot as plt
import Hapi.visualizer as V
Vis = V.Visualize(1)

class SensitivityAnalysis():
    
    def __init__(self, Parameter, LB, UB, Function, Positions=[], NoValues=5, Type=1):
        self.Parameter = Parameter
        self.LB = LB
        self.UB = UB
        self.Function = Function
        self.NoValues = NoValues
        self.Type = Type

        if Positions == []:
            self.NoPar = len(Parameter)
            self.Positions = list(range(len(Parameter)))
        else:
            self.NoPar = len(Positions)
            self.Positions = Positions
    
    
    def OAT(self, *args,**kwargs):
        """
        ======================================================================
           OAT(Parameter, LB, UB, Function,*args,**kwargs)
        ======================================================================
        OAT one-at-a-time sensitivity analysis
        Parameters
        ----------
        Parameter : [dataframe]
            parameters dataframe including the parameters values in a column with
            name 'value' and the parameters name as index.
        LB : [List]
            parameters upper bounds.
        UB : [List]
            parameters lower bounds.
        Function : [function]
            DESCRIPTION.
        *args : TYPE
            arguments of the function with the same exact names inside the function.
        **kwargs : TYPE
            keyword arguments of the function with the same exact names inside the function.

        Returns
        -------
        sen : [Dictionary]
            DESCRIPTION.

        """        
            
        self.sen={}
        
        for i in range(self.NoPar):
            k = self.Positions[i]
            if self.Type == 1:
                self.sen[self.Parameter.index[k]]=[[],[],[]]
            else:
                self.sen[self.Parameter.index[k]]=[[],[],[],[]]
            # generate 5 random values between the high and low parameter bounds
            rand_value = np.linspace(self.LB[k],self.UB[k],self.NoValues)
            # add the value of the calibrated parameter and sort the values
            rand_value = np.sort(np.append(rand_value,self.Parameter['value'][k]))
            # store the relative values of the parameters in the first list in the dict
            self.sen[self.Parameter.index[k]][0] = [((h)/self.Parameter['value'][k]) for h in rand_value]

            Randpar = self.Parameter['value'].tolist()
            for j in range(len(rand_value)):
                Randpar[k] = rand_value[j]
                # args = list(args)
                # args.insert(Position,Randpar)
                if self.Type == 1:
                    metric = self.Function(Randpar,*args,**kwargs)
                else:
                    metric, CalculatedValues= self.Function(Randpar,*args,**kwargs)
                    self.sen[self.Parameter.index[k]][3].append(CalculatedValues)
                # store the metric value in the second list in the dict
                self.sen[self.Parameter.index[k]][1].append(round(metric,3))
                # store the real values of the parameter in the third list in the dict
                self.sen[self.Parameter.index[k]][2].append(round(rand_value[j],4))
                print( str(k)+'-'+self.Parameter.index[k]+' -'+ str(j))
                print(round(metric,3))

    
    def Sobol(self, RealValues=False, CalculatedValues=False, Title='', 
              xlabel='xlabel', ylabel='Metric values', labelfontsize=12,
              From='', To='',Title2='', xlabel2='xlabel2', ylabel2='ylabel2', 
              spaces=[None,None,None,None,None,None]):
        
        if self.Type == 1:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
            
                        
            for i in range(self.NoPar):
                k = self.Positions[i]
                if RealValues:
                    ax.plot(self.sen[self.Parameter.index[k]][2],self.sen[self.Parameter.index[k]][1],
                             Vis.MarkerStyle[k],linewidth=3,markersize=10, label=self.Parameter.index[k])
                else:
                    ax.plot(self.sen[self.Parameter.index[k]][0],self.sen[self.Parameter.index[k]][1],
                             Vis.MarkerStyle[k],linewidth=3,markersize=10, label=self.Parameter.index[k])
            
            
            ax.set_title(Title,fontsize=12)
            ax.set_xlabel(xlabel,fontsize=12)
            ax.set_ylabel(ylabel,fontsize=12)
            
            ax.tick_params(axis='both', which='major', labelsize=labelfontsize)
            
            ax.legend(fontsize=12)
            plt.tight_layout()
            return fig, ax    
        else : #self.Type == 2 and CalculatedValues
            try:
                fig, (ax1,ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 6))
                            
                for i in range(self.NoPar):
                    k = self.Positions[i]
                    if RealValues:
                        ax1.plot(self.sen[self.Parameter.index[k]][2],self.sen[self.Parameter.index[k]][1],
                                 linewidth=3,markersize=10, label=self.Parameter.index[k]) #MarkerStyle[k],
                    else:
                        ax1.plot(self.sen[self.Parameter.index[k]][0],self.sen[self.Parameter.index[k]][1],
                                 linewidth=3,markersize=10, label=self.Parameter.index[k]) #MarkerStyle[k],
                
                
                ax1.set_title(Title,fontsize=12)
                ax1.set_xlabel(xlabel,fontsize=12)
                ax1.set_ylabel(ylabel,fontsize=12)
                ax1.tick_params(axis='both', which='major', labelsize=labelfontsize)
                
                ax1.legend(fontsize=12)
                
                for i in range(self.NoPar):
                    k = self.Positions[i]
                    for j in range(self.NoValues):
                        if From == '':
                            From = 0
                        if To == '':
                            To = len(self.sen[self.Parameter.index[k]][3][j].values)
                        ax2.plot(self.sen[self.Parameter.index[k]][3][j].values[From:To], label=self.sen[self.Parameter.index[k]][2][j])
                        
                # ax2.legend(fontsize=12)
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0,box.width*0.8,box.height])
                ax2.legend(loc=6, fancybox=True, bbox_to_anchor = (1.015,0.5))
                
                ax2.set_title(Title2,fontsize=12)
                ax2.set_xlabel(xlabel2,fontsize=12)
                ax2.set_ylabel(ylabel2,fontsize=12)
                
                plt.subplots_adjust(left=spaces[0], bottom=spaces[1], right=spaces[2], 
                                    top=spaces[3], wspace=spaces[4], hspace=spaces[5])
                
            except ValueError:
                assert False, "to plot Calculated Values you should choose Type==2 in the sentivivity object"
                
            plt.tight_layout()
            return fig, (ax1,ax2)
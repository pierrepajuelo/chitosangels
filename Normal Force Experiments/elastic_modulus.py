# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:57:46 2023

@author: Pierre PAJUELO
@subject: Recovering the elastic modulus from the normal force experiments 
"""
# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Definition of functions
def line(t,a):
    return(t*a)
# Principal program
if __name__=='__main__':
    folder = 'D:/Documents/GitHub/chitosangels/Figures_Final'
    data = pd.read_csv(folder+'/RESULTS.txt',
                       sep='-', lineterminator='\r',encoding = "utf-8",header=None,
                       low_memory=False)
    diameter = 1.5e-3
    stress = data.iloc[:,3].to_numpy(dtype=float)*1e-3/(np.pi*(diameter/2)**2)
    strain = data.iloc[:,1].to_numpy(dtype=float)
    errbars = data.iloc[:,2].to_numpy(dtype=float)/100
    # Data values
    # normal_force = np.array([6.9,19.9,19.99,20.06])
    # strain = np.array([0.8707,0.795,0.845,0.80])
    # errbars = np.array([3.86,1.21,1.49,1.83])/100
    # stress  = np.array([6.9,19.9,19.99,20.06])
    # folder = 'D:/Documents/GitHub/chitosangels/Figures'
    param,cov = curve_fit(line,strain[strain<0.3],stress[strain<0.3],p0=(10e3),maxfev=10000)
    # PLOT
    plt.close('all')
    plt.figure()
    plt.errorbar(strain,stress,c='blue',marker='o',xerr=errbars,lw=1,ls='None',ms=5,capsize=3)
    plt.plot(np.linspace(0,0.6,100),line(np.linspace(0,0.6,100),*param),ls='--',label=r'Linear fit, $E = %s$ kPa'%round(param[0]/1e3,3))
    plt.xlabel(r'Strain, $\gamma$')
    plt.ylabel(r'Stress, $\sigma$ (Pa)')
    plt.grid()
    plt.legend()
    # plt.savefig(folder+'/'+'Result_elastic_modulus.png')
    
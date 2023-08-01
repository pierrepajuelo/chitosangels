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
    
    modulusw9asph = 13.235997866493843e3 # in Pa
    stdmodulusw9asph = 3.1641026956818994e3
    modulusw9acyl = 6.403508402861586e3 # in Pa
    stdmodulusw9acyl = 2.47725625147321e3
    # For incompressible
    E1 = modulusw9asph*(1-(0.5)**2)*3 # 3 accounts for a scaling factor
    # For compressible
    E2 = modulusw9asph*(1-(0.35)**2)*3 # 3 accounts for a scaling factor
    
    E3 = modulusw9acyl*3 # 3 accounts for a scaling factor
    # Data values
    # normal_force = np.array([6.9,19.9,19.99,20.06])
    # strain = np.array([0.8707,0.795,0.845,0.80])
    # errbars = np.array([3.86,1.21,1.49,1.83])/100
    # stress  = np.array([6.9,19.9,19.99,20.06])
    # folder = 'D:/Documents/GitHub/chitosangels/Figures'
    param,cov = curve_fit(line,strain[strain<0.2],stress[strain<0.2],p0=(10e3),maxfev=10000)
    # PLOT
    plt.close('all')
    plt.figure(figsize=(10,5))
    plt.errorbar(strain,stress,c='blue',marker='o',xerr=errbars,yerr=0.3e-3/(np.pi*(diameter/2)**2)*np.ones(strain.shape[0]),lw=1,ls='None',ms=5,capsize=3)
    plt.plot(np.linspace(0,0.6,100),line(np.linspace(0,0.6,100),*param),ls='--',
             label=r'Linear fit for $\gamma <0.2$, $E = %s$ kPa'%round(param[0]/1e3,3),
             alpha = 0.5)
    end = 0.2
    plt.fill_between(np.linspace(0,end,100),line(np.linspace(0,end,100),E1),
                     line(np.linspace(0,end,100),E1 - stdmodulusw9asph), 
                     line(np.linspace(0,end,100),E1 + stdmodulusw9asph), 
                     color='blue', alpha=0.3, label=r"Sphere Incompressible $\nu = 0.5$, Scaling factor $\alpha = 3$")
    plt.fill_between(np.linspace(0,end,100),line(np.linspace(0,end,100),E2),
                     line(np.linspace(0,end,100),E2 - stdmodulusw9asph), 
                     line(np.linspace(0,end,100),E2 + stdmodulusw9asph), 
                     color='red', alpha=0.3, label=r"Sphere Compressible $\nu = 0.35$, Scaling factor $\alpha = 3$")
    plt.fill_between(np.linspace(0,end,100),line(np.linspace(0,end,100),E3),
                     line(np.linspace(0,end,100),E3 - stdmodulusw9acyl), 
                     line(np.linspace(0,end,100),E3 + stdmodulusw9acyl), 
                     color='green', alpha=0.3, label=r"Cylinder, Scaling factor $\alpha = 3$")
    plt.xlabel(r'Strain, $\gamma$')
    plt.ylabel(r'Stress, $\sigma$ (Pa)')
    plt.grid()
    plt.axhline(y=5*1e-3/(np.pi*(diameter/2)**2),c='navy',label='Technical limit, 5 mN')
    plt.axhline(y=2*1e-3/(np.pi*(diameter/2)**2),c='red',label='Noise limit, 2 mN')
    plt.axhline(y=0.1
                *1e-3/(np.pi*(diameter/2)**2),c='green',label='Technical limit DMA Q800, 0.1 mN')
    plt.legend(loc=1)
    # plt.savefig(folder+'/'+'Result_elastic_modulus.png')
    
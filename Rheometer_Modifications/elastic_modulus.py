# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 18:08:05 2023

@author: afn
"""
# Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Definition of the functions
def line(t,a):
    return(t*a)
# Principal program
if __name__=='__main__':
    strain = np.array([0.07318,0.08469,0.1127])
    stress = np.array([4.93,4.96,5.1])*1e-3/(1e-3*np.pi)**2
    
    param,cov = curve_fit(line,strain,stress,p0 = (10e3),
                          maxfev = 10000)
    # Plot
    plt.figure()
    plt.plot(strain,stress,marker='o',ls='None',label='Exp',ms=2)
    plt.plot(strain,line(strain,*param),ls='--',label='Fit, E = %s'%param[0])
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.legend()
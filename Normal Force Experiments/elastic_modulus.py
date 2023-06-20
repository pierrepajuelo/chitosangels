# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:57:46 2023

@author: Pierre PAJUELO
@subject: Recovering the elastic modulus from the normal force experiments 
"""
# Importing modules
import numpy as np
import matplotlib.pyplot as plt

# Definition of functions

# Principal program
if __name__=='__main__':
    # Data values
    normal_force = np.array([6.9,19.9,19.99,20.06])
    strain = np.array([0.8707,0.795,0.845,0.80])
    errbars = np.array([3.86,1.21,1.49,1.83])/100
    folder = 'D:/Documents/GitHub/chitosangels/Figures'
    # PLOT
    plt.close('all')
    plt.figure()
    plt.errorbar(strain,normal_force,xerr=errbars,c='blue',marker='o',lw=1,ls='--',ms=5,capsize=3)
    plt.xlabel(r'Strain, $\gamma$')
    plt.ylabel(r'Normal force, $F_{\mathrm{N}}$ (mN)')
    plt.grid()
    plt.savefig(folder+'/'+'Result_elastic_modulus.png')
    
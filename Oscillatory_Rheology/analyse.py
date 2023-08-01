# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:00:20 2023

@author: Pierre PAJUELO
@subject: Oscillatory rheology
"""
# Importing modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import matplotlib as mpl

# Defining functions
def data_format_bis(data):
    """
    Formats the data.

    Parameters
    ----------
    data : pandas dataframe
        Table with all the values from the rheometer (Rheocompass>Table>Export).
        
    Returns
    -------
    Time (in sec.), Gap (in mm), Normal force (in N) from the table. Cleans also the nan values.
    """
    list_nan = np.where(data['[rad/s]'].astype(str).str.find('nan').to_numpy(dtype=float)==0)[0]
    if len(list_nan)!=0:
        for i in list_nan:
            data = data.drop(data.index[i])
            list_nan-=1
    angular_frequency = data.iloc[:,0].to_numpy(dtype=float)
    storage_modulus = data.iloc[:,1].to_numpy(dtype=float)
    loss_modulus = data.iloc[:,3].to_numpy(dtype=float)
    return(angular_frequency,storage_modulus,loss_modulus)
def G_prime_maxwell(omega,taui,Gi):
    Gprime = (Gi*(taui*omega)**2)/(1+(omega*taui)**2) 
    return(Gprime)
# Principal program
if __name__=='__main__':
    folder = 'D:/Documents/GitHub/chitosangels/Oscillatory_Rheology'
    ###
    # result_rheometer = [f for f in os.listdir(folder) if f.endswith("30_2023 10_07 AM.csv")][0]
    
    result_rheometer = [f for f in os.listdir(folder) if f.endswith("13_2023 12_18 PM.csv")][0]
    data = pd.read_csv(folder+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False).drop(columns='\n')
    result_rheometer = [f for f in os.listdir(folder) if f.endswith("13_2023 3_33 PM.csv")][0]
    data2 = pd.read_csv(folder+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False).drop(columns='\n')
    angular_frequency,storage_modulus,loss_modulus = data_format_bis(data)
    angular_frequency2,storage_modulus2,loss_modulus2 = data_format_bis(data2)
    
    param,cov = curve_fit(G_prime_maxwell,angular_frequency,storage_modulus,p0=(0.5,10),maxfev=10000)
    # PLOT
    ### PLOT
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{lmodern} \usepackage{bm} \usepackage{xcolor}')
    #Options
    params = {'text.usetex' : True,
              'font.size' : 18,
              'font.family' : 'lmodern',
              }    
    plt.rcParams.update(params)
    mpl.rcParams['axes.linewidth'] = 1.
    
    plt.close('all')
    
    # plt.figure()
    fig,ax = plt.subplots()
    ax2 = ax.twinx()
    # ax.plot(angular_frequency,storage_modulus,'red',marker='+')
    # ax2.plot(angular_frequency,loss_modulus,'blue',marker='+')
    ax.plot(angular_frequency2,storage_modulus2,'red',marker='X',lw=1)
    ax2.plot(angular_frequency2,loss_modulus2,'blue',marker='X',lw=1)
    # plt.plot(angular_frequency,G_prime_maxwell(angular_frequency,*param),ls='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Angular frequency, $\omega$ (rad/s)')
    # plt.ylabel(r"$ \color{red} G^{(1)}$,$G^{(2)}$ (Pa)")
    ax.set_ylabel(r"$G'(\mathrm{Pa})$")
    ax2.set_ylabel(r"$G''(\mathrm{Pa})$")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax.yaxis.label.set_color('red') 
    ax2.yaxis.label.set_color('blue') 
    ax.set_ylim((0.01,10000))
    ax2.set_ylim((0.01,10000))
    plt.tight_layout()

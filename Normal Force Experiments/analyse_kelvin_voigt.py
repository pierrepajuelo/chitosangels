# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:56:32 2023

@author: Pierre PAJUELO
@subject: Reading and analyzing the data from normal force experiments
"""
# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
from scipy.optimize import curve_fit
from tqdm import tqdm
from chisquare_old_version import * # Be sure that your working folder is the right one

# Definition of functions
def data_format(data):
    """
    Formats the data for multiple instructions measurement (Multiple intervals).

    Parameters
    ----------
    data : pandas dataframe
        Table with all the values from the rheometer (Rheocompass>Table>Export).
        
    Returns
    -------
    Time (in sec.), Gap (in mm), Normal force (in N) from the table. Cleans also the nan values. 
    Each interval can be accessed by using X[i] where X is the variable and i the i-th interval.

    """
    limits = np.array([-1])
    limits = np.hstack((limits,np.where(data['[min]'].str.find('[min]').to_numpy(dtype=float)==0)[0]))
    limits = np.hstack((limits,np.array([3])))
    N = limits.shape[0]-1
    time = np.zeros(N,dtype=object)
    gap = np.zeros(N,dtype=object)
    normal_force = np.zeros(N,dtype=object)
    for i in range(N):
        time[i] = data.iloc[limits[i]+1:limits[i+1]-4,0].to_numpy(dtype=float)*60
        gap[i] = data.iloc[limits[i]+1:limits[i+1]-4,1].to_numpy(dtype=float)
        normal_force[i] = data.iloc[limits[i]+1:limits[i+1]-4,3].to_numpy(dtype=float)
    return(time,gap,normal_force)

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
    list_nan = np.where(data['[min]'].astype(str).str.find('nan').to_numpy(dtype=float)==0)[0]
    if len(list_nan)!=0:
        for i in list_nan:
            data = data.drop(data.index[i])
    time = data.iloc[:,0].to_numpy(dtype=float)*60
    gap = data.iloc[:,1].to_numpy(dtype=float)
    normal_force = data.iloc[:,3].to_numpy(dtype=float)
    return(time,gap,normal_force)

def gaussian(t,tau,a):
    return(a*(np.exp(-(t-t[0])/tau)+1))
def line(t,a,t0):
    return((t-t0)*a)
def kv(t,G,t0):
    return(tau0/G*(1-np.exp(-(t-t0)*G/eta)))
def burger(t,G1):
    return(tau0*(1/G0+1/G1*(1-np.exp(-t*G1/eta))+t/eta0))
# Principal program
if __name__=='__main__':
    # Import the data
    # folder = 'D:/STAGE M1/CHITOSAN/NORMALFORCE'
    folder = 'D:/Documents/GitHub/chitosangels/Data'
    # The list was initially to automize a set of experiments :
    result_rheometer = [f for f in os.listdir(folder) if f.endswith("6_43 PM.csv")][0]
    data = pd.read_csv(folder+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False) #,sep=',',encoding='latin1',
    # Encoding need to be changed according to the one from the csv file (can be seen using a text editor)
    # Skiprows = 9 corresponds to the heading part, the separator is tabulation 
    # Cleaning of the data (remove first column)
    data = data.drop(columns=data.columns[0])
    # Recovering the variables from the table
    time,gap,normal_force = data_format_bis(data)
    # Detecting the contact gap (Arbitrary treshold)
    time_start = np.where(normal_force>=0.0025)[0][0]
    # Need to verifiy it!
    # Equilibrium check on raw data with relaxation time (approximation) = No rescale of the curve!
    time_start1 = np.where(gap<=1)[0][0] #Ex. for 6_43 PM
    time_stop = np.where(time>=2000)[0][0] #Ex. for 6_43 PM
    X = time[time_start1:time_stop]
    Y = gap[time_start1:time_stop]
    param_exp,cov_exp = curve_fit(gaussian,X,Y,p0=(100,5),maxfev=10000)
    # Note : to plot
    # plt.close('all')
    # plt.figure()
    # plt.plot(time,gap,c='green')
    # plt.plot(X,Y,label='Exp.',c='red')
    # plt.plot(X,gaussian(X,*param),c='blue')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Gap (mm)')
    '''
    # Chi test section (you're not obliged to use it)
    # Initialize your chi list
    Chi = []
    time_stop1 = 1750
    time_stop2 = 10000
    for i in tqdm(range(time_stop1,time_stop2)): 
        # First method (if your time is not the same as before)
        time_stop = np.where(time>=i)
        X = time[time_start1:time_stop]
        Y = gap[time_start1:time_stop]
        param_exp,cov_exp = curve_fit(gaussian,X,Y,p0=(100,5),maxfev=10000)
        # Second method
        X = time[time_start1:i]
        Y = gap[time_start1:i]
        param_exp,cov_exp = curve_fit(gaussian,X,Y,p0=(100,5),maxfev=10000)
        # Calculus of the chi value
        chi_square_test_statistic, p_value = chisquare(Y, gaussian(X,*param))
        Chi.append(chi_square_test_statistic)
    # PLOT SECTION
    plt.close('all')
    time_list = np.arange(time_stop1,time_stop2)
    plt.plot(time_list,Chi)
    plt.xlabel('Time stop (s)')
    plt.yscale('log')
    plt.ylabel(r'$\chi$')
    print('Stop time should be : %s s'%(time_list[np.argmin(Chi)])) # In order to maximize your chi test
    '''
    # Convertion to strain versus time   
    Time = time[time_start:]
    Strain = (gap[time_start] - gap[time_start:])/gap[time_start] # Full-time strain
    # If we want to fit only the end of the curve (curly part, removing linear part):    
    time_goal = time_start + np.where(Strain>=0.645)[0][0] # 0.63 is a arbitrary treshold, it depends on each experiment
    # If we want to fit the line of the beginning
    time_droite = time_start + np.where(Time>=7792)[0][0] # 7792 is a arbitrary treshold, it depends on each experiment
    # If we want to remove the last part of the curve
    time_stop = np.where(Time>=15850)[0][0] # Ex. previous one 16120, used for "6_43 PM" file
    
    ### Initializing all the curve parts to analyze ###
    # Kelvin-Voigt Model for creep compliance (Data points)
    Xkv = Time[time_goal:]
    Ykv = Strain[time_goal:]
    # Depending if you want to remove the last part of the curve, you can fit the last part of the curve (straight-like): 
    # Xd = Xs[time_droite:]
    Xd = Time[time_droite:time_stop]
    # Yd = Ys[time_droite:]
    Yd = Strain[time_droite:time_stop]
    # To fit the first straight part (useful for fitting the parameter in KV model)
    Strain_begin = 0.35
    Strain_end = 0.6
    Xl = time[time_start+np.where(Strain>Strain_begin)[0][0]:time_start+np.where(Strain<Strain_end)[0][-1]]
    Yl = Strain[time_start+np.where(Strain>Strain_begin)[0][0]:time_start+np.where(Strain<Strain_end)[0][-1]]
    # Depending if you want to erase the last part of the curve
    Xp = Time[time_goal:time_stop]
    Yp = Strain[time_goal:time_stop]
    ###
    
    ### Fitting all your data, according to lines, Kelvin-Voigt and Burgers models
    param2,cov2 = curve_fit(kv,Xp,Yp,p0=(1e3,100),maxfev=10000)
    param3,cov3 = curve_fit(line,Xl,Yl,p0=(1e-3,150),maxfev=10000)
    # For Burgers, you can choose to fit all the curve or at least the beginning part
    # param4,cov4 = curve_fit(burger, Time, Strain, p0=(1e5),maxfev=10000)
    param4,cov4 = curve_fit(burger, Time[:time_stop], Strain[:time_stop], p0=(1e5),maxfev=10000)
    
    param5,cov5 = curve_fit(line,Xd,Yd,p0=(1e-3,150),maxfev=10000)   
    param42,cov42 = curve_fit(line,Time[time_stop:],Strain[time_stop:],p0=(1e-3,150),maxfev=10000)
    ###
    
    ### Initializing your general data, either from fits or from experiments
    Normal_force_instruction = 20e-3 # in N
    Gel_diameter = 1e-3 # in m
    tau0 = Normal_force_instruction/(np.pi*(Gel_diameter)**2) 
    eta = tau0/param3[0]
    eta0 = tau0/param5[0]
    G0 = tau0/Strain[time_start+np.where(Strain>Strain_begin)[0][0]] # To get an estimation of the offset at the beginning
    
    # PLOT
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{lmodern} \usepackage{bm} \usepackage{xcolor}')
    #Options
    params = {'text.usetex' : True,
              'font.size' : 15,
              'font.family' : 'lmodern',
              }
    plt.rcParams.update(params)
    mpl.rcParams['axes.linewidth'] = 1.
    
    plt.close('all')
    
    # PLOT OF CALCULATED DATA   
    fig, ax1 = plt.subplots()
    fig.set_size_inches([11,9])
    
    ax1.plot(Time,Strain,c='red',label='Exp. (All)')
    ax1.plot(Xl,line(Xl,*param3),c='green',label=r'Line begin, $\eta = %s$'%(np.format_float_scientific(tau0/param3[0],precision=2)),ls='--',lw=4)
    ax1.plot(Xd,line(Xd,*param5),c='orange',label=r'Line end, $\eta = %s$'%(np.format_float_scientific(tau0/param5[0],precision=2)),lw=4,ls='--')
    ax1.plot(Time,burger(Time,*param4),c='purple',label=r'Fit Burger, $G= %s \pm %s$'%(round(param4[0],2),round(np.sqrt(np.diag(cov4)[0]),2)))
    ax1.plot(Xp,kv(Xp,*param2),c='blue',label=r'Fit KV., $G=%s \pm %s$'%(round(param2[0],2),round(np.sqrt(np.diag(cov2)[0]),2)))
    ax1.plot(Time[time_stop:],line(Time[time_stop:],*param42),ls='--',c='red',label='End behavior')
    ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'Strain, $\gamma$')
    ax1.set_title(r'Strain vs. Time for $F_{\mathrm{N}}=20\,\mathrm{mN}$')
    ax1.axvline(x=time_goal,ls='--',lw=1,c='red')
    
    left, bottom, width, height = [0.30, 0.25, 0.30, 0.30]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(Xp,kv(Xp,*param2),c='blue',label=r'Fit KV., $G=%s \pm %s$'%(round(param2[0],2),round(np.sqrt(np.diag(cov2)[0]),2)))
    ax2.plot(Xp,burger(Xp,*param4),c='purple',label=r'Fit Burger, $G= %s \pm %s$'%(round(param4[0],2),round(np.sqrt(np.diag(cov4)[0]),2)))
    ax2.plot(Xp,Yp,c='red',label='Exp. (End)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'Strain, $\gamma$')
    ax2.set_title('Inset of End Region')
    ax2.set_xlim([150,10800])
    ax2.set_ylim([0.6,0.9])
    ax2.legend()
    plt.tight_layout()
    save_fig_folder = 'D:/Documents/GitHub/chitosangels/Figures'
    plt.savefig(save_fig_folder+'/'+'Fitted_curves_%s.png'%(result_rheometer))
    # PLOT OF RAW DATA
    plt.figure('Raw Data')
    ax = plt.subplot(111)
    ax.plot(time,normal_force,c='red')
    ax.set_xlabel('Time (sec.)')
    ax.set_ylabel(r'Normal force $F_{\mathrm{N}}$ (N)')
    ax2=ax.twinx()
    ax.yaxis.label.set_color('red') 
    ax2.yaxis.label.set_color('blue') 
    ax2.plot(time,gap,c='blue',label='Exp.')
    # ax2.plot(X,Yfit,c='blue',ls='--',label=r'Fit, $\tau = %s \pm %s$'%(round(param[0],2),round(np.sqrt(np.diag(cov)[0]),2)))
    plt.legend()
    ax2.set_ylabel('Gap (mm)')
    ax2.grid(ls='--')
    ax.grid(ls='-')
    plt.savefig(save_fig_folder+'/'+'Raw_data_%s.png'%(result_rheometer))
    
    
    
    


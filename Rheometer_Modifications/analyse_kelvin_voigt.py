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
# from chisquare_old_version import * # Be sure that your working folder is the right one
import scipy.special as ssp
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
            list_nan-=1
    time = data.iloc[:,0].to_numpy(dtype=float)*60
    gap = data.iloc[:,1].to_numpy(dtype=float)
    normal_force = data.iloc[:,3].to_numpy(dtype=float)
    return(time,gap,normal_force)

def analysis(time,gap,normal_force,
             end_time_relu=500,
             linear_beginning=0.1067,
             linear_end=0.1287,
             optim_evap_torque=False,
             optim_value=-1):
    time_start = np.where(normal_force>=0.0025)[0][0]
    ## NEW WAY TO FIND THE BEGIN POINT OF EXPERIMENT
    param_relu,cov_relu = curve_fit(ReLU,time[:500],normal_force[:500],p0=(10,10,10),
                                    maxfev = 10000)
    time_start_relu = np.where(time>=(param_relu[1]-param_relu[2])/param_relu[0])[0][0]
    # print('Time of start with a criterium : %s'%(time_start))
    # print('Time of start with ReLU : %s'%(time_start_relu))
    
       
    # Convertion to strain versus time   
    Time = time[time_start_relu:]
    Strain = (gap[time_start_relu] - gap[time_start_relu:])/gap[time_start_relu] # Full-time strain
    time_goal = np.where(normal_force>=Normal_force_instruction)[0][0]
    
    plt.figure()
    plt.plot(Time,Strain)
    
    Xl = Time[np.where(Strain>linear_beginning)[0][0]:np.where(Strain>=linear_end)[0][0]]
    Yl = Strain[np.where(Strain>linear_beginning)[0][0]:np.where(Strain>=linear_end)[0][0]]
    param3,cov3 = curve_fit(line,Xl,Yl,p0=(1e-3,150),maxfev=100000)
    eta = tau0/param3[0]
    
    ### Finding the better fit possible !
    if optim_evap_torque:
        Chi = []
        time_fit_first = np.where(Time>=Time[-1]/2)[0][0] # 207, 3000 # PREV 4037
        time_fit_end = np.where(Time>=Time[-1]*(9/10))[0][0] # 782, 6000 # PREV 6000
        endend = np.where(Time>=Time[-1]*(99/100))[0][0] # 7257 # PREV 7000
        for end in tqdm(range(time_fit_first,time_fit_end,100)):
            X1 = Time[time_goal:end]
            Y1 = Strain[time_goal:end]
            X2 = Time[end:endend]
            Y2 = Strain[end:endend]
            param7,cov7 = curve_fit(kv,X1,Y1,p0=(1e3,100,eta),maxfev=10000)
            param8,cov8 = curve_fit(line,X2,Y2,p0=(1e-3,150),maxfev=100000)
            Y_origin = Strain[time_goal:endend]
            Y_fit = np.hstack((kv(X1,*param7),line(X2,*param8)))
            Chi.append(np.sum((Y_fit-Y_origin)**2)/(Time[time_goal:endend].shape[0]-4))
        time_stop = np.arange(time_fit_first,time_fit_end)[np.argmin(Chi)] # eg. 4162 for 2_34 PM
        # print('\n Optimal end point : %s'%np.arange(time_fit_first,time_fit_end)[np.argmin(Chi)])
    else:
        time_stop = optim_value
    
    
    
    Xkv = Time[time_goal:time_stop]
    Ykv = Strain[time_goal:time_stop]
    param2,cov2 = curve_fit(kv,Xkv,Ykv,p0=(1e3,100,eta),maxfev=10000)
    return(time_goal,Xl,Yl,param3,Xkv,Ykv,param2,cov2,Strain,Time,param_relu,time_start_relu)

def plottingresult(Xl,param3,Xp,Yp,param2,Time,Strain,
                   time,normal_force,gap,
                   time_goal,param_relu,time_start_relu):
    ### PLOT
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
    # ax1.plot(Xd,line(Xd,*param5),c='orange',label=r'Line end, $\eta = %s$'%(np.format_float_scientific(tau0/param5[0],precision=2)),lw=4,ls='--')
    # ax1.plot(Time,burger(Time,*param4),c='purple',label=r'Fit Burger, $G= %s \pm %s$'%(round(param4[0],2),round(np.sqrt(np.diag(cov4)[0]),2)))
    ax1.plot(Xp,kv(Xp,*param2),c='blue',label=r'Fit KV., $G=%s \pm %s$'%(round(param2[0],2),round(np.sqrt(np.diag(cov2)[0]),2)))
    #Springpot
    # ax1.plot(Time,springpot(Time,*param6),c='black',label=r'Fit Springpot, $\alpha = %s$'%(round(param6[0],3)))
    # ax1.plot(Time[time_stop:],line(Time[time_stop:],*param42),ls='--',c='red',label='End behavior')
    ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'Strain, $\gamma$')
    ax1.set_title(r'Strain vs. Time for $F_{\mathrm{N}}=%s\,\mathrm{mN}$'%(round(np.mean(normal_force[300:])*1000,2)))
    ax1.axvline(x=Time[time_goal],ls='--',lw=1,c='red')
    
    left, bottom, width, height = [0.3, 0.3, 0.5, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(Xp,kv(Xp,*param2),c='blue',label=r'Fit KV., $G=%s \pm %s$'%(round(param2[0],2),round(np.sqrt(np.diag(cov2)[0]),2)))
    # ax2.plot(Xp,burger(Xp,*param4),c='purple',label=r'Fit Burger, $G= %s \pm %s$'%(round(param4[0],2),round(np.sqrt(np.diag(cov4)[0]),2)))
    ax2.plot(Xp,Yp,c='red',label='Exp. (End)')
    #Springpot
    # ax2.plot(Xp,springpot(Xp,*param6),c='black',label=r'Fit Springpot, $\alpha = %s$'%(round(param6[0],3)))
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'Strain, $\gamma$')
    ax2.set_title('Inset of End Region')
    # ax2.set_xlim([150,np.max(Time)])
    # ax2.set_ylim([,0.9])
    ax2.legend()
    plt.tight_layout()
    save_fig_folder = 'D:/Documents/GitHub/chitosangels/Figures'
    # plt.savefig(save_fig_folder+'/'+'Fitted_curves_%s.png'%(result_rheometer))
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
    ax1.grid()
    ax2.grid()
    # ax2.plot(X,Yfit,c='blue',ls='--',label=r'Fit, $\tau = %s \pm %s$'%(round(param[0],2),round(np.sqrt(np.diag(cov)[0]),2)))
    plt.legend()
    ax2.set_ylabel('Gap (mm)')
    ax2.grid(ls='--')
    ax.grid(ls='-')
    ## ReLU TESTS
    ax.plot(time[:500],ReLU(time[:500],*param_relu),ls='--',c='green')
    # ax.axvline(x=time[time_start],ls='-',c='cyan')
    ax.axvline(x=time[time_start_relu],ls='-',c='purple')
    
# Functions for fitting
def gaussian(t,tau,a):
    return(a*(np.exp(-(t-t[0])/tau)+1))
def line(t,a,t0):
    return((t-t0)*a)
def kv(t,G,t0,eta):
    return(tau0/G*(1-np.exp(-(t-t0)*G/eta)))
def ReLU(t,a,b,c):
    return(np.maximum(b,a*t+c))
def gaussian_function(t,tau,mu):
    return(1/(tau*np.sqrt(np.pi*2))*np.exp(-1/2*((t-mu)/tau)**2))
# def burger(t,G1):
#     return(tau0*(1/G0+1/G1*(1-np.exp(-t*G1/eta))+t/eta0))
def springpot(t,alpha,lambda1):
    J = tau0*(1/ssp.gamma(alpha+1)*(lambda1*t**alpha*np.heaviside(t,1)))
    return(J)
def cauchy(x,a,b,c):
    return(a/(np.pi*(1+b*x**2))+c)
#%%
# Principal program
if __name__=='__main__':
    # Import the data
    # folder = 'D:/STAGE M1/CHITOSAN/NORMALFORCE'
    # folder = 'D:/Documents/GitHub/chitosangels/Data'
    folder = "C:/Users/afn/Documents/FractureDynamics/Reometer_Data" 
    # The list was initially to automize a set of experiments :
    result_rheometer = [f for f in os.listdir(folder) if f.endswith("30_2023 10_07 AM.csv")][0]
    data = pd.read_csv(folder+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False).drop(columns='\n') #,sep=',',encoding='latin1',
    # Encoding need to be changed according to the one from the csv file (can be seen using a text editor)
    # Skiprows = 9 corresponds to the heading part, the separator is tabulation 
    # Cleaning of the data (remove first column)
    # Recovering the variables from the table
    time,gap,normal_force = data_format_bis(data)
    # Detecting the contact gap (Arbitrary treshold)
    
    ### ANALYSE ON RAW DATA
    # Need to verifiy it!
    # Equilibrium check on raw data with relaxation time (approximation) = No rescale of the curve!
    # time_start1 = np.where(gap<=1)[0][0] #Ex. for 6_43 PM
    # time_stop = np.where(time>=2000)[0][0] #Ex. for 6_43 PM
    # X = time[time_start1:time_stop]
    # X = time[time_start_relu:]
    # Y = gap[time_start_relu:]
    # param_exp,cov_exp = curve_fit(gaussian,X,Y,p0=(100,5),maxfev=10000)
    # Y = gap[time_start1:time_stop]
    # Note : to plot
    # plt.close('all')
    # plt.figure()
    # plt.plot(time,gap,c='green')
    # plt.plot(X,Y,label='Exp.',c='red')
    # plt.plot(X,gaussian(X,*param),c='blue')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Gap (mm)')
    ###
    
    ### DETECT START OF THE CURVE
    # time_start = np.where(normal_force>=0.0025)[0][0]
    ## NEW WAY TO FIND THE BEGIN POINT OF EXPERIMENT
    # param_relu,cov_relu = curve_fit(ReLU,time[:500],normal_force[:500],p0=(10,10,10),
                                    # maxfev = 10000)
    # tim2e_start_relu = np.where(time>=(param_relu[1]-param_relu[2])/param_relu[0])[0][0]
    # print('Time of start with a criterium : %s'%(time_start))
    # print('Time of start with ReLU : %s'%(time_start_relu))
    ###
    
    ### Initializing your general data, either from fits or from experiments
    # Normal_force_instruction = 20e-3 # in N
    Normal_force_instruction = round(np.mean(normal_force[300:]),6)
    Gel_diameter = 1.5e-3/2 # in m
    tau0 = Normal_force_instruction/(np.pi*(Gel_diameter)**2) 
    ###
    # time_goal,Xl,Yl,param3,Xp,Yp,param2,cov2,Strain,Time = analysis(time[:28799],
    #                                                                 gap[:28799],
    #                                                                 normal_force[:28799])
    time_goal,Xl,Yl,param3,Xp,Yp,param2,cov2,Strain,Time = analysis(time,
                                                                    gap,
                                                                    normal_force)
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
   
    # If we want to fit only the end of the curve (curly part, removing linear part):    
    # time_goal = np.where(Strain>=0.0788)[0][0] # small one : 0.0852 other 0.5 #other prev 0.7 # 0.63 is a arbitrary treshold, it depends on each experiment
    
    # Prev. 0.645 (6_43 PM) ; 0.7 others # 0.65 
    # If we want to fit the line of the end
    # time_droite = np.where(Time>=7792)[0][0] # 7792 is a arbitrary treshold, it depends on each experiment
    # If we want to remove the last part of the curve
    # time_stop = np.where(Time>=15850)[0][0] # Ex. previous one 16120, used for "6_43 PM" file
    # time_stop = np.where(Strain>=0.7)[0][0]
    
    ### First of all: fitting the first part of the curve
    # To fit the first straight part (useful for fitting the parameter in KV model)
    # Strain_begin = 0.1226 # Small strain : 0.050 # 0.3 # other prev. 0.340 # prev. 0.35
    # Strain_end = 0.1330 # Small strain : 0.080 # 0.4 # other prev. 0.556 # prev. 0.6
    # Xl = Time[np.where(Strain>Strain_begin)[0][0]:np.where(Strain>=Strain_end)[0][0]]
    # Yl = Strain[np.where(Strain>Strain_begin)[0][0]:np.where(Strain>=Strain_end)[0][0]]
    # Strain_end = time_goal + np.where(Time>=Time[time_goal]+1e1)[0][0]# Modified name in reality 
    # Xl = Time[time_goal:Strain_end]
    # Yl = Strain[time_goal:Strain_end]
    
    # param3,cov3 = curve_fit(line,Xl,Yl,p0=(1e-3,150),maxfev=100000)
    
    
    ### Calculating first a paramater
    # eta = tau0/param3[0]
    ###
    
    ### Finding the better fit possible !
    # Chi = []
    # time_fit_first = np.where(Time>=7530)[0][0] # 207, 3000
    # time_fit_end = np.where(Time>=17500)[0][0] # 782, 6000
    # endend = np.where(Time>=18000)[0][0] # 7257
    # for end in tqdm(range(time_fit_first,time_fit_end,100)):
    #     X1 = Time[time_goal:end]
    #     Y1 = Strain[time_goal:end]
    #     X2 = Time[end:endend]
    #     Y2 = Strain[end:endend]
    #     param7,cov7 = curve_fit(kv,X1,Y1,p0=(1e3,100),maxfev=10000)
    #     param8,cov8 = curve_fit(line,X2,Y2,p0=(1e-3,150),maxfev=10000)
    #     Y_origin = Strain[time_goal:endend]
    #     Y_fit = np.hstack((kv(X1,*param7),line(X2,*param8)))
    #     Chi.append(np.sum((Y_fit-Y_origin)**2)/(Time[time_goal:endend].shape[0]-4))
    # time_stop = np.arange(time_fit_first,time_fit_end)[np.argmin(Chi)] # eg. 4162 for 2_34 PM
    # print('\n Optimal end point : %s'%np.arange(time_fit_first,time_fit_end)[np.argmin(Chi)])
    # time_stop = 3903
    ### Initializing all the curve parts to analyze ###
    # Kelvin-Voigt Model for creep compliance (Data points)
    # Xkv = Time[time_goal:]
    # Ykv = Strain[time_goal:]
    # Depending if you want to remove the last part of the curve, you can fit the last part of the curve (straight-like): 
    # Xd = Time[time_droite:]
    # Xd = Time[time_droite:time_stop]
    # Yd = Strain[time_droite:]
    # Yd = Strain[time_droite:time_stop]
    
    # Depending if you want to erase the last part of the curve
    # Tstop = np.where(Time>=1374)[0][0]
    # Xp = Time[time_goal:Tstop]
    # Xp = Time[time_goal:]
    # Xp = Time
    # Xp = Time[time_goal:time_stop]
    # Xp = np.copy(Xkv)
    # Yp = Strain[time_goal:Tstop]
    # Yp = Strain[time_goal:]
    # Yp = Strain
    # Yp = Strain[time_goal:time_stop]
    # Yp = np.copy(Ykv)
    ###
    
    
    
    ### Fitting all your data, according to lines, Kelvin-Voigt and Burgers models
    
    # param5,cov5 = curve_fit(line,Xd,Yd,p0=(1e-3,150),maxfev=10000)   
    # param42,cov42 = curve_fit(line,Time[time_stop:],Strain[time_stop:],p0=(1e-3,150),maxfev=10000)
    ###
    
    ### Calculating some of the parameters
    # eta0 = tau0/param5[0]
    # G0 = tau0/Strain[time_start+np.where(Strain>Strain_begin)[0][0]] # To get an estimation of the offset at the beginning
    ###
    
    ### Fitting the data (Bis)
    # param2,cov2 = curve_fit(kv,Xp,Yp,p0=(1e3,100),maxfev=10000)
    # param6,cov6 = curve_fit(springpot,Time[Strain>0.4],Strain[Strain>0.4],p0=(0.5,10),maxfev=10000)
    # For Burgers, you can choose to fit all the curve or at least the beginning part
    # param4,cov4 = curve_fit(burger, Time[:time_stop], Strain[:time_stop], p0=(1e5),maxfev=10000)
    # param4,cov4 = curve_fit(burger, Time, Strain, p0=(1e5),maxfev=10000)
    ###
    
    plottingresult(Xl,param3,Xp,Yp,param2,Time,Strain,
                       time,normal_force,gap,
                       time_goal)
    
    
    
    
    # plt.tight_layout()
    # plt.savefig(save_fig_folder+'/'+'Raw_data_%s.png'%(result_rheometer))
    
    
    ### STUDY OF THE EQUILIBRIUM
    # Put the final time you're interested in
    Time_end = np.argmax(Time) #prev. 9020 for 9_45 AM
    # Time_end = np.where(Time>=9330)[0][0] # for 9_45 
    # Time_end = np.where(Time>=15830)[0][0] 
    # According to the KV model:
    Strain_end_infty = tau0/param2[0]
    Srain_end_experiment = Strain[Time_end]
    relative_error = np.abs(Strain_end_infty-Strain)/Strain_end_infty*100
    Strain_end_estimation = Strain[np.where(Strain>=kv(Xp,*param2)[-1])[0][0]]
    print('\n Strain at the end : %s'%Strain_end_estimation)
    print('\n Error the strain made : %s %%'%(np.abs(Strain_end_infty-Strain_end_estimation)/Strain_end_infty*100))
    
    # print('Error : %s %%'%(np.abs(Strain_end_infty-Srain_end_experiment)/Strain_end_infty*100))
    # plt.figure()
    # plt.plot(Time,relative_error)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Relative error %% ')
    # plt.ylim((0,100))
    # relative_error = relative_error[Time<=np.argmin(relative_error)]
    # error1 = Time[np.where(relative_error>=3)[0][-1]]
    # print('Time for 3%% : %s hours'%(error1/3600))
    # error2 = Time[np.where(relative_error>=5)[0][-1]]
    # print('Time for 5%% : %s hours'%(error2/3600))
    # error3 = Time[np.where(relative_error>=10)[0][-1]]
    # print('Time for 10%% : %s hours'%(error3/3600))
    # error4 = Time[np.where(relative_error>=20)[0][-1]]
    # print('Time for 20%% : %s hours'%(error4/3600))
    # error5 = Time[np.where(relative_error>=30)[0][-1]]
    # print('Time for 30%% : %s hours'%(error5/3600))
    
    ### PLOT OF THE CHI PARAMATER
    # plt.figure()
    # plt.semilogy(np.arange(time_fit_first,time_fit_end),Chi)
    # plt.xlabel('End point')
    # plt.ylabel(r'$\chi^2$')
    # plt.axvline(x=minimum,c='red',ls='--')
    ###
    
    ### FOR EQUILIBRIUM EXP.
    # plt.figure()
    # Noise = Yp - kv(Xp,*param2)
    # plt.hist(Noise,bins=100)
    # hist,bins = np.histogram(Noise,bins=100)
    
    # param_gaussian,cov_gaussian = curve_fit(gaussian_function,bins[1:],hist,
    #                                         p0 = (100,10),
    #                                         maxfev = 10000) 
    # param_cauchy,cov_cauchy = curve_fit(cauchy,bins[1:],hist,
    #                                         p0 = (100,10,10),
    #                                         maxfev = 10000) 
    # mean = np.mean(Noise)
    # std = np.std(Noise)
    # fit_gaussian = gaussian_function(bins,*param_gaussian) 
    # plt.plot(bins,fit_gaussian*np.max(hist)/np.max(fit_gaussian),c='black',ls='--',
    #          label=r'Gaussian fit, $\sigma = %s$'%(param_gaussian[0]))
    # plt.plot(bins,cauchy(bins,*param_cauchy),c='navy',ls='-',
    #          label='Cauchy modified fit')
    # plt.axvline(x=mean,ls='--',c='red',
    #             label='Strain mean, $\gamma = %s \pm %s$'%(round(mean,3),round(std,3)))
    # plt.legend()
    # plt.xlabel(r'Strain, $\gamma$')
    # plt.ylabel(r'$\#$ of points')
    # plt.title('Histogram for %s mN'%(Normal_force_instruction*1e3))
    ###
    #%% Update of the program
    # General informations
    # folder = "C:/Users/afn/Documents/FractureDynamics/Reometer_Data" 
    folder = 'D:/Documents/GitHub/chitosangels/Rheometer_Modifications'
    ###
    # result_rheometer = [f for f in os.listdir(folder) if f.endswith("30_2023 10_07 AM.csv")][0]
    
    ### In the following, we'll analyze only the 1.5mm diameter gels :
    Gel_diameter = 1.5e-3/2 # in m
    ###
    result_rheometer = [f for f in os.listdir(folder) if f.endswith("4_2023 2_17 PM.csv")][0]
    data = pd.read_csv(folder+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False).drop(columns='\n')
    time,gap,normal_force = data_format_bis(data)
    Normal_force_instruction = round(np.mean(normal_force[300:]),6)
    
    tau0 = Normal_force_instruction/(np.pi*(Gel_diameter)**2) 
    time_goal,Xl,Yl,param3,Xp,Yp,param2,cov2,Strain,Time,param_relu,time_start_relu = analysis(time,
                                                                    gap,
                                                                    normal_force,optim_evap_torque=True)
    plottingresult(Xl,param3,Xp,Yp,param2,Time,Strain,
                       time,normal_force,gap,
                       time_goal,param_relu,time_start_relu)
    Strain_end_infty = tau0/param2[0]
    relative_error = np.abs(Strain_end_infty-Strain)/Strain_end_infty*100
    Strain_end_estimation = Strain[np.where(Strain>=kv(Xp,*param2)[-1])[0][0]]
    print('\n Strain at the end : %s'%Strain_end_estimation)
    print('\n Error the strain made : %s %%'%(np.abs(Strain_end_infty-Strain_end_estimation)/Strain_end_infty*100))
    
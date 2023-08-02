# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:35:37 2023

@author: Pierre PAJUELO
"""
# Importing modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
from scipy.optimize import curve_fit

# Definition of functions
def ReLU(t,a,b,c):
    return(np.maximum(b,a*t+c))

def kv(t,G,t0,eta):
    return(tau0/G*(1-np.exp(-(t-t0)*G/eta)))

def data_format_DMA(data):
    """
    Formats the data.

    Parameters
    ----------
    data : pandas dataframe
        Table with all the values from the rheometer (DMA).
        
    Returns
    -------
    Time (in sec.), Gap (in mm), Normal force (in N) from the table.
    """
    time = data.iloc[:,0].to_numpy(dtype=float)*60
    gap = data.iloc[:,5].to_numpy(dtype=float)
    normal_force = data.iloc[:,3].to_numpy(dtype=float)
    return(time,gap,normal_force)

# Principal program
if __name__=='__main__':
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
    
    folder = 'C:/Users/pierr/Downloads/DMA Alberto'
    result_rheometer = [f for f in os.listdir(folder) if f.endswith('.txt')][0]
    data = pd.read_csv(folder+'/'+result_rheometer,
                       skiprows=57,
                       sep=' ', lineterminator='\n',encoding = "utf-16-le",
                       low_memory=False,header=None).iloc[:-1,:]
    time,gap,normal_force = data_format_DMA(data)
    relu_end = 30
    
    param_relu,cov_relu = curve_fit(ReLU,time[:relu_end],gap[:relu_end],p0=(10,10,10),
                                    maxfev = 10000)
    # IF NEEDED :
    time_start_relu = np.where(time>=(param_relu[1]-param_relu[2])/param_relu[0])[0][0]
    time_start_relu = 0
    Time = time[time_start_relu:]
    Strain = (gap[time_start_relu] - gap[time_start_relu:])/gap[time_start_relu]
    time_stop = np.where(Strain>=0.05)[0][0]
    X1 = Time[:time_stop]
    Y1 = Strain[:time_stop]
    # FIT KV
    # param7,cov7 = curve_fit(kv,X1,Y1,p0=(1e3,100,1e8),maxfev=10000)
    
    
    Gel_diameter = 1.5e-3
    Normal_force_instruction = round(np.mean(normal_force[700:]),6)
    tau0 = Normal_force_instruction/(np.pi*(Gel_diameter)**2) 
    
    
    
    
    print('\n Strain at the end : %s'%np.mean(Strain[time_stop-500:time_stop]))
    print('\n Error on strain : %s'%(np.std(Strain[:time_stop],ddof=1)))
    print('\n Normal Force applied : %s'%(np.mean(normal_force[time_start_relu:])))
    print('\n Error Normal Force applied : %s'%(np.std(normal_force[time_start_relu:],ddof=1)))
    # PLOT
    plt.close('all')
    # PLOT OF CALCULATED DATA   
    fig, ax1 = plt.subplots()
    fig.set_size_inches([11,9])
    
    ax1.plot(time,normal_force,c='red',marker='+')
    ax2 = ax1.twinx()
    ax2.plot(time,gap,c='blue',marker='+')
    ax2.set_xlabel('Time (s)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Normal Force (N)')
    ax2.set_ylabel('Gap (mm)')
    ax1.yaxis.label.set_color('red') 
    ax2.yaxis.label.set_color('blue') 
    ax2.plot(time[:relu_end],ReLU(time[:relu_end],*param_relu),ls='--',c='green',label='ReLU fit')
    ax2.axvline(x=time[time_start_relu],ls='-',c='purple',label='ReLU contact gap')
    
    fig, ax = plt.subplots()
    ax.plot(Time,Strain,c='red',label='Exp. (All)')
    # ax1.plot(Xl,line(Xl,*param3),c='green',label=r'Line begin, $\eta = %s\,\mathrm{Pa}\cdot\mathrm{s}$'%(np.format_float_scientific(tau0/param3[0],precision=2)),ls='--',lw=4)
    # # ax1.plot(Xd,line(Xd,*param5),c='orange',label=r'Line end, $\eta = %s$'%(np.format_float_scientific(tau0/param5[0],precision=2)),lw=4,ls='--')
    # # ax1.plot(Time,burger(Time,*param4),c='purple',label=r'Fit Burger, $G= %s \pm %s$'%(round(param4[0],2),round(np.sqrt(np.diag(cov4)[0]),2)))
    # ax1.plot(Xp,kv(Xp,*param2),c='blue',label=r'Fit KV., $G=%s \pm %s\,\mathrm{kPa}$'%(round(param2[0]/1e3,3),round(np.sqrt(np.diag(cov2)[0])/1e3,3)))
    # #Springpot
    # # ax1.plot(Time,springpot(Time,*param6),c='black',label=r'Fit Springpot, $\alpha = %s$'%(round(param6[0],3)))
    # # ax1.plot(Time[time_stop:],line(Time[time_stop:],*param42),ls='--',c='red',label='End behavior')
    # ax1.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Strain, $\gamma$')
    # ax1.set_title(r'Strain vs. Time for $F_{\mathrm{N}}=%s\,\mathrm{mN}$'%(round(np.mean(normal_force[300:])*1000,2)))
    # ax1.axvline(x=Time[time_goal],ls='--',lw=1,c='red')
    # ax1.axvline(x=Time[time_stop],ls='--',lw=1,c='blue')
    # left, bottom, width, height = [0.3, 0.4, 0.5, 0.2]
    # ax2 = fig.add_axes([left, bottom, width, height])
    Xp = np.linspace(np.min(X1),np.max(X1),1000)
    # ax.plot(Xp,kv(Xp,*param7),c='blue',label=r'Fit KV., $G=%s \pm %s\,\mathrm{kPa}$'%(round(param7[0]/1e3,3),round(np.sqrt(np.diag(cov7)[0])/1e3,3)))
    # # ax2.plot(Xp,burger(Xp,*param4),c='purple',label=r'Fit Burger, $G= %s \pm %s$'%(round(param4[0],2),round(np.sqrt(np.diag(cov4)[0]),2)))
    # ax2.plot(Xp,Yp,c='red',label='Exp. (End)')
    # #Springpot
    # # ax2.plot(Xp,springpot(Xp,*param6),c='black',label=r'Fit Springpot, $\alpha = %s$'%(round(param6[0],3)))
    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel(r'Strain, $\gamma$')
    # ax2.set_title('Inset of End Region')
    # # ax2.set_xlim([150,np.max(Time)])
    # # ax2.set_ylim([,0.9])
    # ax2.legend()
    ax.grid(ls='--')
    # ax2.grid(ls='-')
    plt.tight_layout()
    # save_fig_folder = 'D:/Documents/GitHub/chitosangels/Figures'
    # # plt.savefig(save_fig_folder+'/'+'Fitted_curves_%s.png'%(result_rheometer))
    # # PLOT OF RAW DATA
    # # plt.figure('Raw Data')
    # 
    # ax.plot(time,normal_force,c='red')
    # ax.set_xlabel('Time (sec.)')
    # ax.set_ylabel(r'Normal force $F_{\mathrm{N}}$ (N)')
    # ax2=ax.twinx()
    
    # ax2.plot(time,gap,c='blue',label='Exp.')
    # # ax1.grid()
    # # ax2.grid()
    # # ax2.plot(X,Yfit,c='blue',ls='--',label=r'Fit, $\tau = %s \pm %s$'%(round(param[0],2),round(np.sqrt(np.diag(cov)[0]),2)))
    
    # ax2.set_ylabel('Gap (mm)')
    # ax2.grid(ls='--')
    # ax.grid(ls='-')
    # ## ReLU TESTS
    
    # # ax.axvline(x=time[time_start],ls='-',c='cyan')
    # ax.axvline(x=time[time_start_relu],ls='-',c='purple',label='ReLU contact gap')
    # ax.legend(facecolor='white', framealpha=1)
    # left, bottom, width, height = [0.22, 0.6, 0.5, 0.2]
    # ax3 = fig.add_axes([left, bottom, width, height])
    # ax3.plot(time,normal_force,c='red')
    # ax3.plot(time[:end_relu],ReLU(time[:end_relu],*param_relu),ls='--',c='green',label='ReLU fit')
    
    # ax3.set_xlim((0,300))
    # ax3.set_ylim((-0.001,0.006))
    # ax3.grid(ls='-')
    # ax3.set_title('Inset of First Region')
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:03:45 2023

@author: Pierre PAJUELO
"""
# Importation des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.optimize import curve_fit

# Définition des fonctions
def data_format(data):
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
    list_nan = np.where(data['[min]'].astype(str).str.find('nan').to_numpy(dtype=float)==0)[0]
    if len(list_nan)!=0:
        for i in list_nan:
            data = data.drop(data.index[i])
    time = data.iloc[:,0].to_numpy(dtype=float)*60
    gap = data.iloc[:,1].to_numpy(dtype=float)
    normal_force = data.iloc[:,3].to_numpy(dtype=float)
    return(time,gap,normal_force)
def line(t,a,b):
    return(t*a+b)
def exponential(t,tau):
    t0 = time4[np.argmax(normal_force4)]
    b = normal_force4[int(t[-1])]
    a = np.max(normal_force4) - b
    n = 1
    return(b+a*np.exp(-(t-t0)**n/tau))
def st_exp(x,c,tau,beta,y_offset):
    return c*(np.exp(-(x/tau)**beta))+y_offset

# Programme principal
if __name__=='__main__':
    # Paramétrage des dossiers
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{lmodern} \usepackage{bm} \usepackage{xcolor}')
    #Options
    params = {'text.usetex' : True,
              'font.size' : 20,
              'font.family' : 'lmodern',
              }
    plt.rcParams.update(params)
    mpl.rcParams['axes.linewidth'] = 1.

    dossier = 'D:/STAGE M1/CHITOSAN/Strain_imposed_immerged'
    result_rheometer = [f for f in os.listdir(dossier) if f.endswith(
        "Strain_Normal_Force_Immerged_6_2_2023 11_41 AM.csv")][0]
    data = pd.read_csv(dossier+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False)#,sep=',',encoding='latin1',
    
    # Cleaning of the data
    
    data = data.drop(columns=data.columns[0])
    
    # DISPLAY Strain_Normal_Force_Immerged_6_2_2023 11_41 AM
    # time,gap,normal_force = data_format(data)
    offset = -1
    time = data.iloc[:offset,0].to_numpy(dtype=float)*60
    gap = data.iloc[:offset,1].to_numpy(dtype=float)
    normal_force = data.iloc[:offset,3].to_numpy(dtype=float)
    plt.close('all')
    plt.figure()
    plt.plot(time,normal_force,marker='+',ms=3,label=r'Gap, $d = %s$ mm'%(np.mean(gap)))
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normal Force, $F_{\mathrm{N}}$ (N)')
    plt.grid()
    plt.legend()
    
    end_time = time[-1]
    # DISPLAY Strain_Normal_Force_Immerged_6_2_2023 6_59 PM
    result_rheometer = [f for f in os.listdir(dossier) if f.endswith(
        "PM.csv")][0]
    data2 = pd.read_csv(dossier+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False)#,sep=',',encoding='latin1',
    data2 = data2.drop(columns=data2.columns[0])
    time2,gap2,normal_force2 = data_format(data2)
    
    for i in range(4):
        time2[i] += end_time
    # Plotting the result
    # plt.close('all')
    # plt.figure()
    color = plt.cm.rainbow(np.linspace(0, 1, 4))
    for i, c in zip(range(4), color):
        plt.plot(time2[i],normal_force2[i],c=c,marker='+',ms=3,label=r'Gap, $d = %s$ mm'%(np.mean(gap2[i])))
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normal Force, $F_{\mathrm{N}}$ (N)')
    # plt.grid()
    plt.legend()
    
    # SECTION DATA
    Slopes = []
    for i in range(4):
        param,cov = curve_fit(line,time2[i],normal_force2[i],p0=(-1e-3,10),maxfev=10000)
        plt.plot(time2[i],line(time2[i],*param),ls='--',c='orange',lw=2,label='Fit segment %s'%(i))
        Slopes.append(param[0])
    # FULL DATA
    # Full_time = np.hstack((time,np.concatenate((list(map(np.float64, time2))))))
    # Full_FN = np.hstack((normal_force,np.concatenate((list(map(np.float64, normal_force2))))))
    Full_time = np.concatenate((list(map(np.float64, time2))))
    Full_FN = np.concatenate((list(map(np.float64, normal_force2))))
    param,cov = curve_fit(line,Full_time,Full_FN,p0=(-1e-3,10),maxfev=10000)
    print('Pente par segment : %s \pm %s'%(np.mean(Slopes),np.std(Slopes,ddof=1)))
    print('Pente totale : %s \pm %s'%(param[0],np.sqrt(np.diag(cov)[0])))
    plt.plot(Full_time,line(Full_time,*param),ls='--',c='green',lw=2,label='Fit line')
    plt.legend()
    plt.tight_layout()
    
    # OTHER ANALYSIS
    plt.figure()
    plt.plot(time,normal_force,marker='+',ms=3,label=r'Gap, $d = %s$ mm'%(np.mean(gap)))
    plt.title('Analysing the normal force curve')
    color = plt.cm.rainbow(np.linspace(0, 1, 4))
    for i, c in zip(range(4), color):
        plt.plot(time2[i],normal_force2[i],c=c,marker='+',ms=3)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normal Force, $F_{\mathrm{N}}$ (N)')
    plt.grid()
    # input_user = plt.ginput(4)
    # SAVE
    input_user = [(59386.30461169353, 0.003014253879310355),
     (150072.3304184139, -0.008203880916927895),
     (239761.80649099447, -0.015792619161442003),
     (319485.7852221772, -0.019224048628526644)]
    Slopes2 = []
    for i, c in zip(range(4), color):
        # plt.plot(time2[i],gap2[i],c=c,marker='+',ms=3,label=r'Gap, $d = %s$ mm'%(np.mean(gap2[i])))
        if i!=0:
            gap_value = 1-0.1*i
            markers = np.where(gap2[i]>gap_value)[0][-1]+1
            # plt.scatter(time2[i][markers],gap2[i][markers],marker='+')
            plt.axvline(x=time2[i][markers],ls='--',c='red')
        time_start = np.where(time2[i]>=input_user[i][0])[0][0]
        param,cov = curve_fit(line,time2[i][time_start:],
                              normal_force2[i][time_start:],
                              p0=(-1e-3,10),maxfev=10000)
        Slopes2.append(param[0])
        plt.plot(time2[i][time_start:],
                 line(time2[i][time_start:],*param),ls='--',
                 c='orange',lw=2,label='Fit segment %s'%(i))
        # Slopes2.append(param[0])
    param,cov = curve_fit(line,Full_time,Full_FN,p0=(-1e-3,10),maxfev=10000)
    plt.plot(Full_time,line(Full_time,*param),ls='--',c='green',lw=2,label='Fit line')        
                
    
    
    # COMPARISON
    dossier = 'D:/STAGE M1/CHITOSAN/Strain_imposed_immerged/Others'
    result_rheometer = "Boyancy_Evaporation_Calibration_6_6_2023 4_29 PM.csv"
    data3 = pd.read_csv(dossier+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False)#,sep=',',encoding='latin1',
    # Cleaning of the data
    data3 = data3.drop(columns=data3.columns[0])
    time3,gap3,normal_force3 = data_format(data3)
    plt.figure()
    plt.plot(time3[0],normal_force3[0],c='red',label=r'Boyancy at speed $v = %s$ mm/s'%(round(np.mean(np.gradient(gap3[0],time3[0],edge_order=2)),2)))
    param3,cov3 = curve_fit(line,time3[0],normal_force3[0],p0=(-1e-3,10),maxfev=10000)
    plt.plot(time3[0],line(time3[0],*param3),ls='--',c='green',lw=2,label = 'Fit line')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normal Force, $F_{\mathrm{N}}$ (N)')
    plt.legend()
    print('Pente référence : %s \pm %s'%(param3[0],np.sqrt(np.diag(cov3)[0])))
    
    # Acquisition X1
    dossier = 'D:/STAGE M1/CHITOSAN/Strain_imposed_immerged/Others'
    result_rheometer = "Strain_Normal_Force_Immerged_6_6_2023 9_42 AM.csv"
    data4 = pd.read_csv(dossier+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False)#,sep=',',encoding='latin1',
    # Cleaning of the data
    data4 = data4.drop(columns=data4.columns[0])
    time4,gap4,normal_force4 = data_format_bis(data4)
    
    plt.figure()
    plt.title(r'Small gel with dimensions $\sim 1.5\,\mathrm{mm}\times %s\,\mathrm{mm}$'%(round(gap4[np.where(normal_force4>=0.00345)[0][0]],2)))# Criterium on graph
    plt.plot(time4,normal_force4,c='red',label=r'Evaporation at gap $d =$ %s'%(round(np.mean(gap4),2)))
    time_stop = 1000
    param6,cov6 = curve_fit(st_exp,
                            time4[np.argmax(normal_force4):time_stop]-time4[np.argmax(normal_force4)], 
                            normal_force4[np.argmax(normal_force4):time_stop],
                            p0 = (10,100,15,20), maxfev=10000)
    plt.plot(time4[np.argmax(normal_force4):time_stop],
             st_exp(time4[np.argmax(normal_force4):time_stop]-time4[np.argmax(normal_force4)],*param6),
             ls='-',c='blue',label='Fit Expo.')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normal Force, $F_{\mathrm{N}}$ (N)')
    param,cov = curve_fit(line,time4[time_stop:],
                          normal_force4[time_stop:],p0=(-1e-3,10),maxfev=10000)
    plt.plot(time4[time_stop:],line(time4[time_stop:],*param),ls='--',c='green',lw=2,label = 'Fit line')
    # param6,cov6 = curve_fit(exponential,time4[:2970],normal_force4[:2970],
    #                         p0 = (1e3),
    #                         maxfev=100000)
    
    plt.legend()
    print('Pente expérience : %s \pm %s'%(param[0],np.sqrt(np.diag(cov)[0])))
    # plt.figure('Line substraction')
    plt.figure()
    ax = plt.subplot(111)
    normal_force5 = normal_force4-param[0]*time4
    ax.plot(time4,normal_force5,label='Sub.')
    plt.title(r'Small gel with dimensions $\sim 1.5\,\mathrm{mm}\times %s\,\mathrm{mm}$'%(round(gap4[np.where(normal_force4>=0.00345)[0][0]],2)))# Criterium on graph
    param7,cov7 = curve_fit(st_exp,
                            time4[np.argmax(normal_force5):]-time4[np.argmax(normal_force5)], 
                            normal_force5[np.argmax(normal_force5):],
                            p0 = (10,100,15,20), maxfev=10000)
    ax.plot(time4[np.argmax(normal_force5):],
             st_exp(time4[np.argmax(normal_force5):]-time4[np.argmax(normal_force5)],*param7),
             ls='-',c='blue',label='Fit Expo.')
    ax.yaxis.label.set_color('blue') 
   
    plt.legend()
    ax2 = ax.twinx()
    ax2.yaxis.label.set_color('red')
    ax2.plot(time4,gap4,c='red')
    ax2.set_ylim((0,2))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normal force (N)')    
    ax2.set_ylabel('Gap (mm)')
    
    # EXPERIENCES SEMI IMMERGED (NORMAL FORCE CTE)
    dossier = 'D:/STAGE M1/CHITOSAN/Strain_imposed_immerged/Semi immerged'
    # result_rheometer = "10mN_PP25S_4mm_4h_6_7_2023 10_17 AM.csv" # Strange peak
    result_rheometer = '10mN_PP25S_4mm_4h_6_7_2023 2_40 PM.csv' # No so long
    # result_rheometer = '10mN_PP25S_4mm_4h_6_7_2023 3_43 PM.csv'
    data6 = pd.read_csv(dossier+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False)#,sep=',',encoding='latin1',
    # Cleaning of the data
    data6 = data6.drop(columns=data6.columns[0])
    time6,gap6,normal_force6 = data_format_bis(data6)
    plt.figure()
    plt.title('Normal force imposed, semi-immerged')
    ax = plt.subplot(111)
    ax.plot(time6,normal_force6,c='Blue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Normal Force, $F_{\mathrm{N}}$')
    ax.yaxis.label.set_color('blue') 
    ax2 = ax.twinx()
    ax2.set_ylabel('Gap (mm)')
    ax2.plot(time6,gap6,c='red')
    ax2.yaxis.label.set_color('red')
    # time_start = np.where(time6>=6400)[0][0]
    # param444,cov444 = curve_fit(line,time6[time_start:],gap6[time_start:],p0=(-1e-6,10),maxfev=10000)
    # ax2.plot(time6[time_start:],line(time6[time_start:],*param444),lw=1,ls='--')
    
    
    # 0106 Experiment
    dossier = 'D:/STAGE M1/CHITOSAN/Strain_imposed_immerged'
    result_rheometer = 'Strain_Normal_Force_Immerged_6_1_2023 11_41 AM.csv'
    data7 = pd.read_csv(dossier+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False)#,sep=',',encoding='latin1',
    # Cleaning of the data
    data7 = data7.drop(columns=data7.columns[0])
    time7,gap7,normal_force7 = data_format_bis(data7)
    plt.close('all')
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(time7,normal_force7,marker='+',ms=3,c='blue',label=r'Gap, $d = %s$ mm'%(round(np.mean(gap7),2)))
    time_start = np.where(time7>=3300)[0][0]
    param6,cov6 = curve_fit(st_exp,
                            time7[np.argmax(normal_force7):time_start]-time7[np.argmax(normal_force7)], 
                            normal_force7[np.argmax(normal_force7):time_start],
                            p0 = (10,100,15,20), maxfev=10000)
    ax.plot(time7[np.argmax(normal_force7):time_start],
             st_exp(time7[np.argmax(normal_force7):time_start]-time7[np.argmax(normal_force7)],*param6),
             ls='-',c='red',label='Fit Expo.',lw=3)
    param777,cov777 = curve_fit(line,time7[time_start:],normal_force7[time_start:],p0=(-1e-7,10),maxfev=10000)
    ax.plot(time7[time_start:],line(time7[time_start:],*param777),ls='--',lw=2,c='purple',label='Linear Fit')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Normal Force, $F_{\mathrm{N}}$ (N)')
    
    ax2 = ax.twinx()
    ax2.set_ylabel('Gap (mm)')
    ax2.plot(time7,gap7,c='red')
    ax2.yaxis.label.set_color('red')
    ax.yaxis.label.set_color('blue') 
    normal_force8 = normal_force7-param777[0]*time7
    ax.plot(time7,normal_force8,marker='+',ms=3,c='orange',label='Line sub.')
    time_start = np.where(time7>=3300)[0][0]
    param888,cov888 = curve_fit(st_exp,
                            time7[np.argmax(normal_force8):]-time7[np.argmax(normal_force8)], 
                            normal_force8[np.argmax(normal_force8):],
                            p0 = (10,100,15,20), maxfev=10000)
    ax.plot(time7[np.argmax(normal_force8):],
             st_exp(time7[np.argmax(normal_force8):]-time7[np.argmax(normal_force8)],*param888),
             ls='-',c='green',label='Fit Expo. Sub')
    ax.grid()
    ax.legend()
    
    
    # 2905 Experiment
    dossier = 'D:/STAGE M1/CHITOSAN/NORMALFORCE'
    result_rheometer = '25mN_PP25S_2mm_.csv'
    data10 = pd.read_csv(dossier+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False)#,sep=',',encoding='latin1',
    # Cleaning of the data
    data10 = data10.drop(columns=data10.columns[0])
    time10,gap10,normal_force10 = data_format(data10)
    plt.close('all')
    plt.figure()
    ax = plt.subplot(111)
    
    ax.plot(time10[1],normal_force10[1],marker='+',ms=3,c='blue')
    ax.yaxis.label.set_color('blue') 
    ax2 = ax.twinx()
    ax2.plot(time10[1],gap10[1],marker='+',ms=3,c='red')
    ax2.yaxis.label.set_color('red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Normal Force, $F_{\mathrm{N}}$ (N)')
    ax2.set_ylabel('Gap (mm)')
    plt.tight_layout()
    ax.grid() 
    ax2.grid()
    
    
    # 2605 Experiment
    dossier = 'D:/STAGE M1/CHITOSAN/NORMALFORCE'
    result_rheometer = 'TEST_VIDEOS_5_26_2023 8_32 PMx.csv'
    data11 = pd.read_csv(dossier+'/'+result_rheometer,skiprows=9,
                       sep='\t', lineterminator='\r',encoding = "utf-16",
                       low_memory=False)#,sep=',',encoding='latin1',
    # Cleaning of the data
    data11 = data11.drop(columns=data11.columns[0])
    time11,gap11,normal_force11 = data_format(data11)
    plt.close('all')
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(time11[1],normal_force11[1],marker='+',ms=3,c='blue')
    ax.yaxis.label.set_color('blue') 
    ax2 = ax.twinx()
    ax2.plot(time11[1],gap11[1],marker='+',ms=3,c='red')
    ax2.yaxis.label.set_color('red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Normal Force, $F_{\mathrm{N}}$ (N)')
    ax2.set_ylabel('Gap (mm)')
    plt.tight_layout()
    ax.grid() 
    ax2.grid()
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:50:47 2023

@author: afn
"""
# Importing modules
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# Definitions of functions
def importls(angles,loc):
    SLS = [[],[],[],[],[],[],[],[]]
    Time = []
    Angles = []
    SLS3 = []
    for angle in angles:
        # angle = angles[0]
        if angle not in angles[0:1] and angle not in angles[3:7]:
            continue
        # elif angle in angles[19:]:
        #     continue
        else:
            experiments = [f.path for f in os.scandir(angle) if f.is_dir()]
            SLS2 = [[],[],[],[],[],[],[]]
            for iex,experiment in tqdm(enumerate(experiments)):
                # print(experiment)
                fichier = pd.read_csv(experiment+'/'+'Raw Data.csv',sep=',',skiprows=5)
                limit = np.where(fichier['Count Trace ChA [kHz]'].str.find('Lag').to_numpy(dtype=float)==0)[0][0]-1
                dataA = fichier.iloc[:limit,0].to_numpy(dtype=float)
                dataB = fichier.iloc[:limit,1].to_numpy(dtype=float)
                lagtime = fichier.iloc[limit+2:,0].to_numpy(dtype=float)
                correlation = fichier.iloc[limit+2:,1].to_numpy(dtype=float)
                
                # angle_num = int(experiment[loc:loc+3]) # For Old protocol
                angle_num = int(experiment[loc:loc+2])
                if experiment[loc+2]=='.':
                    angle_num = float(experiment[loc:loc+5])
                q_value = 4*np.pi*1.33/(632.8e-9)*np.sin(angle_num*np.pi/360)
                Amean = np.mean(dataA)
                Bmean = np.mean(dataB)
                Astd = np.std(dataA,ddof=1)
                Bstd = np.std(dataB,ddof=1)
                SLS2[0].append(q_value)
                SLS2[1].append(Amean)
                SLS2[2].append(Bmean)
                SLS2[3].append(Astd)
                SLS2[4].append(Bstd)
                if iex==0:
                    Angles.append(angle_num)
                # print(experiment)
                
                    
                # Correlation fnction
                # if iex==0:
                # PLOT
                # plt.close('all')
                stop_fit = np.where(lagtime>=0.001)[0][0]
                # plt.figure()
                # plt.plot(lagtime[:stop_fit],correlation[:stop_fit])
                paramline,covline = curve_fit(line,lagtime[:stop_fit],correlation[:stop_fit],
                                              p0 = (correlation[stop_fit]),
                                              maxfev=10000)
                # if experiment.endswith('n4_2'):
                #     plt.close('all')
                #     plt.figure()
                #     plt.plot(lagtime,correlation/paramline[0],'red',marker='x',lw=1)
                #     plt.xscale('log')
                #     plt.xlabel(r'Lagtime, $\tau$ (s)')
                #     plt.ylabel(r'Correlation function, $g^{(2)}(\tau)-1$')
                #     plt.ylim((0,1.1))
                # X = np.linspace(np.min(lagtime),lagtime[stop_fit],100)
                # plt.plot(X,line(X,*paramline),'red',label=r'Intercept, $\beta = %s$ '%round(paramline[0],3))
                # plt.legend()
                # plt.xscale('log')
                # plt.xlabel(r'Lagtime, $\tau$ (s)')
                # plt.ylabel(r'Correlation function, $g^{(2)}(\tau)-1$')
                
                if pd.read_csv(experiment+'/'+'Cumulant Results.csv',sep=',',encoding='UTF-8',header=None,
                                              index_col=False,skiprows=2,skipfooter=8).iloc[2,0]!='NaN':
                    fichierresults = float(pd.read_csv(experiment+'/'+'Cumulant Results.csv',sep=',',encoding='UTF-8',header=None,
                                                  index_col=False,skiprows=2,skipfooter=8).iloc[2,0])#.to_numpy(dtype=float)
                    # print('\n From fit : %s'%round(paramline[0],3))
                    # print('\n From CONTIN : %s'%round(fichierresults,3))
                correlation_normalize = correlation/paramline[0]
                # for tau_interest in np.where(lagtime>=0.01)[0]:
                #     if correlation_normalize[tau_interest]>=1:
                #         print('Aiieee caramba')
                #     else:
                #         SLS2[5].append(lagtime)
                #         SLS2[6].append(correlation_normalize)
                # tau_interest = np.where(lagtime>=0.15)[0][0]
                # if correlation_normalize[tau_interest]>=1:
                #     print('Aiieee caramba, numero %s'%iex)
                # else:
                #     SLS2[5].append(lagtime)
                #     SLS2[6].append(correlation_normalize)
                if iex==0:
                    SLS2[5].append(lagtime)
                    begin = np.char.find(experiment,'3DDLS',start=0)+6
                    end = np.char.find(experiment,'s CSV',start=0)
                    time = int(experiment[begin:end])
                    Time.append(time)
                compteur = 0
                for tau_interest in np.where(lagtime>=0.01)[0]:
                    if correlation_normalize[tau_interest]<=1 and correlation_normalize[tau_interest]>0:
                        compteur+=1
                # print('Score :', compteur)
                # print('Goal :', len(np.where(lagtime>=0.01)[0]))
                if compteur>=len(np.where(lagtime>=0.01)[0])-30:
                    SLS2[6].append(correlation_normalize)
            SLS[0].append(np.mean(SLS2[0]))
            SLS[1].append(np.mean(SLS2[1]))
            SLS[2].append(np.mean(SLS2[2]))
            SLS[3].append(np.linalg.norm(SLS2[3],ord=2))
            SLS[4].append(np.linalg.norm(SLS2[4],ord=2))
            
            print(np.shape(SLS2[6]))
            if len(np.shape(SLS2[6]))==2:
                   SLS3.append(SLS2[6])
                   SLS[5].append(SLS2[5])
                   SLS[6].append(np.mean(SLS2[6],axis=0))
                   SLS[7].append(np.std(SLS2[6],axis=0,ddof=1))
            elif len(np.shape(SLS2[6]))==1:
                   # SLS[6].append(np.mean(SLS2[6],axis=0))
                   # SLS[7].append(np.std(SLS2[6],axis=0,ddof=1))
                   # print(SLS2[6])
                   
                   print('Ok captain, T = %s min'%(time/60))
                   if time!=0:
                       for i in SLS2[6]:
                           print(len(i))
                       raise Exception("Sorry")
        
    return(SLS,Angles,SLS3,Time)

def reshape(SLS,Angles,exp,nbangle):
    for i in range(3):
        SLS[i] = np.array(SLS[i]).reshape((nbangle,exp))
    SLS = np.array(SLS)
    Angles = np.array(Angles).reshape((nbangle,exp))
    return(SLS,Angles)

def PqFit(q,c,R):
    return(c*((3*(np.sin(q*R)-(q*R)*np.cos(q*R)))/((q*R)**3))**2)
def line(tau,a):
    return(a*np.ones(tau.shape[0]))
def Debye(q,c,R):
    return(c*(np.exp(-(q*R)**2)-1+(q*R)**2)/((q*R)**4))
def g2fit(tau,gamma):
    return(np.exp(-2*gamma*tau))
def g2fitstretch(tau,gamma,beta):
    return(np.exp(-2*(gamma*tau)**beta))
def g2fitm(tau,gamma,b):
    return(np.exp(-2*gamma*tau)+b)
def g2fitstretchm(tau,gamma,beta,b):
    return(np.exp(-2*(gamma*tau)**beta)+b)
def g2fitmm(tau,gamma,b,a):
    return(a*np.exp(-2*gamma*(tau-b)))
def g2fitstretchmm(tau,gamma,beta,b,a):
    return(a*np.exp(-2*(gamma*(tau-b))**beta))

# Principal program
if __name__=='__main__':
    # Parameters for the plot
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{lmodern} \usepackage{bm} \usepackage{xcolor}')
    #Options
    params = {'text.usetex' : True,
              'font.size' : 15,
              'font.family' : 'lmodern',
              }
    plt.rcParams.update(params)
    mpl.rcParams['axes.linewidth'] = 1.
    
    # Importation des fichiers
    # dossier = 'D:/STAGE M1/DLS\correction/OLD/230512'
    # dossier = 'C:/DLS/070723_Pierre/CSV_Files'
    # dossier = 'C:/DLS/070723_Pierre/100723/CSV_Files'
    # dossier = 'C:/DLS/120723_Pierre/Amplitude_Sweep_SLS/CSV_Files'
    # dossier = 'C:/DLS/120723_Pierre/SOL_TO_GEL_TRANSITION/CSV_Files'
    # path = 'D:\STAGE M1\CHITOSAN\DLS\\120723_Pierre\\120723_Pierre\\SOL_TO_GEL_TRANSITION\CSV_Files'
    path = 'D:\STAGE M1\CHITOSAN\DLS\FINAL DATA\SOL_TO_GEL_TRANSITION\CSV_Files'
    dossier = path.replace(os.sep, '/')
    angles = [f.path for f in os.scandir(dossier) if f.is_dir() and f.path.endswith('CSV')]
    # loc = 40
    # loc = 47
    # loc = 60
    # loc = 62
    # loc = 85+9
    loc = 77 
    Rg = 5e-6
    # Rg = 500e-9
    SLS,Angles,SLS2,time = importls(angles,loc)
    # Angles = np.array(Angles)
    # Scatteringvectors = np.array(SLS[0])
    # DetectorA = np.array(SLS[1])*np.sin(Angles*np.pi/180)
    # MaxA = np.max(DetectorA)
    # DetectorA = DetectorA/MaxA
    # ErrA = np.array(SLS[3])/MaxA
    # DetectorB = np.array(SLS[2])*np.sin(Angles*np.pi/180)
    # MaxB = np.max(DetectorB)
    # DetectorB = DetectorB/MaxB
    # ErrB = np.array(SLS[4])/MaxB
    # Scatteringvectorslist = np.linspace(np.min(Scatteringvectors),np.max(Scatteringvectors),1000)
    # %% FIT
    import scipy.constants as cte
    plt.close('all')
    N_runs = np.shape(np.array(SLS[6],dtype=object))[0]
    Rh = []
    Rhe = []
    Beta = []
    Betae = []
    Rh2 = []
    Rh2e = []
    color = plt.cm.rainbow(np.linspace(0,1,N_runs))
    # plt.figure(figsize=(10,7))
    for i in tqdm(range(0,N_runs,1)):
        # if i!=9 or i!=11:
        #     stop_fit = np.where(SLS[5][i][0]>=1*10**(-0.69))[0][0]
        #     start_fit = np.where(SLS[5][i][0]>=1e-3)[0][0]
        #     param,cov = curve_fit(g2fit,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],p0=(10),maxfev=10000)
        #     param2,cov2 = curve_fit(g2fitstretch,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],
        #                           p0=(10,0.8),maxfev=10000)
        # else:
        #     stop_fit = np.where(SLS[5][i][0]>=1*10**(-0.69))[0][0]
        #     start_fit = np.where(SLS[5][i][0]>=1e-4)[0][0]
        #     param,cov = curve_fit(g2fit,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],p0=(10),maxfev=10000)
        #     param2,cov2 = curve_fit(g2fitstretch,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],
        #                           p0=(10,0.8),maxfev=10000)
        # tautimearr = np.logspace(-1.5,-0.69,100)
        # coucou = []
        # coucou2 = []
        # for tautime in tautimearr:
        #     # stop_fit = np.where(SLS[5][i][0]>=1e-1)[0][0]
        #     stop_fit = np.where(SLS[5][i][0]>=tautime)[0][0]
        #     start_fit = np.where(SLS[5][i][0]>=1e-3)[0][0]
        #     param,cov = curve_fit(g2fitm,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],p0=(10,1),maxfev=100000)
        #     param2,cov2 = curve_fit(g2fitstretchm,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],
        #                           p0=(10,0.8,1),maxfev=100000)
        #     coucou.append(np.sqrt(np.diag(cov2)[0]))
        #     coucou2.append(param2[0])
        # coucou = np.array(coucou)
        
        stop_fit = np.where(SLS[5][i][0]>=0.1)[0][0]
        start_fit = np.where(SLS[5][i][0]>=1e-4)[0][0]
        param,cov = curve_fit(g2fitm,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],
                              p0=(10,1),maxfev=100000)
        param2,cov2 = curve_fit(g2fitstretchm,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],
                                  p0=(10,0.8,1),maxfev=100000)
        
        plt.figure(figsize=(10,7))
        lagtimearr = np.logspace(-8,2,1000)
        plt.plot(lagtimearr,g2fitm(lagtimearr,*param),c=color[i],label=r'Fit exp., $T=%s$ min'%(time[i]/60),ls='--')
        plt.plot(lagtimearr,g2fitstretchm(lagtimearr,*param2),c=color[i],
                  label=r'Fit stretch exp., $T=%s$ min, $\beta = %s$'%(time[i]/60,param2[1]),ls=':')
        plt.plot(SLS[5][i][0],SLS[6][i],c=color[i],label='Mean value, $T=%s$ min'%(time[i]/60))
        plt.axvline(x=SLS[5][i][0][start_fit],ls='--',c='red')
        plt.axvline(x=SLS[5][i][0][stop_fit],ls=':',c='blue')
        plt.xscale('log')
        plt.xlabel(r'Lagtime, $\tau$ (s)')
        plt.ylabel(r'Correlation function, $\langle (g^{(2)}(\tau)-1)/\beta\rangle$')
        plt.legend(ncol=2,fontsize='10')
        
        # # FIRST EXP
        
        # if i==0:
        #     stop_fit = np.where(SLS[5][i][0]>=0.1)[0][0]
        #     start_fit = np.where(SLS[5][i][0]>=1e-4)[0][0]
        # else:
        #     stop_fit = np.where(SLS[5][i][0]>=0.6)[0][0]
        #     start_fit = np.where(SLS[5][i][0]>=1e-4)[0][0]
        # param,cov = curve_fit(g2fitm,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],
        #                       p0=(10,1),maxfev=100000)
        # param2,cov2 = curve_fit(g2fitstretchm,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],
        #                           p0=(10,0.8,1),maxfev=100000)
        # lagtimearr = np.logspace(-8,2,1000)
        # # if i == 3:
        # plt.figure(figsize=(10,7))
        # plt.plot(lagtimearr,g2fitm(lagtimearr,*param),c=color[i],label=r'Fit exp., $T=%s$ min'%(time[i]/60),ls='--')
        # plt.plot(lagtimearr,g2fitstretchm(lagtimearr,*param2),c=color[i],
        #          label=r'Fit stretch exp., $T=%s$ min, $\beta = %s$'%(time[i]/60,param2[1]),ls=':')
        # plt.plot(SLS[5][i][0],SLS[6][i],c=color[i],label='Mean value, $T=%s$ min'%(time[i]/60))
        # plt.axvline(x=SLS[5][i][0][start_fit],ls='--',c='red')
        # plt.axvline(x=SLS[5][i][0][stop_fit],ls=':',c='blue')
        
        # # SECOND EXP
        # if i==0:
        #     stop_fit = np.where(SLS[5][i][0]>=0.7)[0][0]
        #     start_fit = np.where(SLS[5][i][0]>=0.4)[0][0]
        # else:
        #     stop_fit = np.where(SLS[5][i][0]>=200)[0][0]
        #     start_fit = np.where(SLS[5][i][0]>=3)[0][0]
        # param3,cov3 = curve_fit(g2fitmm,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],
        #                       p0=(10,10,0.9),maxfev=10000000)
        # param4,cov4 = curve_fit(g2fitstretchmm,SLS[5][i][0][start_fit:stop_fit],SLS[6][i][start_fit:stop_fit],
        #                           p0=(10,0.8,10,0.9),maxfev=10000000)
        # lagtimearr = np.logspace(-8,3,1000)
        # plt.plot(lagtimearr,g2fitmm(lagtimearr,*param3),c=color[i],label=r'Fit exp., $T=%s$ min'%(time[i]/60),ls='--')
        # plt.plot(lagtimearr,g2fitstretchmm(lagtimearr,*param4),c=color[i],
        #          label=r'Fit stretch exp., $T=%s$ min, $\beta = %s$'%(time[i]/60,param2[1]),ls=':')
        # plt.axvline(x=SLS[5][i][0][start_fit],ls='--',c='red')
        # plt.axvline(x=SLS[5][i][0][stop_fit],ls=':',c='blue')
        
        
        # plt.xscale('log')
        # plt.xlabel(r'Lagtime, $\tau$ (s)')
        # plt.ylabel(r'Correlation function, $\langle (g^{(2)}(\tau)-1)/\beta\rangle$')
        # plt.legend(ncol=2,fontsize='10')
        # plt.yscale('log')
        R = param[0]
        R2 = param2[0]
        # D = param[0]/SLS[0][i]**2
        # R = cte.k*300/(6*np.pi*0.83e-3*D)
        # Rh.append(np.mean(coucou2))
        Rh.append(1/R)
        Rh2.append(1/R2)
        # Rhe.append(np.linalg.norm(coucou,ord=2))
        Rhe.append(np.sqrt(np.diag(cov)[0]))
        Rh2e.append(np.sqrt(np.diag(cov2)[0]))
        Beta.append(param2[1])
        Betae.append(np.sqrt(np.diag(cov2)[1]))
        plt.ylim((-0.1,1.1))
    # plt.close('all')
    
    plt.figure()
    # for i,t in enumerate(time[1:N_runs-5]):
    plt.errorbar(np.array(time)/60,Rh,
                  yerr=Rhe,
                  c='red',marker='+',ls='None',ms=10,capsize=4,ecolor='navy'
                   ,label='Exp.')
    plt.errorbar(np.array(time)/60,Rh2,
                   yerr=Rh2e,
                  c='green',marker='+',ls='None',ms=10,capsize=4,ecolor='navy',
                  label='Stretch Exp.')
    plt.xlabel('Polymerisation time (min.)')
    plt.ylabel(r'Inverse relaxation decay, $\Gamma^{-1}$ (s)')
    plt.grid()
    plt.legend()
    # plt.ylim((1,2))
    plt.figure()
    # for i,t in enumerate(time[1:N_runs-5]):
    plt.errorbar(np.array(time)/60,Beta,
                  yerr=Betae,
                 c='red',marker='+',ls='None',ms=10,capsize=4,ecolor='navy')
    plt.xlabel('Polymerisation time (min.)')
    plt.ylabel(r'Stretch exponent, $\beta$')
    plt.grid()
    # plt.ylim((1,2))
        # plt.scatter(t/60,Beta[i],c='blue',marker='+',ls='None')
    # plt.xscale('log')
    # plt.yscale('log')
    #%% PLOT with check boxes
    from matplotlib.widgets import CheckButtons,AxesWidget
    N_runs = np.shape(np.array(SLS[6],dtype=object))[0]
    plt.close('all')
    for i in range(N_runs):
        if i == 0 or i==1:
            fig, ax = plt.subplots()
            # fig, (rax,ax) = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios = [0.1,1]) )
            ax.set_title('Folder %s'%i)
            N_repetition = np.shape(SLS2[i])[0]
            color = plt.cm.rainbow(np.linspace(0,1,N_repetition))
            lines = [ax.plot(SLS[5][i][0],SLS2[i][j],c=color[j],
                             marker='+',label='Repetition %s'%j)[0] for j in range(N_repetition)]
            stop_fit = np.where(SLS[5][i][0]>=1e-1)[0][0]
            param,cov = curve_fit(g2fit,SLS[5][i][0][:stop_fit],SLS[6][i][:stop_fit],p0=(10),maxfev=10000)
            param2,cov2 = curve_fit(g2fitstretch,SLS[5][i][0][:stop_fit],SLS[6][i][:stop_fit],
                                      p0=(10,0.8),maxfev=10000)
            lagtimearr = np.logspace(-8,2,1000)
            ax.plot(lagtimearr,g2fit(lagtimearr,*param),'black',ls='--',
                     label=r'$\Gamma = %s \pm %s$'%(round(param[0],3),round(np.sqrt(np.diag(cov)[0]),3)))
            ax.plot(lagtimearr,g2fitstretch(lagtimearr,*param2),'blue',ls='--',
                    label=r'$\Gamma = %s \pm %s$, $\beta = %s \pm %s$'%(round(param2[0],3),round(np.sqrt(np.diag(cov2)[0]),3),
                    round(param2[1],3),round(np.sqrt(np.diag(cov2)[1]),3)))
            ax.set_xlabel(r'Lag time, $\tau$')
            ax.set_ylabel(r'Correlation function, $(g^{(2)}(\tau)-1)/\beta$')
            ax.set_xscale('log')
            ax.legend()
            # plt.legend(ncol=3,bbox_to_anchor=(0.55, 0.6, 0, 0),fontsize="10")
            # Make checkbuttons with all plotted lines with correct visibility
            # rax = plt.axes([0.01, 0.01, 0.2, 3])
            # labels = [str(line.get_label()) for line in lines]
            # visibility = [line.get_visible() for line in lines]
            # check = CheckButtons(rax, labels, visibility)
            # def func(label):
            #     index = labels.index(label)
            #     lines[index].set_visible(not lines[index].get_visible())
            #     # plt.draw()
            #     fig.canvas.draw_idle()
            
            # check.on_clicked(func)
            ax.set_ylim((0,1.1))
            
            
            
            
            

    #%% PLOT
    plt.close('all')
    plt.figure(figsize=(14,7))
    listindices = np.argsort(time)
    color = plt.cm.brg(np.linspace(0,1,np.shape(np.array(SLS[6],dtype=object))[0]))
    # color = plt.cm.hsv('hsv',np.shape(SLS[5])[0])
    for i,c in zip(listindices,color):
        # if i==len(listindices)-3 or i==len(listindices)-2:
        # if i < len(listindices)-5:# or i==0:
        # if i==0:
        # if i>=1 and i<N_runs-5:
        # if i==0 or i==1:
        plt.errorbar(SLS[5][i][0],SLS[6][i],
                  yerr = SLS[7][i],
                  marker='X',c=c,ls='None',ms=10,capsize=4,lw=1,ecolor='navy',
                  label = r"$\vartheta = %s^{\circ}$, UV Time $T=%s$ min"%(Angles[i],time[i]/60))
    plt.xscale('log')
    plt.legend()
    # plt.yscale('log')
    plt.xlabel(r'Lagtime, $\tau$ (s)')
    plt.ylabel(r'Correlation function, $\langle (g^{(2)}(\tau)-1)/\beta\rangle$')
    # plt.axvline(x=0.5,c='magenta',ls='--')
    plt.grid()
    # color = plt.cm.rainbow(np.linspace(0,1,np.shape(SLS2[6])[0]))
    # BANNED = [15,0,72,11,14,17,18,31,55,69,98,145]
    # for i,c in zip(range(np.shape(SLS2[6])[0]),color):
    #     if i not in BANNED:
        # plt.plot(SLS[5][0][0],SLS2[6][i],c=c,lw=1)
        # plt.plot(SLS[5][0],SLS2[6][72],c=c,lw=1)
        
    plt.ylim((0,1.1))
    # plt.ylim((0.5,1))
    plt.xlim((1e-7,1e4))
    # plt.xlim((1e-2,100))
    plt.tight_layout()
    #%%
    # stop = 25
    # stop = -1
    # param,cov = curve_fit(PqFit,Scatteringvectors[:stop],DetectorA[:stop],p0=(500,Rg),maxfev=100000)
    # locmaxA = np.where(Scatteringvectorslist>=Scatteringvectors[np.argmax(SLS[1])])[0][0]
    # fitA = PqFit(Scatteringvectorslist,*param)/PqFit(Scatteringvectorslist,*param)[locmaxA]
    # param2,cov2 = curve_fit(PqFit,Scatteringvectors[:stop],DetectorB[:stop],p0=(500,Rg),maxfev=100000)
    # locmaxB = np.where(Scatteringvectorslist>=Scatteringvectors[np.argmax(SLS[2])])[0][0]
    # fitB = PqFit(Scatteringvectorslist,*param2)/PqFit(Scatteringvectorslist,*param2)[locmaxB]
    # param,cov = curve_fit(Debye,Scatteringvectors,DetectorA,p0=(500,5e-6),maxfev=100000)
    # param2,cov2 = curve_fit(Debye,Scatteringvectors,DetectorA,p0=(500,5e-6),maxfev=100000)
    plt.close('all')
    fig = plt.figure()
    ax = plt.subplot(211)
    # R = param[1]
    # plt.plot(np.array(SLS[0])/1e4,SLS[1],marker='+',label='Detector A')
    # plt.plot(np.array(SLS[0])/1e4,SLS[2],marker='+',label='Detector B')
    ax.errorbar(Scatteringvectors,DetectorA,yerr=ErrA,marker='+',label='Detector A',c='red',ls='None',ms=5,capsize=4)
    # ax.plot(Scatteringvectorslist,fitA,ls='--',label='Fit Detector A, $R_g = %s \pm %s$ nm'%(round(param[1]*1e9,4),round(np.sqrt(np.diag(cov)[1]*1e9),4)), c ='green',lw=1)
    # ax.plot(Scatteringvectorslist,fitB,ls='--',label='Fit Detector B, $R_g = %s \pm %s$ nm'%(round(param2[1]*1e9,4),round(np.sqrt(np.diag(cov2)[1]*1e9),4)), c ='navy',lw=1)
    ax.errorbar(Scatteringvectors,DetectorB,yerr=ErrB,marker='+',label='Detector B',c='blue',ls='None',ms=5,capsize=4)
    ax.set_title('SLS Study')
    # ax.set_xlabel(r'$q (\mathrm{m}^{-1})$')
    # ax.set_ylabel(r'$\langle I \rangle /I_0$ (kHz)')
    # plt.xlabel(r'$qR$')
    #
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    # ax.errorbar(Scatteringvectors,DetectorA,yerr=ErrA,marker='+',label='Detector A',c='red',ls='None',ms=5,capsize=3)
    # ax.plot(Scatteringvectorslist,Debye(Scatteringvectorslist,*param),ls='--',label='Fit Detector A',c='green',lw=1)
    # ax.plot(Scatteringvectorslist,Debye(Scatteringvectorslist,*param2),ls='--',label='Fit Detector B',c='navy',lw=1)
    # ax.errorbar(Scatteringvectors,DetectorB,yerr=ErrB,marker='+',label='Detector B',c='blue',ls='None',ms=5,capsize=3)
    ax.set_xlabel(r'$q (\textrm{m}^{-1})$')
    ax.set_ylabel(r'$ \langle I \rangle /I_0$ (kHz)')
    plt.legend()
    plt.grid()
    ax2 = plt.subplot(212)
    Angleslist = np.linspace(np.min(Angles),np.max(Angles),1000)
    ax2.errorbar(Angles,DetectorA,yerr=ErrA,marker='+',label='Detector A',c='red',ls='None',ms=5,capsize=4,lw=1)
    # ax2.plot(Angleslist,fitB,ls='--',label='Fit Detector B',c='navy',lw=1)
    ax2.errorbar(Angles,DetectorB,yerr=ErrB,marker='+',label='Detector B',c='blue',ls='None',ms=5,capsize=4,lw=1)
    plt.legend()
    # ax2.set_xlim((19,27))
    plt.grid()
    ax2.set_xlabel(r'$\vartheta$, Scattering angle')
    ax2.set_ylabel(r'$\langle I \rangle /I_0$ (kHz)')
    ax2.set_yscale('log')
    # plt.tight_layout()
    # print('Radius of form factor A : R = %s +- %s nm'%(round(param[1]*1e9,4),round(np.sqrt(np.diag(cov)[1]*1e9),4)))
    # print('Radius of form factor B : R = %s +- %s nm'%(round(param2[1]*1e9,4),round(np.sqrt(np.diag(cov2)[1]*1e9),4)))
    #%% Temporal study of the intensity
    # Import the data
    dossier = 'C:/DLS/120723_Pierre/Intensity_Temporal'
    angle = [f.path for f in os.scandir(dossier) if f.is_dir() and f.path.endswith('CSV')]
    experiment = [f.path for f in os.scandir(angle) if f.is_dir()][0]
    fichier = pd.read_csv(experiment+'/'+'Raw Data.csv',sep=',',skiprows=5)
    limit = np.where(fichier['Count Trace ChA [kHz]'].str.find('Lag').to_numpy(dtype=float)==0)[0][0]-1
    dataA = fichier.iloc[:limit,0].to_numpy(dtype=float)
    dataB = fichier.iloc[:limit,1].to_numpy(dtype=float)
    lagtime = fichier.iloc[limit+2:,0].to_numpy(dtype=float)
    correlation = fichier.iloc[limit+2:,1].to_numpy(dtype=float)
    angle_num = int(experiment[loc:loc+2])
    
    
    #%%
    exp = 5
    nbangle = len(angles)
    loc = 9
    SLS,Angles = reshape(importls(angles,loc)[0],importls(angles,loc)[1],exp,nbangle)
    Ian = np.mean(SLS[1,:,:],axis=1)
    Ibn = np.mean(SLS[2,:,:],axis=1)
   
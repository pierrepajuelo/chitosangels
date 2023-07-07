# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:11:14 2023

@author: Pierre PAJUELO
Analysis of the DATA from the rheometer
Update 06/06/23
"""
# Importation des modules
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.signal import welch
# Définition des fonctions
def fit_function(t,tau,b):
    return(np.exp(t/tau+b))
def fit_derivative(t,tau,b):
    return(np.exp(t/tau+b)/tau)
def fit_intercept(t,a,b):
    return(a*t+b)
def CI_intercept(t,Fn):
    a = 0.01
    b = 1
    return(a,b)
def CI_test(t,Fn):
    tau = 10
    b = 10
    return(tau,b)
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def loinormale(x,mu,sigma):
    return(1/(np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2))

def droite(t,a):
    return(a*t)
def powerlaw(g,a):
    return(a*g**3)
def analysis(normal_force,gap,time,ifile,folder,speed=0.001,record=False,
             moving=False,n_average=10,plot=False,tau_method=False,exponential=False):
    # BEFORE ANYTHING :
    # Calculating the speed 
    X = time
    Y = gap
    param,cov = curve_fit(fit_intercept,X,Y,p0=(0.001,10),maxfev=10000)
    print('Speed = %s mm/s'%(-param[0]))
    if exponential:
        # Ask the user to input the starting time
        plt.close('all')
        plt.figure('Choose starting time of exponential fit')
        plt.plot(time,normal_force,marker='+',ms=4,c='red')
        plt.xlabel('Time (s)')
        plt.ylabel(r'Normal Force, $F_{\mathrm{N}}$ (N)')
        plt.grid()
        plt.ylim((-0.05,0.05))
        # INPUT
        start_fit = np.where(time>=plt.ginput(n=1)[0][0])[0][0]
        # STOP TIME IS AUTOMATIC
        stop_fit = np.where(normal_force==np.max(normal_force))[0][-1]
        # MOVING AVERAGE
    if moving:
        # stoptime = np.where(time>=800)[0][0] # To stop before
        mov = moving_average(normal_force,n=n_average)
        normal_force = mov
        time = time[n_average-1:]
        gap = gap[n_average-1:]
    if plot: # Plotting the moving average
        plt.close('all')
        plt.figure()
        plt.plot(time,normal_force,c='red',label='Exp.')
        # plt.scatter(time[start_fit],normal_force[start_fit],c='blue',marker='+',label='Contact gap') 
        if moving:
            plt.plot(time[n_average-1:],mov,label='Mov. Aver.')
        plt.xlabel('Time (s)')
        plt.ylabel(r'Normal force, $F_{\mathrm{N}}$ (N)')
        plt.legend()
        plt.grid()
    
    if exponential:
        # FITTING AN EXPONENTIAL CURVE
        x = time[start_fit:stop_fit] # for aving the gap : *speed # time ==> gap (mm)
        y = normal_force[start_fit:stop_fit]
        param,covariance = curve_fit(fit_function,x,y,p0=CI_test(x,y),maxfev=100000)
        xi = min(x)
        xend = max(x)
        time_fit = np.linspace(xi,xend,200)
        fit = fit_function(time_fit,*param)
    
    if plot:
        plt.close('all')
        plt.figure()
        plt.plot(time,normal_force,c='red',label='Exp.')
        plt.plot(time_fit,fit,c='blue',ls='--',label='Fit exponential')
        plt.xlabel('Time (s)')
        plt.ylabel(r'Normal force, $F_{\mathrm{N}}$ (N)')
        plt.legend()
        plt.grid()
        # plt.ginput()
        # Other method  
        # dfit = fit_derivative(time_fit, *param)
        # gradient = np.gradient(y,edge_order=2) # pretty messy
    if tau_method:
        ntau = 3
        if param[1]<-7:
            ntau+=1
        start_fit = np.where(time>=ntau*param[0])[0][0]
        stop_fit = np.where(time>=(ntau+1)*param[0])[0][0]
    
    # Fitting the linear regime
    # Ask the user to point and click
    plt.close('all')
    plt.figure('Choose linear regime limits')
    plt.plot(time,normal_force,c='blue',label='Exp.')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Normal force, $F_{\mathrm{N}}$ (N)')
    plt.grid()
    plt.legend()
    plt.ylim((-0.05,0.05))
    
    inputvalues = plt.ginput(n=2)
    start_fit = np.where(time>=inputvalues[0][0])[0][0]
    stop_fit = np.where(time>=inputvalues[1][0])[0][0]
    
    # Previous versions
    # start_fit = 15000
    # start_fit = np.where(time>=608)[0][0] # 060223
    # stop_fit = 20000
    # stop_fit = np.where(time>=1412)[0][0] # 060223
    
    x = time[start_fit:stop_fit] # for aving the gap : *speed # time ==> gap (mm)
    y = normal_force[start_fit:stop_fit]
    param2,covariance2 = curve_fit(fit_intercept,x,y,p0=CI_intercept(x,y),maxfev=100000)
    xi = min(x)
    xend = max(x)
    time_fit2 = np.linspace(xi,xend,200)
    fit_inter = fit_intercept(time_fit2, *param2)
    gap_intercept = 10+param2[1]/param2[0]*speed # in mm # NOT NECESSARY
    print('Gap contact : ', gap_intercept, ' mm')
    print('Actual gap intercept :', gap[np.where(time>=-param2[1]/param2[0])][0], 'mm')
    contact_gap = gap[np.where(time>=-param2[1]/param2[0])][0]
    strain = (contact_gap-gap)/contact_gap
    print('Maximal strain :', strain[np.argmax(normal_force)])
    if plot:
        # Plotting the data
        plt.close('all')
        plt.figure(figsize=(6,4))
        plt.plot(time,normal_force,'+',label='Exp.')
        # plt.plot(data[:,3],data[:,2],'+',label='Gap')
        plt.plot(time_fit,fit,'lime',ls='--',label=r'Fit exponential, $\tau = %s \pm %s\,\mathrm{s}$'%(round(param[0],2),round(np.sqrt(np.diag(covariance)[0]),2)))
        # plt.plot(time_fit,dfit,label='fit derivative')
        plt.plot(time_fit2,fit_inter,'red',ls='-',label='Fit line')
        # plt.plot(x,gradient,label='gradient')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel(r'Normal Force $F_{\mathrm{N}}$ (N)')    
        plt.grid()
        # plt.title('File %s'%(files[ifile-1]))
        plt.title('Compression at %s'%round(0.001249,4)+' mm/s')
        plt.tight_layout()
        # plt.ginput()
    # plt.savefig(folder+'/IMAGES/'+str(ifile)+'.pdf',format='pdf')
    
    # STRAIN STRESS
    surface = (1.5e-3/2)**2*np.pi
    # print('Elastic modulus : %s Pa'%(np.max(normal_force)/(surface*strain[np.argmax(normal_force)])))
    stress = normal_force/surface
    
    interval_fit = np.where((strain>0) & (strain<0.2))[0]
    X = strain[interval_fit]
    Y = stress[interval_fit]
    param,cov = curve_fit(droite,X,Y,p0=(10e3),maxfev=10000)
    
    plt.close('all')
    plt.figure(figsize=(6,4))
    plt.plot(strain,stress,'+',label='Exp.')
    plt.axvline(x=0,ls='--',c='red',lw=1)
    plt.plot(X,droite(X,*param),label=r'Linear Regression, $E = %s \pm %s$ kPa'%(round(param[0]/1000,2),round(np.sqrt(np.diag(cov)[0])/1000,2)))
    # begin_powerlaw = np.where(strain>=0.3)[0][0]
    # param3,cov3 = curve_fit(powerlaw,strain[begin_powerlaw:np.argmax(stress)],stress[begin_powerlaw:np.argmax(stress)],p0=(1000),maxfev=10000)
    # plt.plot(strain[begin_powerlaw:np.argmax(stress)],powerlaw(strain[begin_powerlaw:np.argmax(stress)],*param3),'lime',ls='--',label=r'Power law, $\sigma = \gamma^{%s}$'%(param3[0]))
    # plt.plot(strain[begin_powerlaw:np.argmax(stress)],powerlaw(strain[begin_powerlaw:np.argmax(stress)],2),
             # 'lime',ls='--',label=r'Power law, $\sigma = \gamma^{2}$')
    plt.legend()
    plt.grid()
    plt.xlabel(r'Strain, $\gamma$')
    plt.ylabel(r'Stress, $\sigma$ (Pa)')
    # plt.title('Fracture experiment at 0.001 mm/s')
    plt.title('Fracture experiment at %s'%round(0.001249,4)+' mm/s')
    plt.tight_layout()
    plt.ginput()
    # plt.savefig(folder+'/IMAGES'+str(ifile)+'.pdf', format='pdf')
    print('Elastic modulus : %s Pa'%(param[0]))
    print('Maximal normal force : %s N'%(np.max(normal_force)))
    
    return(param[0])
    
# Programme principal
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
    
    # Folder and data collection
    # folder = "C:/Users/pierr/iCloudDrive/Stages/Stage_M1/Chitosan_Gels_Alexis_DelaCotte/DATA_TEST" # DATA TESTS
    folder = 'D:/STAGE M1/CHITOSAN/Fracture/Small Gels'
    # files = [f for f in os.listdir(folder)[:-1] if f.endswith('.csv')]
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    # file = files[2]
    ANALYSIS = np.zeros((len(files)+1,5),dtype=object)
    ANALYSIS[0,0] = 'File name'
    ANALYSIS[0,1] = 'Max. Normal Force in N'
    ANALYSIS[0,2] = 'Contact gap in mm'
    ANALYSIS[0,3] = 'Incertitude of Contact gap in mm'
    ANALYSIS[0,4] = 'Break gap in mm'
    Elastic = []
    plt.close('all')
    for ifile,file in enumerate(files) :
        ifile += 1
        ANALYSIS[ifile,0] = files[ifile-1]
        # Import the data
        data = pd.read_csv(folder+'/'+file,sep='\t',skiprows=9,encoding='utf-16')
        
        # Clean the data
        # Remove nan values
        nan = np.where(np.isnan(data.iloc[:,1].to_numpy(dtype=float))==True)[0]
        if len(nan)!=0:
            for rem in nan:
                data = data.drop(rem)
                
        # data = data.drop(data.index[0])
        data = data.drop(columns=data.columns[0])
        # limit = np.where(data['Unnamed: 5'].astype(str).str.find('invalid').to_numpy(dtype=float)==0)[0]
        
        # Extract !
        normal_force = data.iloc[:,1].to_numpy(dtype=float)
        gap = data.iloc[:,2].to_numpy(dtype=float) # in mm.
        time = data.iloc[:,3].to_numpy(dtype=float) # in sec.
        # plt.figure(figsize=(6,4))
        # plt.plot(gap,normal_force)
        # plt.xlabel('Gap (mm)')
        # plt.ylabel(r'Normal force, $F_{\mathrm{N}}$ (mN)')
        # plt.grid()
        # plt.tight_layout()
        Elastic.append(analysis(normal_force, gap, time,ifile,folder,plot=True,exponential=True))
    
    print('Mean value : %s Pa'%(np.mean(Elastic)))
    print('Error at 95 percent: %s Pa'%(np.std(Elastic,ddof=1)*1.96))
        # # Just to verify
        # if ifile==0:
        #     plt.close('all')
        # plt.figure('File %s'%(files[ifile-1]))
        # plt.plot(time,normal_force)
        
        # X = time
        # Y = gap
        # param,cov = curve_fit(fit_intercept,X,Y,p0=(0.001,10),maxfev=10000)
        # print('File %s'%(files[ifile-1]))
        # print('Speed = %s mm/s'%(-param[0]))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# BROUILLON
        # Fitting the data
        
        # stop_fit = np.argmax(normal_force)
        # start_fit = np.where(normal_force>0.005)[0][0]
        
        # # HISTOGRAMME
        # if ifile==len(files)-1:
        #     
            
        #     bruit = normal_force[n_average-1:stop]-mov
        #     plt.close('all')
        #     plt.figure()
        #     plt.plot(time[n_average-1:stop],bruit)
        #     plt.figure()
        #     hist = np.histogram(bruit/np.std(bruit),bins=100,density=True)
        #     X = hist[1][:-1]
        #     Y = hist[0]
        #     plt.plot(X,Y,label='Hist.')
        #     param,cov = curve_fit(loinormale,X,Y,p0=(1e-2,10),maxfev=10000)
        #     plt.plot(X,loinormale(X,*param),label='Fit')
        #     plt.legend()
        #     N = len(normal_force)
        #     b = np.random.randn(N)*(param[1])**2
        #     freq,gammab = welch(b,fs=128,nperseg=32,nfft=N,return_onesided=False)
        #     freq,gammay = welch(normal_force,fs=128,nperseg=32,nfft=N,return_onesided=False)
        #     hfreq = (gammay-gammab)/gammay
        #     xdebruit = np.abs(np.fft.ifft(hfreq*np.fft.fft(normal_force)))
        #     plt.figure()
        #     plt.plot(time,normal_force)
        #     plt.figure()
        #     plt.plot(time,xdebruit)
        # if ifile==len(files)-2:
            
            # plt.figure()
            # plt.plot(time[n_average-1:stoptime],np.gradient(mov,edge_order=2),label='Gradient')
            # plt.xlabel('Time (s)')
            # plt.ylabel(r'Gradient Normal force, $F_{\mathrm{N}}$ (N)')
            # plt.legend()
            # plt.grid()
            # start_fit = 0
            
        
        
        
        # if ifile!=len(files)-2:
        #     continue

            
        # speed = 0.001 # mm/s
        
        
        
        
        
        
        
        
            # print(np.max(data[:,1]))
        # plt.close('all')
        # ANALYSIS[ifile,1] = np.max(data[:,1]) # Adding the max. of normal force
        # ANALYSIS[ifile,2] = data[np.where(time[:,3]>=-param2[1]/param2[0]),2][0][0] # Adding the intercept, which stands for the contact gap in mm
        # ANALYSIS[ifile,3] = data[np.where(data[:,3]>=-param2[1]/param2[0]),2][0][0]*np.sqrt((np.diag(covariance2)[0]/param2[0])**2+
        #                                                                                     (np.diag(covariance2)[1]/param2[1])**2) # Incertitude on the contact gap
        # ANALYSIS[ifile,4] = data[np.where(data[:,1]==np.max(data[:,1]))[0][0],2] # Adding the max gap corresponding of normal force maximum
    # np.savetxt(folder+'/IMAGES/Results.txt',ANALYSIS,fmt='%s',delimiter=';')
        
    
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:28:25 2023

@author: Pierre PAJUELO
@subject: Data and analysis of evaporation results on chitosan gels
@version: 01/08/23 (Last Version)
"""
# Importation des modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# Définition des fonctions
def incert(sech,sigma_sec,dry,sigm_dry,total):
    """
    Returns the uncertainty of the ratio according the other uncertainties

    Parameters
    ----------
    sech : array or float
        Numerator of the ratio.
    sigma_sec : array or float
        Uncertainty of the numerator.
    dry : array or float
        Denominator of the ratio.
    sigm_dry : array or float
        Uncertainty of the denominator.
    total : array or float
        Value of the ratio.

    Returns
    -------
    Uncertainty of the ratio.

    """
    return(total*np.sqrt((sigma_sec/sech)**2+(sigm_dry/dry)**2))

# Programme principal
if __name__=='__main__':
    # FIGURES OPTIONS
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{lmodern} \usepackage{bm} \usepackage{xcolor}')
    #Options
    params = {'text.usetex' : True,
              'font.size' : 20,
              'font.family' : 'lmodern',
              }
    plt.rcParams.update(params)
    mpl.rcParams['axes.linewidth'] = 1.
    # DATA
    # FIRST BATCH (W9a)
    percentage = np.array([0.64958,0.64141,0.6297,0.4955,
                           0.4092,0.4047,0.37,0.352,0.33881,0.3387,
                           0.2494,0.228,0.17,0.14,0.11827,0.064,0.0603,0.06,0.00458])     
    recov = np.array([0.4858,0.5433,0.5171,0.4201,
                      0.4101,0.4969,0.4214,0.4651,0.5182,0.5017,
                      0.4590,0.5096,0.4087,0.4326,0.4534,0.4661,0.4320,0.2641,0.1443])
    dry = np.array([0.3029,0.3073,0.3127,0.1915,
                    0.1545,0.1609,0.1575,0.1574,0.1587,0.1654,
                    0.1155,0.1003,0.079,0.069,0.0525,0.0356,0.0332,0.0271,0.0018])
    init = np.array([0.4663,0.4791,0.4966,0.3865,
                     0.3776,0.3976,0.4278,0.4473,0.4684,0.4883,
                     0.4631,0.4409,0.4615,0.4802,0.4439,0.5525,0.5508,0.4476,0.3926])
    # FIRST BATCH but with wrong protocol
    percentage2 = np.array([0.75,0.5,0.25])
    recov2 = np.array([0.5200,0.3860,0.167])
    dry2 = np.array([0.3117,0.23,0.0625])
    init2 = np.array([0.4156,0.4591,0.25])
    # FIRST BATCH but with wrong measurements
    percentage3 = np.array([0.44482,0.30744])
    recov3 = np.array([0.4193,0.3174])
    dry3 = np.array([0.2124,0.1199])
    init3 = np.array([0.4775,0.39])
    # SECOND BATCH : W10 
    percentage4 = np.array([0.64343,0.38081,0.3285,0.3146,0.2761,0.2644,0.25,0.2461,
                            0.24026,0.2315,0.2308,0.2301,0.2255,
                            0.2183,0.2,0.1709,0.1361,0.1237,
                            0.10721,0.0963,0.0491,0.0486,0.0458])
    recov4 = np.array([0.5497,0.5123,0.5193,0.5705,0.3160,0.5233,0.6148,0.6711,
                       0.5328,0.4668,0.6790,0.4739,0.7408,
                       0.4645,0.4937,0.5503,0.4763,0.4902,
                           0.4086,0.5138,0.4239,0.3929,0.3064])
    dry4 = np.array([0.3739,0.1973,0.1685,0.1886,0.0981,0.1412,0.1558,0.1717,
                     0.1313,0.115,0.1657,0.1226,0.1737,
                     0.108,0.1077,0.1,0.0717,0.068,
                         0.0492,0.0527,0.025,0.0249,0.0187])
    init4 = np.array([0.5811,0.5181,0.5129,0.5995,0.3553,0.5340,0.6230,0.6976,
                      0.5465,0.4968,0.7180,0.5328,0.7703,
                      0.4947,0.5383,0.5853,0.5267,0.5497,
                      0.4589,0.5475,0.5095,0.5123,0.4086])
    
    sigma_sech = 0.01
    sigma_pertes = 0.001
    
    # CALCULUS
    recovery_ratio = recov/dry
    recovery_err = incert(recov,sigma_sech,dry,sigma_pertes,recovery_ratio)
    initial_mass_ratio = recov/init
    initial_mass_err = incert(recov,sigma_sech,init,sigma_sech,initial_mass_ratio)
    percent_err = incert(dry,sigma_pertes,init,sigma_sech,percentage)
    
    recovery_ratio2 = recov2/dry2
    recovery_err2 = incert(recov2,sigma_sech,dry2,sigma_pertes,recovery_ratio2)
    initial_mass_ratio2 = recov2/init2
    initial_mass_err2 = incert(recov2,sigma_sech,init2,sigma_sech,initial_mass_ratio2)
    percent_err2 = incert(dry2,sigma_pertes,init2,sigma_sech,percentage2)
    
    sigma_sech = 0.06
    sigma_pertes = 0.1
    recovery_ratio3 = recov3/dry3
    recovery_err3 = incert(recov3,sigma_sech,dry3,sigma_pertes,recovery_ratio3)
    initial_mass_ratio3 = recov3/init3
    initial_mass_err3 = incert(recov3,sigma_sech,init3,sigma_sech,initial_mass_ratio3)
    percent_err3 = incert(dry3,sigma_pertes,init3,sigma_sech,percentage3)
    
    sigma_sech = 0.01
    sigma_pertes = 0.001
    recovery_ratio4 = recov4/dry4
    recovery_err4 = incert(recov4,sigma_sech,dry4,sigma_pertes,recovery_ratio4)
    initial_mass_ratio4 = recov4/init4
    initial_mass_err4 = incert(recov4,sigma_sech,init4,sigma_sech,initial_mass_ratio4)
    percent_err4 = incert(dry4,sigma_pertes,init4,sigma_sech,percentage4)
    
    # PLOT
    plt.close('all')
    fig = plt.figure(figsize=(13,8))
    ax = plt.subplot(111)
    ax.errorbar(percentage,recovery_ratio,xerr=percent_err, yerr=recovery_err,lw=1,
                marker='o',ms=5,c='red',capsize=3,label='New protocol',ecolor='navy')
    ax.errorbar(percentage2,recovery_ratio2,xerr=percent_err2,yerr=recovery_err2,lw=1,
                marker='o',ms=5,c='chocolate',capsize=3,ls='None',label='Old protocol',ecolor='rosybrown')
    ax.errorbar(percentage3,recovery_ratio3,xerr=percent_err3,yerr=recovery_err3,lw=1,
                marker='o',ms=5,c='orange',capsize=3,ls='None',label='Incorrect measurement',ecolor='gray')
    ax.errorbar(percentage4,recovery_ratio4,xerr=percent_err4,yerr=recovery_err4,lw=1,
                marker='o',ms=5,c='gold',capsize=3,label='W10',ecolor='navy')
    ax.set_xlabel('Percentage of initial mass, $m_{\mathrm{dry}}/m_{\mathrm{init.}}$')
    ax.set_ylabel(r'Recovery ratio, $m_{\mathrm{recov.}}/m_{\mathrm{dry}}$')
    percentage_list = np.linspace(1e-10,1,100)
    ax.plot(percentage_list,1/percentage_list,c='black',ls='--',label='Theoritical, perfect recovery')
    ax.yaxis.label.set_color('red') 
    ax.set_ylim((0,np.max(recovery_ratio)+2))
    ax2 = ax.twinx()
    ax2.errorbar(percentage,initial_mass_ratio,xerr=percent_err,yerr=initial_mass_err,lw=1,
                marker='o',ms=5,c='blue',capsize=3,label='New protocol',ecolor='navy')
    ax2.errorbar(percentage2,initial_mass_ratio2,xerr=percent_err2,yerr=initial_mass_err2,lw=1,
                marker='o',ms=5,c='purple',capsize=3,ls='None',label='Old protocol',ecolor='rosybrown')
    ax2.errorbar(percentage3,initial_mass_ratio3,xerr=percent_err3,yerr=initial_mass_err3,lw=1,
                marker='o',ms=5,c='black',capsize=3,ls='None',label='Incorrect measurement',ecolor='gray')
    ax2.errorbar(percentage4,initial_mass_ratio4,xerr=percent_err4,yerr=initial_mass_err4,lw=1,
                marker='o',ms=5,c='teal',capsize=3,label='W10',ecolor='navy')
    ax2.set_ylabel(r'Irreversibility ratio, $m_{\mathrm{recov.}}/m_{\mathrm{init.}}$')
    ax2.axvline(x=0.245,c='red',ls=':',label=r'Non-return point, $\%_{\mathrm{init.}}=24.5$')
    ax2.legend(bbox_to_anchor=(0.55, 0.6, 0, 0),fontsize="15")
    ax.legend(bbox_to_anchor=(0.3, 0.1, 0, 0),fontsize="15")
    ax2.axhline(y=1,ls='--')
    ax2.yaxis.label.set_color('blue')
    ax2.set_ylim((0,1.4))
    left, bottom, width, height = [0.65, 0.2, 0.2, 0.2]
    ax3 = fig.add_axes([left, bottom, width, height])
    # ax3.set_xlabel(r'$m_{\mathrm{dry}}/m_{\mathrm{init.}}$')
    # ax3.set_ylabel(r'$m_{\mathrm{recov.}}/m_{\mathrm{dry}}$')
    ax3.errorbar(percentage,initial_mass_ratio,xerr=percent_err,yerr=initial_mass_err,lw=1,
                marker='o',ms=5,c='blue',capsize=3,label='New protocol',ecolor='navy')
    ax3.errorbar(percentage4,initial_mass_ratio4,xerr=percent_err4,yerr=initial_mass_err4,lw=1,
                marker='o',ms=5,c='teal',capsize=3,label='W10',ecolor='navy')
    ax3.axvline(x=0.245,c='red',ls=':')
    ax3.axhline(y=1,ls='--')
    ax3.set_xlim((0.15,0.35))
    ax3.set_ylim((0.841,1.1))
    ax3.set_title('Zoom in')
    plt.tight_layout()
    # plt.savefig('D:/Downloads/Evaporation_results.png', transparent = True)
    #%%
    # STABILIZATION STUDY
    # DATA
    N_times = 3
    init_evap = np.hstack([np.array([[0.4278],[0.4802]])]*N_times)
    dry_evap = np.hstack([np.array([[0.1575],[0.069]])]*N_times)
    recov_evap = np.array([[0.4434,0.4214,0.4152],[0.3890,0.4326,0.3947]])
    sigma_sech = 0.01
    sigma_pertes = 0.01
    
    # CALCULUS
    percentage_evap = np.hstack([np.array([[0.37],[0.14]])]*N_times)
    recovery_ratio_evap = recov_evap/dry_evap
    recovery_err_evap = incert(dry_evap,sigma_sech,dry_evap,sigma_pertes,recovery_ratio_evap)
    initial_mass_ratio_evap = recov_evap/init_evap
    initial_mass_err_evap = incert(recov_evap,sigma_sech,init_evap,sigma_sech,initial_mass_ratio_evap)
    
    plt.figure()
    
    ax = plt.subplot(111)
    color = plt.cm.rainbow(np.linspace(0, 0.4, 3))
    for i, c in zip(range(3), color):
        ax.errorbar(percentage_evap[:,i],recovery_ratio_evap[:,i],yerr=recovery_ratio_evap[:,i],lw=1,
                    marker='o',ms=5,c=c,capsize=3)
    ax.set_xlabel('Percentage of initial mass')
    ax.set_ylabel(r'Recovery ratio, $m_{\mathrm{recov.}}/m_{\mathrm{dry}}$')
    ax.yaxis.label.set_color('red') 
    ax.set_ylim((0,np.max(recovery_ratio_evap)+2))
    ax2 = ax.twinx()
    color = plt.cm.rainbow(np.linspace(0.5, 1, 3))
    for i, c in zip(range(3), color):
        ax2.errorbar(percentage_evap[:,i],initial_mass_ratio_evap[:,i],yerr=initial_mass_err_evap[:,i],lw=1,
                    marker='o',ms=5,c=c,capsize=3)
    ax2.set_ylabel(r'Irreversibility ratio, $m_{\mathrm{recov.}}/m_{\mathrm{init.}}$')
    ax2.yaxis.label.set_color('blue')
    ax2.set_ylim((0,1.4))
    plt.tight_layout()
    # plt.savefig('D:/Downloads/Evaporation_results.png', transparent = True)
    
    plt.figure()
    
    # TIME SCALE
    begin = datetime(2023,6,7,14,27)
    first_meas = datetime(2023,6,9,17)
    second_meas = datetime(2023,6,12,11,20)
    third_meas = datetime(2023,6,13,14,30)
    time_list = np.array([(first_meas-begin).total_seconds(),(second_meas-begin).total_seconds(),(third_meas-begin).total_seconds()])/(3600*24)
    plt.errorbar(time_list,recovery_ratio_evap[0,:],yerr=recovery_err_evap[0,:],c='blue',marker='o',ms=5,capsize=3)
    plt.errorbar(time_list,recovery_ratio_evap[1,:],yerr=recovery_err_evap[1,:],c='red',marker='o',ms=5,capsize=3)
    plt.xlabel('Time of recovery (days)')
    plt.ylabel(r'Recovery ratio, $m_{\mathrm{recov.}}/m_{\mathrm{dry}}$')
    plt.ylim((0,12))

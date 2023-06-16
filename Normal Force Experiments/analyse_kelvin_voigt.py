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
###
from collections import namedtuple
import numpy as np
from scipy import stats, special


def chisquare(f_obs, f_exp=None, ddof=0, axis=0):
    """
    Calculate a one-way chi-square test.
    The chi-square test tests the null hypothesis that the categorical data
    has the given frequencies.
    Parameters
    ----------
    f_obs : array_like
        Observed frequencies in each category.
    f_exp : array_like, optional
        Expected frequencies in each category.  By default the categories are
        assumed to be equally likely.
    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0.
    axis : int or None, optional
        The axis of the broadcast result of `f_obs` and `f_exp` along which to
        apply the test.  If axis is None, all values in `f_obs` are treated
        as a single data set.  Default is 0.
    Returns
    -------
    chisq : float or ndarray
        The chi-squared test statistic.  The value is a float if `axis` is
        None or `f_obs` and `f_exp` are 1-D.
    p : float or ndarray
        The p-value of the test.  The value is a float if `ddof` and the
        return value `chisq` are scalars.
    See Also
    --------
    scipy.stats.power_divergence
    Notes
    -----
    This test is invalid when the observed or expected frequencies in each
    category are too small.  A typical rule is that all of the observed
    and expected frequencies should be at least 5.
    The default degrees of freedom, k-1, are for the case when no parameters
    of the distribution are estimated. If p parameters are estimated by
    efficient maximum likelihood then the correct degrees of freedom are
    k-1-p. If the parameters are estimated in a different way, then the
    dof can be between k-1-p and k-1. However, it is also possible that
    the asymptotic distribution is not chi-square, in which case this test
    is not appropriate.
    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 8.
           https://web.archive.org/web/20171022032306/http://vassarstats.net:80/textbook/ch8pt1.html
    .. [2] "Chi-squared test", https://en.wikipedia.org/wiki/Chi-squared_test
    Examples
    --------
    When just `f_obs` is given, it is assumed that the expected frequencies
    are uniform and given by the mean of the observed frequencies.
    >>> from scipy.stats import chisquare
    >>> chisquare([16, 18, 16, 14, 12, 12])
    (2.0, 0.84914503608460956)
    With `f_exp` the expected frequencies can be given.
    >>> chisquare([16, 18, 16, 14, 12, 12], f_exp=[16, 16, 16, 16, 16, 8])
    (3.5, 0.62338762774958223)
    When `f_obs` is 2-D, by default the test is applied to each column.
    >>> obs = np.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
    >>> obs.shape
    (6, 2)
    >>> chisquare(obs)
    (array([ 2.        ,  6.66666667]), array([ 0.84914504,  0.24663415]))
    By setting ``axis=None``, the test is applied to all data in the array,
    which is equivalent to applying the test to the flattened array.
    >>> chisquare(obs, axis=None)
    (23.31034482758621, 0.015975692534127565)
    >>> chisquare(obs.ravel())
    (23.31034482758621, 0.015975692534127565)
    `ddof` is the change to make to the default degrees of freedom.
    >>> chisquare([16, 18, 16, 14, 12, 12], ddof=1)
    (2.0, 0.73575888234288467)
    The calculation of the p-values is done by broadcasting the
    chi-squared statistic with `ddof`.
    >>> chisquare([16, 18, 16, 14, 12, 12], ddof=[0,1,2])
    (2.0, array([ 0.84914504,  0.73575888,  0.5724067 ]))
    `f_obs` and `f_exp` are also broadcast.  In the following, `f_obs` has
    shape (6,) and `f_exp` has shape (2, 6), so the result of broadcasting
    `f_obs` and `f_exp` has shape (2, 6).  To compute the desired chi-squared
    statistics, we use ``axis=1``:
    >>> chisquare([16, 18, 16, 14, 12, 12],
    ...           f_exp=[[16, 16, 16, 16, 16, 8], [8, 20, 20, 16, 12, 12]],
    ...           axis=1)
    (array([ 3.5 ,  9.25]), array([ 0.62338763,  0.09949846]))
    """
    return power_divergence(f_obs, f_exp=f_exp, ddof=ddof, axis=axis,
                            lambda_="pearson")


# Map from names to lambda_ values used in power_divergence().
_power_div_lambda_names = {
    "pearson": 1,
    "log-likelihood": 0,
    "freeman-tukey": -0.5,
    "mod-log-likelihood": -1,
    "neyman": -2,
    "cressie-read": 2/3,
}


def _count(a, axis=None):
    """
    Count the number of non-masked elements of an array.
    This function behaves like np.ma.count(), but is much faster
    for ndarrays.
    """
    if hasattr(a, 'count'):
        num = a.count(axis=axis)
        if isinstance(num, np.ndarray) and num.ndim == 0:
            # In some cases, the `count` method returns a scalar array (e.g.
            # np.array(3)), but we want a plain integer.
            num = int(num)
    else:
        if axis is None:
            num = a.size
        else:
            num = a.shape[axis]
    return num


Power_divergenceResult = namedtuple('Power_divergenceResult',
                                    ('statistic', 'pvalue'))


def power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=None):
    """
    Cressie-Read power divergence statistic and goodness of fit test.
    This function tests the null hypothesis that the categorical data
    has the given frequencies, using the Cressie-Read power divergence
    statistic.
    Parameters
    ----------
    f_obs : array_like
        Observed frequencies in each category.
    f_exp : array_like, optional
        Expected frequencies in each category.  By default the categories are
        assumed to be equally likely.
    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0.
    axis : int or None, optional
        The axis of the broadcast result of `f_obs` and `f_exp` along which to
        apply the test.  If axis is None, all values in `f_obs` are treated
        as a single data set.  Default is 0.
    lambda_ : float or str, optional
        The power in the Cressie-Read power divergence statistic.  The default
        is 1.  For convenience, `lambda_` may be assigned one of the following
        strings, in which case the corresponding numerical value is used::
            String              Value   Description
            "pearson"             1     Pearson's chi-squared statistic.
                                        In this case, the function is
                                        equivalent to `stats.chisquare`.
            "log-likelihood"      0     Log-likelihood ratio. Also known as
                                        the G-test [3]_.
            "freeman-tukey"      -1/2   Freeman-Tukey statistic.
            "mod-log-likelihood" -1     Modified log-likelihood ratio.
            "neyman"             -2     Neyman's statistic.
            "cressie-read"        2/3   The power recommended in [5]_.
    Returns
    -------
    statistic : float or ndarray
        The Cressie-Read power divergence test statistic.  The value is
        a float if `axis` is None or if` `f_obs` and `f_exp` are 1-D.
    pvalue : float or ndarray
        The p-value of the test.  The value is a float if `ddof` and the
        return value `stat` are scalars.
    See Also
    --------
    chisquare
    Notes
    -----
    This test is invalid when the observed or expected frequencies in each
    category are too small.  A typical rule is that all of the observed
    and expected frequencies should be at least 5.
    When `lambda_` is less than zero, the formula for the statistic involves
    dividing by `f_obs`, so a warning or error may be generated if any value
    in `f_obs` is 0.
    Similarly, a warning or error may be generated if any value in `f_exp` is
    zero when `lambda_` >= 0.
    The default degrees of freedom, k-1, are for the case when no parameters
    of the distribution are estimated. If p parameters are estimated by
    efficient maximum likelihood then the correct degrees of freedom are
    k-1-p. If the parameters are estimated in a different way, then the
    dof can be between k-1-p and k-1. However, it is also possible that
    the asymptotic distribution is not a chisquare, in which case this
    test is not appropriate.
    This function handles masked arrays.  If an element of `f_obs` or `f_exp`
    is masked, then data at that position is ignored, and does not count
    towards the size of the data set.
    .. versionadded:: 0.13.0
    References
    ----------
    .. [1] Lowry, Richard.  "Concepts and Applications of Inferential
           Statistics". Chapter 8.
           https://web.archive.org/web/20171015035606/http://faculty.vassar.edu/lowry/ch8pt1.html
    .. [2] "Chi-squared test", https://en.wikipedia.org/wiki/Chi-squared_test
    .. [3] "G-test", https://en.wikipedia.org/wiki/G-test
    .. [4] Sokal, R. R. and Rohlf, F. J. "Biometry: the principles and
           practice of statistics in biological research", New York: Freeman
           (1981)
    .. [5] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
           pp. 440-464.
    Examples
    --------
    (See `chisquare` for more examples.)
    When just `f_obs` is given, it is assumed that the expected frequencies
    are uniform and given by the mean of the observed frequencies.  Here we
    perform a G-test (i.e. use the log-likelihood ratio statistic):
    >>> from scipy.stats import power_divergence
    >>> power_divergence([16, 18, 16, 14, 12, 12], lambda_='log-likelihood')
    (2.006573162632538, 0.84823476779463769)
    The expected frequencies can be given with the `f_exp` argument:
    >>> power_divergence([16, 18, 16, 14, 12, 12],
    ...                  f_exp=[16, 16, 16, 16, 16, 8],
    ...                  lambda_='log-likelihood')
    (3.3281031458963746, 0.6495419288047497)
    When `f_obs` is 2-D, by default the test is applied to each column.
    >>> obs = np.array([[16, 18, 16, 14, 12, 12], [32, 24, 16, 28, 20, 24]]).T
    >>> obs.shape
    (6, 2)
    >>> power_divergence(obs, lambda_="log-likelihood")
    (array([ 2.00657316,  6.77634498]), array([ 0.84823477,  0.23781225]))
    By setting ``axis=None``, the test is applied to all data in the array,
    which is equivalent to applying the test to the flattened array.
    >>> power_divergence(obs, axis=None)
    (23.31034482758621, 0.015975692534127565)
    >>> power_divergence(obs.ravel())
    (23.31034482758621, 0.015975692534127565)
    `ddof` is the change to make to the default degrees of freedom.
    >>> power_divergence([16, 18, 16, 14, 12, 12], ddof=1)
    (2.0, 0.73575888234288467)
    The calculation of the p-values is done by broadcasting the
    test statistic with `ddof`.
    >>> power_divergence([16, 18, 16, 14, 12, 12], ddof=[0,1,2])
    (2.0, array([ 0.84914504,  0.73575888,  0.5724067 ]))
    `f_obs` and `f_exp` are also broadcast.  In the following, `f_obs` has
    shape (6,) and `f_exp` has shape (2, 6), so the result of broadcasting
    `f_obs` and `f_exp` has shape (2, 6).  To compute the desired chi-squared
    statistics, we must use ``axis=1``:
    >>> power_divergence([16, 18, 16, 14, 12, 12],
    ...                  f_exp=[[16, 16, 16, 16, 16, 8],
    ...                         [8, 20, 20, 16, 12, 12]],
    ...                  axis=1)
    (array([ 3.5 ,  9.25]), array([ 0.62338763,  0.09949846]))
    """
    # Convert the input argument `lambda_` to a numerical value.
    if isinstance(lambda_, str):
        if lambda_ not in _power_div_lambda_names:
            names = repr(list(_power_div_lambda_names.keys()))[1:-1]
            raise ValueError("invalid string for lambda_: {0!r}.  Valid strings "
                             "are {1}".format(lambda_, names))
        lambda_ = _power_div_lambda_names[lambda_]
    elif lambda_ is None:
        lambda_ = 1

    f_obs = np.asanyarray(f_obs)

    if f_exp is not None:
        f_exp = np.asanyarray(f_exp)
    else:
        # Ignore 'invalid' errors so the edge case of a data set with length 0
        # is handled without spurious warnings.
        with np.errstate(invalid='ignore'):
            f_exp = f_obs.mean(axis=axis, keepdims=True)

    # `terms` is the array of terms that are summed along `axis` to create
    # the test statistic.  We use some specialized code for a few special
    # cases of lambda_.
    if lambda_ == 1:
        # Pearson's chi-squared statistic
        terms = (f_obs.astype(np.float64) - f_exp)**2 / f_exp
    elif lambda_ == 0:
        # Log-likelihood ratio (i.e. G-test)
        terms = 2.0 * special.xlogy(f_obs, f_obs / f_exp)
    elif lambda_ == -1:
        # Modified log-likelihood ratio
        terms = 2.0 * special.xlogy(f_exp, f_exp / f_obs)
    else:
        # General Cressie-Read power divergence.
        terms = f_obs * ((f_obs / f_exp)**lambda_ - 1)
        terms /= 0.5 * lambda_ * (lambda_ + 1)

    stat = terms.sum(axis=axis)

    num_obs = _count(terms, axis=axis)
    ddof = np.asarray(ddof)
    p = stats.distributions.chi2.sf(stat, num_obs - 1 - ddof)

    return Power_divergenceResult(stat, p)
###
# Principal program
if __name__=='__main__':
    # Import the data
    folder = 'D:/STAGE M1/CHITOSAN/NORMALFORCE'
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
    time_stop = np.where(time>=2000)[0][0]
   
    
    Chi = []
    for i in tqdm(range(1750,10000)):
    # time_stop = np.where(time>=2000)[0][0] # prev. 4000
        X = time[time_start1:i]
        Y = gap[time_start1:i]
        param,cov = curve_fit(gaussian,X,Y,p0=(100,5),maxfev=10000)
    # plt.close('all')
    # plt.figure()
    # plt.plot(time,gap,c='green')
    # plt.plot(X,Y,label='Exp.',c='red')
    # plt.plot(X,gaussian(X,*param),c='blue')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Gap (mm)')
    # import scipy.stats as stats # For Chi test
        chi_square_test_statistic, p_value = chisquare(Y, gaussian(X,*param))
        Chi.append(chi_square_test_statistic)
    
    plt.close('all')
    plt.plot(np.arange(1750,10000),Chi)
    plt.xlabel('Time stop (s)')
    plt.yscale('log')
    plt.ylabel(r'$\chi$')
    #%%
    Yfit = gaussienne(X,*param)
    
    Xs = time1[time_start:]
    Ys = (gap1[time_start] - gap1[time_start:])/gap1[time_start]# Strain all
    
    time_goal = time_start + np.where(Ys>=0.63)[0][0]
    time_droite = time_start + np.where(time1>=7792)[0][0]
    # Kelvin-Voigt Model for creep compliance
    Xp = Xs[time_goal:]
    Yp = Ys[time_goal:]
    # Time stop for 6_43 PM
    time_stop = np.where(Xs>=16120)[0][0]
    # Xd = Xs[time_droite:]
    Xd = Xs[time_droite:time_stop]
    # Yd = Ys[time_droite:]
    Yd = Ys[time_droite:time_stop]
    
    param5,cov5 = curve_fit(line,Xd,Yd,p0=(1e-3,150),maxfev=10000)
    
    
   
    
    Xl = time1[time_start+np.where(Ys>0.35)[0][0]:time_start+np.where(Ys<0.6)[0][-1]]
    Yl = Ys[time_start+np.where(Ys>0.35)[0][0]:time_start+np.where(Ys<0.6)[0][-1]]
    param3,cov3 = curve_fit(line,Xl,Yl,p0=(1e-3,150),maxfev=10000)
    tau0 = 20e-3/(np.pi*(1e-3)**2)
    eta = tau0/param3[0]
    eta0 = tau0/param5[0]
    G0 = tau0/0.352
    Xp = Xp[:time_stop]
    Yp = Yp[:time_stop]
    
    param2,cov2 = curve_fit(kv,Xp,Yp,p0=(1e3,100),maxfev=10000)
    # param4,cov4 = curve_fit(burger, Xs, Ys, p0=(1e5),maxfev=10000)
    param4,cov4 = curve_fit(burger, Xs[:time_stop], Ys[:time_stop], p0=(1e5),maxfev=10000)
    param42,cov42 = curve_fit(line,Xs[time_stop:],Ys[time_stop:],p0=(1e-3,150),maxfev=10000)
    
    plt.close('all')
    
    # 
    plt.figure()
    plt.plot(Xd,Yd)
    plt.plot(Xd,line(Xd,*param5))
    # PLOT OF CALCULATED DATA   
    fig, ax1 = plt.subplots()
    fig.set_size_inches([9,9])
    
    ax1.plot(Xs,Ys,label='Exp. (All)')
    ax1.plot(Xl,line(Xl,*param3),label=r'Fit Line, $\eta = %s$'%(np.format_float_scientific(tau0/param3[0],precision=2)),ls='--',lw=4)
    ax1.plot(Xd,line(Xd,*param5),label=r'Fit Line, $\eta = %s$'%(np.format_float_scientific(tau0/param5[0],precision=2)),lw=4,ls='--')
    ax1.plot(Xs,burger(Xs,*param4),label=r'Fit Burger, $G= %s \pm %s$'%(round(param4[0],2),round(np.sqrt(np.diag(cov4)[0]),2)))
    ax1.plot(Xs,kv(Xs,*param2),label=r'Fit KV., $G=%s \pm %s$'%(round(param2[0],2),round(np.sqrt(np.diag(cov2)[0]),2)))
    ax1.plot(Xs[time_stop:],line(Xs[time_stop:],*param42),ls='--',c='red')
    ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'Strain, $\gamma$')
    ax1.set_title(r'Strain vs. Time for $F_{\mathrm{N}}=20\,\mathrm{mN}$')
    ax1.axvline(x=time_goal,ls='--',lw=1,c='red')
    
    left, bottom, width, height = [0.40, 0.25, 0.30, 0.30]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(Xp,kv(Xp,*param2),label=r'Fit KV., $G=%s \pm %s$'%(round(param2[0],2),round(np.sqrt(np.diag(cov2)[0]),2)))
    ax2.plot(Xp,burger(Xp,*param4),label=r'Fit Burger, $G= %s \pm %s$'%(round(param4[0],2),round(np.sqrt(np.diag(cov4)[0]),2)))
    ax2.plot(Xp,Yp,label='Exp. (End)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel(r'Strain, $\gamma$')
    ax2.set_title('Inset of End Region')
    ax2.set_xlim([150,10800])
    ax2.set_ylim([0.6,0.8])
    ax2.legend()

    # PLOT OF RAW DATA
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(time1,normalforce1,c='red')
    ax.set_xlabel('Time (sec.)')
    ax.set_ylabel(r'Normal force $F_{\mathrm{N}}$ (N)')
    ax2=ax.twinx()
    ax.yaxis.label.set_color('red') 
    ax2.yaxis.label.set_color('blue') 
    ax2.plot(time1,gap1,c='blue',label='Exp.')
    # ax2.plot(X,Yfit,c='blue',ls='--',label=r'Fit, $\tau = %s \pm %s$'%(round(param[0],2),round(np.sqrt(np.diag(cov)[0]),2)))
    plt.legend()
    ax2.set_ylabel('Gap (mm)')
    ax2.grid(ls='--')
    ax.grid(ls='-')
    
    
    
    # PLOT
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{lmodern} \usepackage{bm} \usepackage{xcolor}')
    #Options
    params = {'text.usetex' : True,
              'font.size' : 20,
              'font.family' : 'lmodern',
              }
    plt.rcParams.update(params)
    mpl.rcParams['axes.linewidth'] = 1.
    


#Created by Michael Cretignier 31.09.2023

import datetime

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def get_phase(array,period):
    new_array = np.sort((array%period))
    j0 = np.min(new_array)+(period-np.max(new_array))
    diff = np.diff(new_array)
    if len(diff):
        if np.max(diff)>j0:
            return 0.5*(new_array[np.argmax(diff)]+new_array[np.argmax(diff)+1])
        else:
            return 0
    else:
        return 0

def return_branching_phase(array):
    array1 = array%360
    array2 = (array1+180)%360-180
    if np.std(array1)<=np.std(array2):
        return array1
    else:
        return array2


def match_nearest(array1, array2,random=True):
    """return a table [idx1,idx2,num1,num2,distance] matching the closest element from two arrays. Remark : algorithm very slow by conception if the arrays are too large."""
    if type(array1)!=np.ndarray:
        array1 = np.array(array1)
    if type(array2)!=np.ndarray:
        array2 = np.array(array2)    
    if not (np.product(~np.isnan(array1))*np.product(~np.isnan(array2))):
        print('there is a nan value in your list, remove it first to be sure of the algorithme reliability')
    index1 = np.arange(len(array1))[~np.isnan(array1)] ; index2 = np.arange(len(array2))[~np.isnan(array2)]  
    array1 = array1[~np.isnan(array1)] ;  array2 = array2[~np.isnan(array2)]
    liste1 = np.arange(len(array1))[:,np.newaxis]*np.hstack([np.ones(len(array1))[:,np.newaxis],np.zeros(len(array1))[:,np.newaxis]])
    liste2 = np.arange(len(array2))[:,np.newaxis]*np.hstack([np.ones(len(array2))[:,np.newaxis],np.zeros(len(array2))[:,np.newaxis]])
    liste1 = liste1.astype('int') ; liste2 = liste2.astype('int')
    
    #ensure that the probability for two close value to be the same is null
    if len(array1)>1:
        dmin = np.diff(np.sort(array1)).min()
    else:
        dmin=0
    if len(array2)>1:
        dmin2 = np.diff(np.sort(array2)).min()
    else:
        dmin2=0
    array1_r = array1 + int(random)*0.001*dmin*np.random.randn(len(array1))
    array2_r = array2 + int(random)*0.001*dmin2*np.random.randn(len(array2))
    #match nearest
    m = abs(array2_r-array1_r[:,np.newaxis])
    arg1 = np.argmin(m,axis=0)
    arg2 = np.argmin(m,axis=1)
    mask = (np.arange(len(arg1)) == arg2[arg1])
    liste_idx1 = arg1[mask]
    liste_idx2 = arg2[arg1[mask]]
    array1_k = array1[liste_idx1]
    array2_k = array2[liste_idx2]

    liste_idx1 = index1[liste_idx1]
    liste_idx2 = index2[liste_idx2] 
    
    mat = np.hstack([liste_idx1[:,np.newaxis],liste_idx2[:,np.newaxis],
                        array1_k[:,np.newaxis],array2_k[:,np.newaxis],(array1_k-array2_k)[:,np.newaxis]]) 
        
    return mat


def mad(array,axis=0):
    """"""
    if axis == 0:
        step = abs(array-np.nanmedian(array,axis=axis))
    else:
        step = abs(array-np.nanmedian(array,axis=axis)[:,np.newaxis])
    return np.nanmedian(step,axis=axis)*1.48

def season_length(jdb):
    phase = get_phase(jdb,365.25)
    yarara_t0 = phase+np.min((jdb-phase)%365.25)+365.25*((jdb-phase)[0]//365.25)
    yarara_t1 = phase+np.max((jdb-phase)%365.25)+365.25*((jdb-phase)[0]//365.25)
    return yarara_t0, yarara_t1

def compute_obs_season(time,t0):
    s_num = (time-t0)//365.25
    s_num-= (np.min(s_num)-1)
    loc = []
    for s in np.sort(np.unique(s_num)):
        loc.append([np.where(s_num==s)[0][0],np.where(s_num==s)[0][-1]])
    borders =  np.array(loc)
    return borders

def rm_outliers(array, m=1.5):
    array[array==np.inf] = np.nan    
    interquartile = np.nanpercentile(array, 75, axis=0) - np.nanpercentile(array, 25, axis=0)
    inf = np.nanpercentile(array, 25, axis=0)-m*interquartile
    sup = np.nanpercentile(array, 75, axis=0)+m*interquartile    
    mask = (array >= inf)&(array <= sup)

    return mask,  array[mask], sup, inf        

def interp(x, y, new_x, kind='linear'):
    new_y = interp1d(x, y, kind=kind, bounds_error=False, fill_value='extrapolate')(new_x)
    return new_y

def format_name(val,k1):
    format_nb = {'period':'%.2f','K':'%.3f','phi':'%.1f','a':'%.3f'}
    if k1 in format_nb.keys():
        return format_nb[k1]%(val)
    else:
        return '%.3f'%(val)

def today():
    today = datetime.datetime.now().isoformat()
    today = float(today[0:4])*365.25+30.5*float(today[5:7])+float(today[8:10])
    today = today*60221.0478530759/739208.75
    return today

def local_max(spectre,vicinity=5):
    vec_base = spectre[vicinity:-vicinity]
    maxima = np.ones(len(vec_base))
    for k in range(1,vicinity):
        maxima *= 0.5*(1+np.sign(vec_base - spectre[vicinity-k:-vicinity-k]))*0.5*(1+np.sign(vec_base - spectre[vicinity+k:-vicinity+k]))
    
    index = np.where(maxima==1)[0]+vicinity
    if len(index)==0:
        index = np.array([0,len(spectre)-1])
    flux = spectre[index]       
    return np.array([index,flux])

def corner(dataframe, score=None, fig=None):
    nb = len(dataframe.keys())
    offset=nb
    if fig is None:
        fig = plt.figure(figsize=(10,10))
        gs_corner = fig.add_gridspec(nb, nb)
        plt.subplots_adjust(hspace=0,wspace=0)
        offset=0
    else:
        gs_corner = fig.add_gridspec(nb, 2*nb)
    mean_df = dataframe.mean()
    std_df = dataframe.std()
    z = (dataframe-mean_df)/std_df

    if score is None:
        score = np.ones(len(dataframe))
        score[0] = 0
        vmin = 0
        vmax = 1
    else:
        vmin = np.percentile(score,5)
        vmax = np.percentile(score,95)

    sub = score>np.percentile(score,33)
    
    fig.add_subplot(5,5,10)
    warning = 0
    for n,kw in enumerate(['period','K']):
        v = np.median(np.array(dataframe[kw])[sub])
        v_std = np.std(np.array(dataframe[kw])[sub])
        tex = [r'$P_{mag}=$%.2f $\pm$ %.2f years'%(v,v_std),r'$K=$%.4f $\pm$ %.4f'%(v,v_std)][n]
        plt.text(-0.9,-0.25*n,tex,ha='left',va='center',fontsize=13) ; plt.xlim(-1,1) ; plt.ylim(-1,1)
        if abs(v)/v_std<3:
            warning = 1
    plt.text(-0.9,0.25,['Cycle detected','Cycle not detected'][warning],color=['g','r'][warning],fontsize=14)
    plt.axis('off')

    for i,k1 in enumerate(dataframe.keys()):
        for j,k2 in enumerate(dataframe.keys()):
            if i==j:
                fig.add_subplot(gs_corner[i,offset+j])
                plt.tick_params(direction='in',top=True,right=True)
                plt.hist(z[k1],bins=15,histtype='step',color='gray',orientation=['vertical','horizontal'][int(i==nb-1)])
                plt.hist(z[k1][sub],bins=15,histtype='step',color='k',orientation=['vertical','horizontal'][int(i==nb-1)])
                plt.tick_params(labelbottom=False)
                plt.title(r'$\frac{%s-%s}{%s}$'%(k1,format_name(mean_df[k1],k1),format_name(std_df[k1],k1)),fontsize=14)
            elif i>j:
                fig.add_subplot(gs_corner[i,offset+j])
                plt.tick_params(direction='in',top=True,right=True)
                plt.scatter(z[k2],z[k1],alpha=0.3,c=score,marker='.',cmap='Greys',vmin=vmin,vmax=vmax)

            if i==nb-1:
                plt.tick_params(labelleft=False)
            else:
                plt.tick_params(labelbottom=False)

            if j!=0:
                plt.tick_params(labelleft=False)
                
            else:
                plt.tick_params(labelleft=True)
                plt.ylabel(r'$\frac{%s-%s}{%s}$'%(k1,format_name(mean_df[k1],k1),format_name(std_df[k1],k1)))

    return 1-warning

def merge_sources(times, values, values_std, max_diff=3):

    t = [x.copy() for x in times]
    s = [y.copy() for y in values]
    s_std = [yerr.copy() for yerr in values_std]

    saves = []
    for i1 in range(len(t)):
        save = []
        for i2 in range(len(t)):
            if i2!=i1:
                match = match_nearest(t[i1],t[i2]) ; match = match[abs(match[:,-1])<max_diff]
                save.append(len(match))
        saves.append(save)
    saves = np.array(saves)
    ref = np.argmax(np.mean(saves,axis=1))
    src_order = [ref]+list(np.delete(np.arange(len(t)),ref)[np.argsort(saves[ref])[::-1]])

    for iteration in range(10):
        for i2 in src_order:
            for i1 in src_order:
                if i2!=i1:
                    match = match_nearest(t[i1],t[i2]) ; match = match[abs(match[:,-1])<max_diff]
                    y1 = s[i1][match[:,0].astype('int')]
                    y2 = s[i2][match[:,1].astype('int')]
                    jump = np.linspace(0,np.median(y2)-np.median(y1),100)
                    offset = jump[np.argmin(mad((y2-jump[:,np.newaxis])/y1,axis=1))]
                    slope = np.median((y2-offset)/y1)
                    comp = np.setdiff1d(np.arange(len(t[i1])),match[:,0])
                    t[i2] = np.hstack([t[i2],t[i1][comp]])
                    s[i2] = np.hstack([s[i2],offset+slope*s[i1][comp]])
                    s_std[i2] = np.hstack([s_std[i2],slope*s_std[i1][comp]])
                    order = np.argsort(t[i2])
                    s[i2] = s[i2][order]
                    t[i2] = t[i2][order]
        if np.std([len(i) for i in t])==0:
            break
    
    merged = [s[src_order[0]]]
    merged_std = [s_std[src_order[0]]]
    for i1 in src_order[1:]:
        y1 = s[i1]
        y2 = s[src_order[0]]
        jump = np.linspace(0,np.median(y2)-np.median(y1),100)
        offset = jump[np.argmin(mad((y2-jump[:,np.newaxis])/y1,axis=1))]
        slope = np.median((y2-offset)/y1)
        merged.append(offset+slope*s[i1])
        merged_std.append(slope*s_std[i1])
    merged = np.array(merged)
    merged_std = np.array(merged_std)
    merged_time = t[src_order[0]]

    if np.sum(merged_std!=0):
        merged_std[merged_std==0] = np.min(merged_std[merged_std!=0])
    else:
        merged_std = 1e6*np.ones(np.shape(merged))
    weights = 1/merged_std**2
    merged = np.sum(merged*weights,axis=0)/np.sum(weights,axis=0)
    merged_std = np.min(merged_std,axis=0)

    return merged_time, merged, merged_std, src_order

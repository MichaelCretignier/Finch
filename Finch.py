#Created by Michael Cretignier 31.09.2023

import datetime

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

__version__ = '1.5.1'

def get_phase(array,period):
    new_array = np.sort((array%period))
    j0 = np.min(new_array)+(period-np.max(new_array))
    diff = np.diff(new_array)
    if np.max(diff)>j0:
        return 0.5*(new_array[np.argmax(diff)]+new_array[np.argmax(diff)+1])
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


def mad(array):
    step = abs(array-np.nanmedian(array))
    return np.nanmedian(step)*1.48

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

def interp(x, y, new_x):
    new_y = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')(new_x)
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


class tableXY(object):


    def __init__(self, x, y, yerr, proxy_name='proxy1'):
        self.y = np.array(y)  
        self.x = np.array(x)  
        self.yerr = np.array(yerr)
        self.yerr_backup = np.array(yerr)
        self.proxy_name = proxy_name
        
        if len(x)!=len(y):
            print('X et Y have no the same lenght')
        
        self.instrument = np.array(['unknown']*len(self.x))


    def copy(self):
        vec = tableXY(self.x,self.y,self.yerr)
        vec.yerr_backup = self.yerr_backup
        vec.instrument = self.instrument
        return vec
    
    def order(self):
        ordering = np.argsort(self.x)
        self.x = self.x[ordering]
        self.y = self.y[ordering]
        self.yerr = self.yerr[ordering]
        self.yerr_backup = self.yerr_backup[ordering]
        self.instrument = self.instrument[ordering]

    def interpolate(self,new_grid):
        newy = interp(self.x,self.y,new_grid)
        newyerr = interp(self.x,self.yerr,new_grid)
        return tableXY(new_grid,newy,newyerr)
        

    def plot(self, color=[], ax=None, alpha=1, fmt=['.'],mec=None, yerr_type='active', zorder=10):
        fmts = np.array(['.','x','^','s','o','v','.','x'])
        colors = np.array(['C%.0f'%(i) for i in range(0,80)])
        fmts[0:len(fmt)] = np.array(fmt)
        colors[0:len(color)] = np.array(color)
        for n,ins in enumerate(np.sort(np.unique(self.instrument))):
            mask_instrument = self.instrument==ins
            x = self.x[mask_instrument]
            y = self.y[mask_instrument]
            only_nan = (len(y)==sum(y!=y))
            
            if yerr_type=='active':
                yerr = self.yerr[mask_instrument]
            else:
                yerr = self.yerr_backup[mask_instrument]
            
            if not only_nan:
                if ax is None:
                    plt.errorbar(x,y,yerr,color=colors[n],capsize=0,fmt=fmts[n],alpha=alpha,ls='',mec=mec,zorder=zorder,label='%s'%(ins))
                else:
                    ax.errorbar(x,y,yerr,color=colors[n],capsize=0,fmt=fmts[n],alpha=alpha,ls='',mec=mec,zorder=zorder,label='%s'%(ins))
        
            
    def masked(self,mask,replace=True):
        x = self.x[mask]
        y = self.y[mask]
        yerr = self.yerr[mask]
        yerr_backup = self.yerr_backup[mask]
        ins = self.instrument[mask]
        if replace:
            self.x = x
            self.y = y
            self.yerr = yerr
            self.yerr_backup = yerr_backup
            self.instrument = ins
        else:
            vec = tableXY(x,y,yerr)
            vec.instrument = ins
            vec.yerr_backup = yerr_backup
            return vec


    def chunck(self,idx1,idx2):
        idx1 = int(idx1)
        idx2 = int(idx2)
        chunk = tableXY(self.x[idx1:idx2],self.y[idx1:idx2],self.yerr[idx1:idx2])
        chunk.yerr_backup = self.yerr_backup[idx1:idx2]
        chunk.instrument = self.instrument[idx1:idx2]
        return chunk


    def night_stack(self,db=0):

        x = []
        y = []
        yerr = []
        instrument = []
        for ins in np.sort(np.unique(self.instrument)):
            mask_ins = self.instrument==ins
            jdb = self.x[mask_ins]
            vrad = self.y[mask_ins]
            vrad_std = self.yerr.copy()[mask_ins]
            
            if sum(vrad_std!=0):
                vrad_std0 = np.nanmax(vrad_std[vrad_std!=0]*10)
            else:
                vrad_std0 = mad(vrad)/5 
            vrad_std[vrad_std==0] = vrad_std0
            

            weights = 1/(vrad_std)**2

            groups = ((jdb-db)//1).astype('int')
            groups -= groups[0]
            group = np.unique(groups)
                
            mean_jdb = []
            mean_vrad = []
            mean_svrad = []
            
            for j in group:
                g = np.where(groups==j)[0]
                mean_jdb.append(np.sum(jdb[g]*weights[g])/np.sum(weights[g]))
                mean_svrad.append(1/np.sqrt(np.sum(weights[g])))
                mean_vrad.append(np.sum(vrad[g]*weights[g])/np.sum(weights[g]))
            
            x.append(np.array(mean_jdb))
            y.append(np.array(mean_vrad))
            yerr.append(np.array(mean_svrad))
            instrument.append(np.array([ins]*len(mean_jdb)))

        self.x = np.hstack(x)
        self.y = np.hstack(y)
        self.yerr = np.hstack(yerr)
        self.yerr_backup = np.hstack(yerr)
        self.instrument = np.hstack(instrument)

    def split_instrument(self):
        self.instrument_splited = {}
        for n,ins in enumerate(np.sort(np.unique(self.instrument))):
            mask_instrument = self.instrument==ins
            self.instrument_splited[ins] = self.masked(mask_instrument,replace=False)

    def merge(self,tableXY2):
        self.x = np.hstack([self.x,tableXY2.x])
        self.y = np.hstack([self.y,tableXY2.y])
        self.yerr = np.hstack([self.yerr,tableXY2.yerr])
        self.instrument = np.hstack([self.instrument,tableXY2.instrument])

    def merge_instrument(self):
        x = [] ; y = [] ; yerr = [] ; instrument = []
        for ins in self.instrument_splited.keys():
            x.append(self.instrument_splited[ins].bin.x)
            y.append(self.instrument_splited[ins].bin.y)
            yerr.append(self.instrument_splited[ins].bin.yerr)
            instrument.append([ins]*len(self.instrument_splited[ins].bin.x))
        x = np.hstack(x) ; y = np.hstack(y) ; yerr = np.hstack(yerr) ; instrument = np.hstack(instrument)
        self.bin = tableXY(x,y,yerr)
        self.bin.instrument = instrument


        x = [] ; y = [] ; yerr = [] ; instrument = []
        for ins in self.instrument_splited.keys():
            x.append(self.instrument_splited[ins].bin.grad.x)
            y.append(self.instrument_splited[ins].bin.grad.y)
            yerr.append(self.instrument_splited[ins].bin.grad.yerr)
            instrument.append([ins]*len(self.instrument_splited[ins].bin.grad.x))
        x = np.hstack(x) ; y = np.hstack(y) ; yerr = np.hstack(yerr) ; instrument = np.hstack(instrument)
        self.bin.grad = tableXY(x,y,yerr)
        self.bin.grad.instrument = instrument

    def rm_seasons_outliers(self,m=3):
        seasons_t0 = season_length(self.x)[0]
        seasons = compute_obs_season(self.x,seasons_t0)
        
        mask_kept = np.ones(len(self.x)).astype('bool')
        for i in np.arange(len(seasons[:,0])):
            if (seasons[i,1]-seasons[i,0])>10:
                sub = self.y[seasons[i,0]:seasons[i,1]+1]
                mask = abs(sub-np.median(sub))<=mad(sub)*m
                mask_kept[seasons[i,0]:seasons[i,1]+1] = mask
        
        self.masked(mask_kept)
        

    def split_seasons(self,Plot=False,seasons_t0=None):
        
        if seasons_t0 is None:
            seasons_t0 = season_length(self.x)[0]
        seasons = compute_obs_season(self.x,seasons_t0)
        
        self.seasons_splited = [self.chunck(seasons[i,0],seasons[i,1]+1) for i in np.arange(len(seasons[:,0]))]

        self.seasons_std = []
        self.seasons_meanx = [] ; self.seasons_meany = []
        self.seasons_medy = []
        self.seasons_maxy = []
        self.seasons_miny = []
        self.seasons_meany_std = []
        self.seasons_nb = []

        for i in range(len(self.seasons_splited)):
            if Plot:
                self.seasons_splited[i].plot(color='C%.0f'%(i))
            self.seasons_std.append(np.std(self.seasons_splited[i].y))
            self.seasons_meanx.append(np.nanmean(self.seasons_splited[i].x))
            self.seasons_meany.append(np.nansum(self.seasons_splited[i].y/self.seasons_splited[i].yerr**2)/np.nansum(1/self.seasons_splited[i].yerr**2))
            self.seasons_medy.append(np.nanmedian(self.seasons_splited[i].y))
            self.seasons_meany_std.append(np.nanmedian(self.seasons_splited[i].yerr))
            self.seasons_miny.append(np.nanmin(self.seasons_splited[i].y))
            self.seasons_maxy.append(np.nanmax(self.seasons_splited[i].y))
            self.seasons_nb.append(len(self.seasons_splited[i].x))
        
        self.seasons_span = [np.max(s.x)-np.min(s.x) for s in self.seasons_splited]

        self.seasons_species = np.hstack([np.ones(m)*(n+1) for n,m in enumerate(self.seasons_nb)])


    def fit_line(self,perm=1000,Plot=False,color='C0'):
        mean_x = np.nanmean(self.x)
        x_recentered = self.x-mean_x
        x_std = np.nanstd(x_recentered)
        z = x_recentered/x_std

        base_vec = np.array([z**i for i in range(0,2)])

        coeff,sample = self.fit_base(base_vec,perm)

        self.offset,self.slope = np.mean(coeff,axis=1)
        self.offset_std,self.slope_std = np.std(coeff,axis=1)

        self.model = np.dot(np.array([self.offset,self.slope])[:,np.newaxis].T,base_vec)[0]
        self.sub_model = self.y-self.model
        models = np.dot(coeff.T,base_vec)

        if Plot:
            self.plot()
            sup = np.nanpercentile(models,84,axis=0)
            inf = np.nanpercentile(models,16,axis=0)
            plt.fill_between(self.x,inf,sup,color=color,alpha=0.5)
            plt.plot(self.x,self.model,color='k',lw=3)
            plt.plot(self.x,self.model,color=color,lw=1.5)

        self.slope/=x_std
        self.slope_std/=x_std

    def match_proxies(self,tableXY_2):
        match = match_nearest(self.x,tableXY_2.x)
        mask1 = np.in1d(np.arange(len(self.x)),match[:,0].astype('int'))
        mask2 = np.in1d(np.arange(len(tableXY_2.x)),match[:,1].astype('int'))
        sub1 = self.masked(mask1,replace=False)
        sub2 = tableXY_2.masked(mask2,replace=False)

        base_vec = np.array([((sub2.x-np.mean(sub2.x))/np.std(sub2.x))**i for i in range(0,2)])
        base_instrument = np.array([(sub2.instrument==ins).astype('float') for ins in minor_instruments])
        base_vec = np.vstack([base_vec,base_instrument])

        sub2.fit_base()


        coeff, sample = timeseries.fit_base(base_vec,perm=perm)

        return sub1,sub2

    def transform_vector(self,Plot=False,data_driven_std=True):
        self.rm_seasons_outliers(m=3)
        self.split_seasons()

        new_x = self.seasons_meanx
        new_y = self.seasons_meany
        new_yerr = self.seasons_meany_std

        new_gradx = []
        new_grady = []
        new_gradyerr = []
        
        if Plot:
            plt.figure(figsize=(18,10))
            plt.subplot(3,1,1)
            ax = plt.gca()

        for n,v in enumerate(self.seasons_splited):
            if (self.seasons_nb[n]>2)&(self.seasons_span[n]>50):
                v.fit_line(Plot=Plot,color='C%.0f'%(n+1))
                new_gradx.append(np.mean(v.x))
                new_grady.append(v.slope)
                new_gradyerr.append(v.slope_std)
            else:
                v.sub_model = v.y
            
            if (self.seasons_nb[n]>4)&data_driven_std: #data driven uncertainties 
                new_yerr[n] = np.std(v.sub_model)

            if Plot:
                v.plot(color='C%.0f'%(n+1))
        
        new_yerr = np.array(new_yerr)
        if data_driven_std:
            yerr_driven = np.mean(new_yerr[np.array(self.seasons_nb)>4])
            new_yerr[np.array(self.seasons_nb)<=4] = np.sqrt((yerr_driven)**2+(new_yerr[np.array(self.seasons_nb)<=4])**2)

        self.bin = tableXY(new_x,new_y,new_yerr)
        self.bin.grad = tableXY(new_gradx,new_grady,new_gradyerr)

        if data_driven_std:
            self.yerr_backup = self.yerr.copy()
            self.yerr = np.sqrt(self.yerr**2+(self.bin.yerr[self.seasons_species.astype('int')-1])**2)

        if Plot:
            plt.subplot(3,1,2,sharex=ax)
            self.bin.plot()
            plt.subplot(3,1,3,sharex=ax)
            self.bin.grad.plot()
            plt.axhline(y=0,color='k',ls=':')


    def bootstrap(self,perm=1000):
        new_y = self.y+np.random.randn(perm,len(self.x))*self.yerr
        return new_y


    def fit_base(self,base_vec,perm=1000):

        sample = self.bootstrap(perm=perm)
        weight = np.ones(len(self.yerr))
        coeff = np.linalg.lstsq(base_vec.T, sample.T,rcond=None)[0]

        #coeff = np.array([np.linalg.lstsq(base_vec[:,i,:].T*np.sqrt(weight[i])[:,np.newaxis], (table[i])*np.sqrt(weight[i]),rcond=None)[0] for i in range(len(table))])

        return coeff, sample


    def fit_sinus(self, pmin=1000, pmax=None, perm=1000, trend_degree=1, ax=None, ax_chi=None, fmt='.', fig=None, offset_instrument=False, predict_today=False):
        
        jdb_today = -1
        if predict_today:
            jdb_today = today()

        warning = 0

        minor_instruments = []
        count = pd.DataFrame(self.instrument).value_counts().sort_values(ascending=False)
        major_instrument = count.keys()[0][0]
        if (offset_instrument)&(len(np.unique(self.instrument))>1):
            count = count[1:]
            if sum(count>1):
                minor_instruments = np.hstack(count[count>1].keys())
                if len(minor_instruments):
                    print('[INFO] Minor instrument detected : ',minor_instruments,' vs. Major instrument : %s'%(major_instrument))

        x_val = self.x.copy()
        y_val = self.y.copy()
        yerr_val = self.yerr.copy()
        ins_val = self.instrument.copy()

        values_rejected = 0
        if offset_instrument: #rm single season instrument if free offset model
            liste = [major_instrument]+list(minor_instruments)
            kept = np.in1d(self.instrument,np.array(liste))
            removed = np.setdiff1d(self.instrument,np.array(liste))
            if len(removed):
                rejected = tableXY(x_val[~kept], y_val[~kept]*np.nan, yerr_val[~kept])
                rejected.instrument = ins_val[~kept]            
                print('[INFO] Instrument removed from the fit because single season',removed)
                values_rejected = 1
            x_val = x_val[kept]
            y_val = y_val[kept]
            yerr_val = yerr_val[kept]
            ins_val = ins_val[kept]
            
        timeseries = tableXY(x_val,y_val,yerr_val)
        timeseries.instrument = ins_val

        baseline = int(np.max(x_val) - np.min(x_val))
        if pmax is None:
            pmax = baseline*1.5
        print('[INFO] Pmin = %.0f - Pmax = %.0f'%(pmin,pmax))
        self.grid_pmin = pmin
        self.grid_pmax = pmax
        
        period_grid = np.linspace(pmin,pmax,500)

        mean_x = np.nanmean(x_val)
        x_recentered = x_val - mean_x
        x_std = np.nanstd(x_recentered)
        z = x_recentered/x_std

        nb_params = 2+trend_degree+1+len(minor_instruments)
        print('\n[INFO] Nb parameter : %.0f | Nb observations : %.0f'%(nb_params,len(x_val)))
        if (nb_params+1)>=len(x_val):
            print('[WARNING] Too much parameters compared to number of observations')
            trend_degree = 0
            offset_instrument = False

        def create_base(x,z,period,trend_degree,ins_offset=True):
            base_sin = np.array([np.sin(2*np.pi/period*x),np.cos(2*np.pi/period*x)])
            base_trend = np.array([z**i for i in range(0,trend_degree+1)])
            base_vec = np.vstack([base_sin,base_trend])
            
            if bool(len(minor_instruments))&ins_offset:
                base_instrument = np.array([(ins_val==ins).astype('float') for ins in minor_instruments])
                base_vec = np.vstack([base_vec,base_instrument])

            return base_vec

        save = []
        all_chi2 = []
        for period in period_grid:

            param_name = ['K','phi','A','B']
            param_name.append([['a','b','c','d','e'][i] for i in range(0,trend_degree+1)])
            param_name.append(['C_{%s}'%(i) for i in minor_instruments])

            base_vec = create_base(x_recentered,z,period,trend_degree)
            coeff, sample = timeseries.fit_base(base_vec,perm=perm)

            params = np.mean(coeff,axis=1)
            params_std = np.std(coeff,axis=1)

            model = np.dot(params,base_vec)
            residu = y_val - model
            chi2 = np.sum(residu**2/yerr_val**2)/perm
            chi2_reduced = chi2/(len(x_val)-len(params))

            residus = sample - np.dot(coeff.T,base_vec)
            all_chi2.append(np.sum(residus**2/yerr_val**2,axis=1))

            K_boot = np.sqrt(coeff[0]**2+coeff[1]**2)
            phi_boot = np.arctan2(coeff[1],coeff[0])

            K = np.mean(K_boot)
            phi = np.mean(phi_boot)
            K_std = np.std(K_boot)
            phi_std = np.std(phi_boot)

            param_name = list(np.hstack(param_name))
            name0 = ['period']+param_name
            param_name = param_name+[i+'_std' for i in param_name]
            param_name = ['period']+param_name+['chi2','chi2_reduced']

            save.append(np.hstack([period,K,phi,params,K_std,phi_std,params_std,chi2,chi2_reduced]))

        save = np.array(save)
        save = pd.DataFrame(save,columns=param_name)

        all_chi2 = np.array(all_chi2)
        med_chi = np.median(all_chi2,axis=1)
        std_chi = np.std(all_chi2,axis=1)
        med_loglk = np.median(-0.5*np.log(all_chi2),axis=1)
        std_loglk = np.std(-0.5*np.log(all_chi2),axis=1)

        metric = med_loglk - np.min(med_loglk)
        metric /= np.max(metric)
        metric = np.mean(metric)
        self.model_metric = metric
        print('[INFO] Metric model = %.3f'%(metric))

        sup = np.min(med_chi)+std_chi[np.argmin(med_chi)]
        kept = med_chi<=sup

        best_fit_period = save.sort_values(by='chi2_reduced')['period'].values[0]
        self.Pmag = best_fit_period
        self.Pmag_sup = best_fit_period
        self.Pmag_inf = best_fit_period

        crit1 = (best_fit_period==np.max(period_grid))&(trend_degree!=0)
        crit2 = (np.max(period_grid[kept])==np.max(period_grid))&(trend_degree!=0)
        if crit1|crit2:
            print('[WARNING] Period larger than the baseline polytrend should be removed')
            warning=1

        density = med_loglk.copy()
        if np.sum(~kept):
            density-=np.max(density[~kept])
            density[~kept] = 0
        density = density/np.sum(density)

        if ax_chi is not None:
            ax_chi.plot(save['period']/365.25,med_loglk,color='k',label='Metric=%.3f'%(metric))
            ax_chi.plot(save['period']/365.25,med_loglk+std_loglk,color='k',alpha=0.6,ls='-.')
            ax_chi.plot(save['period']/365.25,med_loglk-std_loglk,color='k',alpha=0.6,ls='-.')
            ax_chi.fill_between(save['period']/365.25, med_loglk-std_loglk, med_loglk+std_loglk,alpha=0.2,color='k')
            ax_chi.legend()

            p0 = best_fit_period
            p_sup = np.max(save['period'][kept])
            p_inf = np.min(save['period'][kept])
            p_sup_std = p_sup - p0
            p_inf_std = p0 - p_inf

            self.Pmag_sup = p0+p_sup_std
            self.Pmag_inf = p0-p_inf_std

            ax_chi.axhline(y=-0.5*np.log(sup),ls=':',color='k')
            ax_chi.axvline(x=best_fit_period/365.25,ls=':')
            ax_chi.set_title(label=r'$P_{mag}=%.0f^{+%.0f}_{-%.0f}$ days | $P_{mag}=%.2f^{+%.2f}_{-%.2f}$ years'%(p0,p_sup_std,p_inf_std,p0/365.25,p_sup_std/365.25,p_inf_std/365.25))
            ax_chi.set_xlabel('Pmag [days]')
            ax_chi.set_ylabel(r'Likelihood')

        #likelihood plot

        coeff_likelihood = []
        all_model = []
        period_interp = np.linspace(np.min(period_grid[kept]),np.max(period_grid[kept]),100)
        density_interp = interp(period_grid, density, period_interp)
        period_interp = period_interp[density_interp!=0]
        density_interp = density_interp[density_interp!=0]
        density_interp /= np.sum(density_interp)

        model_plot = []
        all_chi2_grad = []
        for period,proba in zip(period_interp,density_interp):

            base_vec = create_base(x_val-mean_x,(x_val-mean_x)/x_std,period,trend_degree)
            coeff,sample = timeseries.fit_base(base_vec,perm=int(perm*proba)+1)
            model = np.dot(coeff.T,base_vec)
            all_model.append(model)

            K = np.sqrt(coeff[0]**2+coeff[1]**2)
            phi = np.arctan2(coeff[1],coeff[0])
            all_coeff = np.vstack([np.ones(len(coeff[0]))*period,K,phi,coeff])
            coeff_likelihood.append(all_coeff)

            mask_coeff = np.array([len(c.split('_'))==1 for c in name0[3:]])
            if ax is not None:
                xmax = np.max(self.x)+0.5*baseline
                if jdb_today > np.max(self.x)+0.5*baseline:
                    xmax = jdb_today+365
                x_interp = np.linspace(np.min(self.x)-0.5*baseline,xmax,300)
                base_vec = create_base(x_interp-mean_x,(x_interp-mean_x)/x_std,period,trend_degree,ins_offset=False)
                model = np.dot(coeff[mask_coeff].T,base_vec)
                ax.plot(x_interp,model.T[:,::1],alpha=0.01,color='k',zorder=1)  
                model_plot.append(model)
        
            if True: #use_gradient to select best params
                x_grad = self.grad.x
                x_grad = np.sort(np.ravel(x_grad+np.array([-0.5,0.5])[:,np.newaxis]))
                z_grad = (x_grad-mean_x)/x_std
                base_for_grad = create_base(x_grad-mean_x, z_grad, period, trend_degree,ins_offset=False)
                model2 = np.dot(coeff[mask_coeff].T,base_for_grad)
                model_grad = np.diff(model2,axis=1)[:,0::2]
                residus_grad = self.grad.y - model_grad         
                chi2_grad = np.sum(residus_grad**2/self.grad.yerr**2,axis=1)
                all_chi2_grad.append(chi2_grad)

        coeff_likelihood = np.hstack(coeff_likelihood)
        coeff_likelihood = pd.DataFrame(coeff_likelihood.T,columns=name0)
        coeff_likelihood = coeff_likelihood.drop(columns=['A','B'])
        coeff_likelihood['phi'] = 180*coeff_likelihood['phi']/np.pi
        coeff_likelihood['period'] = coeff_likelihood['period']/365.25
        
        all_chi2_grad = np.hstack(all_chi2_grad) 
        lk_grad = -0.5*np.log10(all_chi2_grad)

        all_model = np.vstack(all_model)
        model_plot = np.vstack(model_plot)

        Q3 = np.percentile(all_model,75,axis=0)
        Q1 = np.percentile(all_model,25,axis=0)
        IQ = Q3-Q1
        self.env_sup = tableXY(x_val,Q3+1.5*IQ,0*x_val)
        self.env_inf = tableXY(x_val,Q1-1.5*IQ,0*x_val)
        self.master_model = tableXY(x_val,np.mean(all_model,axis=0),IQ*1.5)


        chi2_final = np.sum((y_val-self.master_model.y)**2/yerr_val**2)
        self.bic = chi2_final+nb_params*np.log(len(x_val))
        print('[INFO] BIC = ',self.bic)

        if 'b' in coeff_likelihood.keys():
            coeff_likelihood['b'] *= (365.25/x_std) 
        if 'c' in coeff_likelihood.keys():
            coeff_likelihood['c'] *= (365.25/x_std)**2

        phi_shift = 360/np.median(coeff_likelihood['period']*365.25)*(mean_x-60000) #reference date - 2,400,000 for the phase shift definition
        coeff_likelihood['phi'] -= (phi_shift%360)
        coeff_likelihood['phi'] = return_branching_phase(coeff_likelihood['phi'])
        
        self.mcmc_table = coeff_likelihood

        corner(coeff_likelihood,score=lk_grad,fig=fig)

        if (self.Pmag_sup==self.grid_pmax)|(self.Pmag_inf==self.grid_pmin):
            warning=1

        for m in minor_instruments:
            offset = np.mean(coeff_likelihood['C_{%s}'%(m)])
            timeseries.y[timeseries.instrument==m] -= offset

        sub = lk_grad>np.percentile(lk_grad,33)
        if ax is not None:
            if values_rejected:
                timeseries.merge(rejected)
            ax.plot(x_interp,np.percentile(model_plot[sub],50,axis=0),color='k',ls='-',lw=2)
            ax.plot(x_interp,np.percentile(model_plot[sub],16,axis=0),color='k',ls='-.',lw=1)
            ax.plot(x_interp,np.percentile(model_plot[sub],84,axis=0),color='k',ls='-.',lw=1)
            timeseries.plot(ax=ax,fmt=['o']*(len(np.unique(timeseries.instrument))),mec='k')
            ax.set_title('Degree detrend = %.0f | Offset instrumental = %.0f' %(trend_degree,offset_instrument))
            if predict_today:
                ax.axvline(x=jdb_today,ls=':',color='k',label='today')
                ax.legend()

        Pmag_conservative = (self.Pmag_inf/365.25, self.Pmag/365.25, self.Pmag_sup/365.25)
        Pmag = (np.percentile(coeff_likelihood.loc[sub,'period'],16),np.median(coeff_likelihood.loc[sub,'period']),  np.percentile(coeff_likelihood.loc[sub,'period'],84))

        return warning, Pmag_conservative, Pmag


    def fit_magnetic_cycle(self, data_driven_std=True, trend_degree=1, season_bin=True, offset_instrument='yes', automatic_fit=False, debug=False, fig_title='', predict_today=False):
        """
        data_driven_std [bool] : replace binned data uncertainties by inner dispersion
        trend_degree [int] : polynomial drift
        """
        self.night_stack()
        reference = self.copy()
        seasons_t0 = season_length(self.x)[0]
        self.split_instrument()
        for ins in self.instrument_splited.keys():
            self.instrument_splited[ins].split_seasons(seasons_t0=seasons_t0)
            self.instrument_splited[ins].transform_vector(Plot=debug,data_driven_std=data_driven_std)
            if data_driven_std: #second iteration for uncertainties on slope params
                self.instrument_splited[ins].transform_vector(Plot=debug,data_driven_std=data_driven_std)
        self.merge_instrument()
        
        binned = self.bin.y.copy()
        for ins in np.unique(self.bin.instrument):
            binned[self.bin.instrument==ins] -= np.median(binned[self.bin.instrument==ins])

        if len(binned)>=6:
            #bad season value
            mask = rm_outliers(binned,m=5)[0]
            self.bin.masked(mask)

        if len(self.bin.y)>=6:
            #bad season value
            mask = self.bin.yerr<=mad(self.bin.y)*10
            self.bin.masked(mask)

        self.grad = self.bin.grad
        vec = [self,self.bin]

        def gen_figure(name=None):
            fig = plt.figure(name,figsize=(18,10))
            fig.suptitle(fig_title)
            gs = fig.add_gridspec(11, 10)
            plt.subplots_adjust(hspace=0,wspace=0,bottom=0.09,top=[0.95,0.90][int(fig_title!='')],right=0.97,left=0.06)

            ax2 = fig.add_subplot(gs[6:, 0:4]) ; plt.xlabel('Jdb [days]')
            ax_chi2 = fig.add_subplot(gs[0:5, 0:4])

            return fig,gs,ax2,ax_chi2

        if len(np.unique(vec[int(season_bin)].instrument))>3:
            offset_instrument = 'yes!'

        if automatic_fit:

            if offset_instrument=='no!':
                params = np.array([[0,False],[1,False]])

            if (offset_instrument=='yes!')&(len(np.unique(vec[int(season_bin)].instrument))>1):
                params = np.array([[0,True],[1,True]])
            else:
                if len(np.unique(vec[int(season_bin)].instrument))>1:
                    params = np.array([[0,False],[1,False],[0,True],[1,True]])
                else:
                    params = np.array([[0,False],[1,False]])
            
            metric = []
            code = []
            outputs = []
            count=0
            for deg, offset in params:
                fig,gs,ax,ax_chi = gen_figure(name='automatic')
                code.append('D%.0fO%.0f'%(deg,offset))
                print('\n[INFO] Testing model : instrument_offset = %.0f + Trend_degree = %.0f'%(offset,deg))
                dust = vec[int(season_bin)].fit_sinus(ax=ax, ax_chi=ax_chi, trend_degree=deg, fmt='o', fig=fig, offset_instrument=offset)
                outputs.append([list(dust[1]),list(dust[2])])
                metric.append(vec[int(season_bin)].model_metric)
                plt.close('automatic')
            metric = np.array(metric)
            trend_degree = params[np.argmin(metric)][0]
            offset_instrument = params[np.argmin(metric)][1]

            fig,gs,ax,ax_chi = gen_figure()
            fig.add_subplot(5,5,5)
            for n,l in enumerate(np.array(outputs)):
                plt.errorbar(np.array([-0.15,0.15])+n,l[:,1],yerr=[l[:,1]-l[:,0],l[:,2]-l[:,1]],marker='o',ls='')
            plt.ylabel('Pmag [years]')
            plt_ax = plt.gca() ; ylim = plt_ax.get_ylim()
            if ylim[1]>vec[int(season_bin)].grid_pmax/365.25:
                plt.axhline(y=vec[int(season_bin)].grid_pmax/365.25,lw=1,ls='-.',alpha=0.3,color='k')
            plt.xticks(np.arange(len(code)),code)

            print('\n===========')
            print('[AUTOMATIC] Model selected : instrument_offset = %.0f + Trend_degree = %.0f'%(offset_instrument,trend_degree))
            print('===========\n')
        else:
            fig,gs,ax,ax_chi = gen_figure()

        offset_instrument = {'yes':True,'yes!':True,'no':False, 'no!':False, True:True, False:False}[offset_instrument]

        warning2, pmag_conservative, pmag_extracted = vec[int(season_bin)].fit_sinus(ax=ax, ax_chi=ax_chi, trend_degree=trend_degree, fmt='o', fig=fig, offset_instrument=offset_instrument,predict_today=predict_today)

        if warning2:
            print('\n[INFO] Conservatives values selected')
            pmag_inf,pmag,pmag_sup = pmag_conservative
        else:
            print('\n[INFO] Extracted values selected')
            pmag_inf,pmag,pmag_sup = pmag_extracted

        ylim = ax.get_ylim()
        for n,ins in enumerate(np.sort(np.unique(reference.instrument))):
            mask = reference.instrument==ins
            ax.errorbar(reference.x[mask], reference.y[mask], yerr=reference.yerr_backup[mask], zorder=2, alpha=0.2,fmt='.',ls='',color='C%.0f'%(n))
        ax.set_ylim(ylim)
        ax.legend(loc=1)
        if self.proxy_name:
            ax.set_ylabel(self.proxy_name)

        print('\n==============')
        if pmag_sup==vec[int(season_bin)].grid_pmax/365.25:
            print('[FINAL REPORT] Pmag > %.2f [%.2f - ???]'%(pmag_inf,pmag_inf))
            pmag = pmag_inf
            pmag_sup = np.nan
        elif pmag_inf==vec[int(season_bin)].grid_pmin:
            print('[FINAL REPORT] Pmag < %.2f [??? - %.2f]'%(pmag_sup,pmag_sup))
            pmag = pmag_sup
            pmag_inf = np.nan
        else:
            print('[FINAL REPORT] Pmag = %.2f [%.2f - %.2f]'%(pmag,pmag_inf,pmag_sup))
        print('============== \n')

        bootstrap_table = vec[int(season_bin)].mcmc_table
        output_table = np.array([
            np.percentile(bootstrap_table,16,axis=0) ,
            np.percentile(bootstrap_table,50,axis=0) ,
            np.percentile(bootstrap_table,84,axis=0)])[:,1:]
        pmag_extracted[1],
        output_table = np.vstack([np.array([pmag_inf,pmag,pmag_sup]),np.array(pmag_extracted),np.array(pmag_conservative),output_table.T]).T
        output_table = pd.DataFrame(output_table,columns=['P','P_computed','P_conservative']+list(bootstrap_table.keys()[1:]),index=['16%','50%','84%'])
        print('\n[FINAL TABLE]\n')
        print('-----------------------------------')
        print(output_table[['P','K','phi']])
        print('-----------------------------------')

        return output_table


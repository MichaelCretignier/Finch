"""
@author: Cretignier Michael 
@university: University of Geneva
@date: 31.09.2023
"""

import datetime

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import Finch_functions as ff

__version__ = '1.7.0'

class tableXY(object):


    def __init__(self, x, y, yerr, proxy_name='proxy1'):
        self.y = np.array(y)  
        self.x = np.array(x)  
        self.yerr = np.array(yerr)
        self.yerr_backup = np.array(yerr)
        self.proxy_name = proxy_name
        self.mask_flag = np.zeros(len(self.x)).astype('bool')

        if len(x)!=len(y):
            print('X et Y have no the same lenght')
        
        self.instrument = np.array(['unknown']*len(self.x))
        self.reference = np.array(['unknown']*len(self.x))

    def copy(self):
        vec = tableXY(self.x,self.y,self.yerr)
        vec.yerr_backup = self.yerr_backup
        vec.instrument = self.instrument
        vec.mask_flag = self.mask_flag
        return vec
    
    def order(self):
        ordering = np.argsort(self.x)
        self.x = self.x[ordering]
        self.y = self.y[ordering]
        self.yerr = self.yerr[ordering]
        self.yerr_backup = self.yerr_backup[ordering]
        self.instrument = self.instrument[ordering]
        self.mask_flag = self.mask_flag[ordering]
        self.reference = self.reference[ordering]

    def interpolate(self,new_grid,kind='linear'):
        newy = ff.interp(self.x, self.y, new_grid, kind=kind)
        newyerr = ff.interp(self.x, self.yerr, new_grid, kind=kind)
        return tableXY(new_grid,newy,newyerr)
        

    def plot(self, color=[], ax=None, alpha=1, fmt='.',mec=None, yerr_type='active', zorder=10):
        colors = np.array(['C%.0f'%(i) for i in range(0,80)])
        colors[0:len(color)] = np.array(color)
        fmts = ['.','s','o','^','v'] ; fmts[0] = fmt
        for n,ins in enumerate(np.sort(np.unique(self.instrument))):
            mask_instrument = self.instrument==ins
            mask_flag = self.mask_flag[mask_instrument]
            x = self.x[mask_instrument]
            y = self.y[mask_instrument]
            sources = self.reference[mask_instrument]
            only_nan = (len(y)==sum(y!=y))
            
            if yerr_type=='active':
                yerr = self.yerr[mask_instrument]
            else:
                yerr = self.yerr_backup[mask_instrument]
            
            if not only_nan:
                if ax is None:
                    if np.sum(~mask_flag):
                        for i,s in enumerate(np.unique(sources)):
                            plt.errorbar(x[(~mask_flag)&(sources==s)],y[(~mask_flag)&(sources==s)],yerr[(~mask_flag)&(sources==s)],color=colors[n],capsize=0,fmt=fmts[i],alpha=alpha,ls='',mec=mec,zorder=zorder,label='%s (%s)'%(ins,s))
                    if np.sum(mask_flag):
                        plt.errorbar(x[mask_flag],y[mask_flag],yerr[mask_flag],color=colors[n],capsize=0,fmt='x',alpha=0.2,ls='',mec=mec,zorder=zorder)
                else:
                    if np.sum(~mask_flag):
                        for i,s in enumerate(np.unique(sources)):
                            ax.errorbar(x[(~mask_flag)&(sources==s)],y[(~mask_flag)&(sources==s)],yerr[(~mask_flag)&(sources==s)],color=colors[n],capsize=0,fmt=fmts[i],alpha=alpha,ls='',mec=mec,zorder=zorder,label='%s (%s)'%(ins,s))
                    if np.sum(mask_flag):
                        plt.errorbar(x[mask_flag],y[mask_flag],yerr[mask_flag],color=colors[n],capsize=0,fmt='x',alpha=0.2,ls='',mec=mec,zorder=zorder)
            
    def masked(self,mask,replace=True):
        x = self.x[mask]
        y = self.y[mask]
        yerr = self.yerr[mask]
        yerr_backup = self.yerr_backup[mask]
        ins = self.instrument[mask]
        mask_flag = self.mask_flag[mask]
        source = self.reference[mask]

        if replace:
            self.x = x
            self.y = y
            self.yerr = yerr
            self.yerr_backup = yerr_backup
            self.instrument = ins
            self.mask_flag = mask_flag
            self.reference = source
        else:
            vec = tableXY(x,y,yerr)
            vec.instrument = ins
            vec.yerr_backup = yerr_backup
            vec.mask_flag = mask_flag
            vec.reference = source
            return vec


    def chunck(self,idx1,idx2):
        idx1 = int(idx1)
        idx2 = int(idx2)
        chunk = tableXY(self.x[idx1:idx2],self.y[idx1:idx2],self.yerr[idx1:idx2])
        chunk.yerr_backup = self.yerr_backup[idx1:idx2]
        chunk.instrument = self.instrument[idx1:idx2]
        chunk.mask_flag = self.mask_flag[idx1:idx2]
        return chunk

    def night_stack(self,db=0):
        x = []
        y = []
        yerr = []
        instrument = []
        reference = []

        rejected = self.masked(self.mask_flag,replace=False)

        for ins in np.sort(np.unique(self.instrument[~self.mask_flag])):
            mask_ins = (self.instrument==ins)&(~self.mask_flag)
            jdb = self.x[mask_ins]
            vrad = self.y[mask_ins]
            vrad_std = self.yerr.copy()[mask_ins]
            ref = np.unique(self.reference[mask_ins])[0]
            
            if sum(vrad_std!=0):
                vrad_std0 = np.nanmax(vrad_std[vrad_std!=0]*10)
            else:
                vrad_std0 = ff.mad(vrad)/5 
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
            reference.append(np.array([ref]*len(mean_jdb)))

        self.x = np.hstack(x)
        self.y = np.hstack(y)
        self.yerr = np.hstack(yerr)
        self.yerr_backup = np.hstack(yerr)
        self.instrument = np.hstack(instrument)
        self.reference = np.hstack(reference)
        self.mask_flag = np.zeros(len(self.x)).astype('bool')

        self.merge(rejected)
        self.order()

    def split_instrument(self):
        self.instrument_splited = {}
        for n,ins in enumerate(np.sort(np.unique(self.instrument))):
            mask_instrument = self.instrument==ins
            self.instrument_splited[ins] = self.masked(mask_instrument,replace=False)

    def merge(self,tableXY2):
        self.x = np.hstack([self.x,tableXY2.x])
        self.y = np.hstack([self.y,tableXY2.y])
        self.yerr = np.hstack([self.yerr,tableXY2.yerr])
        self.yerr_backup = np.hstack([self.yerr_backup,tableXY2.yerr_backup])
        self.instrument = np.hstack([self.instrument,tableXY2.instrument])
        self.mask_flag = np.hstack([self.mask_flag,tableXY2.mask_flag])
        self.reference = np.hstack([self.reference,tableXY2.reference])

    def merge_sources(self):
        for ins in np.unique(self.instrument):
            mask_ins = self.instrument==ins
            v1 = self.masked(mask_ins,replace=False)
            sources = np.unique(v1.reference)
            self.masked(~mask_ins,replace=True)
            if len(sources)>1:
                times = [v1.x[(v1.reference==s)&(~v1.mask_flag)] for s in sources]
                values = [v1.y[(v1.reference==s)&(~v1.mask_flag)] for s in sources]
                values_std = [v1.yerr[(v1.reference==s)&(~v1.mask_flag)] for s in sources]
                merged_time, merged, merged_std, src_order = ff.merge_sources(times, values, values_std, max_diff=1)
                print('References for instrument %s = '%(ins),sources[src_order])
                v1 = tableXY(merged_time, merged, merged_std)
                v1.instrument = [ins]*len(v1.x)
                v1.reference = [' & '.join(list(sources))]*len(v1.x)
            
            self.merge(v1)
            self.order()

    def merge_instrument(self):
        x = [] ; y = [] ; yerr = [] ; instrument = [] ; source = []
        for ins in self.instrument_splited.keys():
            x.append(self.instrument_splited[ins].bin.x)
            y.append(self.instrument_splited[ins].bin.y)
            yerr.append(self.instrument_splited[ins].bin.yerr)
            instrument.append([ins]*len(self.instrument_splited[ins].bin.x))
            source.append(self.instrument_splited[ins].bin.reference)

        x = np.hstack(x) ; y = np.hstack(y) ; yerr = np.hstack(yerr) ; instrument = np.hstack(instrument) ; source = np.hstack(source)
        self.bin = tableXY(x,y,yerr)
        self.bin.instrument = instrument
        self.bin.reference = source


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
        seasons_t0 = ff.season_length(self.x)[0]
        seasons = ff.compute_obs_season(self.x,seasons_t0)
        
        mask_kept = np.ones(len(self.x)).astype('bool')
        for i in np.arange(len(seasons[:,0])):
            if (seasons[i,1]-seasons[i,0])>10:
                sub = self.y[seasons[i,0]:seasons[i,1]+1]
                mask = abs(sub-np.median(sub))<=ff.mad(sub)*m
                mask_kept[seasons[i,0]:seasons[i,1]+1] = mask
        self.mask_flag = ~mask_kept
        #self.masked(mask_kept)

    def set_ins_uncertainties(self):
        
        #instrumental_noise obtained from HD1461,HD1388,HD23249,HD10700,HD90156 
        ins_smw_std = {
            'HARPS03':{'DACE':0.0012,'Yu+23':0.0014,'YARARA':0.0012},
            'HARPS15':{'DACE':0.0012,'Yu+23':0.0012,'YARARA':0.0012},
            'HARPN':  {'DACE':0.0012,'YARARA':0.0012},
            'HKP-1':  {'Baum+22':0.0043,'Radick+18':0.0043},
            'HKP-2':  {'Baum+22':0.0028,'Radick+18':0.0034},
            'HIRES-1':{'Baum+22':0.0015,'Butler+17':0.0090,'Isaacson+10':0.0040,'Wright+04':0.0015},
            'HIRES-2':{'Baum+22':0.0015,'Butler+17':0.0050,'Isaacson+10':0.0050},
            }

        for i in np.unique(self.instrument):
            mask_ins = self.instrument==i
            for s in np.unique(self.reference[mask_ins]):
                mask_source = self.reference==s
                try:
                    value = ins_smw_std[i][s]
                    self.yerr[mask_ins&mask_source] = np.sqrt(self.yerr[mask_ins&mask_source]**2+value**2) 
                    print('[INFO] Uncertainties updated to a minimum jitter of %.4f for instrument = %s (%s)'%(value,i,s))
                except:
                    print('[WARNING] instrument = %s (%s) not in the default list of calibrated uncertainties'%(i,s))

    def prune_obs_model(self,zmin=4):
        pmag = self.out_pmag
        model = self.out_model_smooth
        model = model.interpolate(self.x)
        for ins in np.unique(self.instrument):
            mask_ins = (self.instrument==ins)&(~self.mask_flag)
            xx = self.x[mask_ins]
            yy = self.y[mask_ins]
            residu = (yy-model.y[mask_ins])
            deg = int(np.round(5*(np.max(xx)-np.min(xx))/(pmag*365.25),0))
            deg = np.min([deg,6])
            residu = residu - np.polyval(np.polyfit(xx,residu,deg),xx)
            residu_norm = residu/model.yerr[mask_ins]
            #plt.figure() ; plt.subplot(2,1,1) ; plt.scatter(xx,residu)  ; plt.subplot(2,1,2) ; plt.scatter(xx,residu_norm)
            new_yerr = np.std(residu[abs(residu_norm)<zmin])
            new_yerr = new_yerr**2-np.median(self.yerr[mask_ins])**2
            if new_yerr>0:
                new_yerr = np.sqrt(new_yerr)
            else:
                new_yerr = 0
            print('[INFO] Uncertainties updated to %.4f for instrument = %s '%(np.median(np.sqrt(new_yerr**2+self.yerr[mask_ins]**2)),ins))
            self.yerr[mask_ins] = np.sqrt(new_yerr**2+self.yerr[mask_ins]**2)
            index = np.arange(len(self.x))[mask_ins][abs(residu_norm)>zmin]
            self.mask_flag[index] = True


    def split_seasons(self,Plot=False,seasons_t0=None):
        
        if seasons_t0 is None:
            seasons_t0 = ff.season_length(self.x)[0]
        seasons = ff.compute_obs_season(self.x,seasons_t0)
        
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
        match = ff.match_nearest(self.x,tableXY_2.x)
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
        self.bin.reference = [np.unique(self.reference)[0]]*len(self.bin.x)

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


    def fit_sinus(self, pmin=1000, pmax=None, perm=1000, trend_degree=1, ax=None, ax_chi=None, fmt='.', fig=None, offset_instrument=False, predict=False):
        
        jdb_today = -1
        if type(predict)==str:
            if predict=='today':
                jdb_today = ff.today()
        elif type(predict)!=bool:
            jdb_today = predict

        warning = 0

        minor_instruments = []
        count = pd.DataFrame(self.instrument).value_counts().sort_values(ascending=False)
        major_instrument = count.keys()[0][0]
        if (offset_instrument)&(len(np.unique(self.instrument))>1):
            count = count[1:]
            if sum(count>1):
                minor_instruments = np.hstack(count[count>1].keys())
                if len(minor_instruments):
                    print('[INFO] Major instrument detected : %s'%(major_instrument))
                    print('[INFO] Minor instrument detected : ',minor_instruments)

        x_val = self.x.copy()
        y_val = self.y.copy()
        yerr_val = self.yerr.copy()
        ins_val = self.instrument.copy()
        src_val = self.reference.copy()

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
            src_val = src_val[kept]
            
        timeseries = tableXY(x_val,y_val,yerr_val)
        timeseries.instrument = ins_val
        timeseries.reference = src_val

        baseline = int(np.max(x_val) - np.min(x_val))
        if pmax is None:
            pmax = baseline*1.5
        pmax = np.min([pmax,30*365.25]) 

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
            ax_chi.legend(loc=2)

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
            ax_chi.set_xlabel('Pmag [years]')
            ax_chi.set_ylabel(r'Likelihood')

        #likelihood plot

        coeff_likelihood = []
        all_model = []
        period_interp = np.linspace(np.min(period_grid[kept]),np.max(period_grid[kept]),100)
        density_interp = ff.interp(period_grid, density, period_interp)
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
                xmin = np.min(self.x)-0.5*baseline
                if jdb_today > xmax:
                    xmax = jdb_today+365
                x_interp = np.linspace(xmin,xmax,300)
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
                chi2_grad = np.nansum(residus_grad**2/self.grad.yerr**2,axis=1)
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
        self.model_master = tableXY(x_val,np.mean(all_model,axis=0),IQ*1.5)
        med_slope = 0
        if 'b' in coeff_likelihood.keys():
            med_slope = np.median(coeff_likelihood['b'])
        self.model_drift = tableXY(x_val,(x_val-mean_x)/x_std*med_slope,0*x_val)

        chi2_final = np.sum((y_val-self.model_master.y)**2/yerr_val**2)
        self.bic = chi2_final+nb_params*np.log(len(x_val))
        print('[INFO] BIC = ',self.bic)

        if 'b' in coeff_likelihood.keys():
            coeff_likelihood['b'] *= (365.25/x_std) 
        if 'c' in coeff_likelihood.keys():
            coeff_likelihood['c'] *= (365.25/x_std)**2

        phi_shift = 360/np.median(coeff_likelihood['period']*365.25)*(mean_x-60000) #reference date - 2,400,000 for the phase shift definition
        coeff_likelihood['phi'] -= (phi_shift%360)
        coeff_likelihood['phi'] = ff.return_branching_phase(coeff_likelihood['phi'])
        
        self.mcmc_table = coeff_likelihood

        converged = ff.corner(coeff_likelihood,score=lk_grad,fig=fig)

        if (self.Pmag_sup==self.grid_pmax)|(self.Pmag_inf==self.grid_pmin):
            warning=1

        for m in minor_instruments:
            offset = np.mean(coeff_likelihood['C_{%s}'%(m)])
            timeseries.y[timeseries.instrument==m] -= offset

        sub = lk_grad>np.nanpercentile(lk_grad,33)
        if ax is not None:
            if values_rejected:
                timeseries.merge(rejected)
            iq = np.percentile(model_plot[sub],75,axis=0) - np.percentile(model_plot[sub],25,axis=0)
            ax.plot(x_interp,np.percentile(model_plot[sub],50,axis=0),color='k',ls='-',lw=2)
            ax.plot(x_interp,np.percentile(model_plot[sub],16,axis=0),color='k',ls='-.',lw=1)
            ax.plot(x_interp,np.percentile(model_plot[sub],84,axis=0),color='k',ls='-.',lw=1)
            ax.plot(x_interp,np.percentile(model_plot[sub],50,axis=0)+1.5*iq,color='k',ls=':',lw=1,alpha=0.5)
            ax.plot(x_interp,np.percentile(model_plot[sub],50,axis=0)-1.5*iq,color='k',ls=':',lw=1,alpha=0.5)
            timeseries.plot(ax=ax,fmt='o',mec='k')
            ax.set_title('Degree detrend = %.0f | Offset instrumental = %.0f' %(trend_degree,offset_instrument))
            if jdb_today>0:
                ax.axvline(x=jdb_today,ls=':',color='k',label='today')
                ax.legend(loc=3)
            
            self.model_smooth = tableXY(
                x_interp,
                np.percentile(model_plot[sub],50,axis=0),
                2*iq)

        Pmag_conservative = (self.Pmag_inf/365.25, self.Pmag/365.25, self.Pmag_sup/365.25)
        Pmag = (np.percentile(coeff_likelihood.loc[sub,'period'],16),np.median(coeff_likelihood.loc[sub,'period']),  np.percentile(coeff_likelihood.loc[sub,'period'],84))

        return warning, converged, Pmag_conservative, Pmag

    def fit_cycles_extrema(self,minima,maxima,pmag,range_ratio=0.40,Plot=True):
        minima2 = []
        for mini in minima:
            m = mini[0]
            mask = (self.x>(m-365*pmag*range_ratio))&(self.x<(m+365*pmag*range_ratio))
            sub = self.masked(mask,replace=False)
            if len(sub.x)>3:
                sub.order()
                sub.x-=m
                base = np.array([sub.x**i for i in range(0,3)])
                coeff,dust = sub.fit_base(base)
                if Plot:
                    xplot = np.linspace(np.min(sub.x),np.max(sub.x),20)
                    curves = np.dot(np.array([xplot**i for i in range(0,3)]).T,coeff)
                    plt.plot(xplot+m,curves,color='k',alpha=0.01)
                coeff = coeff[:,coeff[2]>0]
                centers = -coeff[1]/(2*coeff[2])
                center = np.median(centers)
                sign_para = np.median(coeff[2])/np.std(coeff[2])
                summits = np.sum(np.array([centers**i for i in range(0,3)])*coeff,axis=0)
                if (sign_para>1)&(abs(center)<np.max(abs(sub.x))):
                    minima2.append([m+center,ff.mad(centers),np.median(summits),ff.mad(summits)])
                    #plt.scatter(centers,summits,alpha=0.1)
                    #plt.scatter(center,np.median(summits),color='k',marker='x')
                else:
                    minima2.append(mini)
            else:
                minima2.append(mini)
        minima2 = np.array(minima2)

        maxima2 = []
        for maxi in maxima:
            m = maxi[0]
            mask = (self.x>(m-365*pmag*range_ratio))&(self.x<(m+365*pmag*range_ratio))
            sub = self.masked(mask,replace=False)
            if len(sub.x)>3:
                sub.order()
                sub.x-=m
                base = np.array([sub.x**i for i in range(0,3)])
                coeff,dust = sub.fit_base(base)
                if Plot:
                    xplot = np.linspace(np.min(sub.x),np.max(sub.x),20)
                    curves = np.dot(np.array([xplot**i for i in range(0,3)]).T,coeff)
                    plt.plot(xplot+m,curves,color='k',alpha=0.01)
                coeff = coeff[:,coeff[2]<0]
                centers = -coeff[1]/(2*coeff[2])
                center = np.median(centers)
                sign_para = np.median(coeff[2])/np.std(coeff[2])
                summits = np.sum(np.array([centers**i for i in range(0,3)])*coeff,axis=0)
                if (sign_para<-1)&(abs(center)<np.max(abs(sub.x))):
                    maxima2.append([m+center,ff.mad(centers),np.median(summits),ff.mad(summits)])
                    #plt.scatter(centers,summits,alpha=0.1)
                    #plt.scatter(center,np.median(summits),color='k',marker='x')
                else:
                    maxima2.append(maxi)
            else:
                maxima2.append(maxi)
        maxima2 = np.array(maxima2)
    
        return minima2, maxima2

    def fit_cycles(self):

        vec = self.out_data_analysed
        output_table = self.out_output_table
        pmag = self.out_pmag

        params = output_table.loc['50%']
        for p in list(params.keys()):
            if p[0:3]=='C_{':
                vec[0].y[vec[0].instrument==p[3:-1]] -= params[p]
                vec[1].y[vec[1].instrument==p[3:-1]] -= params[p]

        drift0 = self.out_model_drift.interpolate(vec[0].y,kind='linear')
        drift1 = self.out_model_drift.interpolate(vec[1].y,kind='linear')

        vec[0].y -= drift0.y
        vec[1].y -= drift1.y

        crit_baseline = (np.max(vec[1].x) - np.min(vec[1].x)) > (2*365*self.out_pmag)
        print('[INFO] Crit baseline = %.0f | Crit convergence = %.0f'%(crit_baseline,self.out_convergence_flag))
        if (self.out_convergence_flag)&(crit_baseline):
            print('\n[INFO] Baseline long enough for cycles study')

            model = vec[1].model_smooth
            time_interp = np.arange(np.min(model.x),np.max(model.x),50)
            curve = model.interpolate(time_interp,kind='cubic')

            index,dust = ff.local_max(-curve.y) ; index = index.astype('int')
            minima1 = curve.x[index] ; minima1y = curve.y[index]
            
            index,dust = ff.local_max(curve.y) ; index = index.astype('int')
            maxima1 = curve.x[index] ; maxima1y = curve.y[index]
            
            idx1 = np.where(minima1<np.min(vec[1].x))[0][-1]
            idx2 = np.where(minima1>np.max(vec[1].x))[0][0]
            minima1 = minima1[idx1:idx2+1]
            minima1y = minima1y[idx1:idx2+1]

            idx1 = np.where(maxima1>minima1[0])[0][0]
            idx2 = np.where(maxima1<minima1[-1])[0][-1]
            maxima1 = maxima1[idx1:idx2+1]
            maxima1y = maxima1y[idx1:idx2+1]

            maxima1 = np.hstack([maxima1[:,np.newaxis],0*maxima1[:,np.newaxis],maxima1y[:,np.newaxis],0*maxima1y[:,np.newaxis]])
            minima1 = np.hstack([minima1[:,np.newaxis],0*minima1[:,np.newaxis],minima1y[:,np.newaxis],0*minima1y[:,np.newaxis]])

            coeff_lin = np.polyfit(maxima1[:,0],np.arange(len(maxima1)),1)
            num = np.polyval(coeff_lin,-36113.5) # 1 january 1760
            cycle_nb = (np.arange(len(maxima1))+1-np.round(num,0)).astype('int')

            plt.figure(figsize=(18,9))
            plt.axes([0.06,0.56,0.5,0.4])
            plt.xlabel('Jdb - 2,400,000 [days]')
            ax = plt.gca()

            minima2, maxima2 = vec[1].fit_cycles_extrema(minima1,maxima1,pmag,range_ratio=0.40,Plot=True)
            minima3, maxima3 = vec[0].fit_cycles_extrema(minima2,maxima2,pmag,range_ratio=0.33,Plot=False)

            minima_slct = minima3
            maxima_slct = maxima3

            minima_slct[0,0] = minima_slct[1,0] - pmag*365.25
            minima_slct[0,2] = minima_slct[1,2] ; minima_slct[0,1] = minima_slct[1,1] ; minima_slct[0,3] = minima_slct[1,3]
            minima_slct[-1,0] = minima_slct[-2,0] + pmag*365.25
            minima_slct[-1,2] = minima_slct[-2,2] ; minima_slct[-1,1] = minima_slct[-2,1] ; minima_slct[-1,3] = minima_slct[-2,3]

            minima_curve = tableXY(minima_slct[:,0],minima_slct[:,2],minima_slct[:,3])
            minima_curve.masked(minima_curve.yerr!=0,replace=True)
            minima_curve = minima_curve.interpolate(minima_slct[:,0])
            minima_slct[minima_slct[:,3]==0,2] = minima_curve.y[minima_slct[:,3]==0]
            
            if maxima_slct[0,3]==0:
                maxima_slct[0,2] = maxima_slct[1,2] 
            if maxima_slct[-1,3]==0:
                maxima_slct[-1,2] = maxima_slct[-2,2] 

            #maxima_curve = tableXY(maxima_slct[:,0],maxima_slct[:,2],maxima_slct[:,3])
            #norm_max = maxima_curve.interpolate(vec[0].x,kind='cubic')
            #plt.plot(norm_max.x,norm_max.y,color='k',ls=':')
            norm_min = minima_curve.interpolate(vec[0].x,kind='cubic')
            #plt.plot(norm_min.x,norm_min.y,color='k',ls=':')            

            amps = maxima_slct[:,2] - 0.5*(minima_slct[1:,2]+minima_slct[:-1,2])
            amp_max = np.max(amps)

            cycle = vec[0].copy()

            vec[0].plot(alpha=0.5)
            plt.legend(fontsize=9)
            plt.errorbar(minima_slct[:,0],minima_slct[:,2],xerr=minima_slct[:,1],yerr=minima_slct[:,3],capsize=0,marker='v',ls='',color='k',zorder=500,markersize=10)
            plt.errorbar(maxima_slct[:,0],maxima_slct[:,2],xerr=maxima_slct[:,1],yerr=maxima_slct[:,3],capsize=0,marker='^',ls='',color='k',zorder=500,markersize=10)
            for i in range(len(maxima_slct)):
                plt.text(maxima_slct[i,0],0.5*(minima_slct[i,2]+minima_slct[i+1,2]),cycle_nb[i],ha='center',zorder=200)

            plt.axes([0.06,0.06,0.5,0.4],sharex=ax)
            plt.xlabel('Jdb - 2,400,000 [days]')
            plt.ylabel(r'$P_{mag}$ [years]')
            plt.axhline(y=output_table['P']['50%'],color='k')
            plt.axhspan(output_table['P']['16%'],output_table['P']['84%'],color='k',alpha=0.25)
            plt.errorbar(maxima_slct[:,0],(minima_slct[1:,0]-minima_slct[:-1,0])/365.25,yerr=np.sqrt(minima_slct[1:,1]**2+minima_slct[:-1,1]**2)/365.25,marker='o',color='k')

            plt.axes([0.62,0.38,0.15,0.58])
            plt.xlabel('Time [years]')
            offset=0
            for i in range(len(maxima_slct)):
                v = vec[0].copy()
                #v.y -= norm_min.y
                #v.y /= (norm_max.y-norm_min.y)
                #v.yerr /= (norm_max.y-norm_min.y)

                mask = (v.x>minima_slct[i,0])&(v.x<minima_slct[i+1,0])
                v.y[~mask] = np.nan
                v.x -= minima_slct[i,0]
                v.x /= 365.25
                zero = 0.5*(minima_slct[i+1,2]+minima_slct[i,2])
                v.y-=zero
                if i:
                    offset-=1.5*amp_max
                plt.plot([-3,pmag+3],[offset,offset],color='k',ls='-',alpha=0.3)
                v.y += offset
                v.plot()
                plt.text(-4,offset,cycle_nb[i],ha='right',va='center')

            plt.axvline(x=0,ls=':',color='k')
            plt.axvline(x=pmag,ls=':',color='k')
            plt.tick_params(labelleft=False,left=False)

            plt.axes([0.82,0.38,0.15,0.58])
            plt.xlabel('Time []')
            offset=0
            save = []
            for i in range(len(minima_slct)-1):
                v = vec[0].copy()
                mask = (v.x>minima_slct[i,0])&(v.x<minima_slct[i+1,0])
                v.y[~mask] = np.nan
                v.x -= minima_slct[i,0]
                v.x /= (minima_slct[i+1,0]-minima_slct[i,0])
                zero = 0.5*(minima_slct[i+1,2]+minima_slct[i,2])
                v.y-=zero
                if i:
                    offset-=amp_max*1.5
                plt.plot([-0.2,1.2],[offset,offset],color='k',ls='-',alpha=0.3)
                v.y += offset
                v.plot(alpha=0.6)
                plt.text(-0.4,offset,cycle_nb[i],ha='right',va='center')
                t = tableXY(v.x,(v.y-offset)/amps[i],v.yerr/amps[i])
                t.instrument = v.instrument
                t.mask_flag = v.mask_flag
                save.append(t)
            
            plt.axvline(x=0,ls=':',color='k')
            plt.axvline(x=1,ls=':',color='k')
            plt.tick_params(labelleft=False,left=False)

            plt.axes([0.62,0.06,0.35,0.25])
            x = np.hstack([v.x for v in save])
            y = np.hstack([v.y for v in save])
            yerr = np.hstack([v.yerr for v in save])
            ins = np.hstack([v.instrument for v in save])
            mask_flag = np.hstack([v.mask_flag for v in save])
            master = tableXY(x,y,yerr)
            master.instrument = ins
            master.mask_flag = mask_flag
            master.masked(master.y==master.y)
            master.order()
            master.plot(alpha=0.2)
            binx = np.arange(0,1,0.1)+0.05
            biny = [np.nanmedian(master.y[~master.mask_flag][master.x[~master.mask_flag]//0.10==b]) for b in np.arange(10)]
            binx = np.hstack([binx-1,binx,binx+1]) 
            biny = np.hstack([biny,biny,biny])
            master_cycle = tableXY(binx,biny,0*binx)
            master_cycle = master_cycle.interpolate(np.linspace(0,1,100),kind='cubic')
            plt.plot(master_cycle.x,master_cycle.y,color='k',zorder=100,marker='o')
            proba = master_cycle.y
            proba-=np.min(proba)
            proba/=np.sum(proba)
            sample = np.random.choice(master_cycle.x,1000,p=proba,replace=True)
            skew = np.sum((sample-np.mean(sample))**3)/(len(sample-1)*np.std(sample)**3)
            plt.title(r'$\gamma$ = %.2f'%(skew))
            plt.axhline(y=0,color='k',alpha=0.75,ls=':',zorder=101)
            plt.axvline(x=0.5,color='k',alpha=0.75,ls=':',zorder=101)
            plt.ylim(-0.4,1.9)

    def fit_period_cycle(self, data_driven_std=True, trend_degree=1, season_bin=True, offset_instrument='yes', automatic_fit=False, debug=False, fig_title='', predict=False, previous_period_estimate=[[None,None,None,'source']]):
        """
        data_driven_std [bool] : replace binned data uncertainties by inner dispersion
        trend_degree [int] : polynomial drift
        """

        self.instrument = np.array([i+'$'+j for i,j in zip(self.instrument,self.reference)])
        self.night_stack()
        self.instrument = np.array([i.split('$')[0] for i in self.instrument])
        self.merge_sources()

        reference = self.copy()
        seasons_t0 = ff.season_length(self.x)[0]
        self.split_instrument()
        for ins in self.instrument_splited.keys():
            self.instrument_splited[ins].split_seasons(seasons_t0=seasons_t0)
            self.instrument_splited[ins].transform_vector(Plot=debug,data_driven_std=data_driven_std)
            if data_driven_std: #second iteration for uncertainties on slope params
                self.instrument_splited[ins].transform_vector(Plot=debug,data_driven_std=data_driven_std)
        self.merge_instrument()
        reference.mask_flag = self.mask_flag

        binned = self.bin.y.copy()
        for ins in np.unique(self.bin.instrument):
            binned[self.bin.instrument==ins] -= np.median(binned[self.bin.instrument==ins])

        if len(binned)>=6:
            #bad season value
            mask = ff.rm_outliers(binned,m=5)[0]
            self.bin.masked(mask)

        if len(self.bin.y)>=6:
            #bad season value
            mask = self.bin.yerr<=ff.mad(self.bin.y)*10
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
                outputs.append([list(dust[2]),list(dust[3])])
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

            if previous_period_estimate[0][0] is not None:
                for n2,l in enumerate(np.array(previous_period_estimate)):
                    print(l)
                    plt.errorbar([n+n2+1],[float(l[0])],yerr=[[abs(float(l[1]))],[abs(float(l[2]))]],marker='o',ls='',color='k')
                    code.append(l[3])

            plt.xticks(np.arange(len(code)),code)

            print('\n===========')
            print('[AUTOMATIC] Model selected : instrument_offset = %.0f + Trend_degree = %.0f'%(offset_instrument,trend_degree))
            print('===========\n')
        else:
            fig,gs,ax,ax_chi = gen_figure()

        offset_instrument = {'yes':True,'yes!':True,'no':False, 'no!':False, True:True, False:False}[offset_instrument]

        warning2, converged, pmag_conservative, pmag_extracted = vec[int(season_bin)].fit_sinus(ax=ax, ax_chi=ax_chi, trend_degree=trend_degree, fmt='o', fig=fig, offset_instrument=offset_instrument,predict=predict)
        self.output_period_detected = converged

        if warning2:
            print('\n[INFO] Conservatives values selected')
            pmag_inf,pmag,pmag_sup = pmag_conservative
        else:
            print('\n[INFO] Extracted values selected')
            pmag_inf,pmag,pmag_sup = pmag_extracted

        ylim = ax.get_ylim()
        for n,ins in enumerate(np.sort(np.unique(reference.instrument))):
            mask = (reference.instrument==ins)&(~reference.mask_flag)
            if sum(mask):
                ax.errorbar(reference.x[mask], reference.y[mask], yerr=reference.yerr_backup[mask], zorder=2, alpha=0.3,fmt='.',ls='',color='C%.0f'%(n))
            mask2 = (reference.instrument==ins)&(reference.mask_flag)
            if sum(mask2):
                ax.errorbar(reference.x[mask2], reference.y[mask2], yerr=reference.yerr_backup[mask2], zorder=2, alpha=0.2,fmt='x',ls='',color='C%.0f'%(n))

        ax.set_ylim(ylim)
        ax.legend(loc=3,fontsize=9)
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

        if predict==False:
            ax.set_xlim(np.min(reference.x)-pmag*365,np.max(reference.x)+pmag*365)

        bootstrap_table = vec[int(season_bin)].mcmc_table
        output_table = np.array([
            np.percentile(bootstrap_table,16,axis=0) ,
            np.percentile(bootstrap_table,50,axis=0) ,
            np.percentile(bootstrap_table,84,axis=0)])[:,1:]
        pmag_extracted[1],
        output_table = np.vstack([np.array([pmag_inf,pmag,pmag_sup]),np.array(pmag_extracted),np.array(pmag_conservative),output_table.T]).T
        output_table = pd.DataFrame(output_table,columns=['P','P_computed','P_conservative']+list(bootstrap_table.keys()[1:]),index=['16%','50%','84%'])

        self.out_output_table = output_table
        self.out_data_analysed = vec
        self.out_pmag = pmag
        self.out_model_master = vec[1].model_master
        self.out_model_smooth = vec[1].model_smooth
        self.out_model_drift = vec[1].model_drift
        self.out_convergence_flag = converged
        self.prune_obs_model()

        print('\n[FINAL TABLE]\n')
        print('-----------------------------------')
        print(output_table[['P','K','phi']])
        print('-----------------------------------\n')


    def fit_period(self, predict=False, debug=False, fig_title='', previous_period_estimate=[[None,None,None,'source']]):
        
        #first iteration
        self.fit_period_cycle(
            automatic_fit=True, 
            predict=predict, 
            data_driven_std=True, 
            fig_title=fig_title, 
            previous_period_estimate=previous_period_estimate) 
        
        if not debug:
            plt.close('all')
        
        #second iteration
        self.fit_period_cycle(
            automatic_fit=True, 
            predict=predict, 
            data_driven_std=False, 
            fig_title=fig_title,
            previous_period_estimate=previous_period_estimate) 


try: 
    db = pd.read_pickle('/Users/cretignier/Documents/Python/database/ACTIVITY/Activity_Sindex_database.p')
    def get_star(starname):
        sub_db = db.loc[db['star']==starname]
        if len(sub_db):
            vec = tableXY(
                np.array(sub_db['jdb']).astype('float'),
                np.array(sub_db['smw']).astype('float'),
                np.array(sub_db['smw_std']).astype('float'),
                proxy_name='S-index'
                )
            vec.instrument = np.array(sub_db['ins']).astype('str')
            vec.reference = np.array(sub_db['source']).astype('str')
            code = vec.instrument.astype('object')+' ('+vec.reference.astype('object')+')'
            cod = np.unique(code)
            print('[INFO] %s found in the database with references:\n'%(starname))
            for c in cod:
                print(' ° ',c)
            print('\n')
            vec.set_ins_uncertainties()
        else:
            print('[INFO] %s not found in the database'%(starname))
            vec = None
        return vec
except:
    pass

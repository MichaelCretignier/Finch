import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel

from . import Finch_functions as ff


# Generate synthetic data
def fit_gp(finch_xy, period_bounds = [7.5,7.7,8.0], baseline_factor=1, length_scale=4.0, alpha=0.6, label_fontsize=12, print_legend=True, runalgo=False, predict='today'):
    ff.printv('[INFO] Fitting GP with Period boundaries %.1f - %.1f years'%(period_bounds[0],period_bounds[2]),verbose=finch_xy.verbose)

    vec = finch_xy.bin.copy()
    vec.x = (vec.x- 51544.5)/365.25+2000
    X = vec.x[:,np.newaxis]
    y = vec.y.copy()
    dy = vec.yerr.copy()

    if predict=='today':
        today = ff.today(fmt='decimalyear')
    else:
        today = predict
    X_pred = np.linspace(np.min(X)-15*baseline_factor, np.max([today,np.max(X)])+15*baseline_factor, 300*baseline_factor).reshape(-1, 1)
    y_pred = np.ones(np.shape(X_pred)[0])*np.nanmean(y)
    sigma = np.ones(np.shape(X_pred)[0])*np.nanmean(dy)*0.5
    next_min = -1
    next_max = -1

    today_pred = np.array([np.mean(np.ravel(y_pred))])
    dphase_today = '?'
    phase_today = -99.9
    phase = y_pred*0+0.5

    Pmag = 0.00
    Amp = 0.00
    Mmean = 0.00
    Kernel = None

    if runalgo:
        factor = np.ones(len(dy))
        if False: #to increase uncertainties at high activity compared to low activity level #Update 20.03.25 : no more need since updated in the main Finch now
            model = finch_xy.out_gp_phase
            factor = ff.interp(model[0], model[1]+0.50, X[:,0], kind='linear')
            factor /= np.mean(factor)
        dy *= factor
        
        k1 = C(1.0, (1e-2, 1e2)) * ExpSineSquared(length_scale=length_scale, periodicity=period_bounds[1],periodicity_bounds=(period_bounds[0],period_bounds[2]))
        k2 = C(1.0, (1e-2, 1e2)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2))
        kernel = k1* k2 + WhiteKernel(noise_level=np.min(dy), noise_level_bounds=(np.min(dy)*0.5, np.mean(dy)))

        gp = GaussianProcessRegressor(kernel=kernel, alpha=(dy)**2, n_restarts_optimizer=10)

        gp.fit(X, y)

        y_pred, sigma = gp.predict(X_pred, return_std=True)

        today_pred, today_sigma = gp.predict(np.array([today])[:,np.newaxis], return_std=True)

        futur = X_pred[:,0]>today

        index_period = int(gp.kernel_.k1.k1.k2.periodicity/np.nanmean(np.diff(X_pred[:,0])))

        maxi = ff.local_max(y_pred,vicinity=int(index_period/4))
        mini = ff.local_max(-y_pred,vicinity=int(index_period/4)) 
        
        if (np.shape(maxi)[1]>1)&(np.shape(mini)[1]>1):
            lower = ff.interp(X_pred[:,0][mini[0].astype('int')], -mini[1], X_pred[:,0], kind='linear')
            upper = ff.interp(X_pred[:,0][maxi[0].astype('int')], maxi[1], X_pred[:,0], kind='linear')
            
            phase = (y_pred - lower)/(upper-lower)
            phase_today = ff.interp(X_pred[:,0], phase, np.array([today]), kind='linear')[0]
            phase_today = phase_today
            dphase_today = ff.interp(X_pred[:,0], np.gradient(phase), np.array([today]), kind='linear')[0]
            if dphase_today!=dphase_today:
                dphase_today = 1
            dphase_today = ['+','+','-'][int(np.sign(dphase_today))]
        
        maxi = maxi[0][maxi[0]>np.sum(~futur)]
        mini = mini[0][mini[0]>np.sum(~futur)]

        if len(maxi):
            next_max = int(maxi[0])
        else:
            next_max = np.argmax(y_pred[futur])+np.sum(~futur)

        if len(mini):
            next_min = int(mini[0])    
        else:
            next_min = np.argmin(y_pred[futur])+np.sum(~futur)

        Pmag = gp.kernel_.k1.k1.k2.periodicity 
        Kernel = gp.kernel_

    # Plot the results
    fig = plt.figure(figsize=(18, 6))
    ax = plt.gca()

    if runalgo:
        plt.scatter(X_pred[next_min],y_pred[next_min],marker='v',color='k',s=70)
        plt.scatter(X_pred[next_max],y_pred[next_max],marker='^',color='k',s=70)
    plt.axvline(x=today,ls=':',color='k',label='today [M = %.1f %% | %.2f%s]'%(today_pred,phase_today,dphase_today))

    vec.plot(fmt='o',mec='k',zorder=100)
    if print_legend:
        plt.legend(fontsize=label_fontsize)
    finch_xy.x = (finch_xy.x - 51544.5)/365.25+2000

    Mmean = np.mean(y_pred)

    finch_xy.print_label = False
    finch_xy.plot(alpha=alpha,zorder=10,yerr_type='null')
    finch_xy.print_label = True
    finch_xy.x = (finch_xy.x - 2000)*365.25+51544.5
    plt.plot(X_pred, y_pred, 'k-', label='GP Prediction',zorder=100)
    sup = y_pred + 1.96 * sigma
    inf = y_pred - 1.96 * sigma
    plt.fill_between(X_pred.ravel(), inf, sup,
                    alpha=0.15, color='k', label='Confidence Interval',zorder=99)

    plt.title('Pmag = %.2f years    |    Next minimum ▼ (%.2f = %.1f %%)   |   Next maximum ▲ (%.2f  = %.1f %%)'%(Pmag,X_pred[next_min],y_pred[next_min],X_pred[next_max],y_pred[next_max]),fontsize=14)
    plt.xlabel('Date [year]',fontsize=15)
    plt.ylabel(ff.ylabel_format(finch_xy.proxy_name),fontsize=15)
    plt.subplots_adjust(left=0.07,right=0.98,top=0.95)
    span = np.max(sup) - np.min(inf)
    ymax = np.max([1.25*np.max(vec.y), np.max(sup)+0.15*span]) 
    plt.ylim(np.min(inf)-0.15*span, ymax)
    ff.printv('[INFO] GP Pmag = %.2f years'%(Pmag),verbose=finch_xy.verbose)
    
    if runalgo:
        print("[INFO] Learned kernel:", gp.kernel_)

    Amp = y_pred[next_max] - y_pred[next_min]
    
    finch_xy.out_gp_pmag = Pmag
    finch_xy.out_gp_ampmag = Amp
    finch_xy.out_gp_meanmag = Mmean
    finch_xy.out_gp_model = np.array([X_pred[:,0],y_pred,sigma])
    finch_xy.out_gp_phase = np.array([X_pred[:,0],phase])
    finch_xy.out_gp_predict = [np.round(today_pred[0],1), np.round(phase_today,2), dphase_today]
    finch_xy.out_gp_kernel = Kernel

    ff.show_sun()
    ax = plt.gca()
    ylim = ax.get_ylim()
    if ylim[0]>0:
        plt.ylim(0,None)
    if ylim[1]<10:
        plt.ylim(None,10)

    return fig 
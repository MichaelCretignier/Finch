import finch as Finch
import matplotlib.pylab as plt
import numpy as np

vec = Finch.import_test()

plt.figure(figsize=(18,6))
# Show the time-series
vec.plot() ; plt.legend() ; plt.xlabel('Jdb - 2,400,000 [days]') ; plt.ylabel('MHK [%%]') ; plt.show()

# Merge SNAKY and YARARA to create a HYDRA time-series
vec.create_hydra()

plt.figure(figsize=(18,6))
# Show the time-series
vec.plot() ; plt.legend() ; plt.xlabel('Jdb - 2,400,000 [days]') ; plt.ylabel('MHK [%%]') ; plt.show()

# Fit cycle if at least 4 years baseline
warning = vec.check_baseline()

# FINCH can fit cycles with or without linear trend + with or without instrumental offsets
# Let's fit a trend, but no instrumental offset (model = D1O0)

vec.fit_period_cycle(
    automatic_fit = False, 
    trend_degree = 1, 
    data_driven_std = True, 
    offset_instrument = 'no', 
    offset_fixed = ['SNAKY','HYDRA'],
    predict = 'today',
    x_unit = 'years')

# FINCH can also test the 4 models and choose the best one using the automatic_fit option
# Let's download again the time-series since fit_period_cycle modify the uncertainties

vec = Finch.import_test(create_hydra=True)

vec.fit_period_cycle(
    automatic_fit = True, 
    data_driven_std = True, 
    offset_fixed = ['SNAKY','HYDRA'],  #only Xlum Ayres will have the offset free to vary
    predict = 'today',
    x_unit = 'years')
plt.show()

# The best model is a trend + instrumental offset (model = D1O1)
# FINCH can now use the cycles properties to fit a GP with the initial guess
# We first remove the instrumental offset that we fit

vec.remove_ins_offset()
 
#Let's fit the GP to have a more precise fit and predict the next maximum and minimum of the cycle

fig_gp = vec.fit_gp(
    baseline_factor=1, 
    runalgo=bool(vec.out_convergence_flag), 
    predict=Finch.today_deciyear)
plt.show()

# From the GP fit:
# The next maximum of Alpha Cen B is predicted around 2027.21
# The next minimum of Alpha Cen B is predicted around 2031.04

print('\n\n[FINAL] Production of the final summary:')
print('[FINAL] %s cycle is around %.2f years with an amplitude of %.1f %% and a mean activity level of %.1f %% (Min=%.1f%% | Max= %.1f %%)'%(vec.star_starname, vec.out_gp_pmag, vec.out_gp_ampmag, vec.out_gp_meanmag, vec.out_gp_meanmag-0.5*vec.out_gp_ampmag, vec.out_gp_meanmag+0.5*vec.out_gp_ampmag))
print('[FINAL] The Sun has a comparison has a cycle period of 11.0 years, an amplitude of 10% and a mean activity level of 5% (Min=0% | Max=10%)')

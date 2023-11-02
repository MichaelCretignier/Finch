"""
@author: Cretignier Michael 
@university: University of Geneva
@date: 31.09.2023
"""

import pandas as pd

import Finch as Finch

# =============================================================================
# Playground
# =============================================================================

star_dataset = '1' #test also 2,3,4,5,6,7
table = pd.read_csv('./dataset_test/table_star%s.csv'%(star_dataset))

vec = Finch.tableXY(
    table['time'].astype('float'), 
    table['proxy'].astype('float'), 
    table['proxy_err'].astype('float'), 
    proxy_name = 'Sindex [a.u]'
    ) 

vec.instrument = table['instrument']                    # initialisation of the instrument vector 
vec.reference = np.array(['YourSource+24']*len(vec.x))  # initialisation of sources or references for the data

# =============================================================================
# Visualisation of the time-series
# =============================================================================

plt.figure('Input')
vec.plot() ; plt.legend()

# let's fit a cycle to test the code speed (binning, merging, fit, bootstrap and plot)
vec.fit_period_cycle(trend_degree=0, automatic_fit=False) # FAST !

#parameters are saved in the vec. class object (all the output follow the vec.out_X name convention)
print(vec.out_output_table) # 16%, 84% being the inf & sup uncertainties of the values (50%)

# =============================================================================
# Fitting magnetic cycle period
# =============================================================================

# Finch contains 4 different models : Trend = Y/N and Instrumental offset = Y/N
# You can fix those parameters by yourself (as above) or let the code selecting the best model

vec.fit_period()

# Slightly longer to run... But Easy ! Isn't it ? :) 

# Warning: Automatic Tools are nice but not perfect !
# There is no garanty for the automatic mode to provide the best result ! 
# Always keep an eye on the top right panel that compare the results of the different models

# =============================================================================
# Predicting current stellar activity level 
# =============================================================================

# time have to be specifed in units of : jdb - 2,400,000

vec.fit_period_cycle(predict='today', trend_degree=0, automatic_fit=False)  # manual

vec.fit_period(predict='today')                                             # automatic

# With example 1, we see quite a difference in the predicted activity level
# This is because there are some instrumental systematic in the first three seasons that bias the model
# Remember that Finch model is simple, YOU have to clean your data of systematics
# Note that it makes sense that the prediction is unsure given that baseline ~ cycle period 

# Try to play with the other datasets ! 

# =============================================================================
# FINCH S-index database
# =============================================================================

# Finch is delivered with an outlier-curated and fully merged database of the most important CaIIH&K program
# you can load your favorite star using its HD number

vec = Finch.get_star('HD109200')

# Please: 
# 1) cite ALL the ReadMe references if you are using such data 
# 2) keep a similar table format if you provide an updated version of the database
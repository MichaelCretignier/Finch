
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

vec.instrument = table['instrument'] # initialisation of the instrument vector 

# see the timeseries with : vec.plot() ; plt.legend()

Output = vec.fit_magnetic_cycle(automatic_fit=True)
print(Output) # 16%,84% being the inf & sup uncertainties of the values (50%)

#  Easy ! Isn't it ? :)

# =============================================================================
# Warning : Automatic Tools are nice but not perfect !
# =============================================================================

# There is no garanty for the automatic mode to provide the best result ! 
# You can choose to add or remove trend manually 

Output = vec.fit_magnetic_cycle(trend_degree=0, automatic_fit=False)


# =============================================================================
# Predicting current stellar activity level 
# =============================================================================

Output = vec.fit_magnetic_cycle(automatic_fit=True, predict_today=True)


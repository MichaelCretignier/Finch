# Finch

<p align="center">
  <img src="logo.png" alt="Project logo" width="750">
</p>

# Contact Me

If you have any problem, please contact me at:

michael.cretignier@physics.ox.ac.uk

# Description

Finch is a Python stand-alone code to fit the stellar magnetic cycle periods on activity proxies' time series.
The code is fast and outlier-robust in order to deliver a period and its related uncertainty in a few seconds.

https://github.com/MichaelCretignier/Finch

————————-————————-—
IMPORTANT INFORMATION :
————————————————-——

Even if Finch has never been properly presented in a paper, since this method was initially a standard analysis of the YARARA pipeline of Cretignier et al., 2021 paper, please cite it as a "publicly available function of the YARARA pipeline".

ADS Link : https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..43C/abstract

—————————————
REQUIREMENT :
—————————————

Standard python libraries : 

numpy (1.20.1) 
scipy (1.7.3)
matplotlib (3.3.4)
pandas (1.4.1)

—————————————
CODE DETAIL :
—————————————

The magnetic cycle model is a simple sinusoids that includes polynomial drift and instrumental offsets.

Uncertainties are derived using intra-season jitter (induced by the instrumental noise and stellar rotation).

The code estimates the uncertainties on all the parameters by bootstrap using the advantage of the simple multilinear model optimized via a least-square matrix inversion. 

The code contains an automatic mode that compares different pre-registered models and selects the one producing the sharpest likelihood. 

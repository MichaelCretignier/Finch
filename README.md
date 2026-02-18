# FINCH (v2.1.0)

<p align="center">
  <img src="logo.png" alt="Project logo" width="750">
</p>

FINCH is a Python package designed to fit stellar magnetic cycle periods from activity proxy time series.
The code is fast and robust to outliers, delivering a period estimate and its associated uncertainty within a few seconds.

https://github.com/MichaelCretignier/FINCH

## Contact Me

If you encounter any issues, please contact me at:

michael.cretignier@physics.ox.ac.uk

## Installation (pip install) 

Using conda:

```bash
conda create -n finch python=3.12.5
conda activate finch
pip install finch
```

## Installation (Git Clone)

Download the directory and try to run the minimal example `example.py` with your own Python installation.
If it crashes, install a Python environment:

[Mac M4 Chip] Python environment (Conda install) (Python 3.12.5)

```bash
[TERMINAL]
conda create -n finch -c conda-forge python=3.12.5 numpy=1.26.4 pandas=2.3.2 scipy=1.16.2 matplotlib=3.10.6 ipython=9.5.0 colorama=0.4.6 scikit-learn=1.7.2 -y 
```

[Mac Intel Chip] Python environment (Conda install) (Python 3.8.8)

```bash
[TERMINAL]
conda create -n finch -c conda-forge python=3.8.8 numpy=1.23.5 pandas=1.4.1 scipy=1.8 matplotlib=3.5 ipython=7.22.0 colorama=0.4.4 scikit-learn=0.24.1 -y 
```

[Alternative to conda] Python environment (Venv install)

```bash
[TERMINAL]
python3 -m venv finch
source finch/bin/activate 
pip install --upgrade pip 
pip install -r requirements_3.12.5.txt
```

## Test minimal example

Move inside the `FINCH/` directory and launch an IPython shell:

```bash
[TERMINAL]
conda activate finch
cd .../GitHub/FINCH/
ipython
```

Then run the example using the magic matplotlib command line `%matplotlib` :

```python
[IPYTHON]
%matplotlib
run example.py
```

## FINCH file format

FINCH input tables are typical `.csv` files containing at minimum 6 columns: 

1) jdb (jdb - 2,400,000)
2) proxy (MHK in %)
3) proxy uncertainties
4) instrument (spectrograph)
5) reference (sources)
6) flag (binary)

Data with `flag=1` are rejected of FINCH analysis, but preserved in the plots.

FINCH can create a `tableXY` object by loading right formatted .csv table and specifying the stellar object:

```python
[IPYTHON]
import finch as Finch
vec = Finch.import_csv(
  'your_file.csv', 
  proxy_name = 'MHK', 
  starname = 'HD128621', 
  teff = 5142, 
  logg = 4.49, 
  feh = 0.15, 
  create_hydra = True)

```

Stellar atmospheric parameters are optional but recommended.

## Citations

Although FINCH has not yet been formally presented in a dedicated paper, as the method originates from the YARARA pipeline described in Cretignier et al. (2021), please cite it as a "publicly available function of the YARARA pipeline". 

The MHK activity index was explained in Cretignier et al. 2024a and 2024b.

ADS Link : 

1) [Cretignier et al. 2021] https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..43C/abstract
2) [Cretignier et al. 2024a] https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.2940C/abstract
3) [Cretignier et al. 2024b] https://ui.adsabs.harvard.edu/abs/2024MNRAS.535.2562C/abstract

## Details Description of the Algorithm

FINCH combines data from different sources while keeping track of instrument-dependent offsets.

The magnetic cycle model is a simple sinusoid that includes polynomial drift and instrumental offsets.

Uncertainties are derived using intra-season jitter (induced by the instrumental noise and stellar rotation).

Parameter uncertainties are estimated via bootstrap, leveraging the simplicity of the multilinear model optimized through least-squares matrix inversion.

The code contains an automatic mode that compares different pre-registered models and selects the one producing the sharpest likelihood. 

A Gaussian Process can then be run using the previous fit as an initial guess, ensuring stability.


```bash
[TERMINAL] 
conda remove --name finch --all
```

from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent)
TEST_DIR = BASE_DIR + "/test_dataset"
test_file = TEST_DIR + "/Finch_input_HD128621.csv"
sun_file = TEST_DIR + "/Solar_Mg2.csv"


ins_mhk_std = {
    'HARPS03':{'YARARA':0.5,'SNAKY':0.5,'HYDRA':0.5},
    'HARPS15':{'YARARA':0.5,'SNAKY':0.5,'HYDRA':0.5},
    'HARPN':  {'YARARA':0.5,'SNAKY':0.5,'HYDRA':0.5},
    'NEID':  {'YARARA':2,'SNAKY':1.0,'HYDRA':1.0},
    'NEID-HE':  {'YARARA':2,'SNAKY':1.0,'HYDRA':1.0},
    'SOPHIE':  {'YARARA':4,'SNAKY':4.0,'HYDRA':4.0},
    'SOPHIE-HE':  {'YARARA':4,'SNAKY':4.0,'HYDRA':4.0},
    'CORALIE98':  {'YARARA':3.5,'SNAKY':3.5,'HYDRA':3.5},
    'CORALIE07':  {'YARARA':1.5,'SNAKY':1.5,'HYDRA':1.5},
    'CORALIE14':  {'YARARA':1.5,'SNAKY':1.5,'HYDRA':1.5},
    'ESPRESSO18':  {'YARARA':0.5,'SNAKY':0.5,'HYDRA':0.5},
    'ESPRESSO19':  {'YARARA':0.5,'SNAKY':0.5,'HYDRA':0.5},
    'Xlum':  {'Ayres+14':0.5,'Ayres+23':0.5},
    'HKP-1':  {'Baum+22':2,'Radick+18':1},
    'HKP-2':  {'Baum+22':2,'Radick+18':1},
    'HIRES-1':{'Baum+22':2,'Butler+17':1,'Isaacson+10':1,'Wright+04':1},
    'HIRES-2':{'Baum+22':2,'Butler+17':2,'Isaacson+10':1,'Teklu+25':2},
    }

ins_smw_std = {
    'HARPS03':{'DACE':0.0012,'Yu+23':0.0014,'YARARA':0.0012},
    'HARPS15':{'DACE':0.0012,'Yu+23':0.0012,'YARARA':0.0012},
    'HARPN':  {'DACE':0.0012,'YARARA':0.0012},
    'ESPRESSO18':  {'DACE':0.0012,'YARARA':0.0012},
    'ESPRESSO19':  {'DACE':0.0012,'YARARA':0.0012},
    'CORALIE98':  {'DACE':0.0024,'YARARA':0.0024},
    'CORALIE07':  {'DACE':0.0024,'YARARA':0.0024},
    'CORALIE14':  {'DACE':0.0024,'YARARA':0.0024},
    'HKP-1':  {'Baum+22':0.0043,'Radick+18':0.0043},
    'HKP-2':  {'Baum+22':0.0028,'Radick+18':0.0034},
    'HIRES-1':{'Baum+22':0.0015,'Butler+17':0.0090,'Isaacson+10':0.0040,'Wright+04':0.0015},
    'HIRES-2':{'Baum+22':0.0015,'Butler+17':0.0050,'Isaacson+10':0.0050,'Teklu+25':0.0050},
    }    
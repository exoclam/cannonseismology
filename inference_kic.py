import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm 
from astropy.io import fits
from astropy.table import Table

import thecannon as tc
print(tc.__version__)
from process_spectra_gaus import *

import loocv
import matplotlib
import matplotlib.pylab as pylab
matplotlib.rcParams.update({'errorbar.capsize': 1})
pylab_params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(pylab_params)

#path = '/Users/chrislam/Desktop/cannon-ages/' 
path = '/home/c.lam/blue/cannon-ages/'

bedell_kic_apogee = pd.read_csv(path+'data/bedell_kic_apogee.csv')
#bedell_kic_apogee = bedell_kic_apogee.loc[bedell_kic_apogee['sdss_id'].isin(np.array([66646541,66647080,66647116,66647134,66647246,66647251]))]
training_names = bedell_kic_apogee['sdss_id'].astype(str)

# use Aida's normalization code on the inference spectra
directory = path+'data/kic_spectra/' 
#spectra_paths = sorted(os.listdir(directory))
spectra_paths = get_files_in_order(directory, training_names)
label_names=["Teff", "logg", "feh", "mg_h", "Age", "Dnu"]

fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)  
fluxes=[]
ivars=[]
success_sdss_ids=[]
teffs = []
e_teffs = []
loggs = []
e_loggs = []
fe_hs = []
e_fe_hs = []
mg_hs = []
e_mg_hs = []
for spectra_path in spectra_paths: # toggle for short or full version
	wl,flux_single,ivar_single = process_spectra(spectra_path,10) # 10 is the width of your Gaussian for continuum normalization
	fluxes.append(flux_single)
	ivars.append(ivar_single)
	     
    # looks like sdss_access failed for six spectra. handle these.
	sdss_id = get_number_between(spectra_path, 'mwmStar-0.6.0-', '.fits')
	success_sdss_ids.append(sdss_id)
     
    # pull ASPCAP stellar params from mwmLite
	teff = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].teff[0]
	e_teff = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].e_teff[0]
	logg = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].logg[0]
	e_logg = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].e_logg[0]
	fe_h = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].fe_h[0]
	e_fe_h = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].e_fe_h[0]
	mg_h = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].mg_h[0]
	e_mg_h = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].e_mg_h[0]
	teffs.append(teff)
	e_teffs.append(e_teff)
	loggs.append(logg)
	e_loggs.append(e_logg)
	fe_hs.append(fe_h)
	e_fe_hs.append(e_fe_h)
	mg_hs.append(mg_h)
	e_mg_hs.append(e_mg_h)


# read in model 
model = tc.CannonModel.read(path+"apogee-serenelli-lite.model")

# inference!
labels_arr = []
cov_arr = []
for i in tqdm(range(len(fluxes))):
    flux = fluxes[i]
    ivar = ivars[i]
    labels, cov, metadata = model.test(flux, ivar)
    print("labels, cov, metadata: ", labels, cov, metadata)
    labels_arr.append(labels)
    
    # use cov to propagate per-star, per-visit uncertainty 
    matrix = np.zeros((len(cov),len(label_names))) # Pre-allocate matrix
    for j in range(0,len(cov)):
        matrix[j,:] = np.sqrt(np.diag(cov[j]))
    cov_arr.append(matrix)

preds = pd.DataFrame()
preds['kepid'] = bedell_kic_apogee['kepid']
preds['source_id'] = bedell_kic_apogee['source_id']
preds['sdss_id'] = bedell_kic_apogee['sdss_id']
preds['teff'] = bedell_kic_apogee['teff']
preds['teff_err1'] = bedell_kic_apogee['teff_err1']
preds['teff_err2'] = bedell_kic_apogee['teff_err2']
preds['logg'] = bedell_kic_apogee['logg']
preds['logg_err1'] = bedell_kic_apogee['logg_err1']
preds['logg_err2'] = bedell_kic_apogee['logg_err2']
preds['feh'] = bedell_kic_apogee['feh']
preds['feh_err1'] = bedell_kic_apogee['feh_err1']
preds['feh_err2'] = bedell_kic_apogee['feh_err2']
preds['mg_h'] = bedell_kic_apogee['mg_h']

# looks like sdss_access failed for six spectra. handle these.
preds = preds.loc[preds['sdss_id'].isin(np.array(success_sdss_ids))]

# those params were from Gaia. throw in ASPCAP parameters
preds['teff'] = teffs
preds['e_teff'] = e_teffs
preds['logg_aspcap'] = loggs
preds['e_logg_aspcap'] = e_loggs
preds['fe_h'] = fe_hs
preds['e_fe_h'] = e_fe_hs
preds['mg_h_aspcap'] = mg_hs
preds['e_mg_h'] = e_mg_hs

preds['Teff_pred'] = np.array(labels_arr)[:,0][:,0]
preds['logg_pred'] = np.array(labels_arr)[:,0][:,1]
preds['fe_h_pred'] = np.array(labels_arr)[:,0][:,2]
preds['mg_h_pred'] = np.array(labels_arr)[:,0][:,3]
preds['Age_pred'] = np.array(labels_arr)[:,0][:,4]
preds['Dnu_pred'] = np.array(labels_arr)[:,0][:,5]
print(preds)
preds.to_csv(path+'data/inferences_kic.csv', index=False)
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

path = '/Users/chrislam/Desktop/cannon-ages/' 

silver_inferences_kic = pd.read_csv(path+'data/silver_inferences_kic_no_rgb.csv')
silver_inferences_kic = silver_inferences_kic.loc[silver_inferences_kic['chisq']<100000]
silver_inferences_kic['Teff_err'] = np.sqrt(silver_inferences_kic['sigma_star_Teff']**2 + 31**2)
silver_inferences_kic['logg_err'] = np.sqrt(silver_inferences_kic['sigma_star_logg']**2 + 0.043**2)
silver_inferences_kic['feh_err'] = np.sqrt(silver_inferences_kic['sigma_star_fe_h']**2 + 0.027**2)
silver_inferences_kic['mg_h_err'] = np.sqrt(silver_inferences_kic['sigma_star_mg_h']**2 + 0.026**2)
silver_inferences_kic['Age_err'] = np.sqrt(silver_inferences_kic['sigma_star_age']**2 + 0.417**2)
silver_inferences_kic['Dnu_err'] = np.sqrt(silver_inferences_kic['sigma_star_Dnu']**2 + 0.751**2)

### compare with LEGACY ages
legacy = pd.read_csv(path+'data/silva-aguirre-legacy.txt',sep='\s+')
print("size of LEGACY: ", len(legacy))
print(list(legacy.columns))

# I don't need to process spectra for all ~3000 test set stars. Just the cross-match of the test set and the Silva Aguirre+ LEGACY sample.
inference_legacy = pd.merge(legacy[['KIC','Age','sAgeP','sAgeM']], silver_inferences_kic, left_on='KIC', right_on='kepid', how='inner')
print(list(inference_legacy.columns))
print(inference_legacy)

# use Aida's normalization code on the inference spectra
directory = path+'data/silva_aguirre_apogee_spectra/' 
bedell_kic_apogee = pd.read_csv(path+'data/bedell_kic_apogee.csv')
training_names = bedell_kic_apogee['sdss_id'].astype(str)
spectra_paths = get_files_in_order(directory, training_names)
label_names=["Teff", "logg", "feh", "mg_h", "Age", "Dnu"]

fluxes=[]
ivars=[]
success_sdss_ids=[]
chisqs = []
for spectra_path in spectra_paths: # toggle for short or full version
	# looks like sdss_access failed for six spectra. handle these.
	sdss_id = get_number_between(spectra_path, 'mwmStar-0.6.0-', '.fits')
	print(sdss_id)

	# if sdss_id is in the silver sample (or whatever the inference sample is), proceed. else, skip to next sdss_id
	if sdss_id in list(inference_legacy['sdss_id']):
		pass
	else:
		continue

	#wl,flux_single,ivar_single = process_spectra(spectra_path,10)
	wl,flux_single,ivar_single = process_spectra_chisq(spectra_path,10) # 10 is the width of your Gaussian for continuum normalization

	fluxes.append(flux_single)
	ivars.append(ivar_single)
	success_sdss_ids.append(sdss_id)
     
print("number of spectra: ", len(fluxes))
print("wl: ", wl)

# read in training set and keep label averages, for test step optimization intialization
training = pd.read_csv(path+'data/4_fold_cv_no_gb.csv') # 4_fold_cv.csv, 4_fold_cv_ruwe.csv, 4_fold_cv_teff.csv, 4_fold_cv_teff_ruwe.csv, 4_fold_cv_no_rgb.csv
training = training.loc[training['KIC']!=7976303]
training = training.loc[training['KIC']!=10454113]
training_labels = [np.nanmean(training['aspcap_teff']), np.nanmean(training['aspcap_logg']), np.nanmean(training['aspcap_fe_h']), np.nanmean(training['aspcap_mg_h']), np.nanmean(training['Age']), np.nanmean(training['Dnu'])]

# read in model 
model = tc.CannonModel.read(path+"no-rgb.model") # apogee-serenelli-lite.model
s2 = np.loadtxt(path+'data/s2-no-rgb.txt',delimiter=',',dtype=float) 

def cov_matrix(cov):
	# model-assigned label scatter: this is for sigma_A, B, as well as errorbars at the individual star level 
	matrix = np.zeros((len(cov),6)) # Pre-allocate matrix
	for i in range(0,len(cov)):
		matrix[i,:] = np.sqrt(np.diag(cov[i]))

	df_sigma = pd.DataFrame(matrix)
	return df_sigma

# inference!
labels_arr = []
sigma_stars = []
for i in tqdm(range(len(fluxes))):
    flux = fluxes[i]
    ivar = ivars[i]
    labels, cov, metadata = model.test(flux, ivar, initial_labels=training_labels) # upon Andy's suggestion, for LEGACY comparison only; these are training set label averages
    #labels, cov, metadata = model.test(flux, ivar) # original
    print("labels, cov, metadata: ", labels, cov, metadata)
    labels_arr.append(labels)
    
    # use cov to propagate per-star, per-visit uncertainty 
    #matrix = np.zeros((len(cov),len(label_names))) # Pre-allocate matrix
    #for j in range(0,len(cov)):
    #    matrix[j,:] = np.sqrt(np.diag(cov[j]))
    #cov_arr.append(matrix)
    	
    sigma_star = np.array(cov_matrix(cov))
    sigma_stars.append(sigma_star)

	# get Cannon-derived model spectra
    model_spectrum = model(labels) 
	
	# chisq of model spectral fit
    spec_fit_chisq = np.sum(((model_spectrum-flux)**2)/(ivar**-1 + s2))
    #print("chisq: ", spec_fit_chisq)
    chisqs.append(spec_fit_chisq)

# joint LEGACY and our test set
silver = pd.DataFrame()
silver['sdss_id'] = success_sdss_ids
silver['chisq'] = chisqs

# map sdss_id to KIC
preds = pd.DataFrame()
preds['kepid'] = bedell_kic_apogee['kepid']
preds['source_id'] = bedell_kic_apogee['source_id']
preds['sdss_id'] = bedell_kic_apogee['sdss_id']

# looks like sdss_access failed for six spectra. handle these.
preds = preds.loc[preds['sdss_id'].isin(np.array(success_sdss_ids))]

# enrich silver inference sample with ASPCAP parameters and Kepid
silver_preds = pd.merge(silver, preds, on='sdss_id', how='inner')

# these are our Cannon-predicted parameters
silver_preds['Teff_pred'] = np.array(labels_arr)[:,0][:,0]
silver_preds['logg_pred'] = np.array(labels_arr)[:,0][:,1]
silver_preds['fe_h_pred'] = np.array(labels_arr)[:,0][:,2]
silver_preds['mg_h_pred'] = np.array(labels_arr)[:,0][:,3]
silver_preds['Age_pred'] = np.array(labels_arr)[:,0][:,4]
silver_preds['Dnu_pred'] = np.array(labels_arr)[:,0][:,5]
silver_preds['sigma_star_Teff'] = np.array(sigma_stars)[:,0][:,0]
silver_preds['sigma_star_logg'] = np.array(sigma_stars)[:,0][:,1]
silver_preds['sigma_star_fe_h'] = np.array(sigma_stars)[:,0][:,2]
silver_preds['sigma_star_mg_h'] = np.array(sigma_stars)[:,0][:,3]
silver_preds['sigma_star_age'] = np.array(sigma_stars)[:,0][:,4]
silver_preds['sigma_star_Dnu'] = np.array(sigma_stars)[:,0][:,5]

print(silver_preds)
silver_preds.to_csv(path+'data/silver_legacy_inferences_kic_no_rgb_init.csv', index=False)
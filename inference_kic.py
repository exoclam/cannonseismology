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
#path = '/home/c.lam/blue/cannon-ages/'

### read in run results
silver_inferences_kic = pd.read_csv(path+'data/silver_inferences_kic_no_rgb.csv')
silver_inferences_kic_head = silver_inferences_kic[['kepid','sdss_id','Teff_pred','logg_pred','fe_h_pred','mg_h_pred','Age_pred','Dnu_pred','chisq','sigma_star_Teff','sigma_star_logg','sigma_star_fe_h','sigma_star_mg_h','sigma_star_age','sigma_star_Dnu']].head()
inferences_latex = silver_inferences_kic_head.style.hide(axis="index").format({
    "aspcap_logg": "{:.2f}",
    "aspcap_fe_h": "{:.2f}",
    "aspcap_mg_h": "{:.2f}",
	"Age": "{:.2f}",
	"Dnu": "{:.2f}",
}).to_latex()
print(inferences_latex)

print(silver_inferences_kic)
print(silver_inferences_kic.loc[silver_inferences_kic['chisq']>100000])
print(list(silver_inferences_kic.columns))
silver_inferences_kic = silver_inferences_kic.loc[silver_inferences_kic['chisq']<100000]

# sigma_inflate_teff = 31.0 K (1.491 dex --> -0.014, -0.038, 1.000) (10**sigma_inflate --> median, mean, std)
# sigma_inflate_logg = 0.043 (-1.371 dex --> -0.000, -0.013, 1.000)
# sigma_inflate_fe_h = 0.027 (-1.565 dex --> 0.005, -0.087, 0.999)
# sigma_inflate_mg_h = 0.026 (-1.586 dex --> -0.037, -0.101 1.000)
# sigma_inflate_age = 0.417 (-0.380 dex --> -0.008, -0.045, 0.998)
# sigma_inflate_Dnu = 0.751 (5.637 dex --> -0.005, 0.038, 1.000)
silver_inferences_kic['Teff_err'] = np.sqrt(silver_inferences_kic['sigma_star_Teff']**2 + 31**2)
silver_inferences_kic['logg_err'] = np.sqrt(silver_inferences_kic['sigma_star_logg']**2 + 0.043**2)
silver_inferences_kic['feh_err'] = np.sqrt(silver_inferences_kic['sigma_star_fe_h']**2 + 0.027**2)
silver_inferences_kic['mg_h_err'] = np.sqrt(silver_inferences_kic['sigma_star_mg_h']**2 + 0.026**2)
silver_inferences_kic['Age_err'] = np.sqrt(silver_inferences_kic['sigma_star_age']**2 + 0.417**2)
silver_inferences_kic['Dnu_err'] = np.sqrt(silver_inferences_kic['sigma_star_Dnu']**2 + 0.751**2)

silver_inferences_kic["Teff"] = silver_inferences_kic.apply(
    lambda row: f"{row['Teff_pred'].astype(int)} \\pm {row['Teff_err'].astype(int)}", axis=1)
silver_inferences_kic["logg"] = silver_inferences_kic.apply(
    lambda row: f"{row['logg_pred']:.3f} \\pm {row['logg_err']:.3f}", axis=1)
silver_inferences_kic["feh"] = silver_inferences_kic.apply(
    lambda row: f"{row['fe_h_pred']:.3f} \\pm {row['feh_err']:.3f}", axis=1)
silver_inferences_kic["mg_h"] = silver_inferences_kic.apply(
    lambda row: f"{row['mg_h_pred']:.3f} \\pm {row['mg_h_err']:.3f}", axis=1)
silver_inferences_kic["Age"] = silver_inferences_kic.apply(
    lambda row: f"{row['Age_pred']:.2f} \\pm {row['Age_err']:.2f}", axis=1)
silver_inferences_kic["Dnu"] = silver_inferences_kic.apply(
    lambda row: f"{row['Dnu_pred']:.2f} \\pm {row['Dnu_err']:.2f}", axis=1)
silver_inferences_kic['chisq_reduced'] = silver_inferences_kic['chisq']/(7409 - 6) # 7409 discrete wavelenghts in each spectra; 6 parameters
#silver_inferences_kic['chisq_reduced'] = np.round(silver_inferences_kic['chisq_reduced'], 2)
print(silver_inferences_kic.loc[silver_inferences_kic['chisq_reduced']>7])

silver_inferences_kic_head = silver_inferences_kic[['kepid','sdss_id','Teff','logg','feh','mg_h','Age','Dnu','chisq_reduced']].head(n=10)
print(silver_inferences_kic_head)
#inferences_latex = silver_inferences_kic_head.style.hide(axis="index").to_latex()
inferences_latex = silver_inferences_kic_head.style.hide(axis="index").format({
	#"Age": "{:.2f}",
	#"Dnu": "{:.2f}",
	"chisq_reduced": "{:.2f}"
}).to_latex()
print(inferences_latex)

print(np.mean(silver_inferences_kic['chisq_reduced']))
print(np.std(silver_inferences_kic['chisq_reduced']))
print(np.median(silver_inferences_kic['chisq_reduced']))
plt.hist(silver_inferences_kic['chisq_reduced'])
plt.show()

### compare with LEGACY ages
legacy = pd.read_csv(path+'data/silva-aguirre-legacy.txt',sep='\s+')
print("size of LEGACY: ", len(legacy))
print(list(legacy.columns))

legacy_silver_inference = pd.merge(legacy[['KIC','Age','sAgeP','sAgeM']], silver_inferences_kic[['kepid','Age_pred','Age_err']], left_on='KIC', right_on='kepid', how='inner')
print("size of LEGACY-silver inference overlap: ", len(legacy_silver_inference))
print(legacy_silver_inference)
plt.plot(np.linspace(-0.4, 14, 10), np.linspace(-0.4, 14, 10), color='k', alpha=0.5)
plt.errorbar(legacy_silver_inference['Age'], legacy_silver_inference['Age_pred'], xerr=[legacy_silver_inference['sAgeP'], -1*legacy_silver_inference['sAgeM']], yerr=legacy_silver_inference['Age_err'], linestyle='', marker='o', color="#1D5A0E", alpha=0.4)
plt.xlabel(r"age [Gyr], Cannon")
plt.ylabel(r"age [Gyr], LEGACY")
plt.xlim([-0.4, 14])
plt.ylim([-0.4, 14.])
x_rms = np.round(np.std(np.sqrt(0.5*(legacy_silver_inference['Age']-legacy_silver_inference['Age_pred'])**2)),2)
x_std = np.round(np.std(legacy_silver_inference['Age']),2)
plt.text(0.25, 13, r'$\sigma_{X_{LEGACY}-X_{Cannon}}$ = ' + f'{x_rms}', fontsize=15, horizontalalignment='left')
plt.text(0.25, 12, r'$\sigma_{X_{LEGACY}}$ = ' + f'{x_std}', fontsize=15, horizontalalignment='left')
plt.tight_layout()
plt.savefig(path+'plots/legacy_age_silver_no_rgb.png')
#plt.savefig(path+'plots/legacy_age_compare_no_rgb_limited.png')
plt.show()
quit()

### run below for the first time only!!!

bedell_kic_apogee = pd.read_csv(path+'data/bedell_kic_apogee.csv')
#bedell_kic_apogee = bedell_kic_apogee.loc[bedell_kic_apogee['sdss_id'].isin(np.array([66646541,66647080,66647116,66647134,66647246,66647251]))]
training_names = bedell_kic_apogee['sdss_id'].astype(str)

inferences_df_culled = pd.read_csv(path+'data/inferences_kic_culled.csv')

# use Aida's normalization code on the inference spectra
directory = path+'data/kic_spectra/' 
#spectra_paths = sorted(os.listdir(directory))
spectra_paths = get_files_in_order(directory, training_names)
label_names=["Teff", "logg", "feh", "mg_h", "Age", "Dnu"]

fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
s2 = np.loadtxt(path+'data/s2-no-rgb.txt',delimiter=',',dtype=float) 
hdul_lite = fits.open(fits_image_filename_lite)  
#sdss_ids = []
source_id_dr2s = []
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
chisqs = []
for spectra_path in spectra_paths: # toggle for short or full version
	#wl,flux_single,ivar_single = process_spectra(spectra_path,10)
	wl,flux_single,ivar_single = process_spectra_chisq(spectra_path,10) # 10 is the width of your Gaussian for continuum normalization
	     
    # looks like sdss_access failed for six spectra. handle these.
	sdss_id = get_number_between(spectra_path, 'mwmStar-0.6.0-', '.fits')

	# if sdss_id is in the silver sample (or whatever the inference sample is), proceed. else, skip to next sdss_id
	#print(sdss_id)
	#print(inferences_df_culled['sdss_id'])
	if sdss_id in list(inferences_df_culled['sdss_id']):
		pass
	else:
		continue

	fluxes.append(flux_single)
	ivars.append(ivar_single)
	success_sdss_ids.append(sdss_id)
     
    # pull Gaia(?) stellar params from mwmLite
	teff = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].teff[0]
	e_teff = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].e_teff[0]
	logg = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].logg[0]
	e_logg = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].e_logg[0]
	fe_h = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].fe_h[0]
	e_fe_h = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].e_fe_h[0]
	mg_h = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].mg_h[0]
	e_mg_h = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].e_mg_h[0]
	source_id_dr2 = hdul_lite[1].data[hdul_lite[1].data.sdss_id==sdss_id].gaia_dr2_source_id[0]
	teffs.append(teff)
	e_teffs.append(e_teff)
	loggs.append(logg)
	e_loggs.append(e_logg)
	fe_hs.append(fe_h)
	e_fe_hs.append(e_fe_h)
	mg_hs.append(mg_h)
	e_mg_hs.append(e_mg_h)
	source_id_dr2s.append(source_id_dr2)
	#sdss_ids.append(sdss_id)

print("number of spectra: ", len(fluxes))
print("wl: ", wl)

# read in model 
model = tc.CannonModel.read(path+"no-rgb.model") # apogee-serenelli-lite.model

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
    labels, cov, metadata = model.test(flux, ivar)
    #print("labels, cov, metadata: ", labels, cov, metadata)
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

#print(len(chisqs))
#print(len(fluxes))
#print(len(teffs))
#print(len(success_sdss_ids))

# these are our actual inference set
silver = pd.DataFrame()
silver['sdss_id'] = success_sdss_ids
silver['source_id_dr2'] = source_id_dr2s
silver['chisq'] = chisqs
silver['mg_h'] = mg_hs # I originally only queried mg_h but not e_mg_h, so we need to quickly grab these for the silver sample
silver['e_mg_h'] = e_mg_hs

# initial inference set, pre-training set parameter space culling
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
#preds['mg_h'] = bedell_kic_apogee['mg_h']

# looks like sdss_access failed for six spectra. handle these.
preds = preds.loc[preds['sdss_id'].isin(np.array(success_sdss_ids))]

# enrich silver inference sample with ASPCAP parameters and Kepid
silver_preds = pd.merge(silver, preds, on='sdss_id', how='inner')

"""
# these are ASPCAP? Gaia? parameters. Check against ASPCAP to be sure.
preds['teff'] = teffs
preds['e_teff'] = e_teffs
preds['logg_aspcap'] = loggs
preds['e_logg_aspcap'] = e_loggs
preds['fe_h'] = fe_hs
preds['e_fe_h'] = e_fe_hs
preds['mg_h_aspcap'] = mg_hs
preds['e_mg_h'] = e_mg_hs
preds['source_id_dr2'] = source_id_dr2s
"""

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
silver_preds.to_csv(path+'data/silver_inferences_kic_no_rgb.csv', index=False)
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

def get_spectra(sdss_id, path, folder, visit_flag=False):

    # do I get the squashed version or the visit-by-visit version?
    if visit_flag==False:
        visit_or_star = 'Star'
    elif visit_flag==True:
        visit_or_star = 'Visit'

    access.add('mwm'+visit_or_star, v_astra='0.6.0', component='', sdss_id=sdss_id)
    access.set_stream()
    access.commit()
    
    mwm_filename = access.full('mwm'+visit_or_star, v_astra='0.6.0', component='', sdss_id=sdss_id)
    print(mwm_filename)
    
    # read to fits, bc actually it'll be easier to handle columns of lists this way
    mwm = fits.open(mwm_filename)
    try:
        mwm.writeto(path+'data/'+folder+'/mwm'+visit_or_star+'-0.6.0-'+str(sdss_id)+'.fits', overwrite=True)
        return path+'data/'+folder+'/mwm'+visit_or_star+'-0.6.0-'+str(sdss_id)+'.fits'
    
    except Exception as e:
        print(e)
        pass
"""
### RUN THIS WHOLE THING ONLY ONCE
fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)  
lite_source_ids = hdul_lite[1].data.gaia_dr3_source_id

# Bedell cross-match has the Gaia DR3 source_id we need 
bedell = Table.read(path+'data/kepler_dr3_good.fits')
bedell_df = bedell.to_pandas()

legacy = pd.read_csv(path+'data/silva-aguirre-legacy.txt',sep='\s+')
legacy_bedell = pd.merge(legacy, bedell_df, left_on='KIC', right_on='kepid', how='left')

# use DR3 source_id to get sdss_id from mwmLite
legacy_bedell_apogee = legacy_bedell.loc[legacy_bedell['source_id'].isin(lite_source_ids)]

source_ids = legacy_bedell_apogee['source_id']
sdss_ids = []
for source_id in tqdm(source_ids):
	sdss_id = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].sdss_id[0]
	sdss_ids.append(sdss_id)
     
	get_spectra(sdss_id, path, 'silva_aguirre_apogee_spectra', visit_flag=False)

legacy_bedell_apogee['sdss_id'] = sdss_ids

training_names = legacy_bedell_apogee['sdss_id'].astype(str)

# use Aida's normalization code on the inference spectra
directory = path+'data/silva_aguirre_apogee_spectra/' 
#spectra_paths = sorted(os.listdir(directory))
spectra_paths = get_files_in_order(directory, training_names)
label_names=["Teff", "logg", "feh", "mg_h", "Age", "Dnu"]

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
for spectra_path in spectra_paths: # toggle for short or full version
	wl,flux_single,ivar_single = process_spectra(spectra_path,10) # 10 is the width of your Gaussian for continuum normalization
	fluxes.append(flux_single)
	ivars.append(ivar_single)
	     
    # looks like sdss_access failed for six spectra. handle these.
	sdss_id = get_number_between(spectra_path, 'mwmStar-0.6.0-', '.fits')
	success_sdss_ids.append(sdss_id)
     
    # pull (Gaia?) stellar params from mwmLite
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

# read in model 
model = tc.CannonModel.read(path+"apogee-serenelli-lite-ruwe.model") # apogee-serenelli-lite.model

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

print("cov: ", cov_arr)

preds = pd.DataFrame()
preds['kepid'] = legacy_bedell_apogee['kepid']
preds['source_id'] = legacy_bedell_apogee['source_id']
preds['sdss_id'] = legacy_bedell_apogee['sdss_id']

# looks like sdss_access failed for six spectra. handle these.
preds = preds.loc[preds['sdss_id'].isin(np.array(success_sdss_ids))]

# mwmLite (GaiaDR3?) params
preds['teff'] = teffs
preds['e_teff'] = e_teffs
preds['logg_aspcap'] = loggs
preds['e_logg_aspcap'] = e_loggs
preds['fe_h'] = fe_hs
preds['e_fe_h'] = e_fe_hs
preds['mg_h_aspcap'] = mg_hs
preds['e_mg_h'] = e_mg_hs
preds['source_id_dr2'] = source_id_dr2s

# Cannon-predicted params
preds['Teff_pred'] = np.array(labels_arr)[:,0][:,0]
preds['logg_pred'] = np.array(labels_arr)[:,0][:,1]
preds['fe_h_pred'] = np.array(labels_arr)[:,0][:,2]
preds['mg_h_pred'] = np.array(labels_arr)[:,0][:,3]
preds['Age_pred'] = np.array(labels_arr)[:,0][:,4]
preds['Dnu_pred'] = np.array(labels_arr)[:,0][:,5]

preds = pd.merge(preds, legacy, left_on='kepid', right_on='KIC', how='left')
print(preds)
preds.to_csv(path+'data/inferences_silva_aguirre.csv', index=False)
"""

preds = pd.read_csv(path+'data/inferences_legacy_ruwe.csv',sep=',') # inferences_silva_aguirre_ruwe.csv
print(list(preds.columns))

cannon_preds = pd.read_csv(path+'data/enriched_lite_visits_chisq.csv', sep=',')
print(cannon_preds)
cannon_preds['age_error'] = np.sqrt(cannon_preds['sigma_star_age']**2 + 0.398**2) # use sigma_inflate-informed age error here
print(cannon_preds['age_error'])

preds['age_error'] = cannon_preds['age_error']

plt.plot(np.arange(0, 14), np.arange(0, 14), color='k', alpha=0.5)
plt.errorbar(preds['Age_pred'], preds['Age'], xerr=preds['age_error'], yerr=[preds['sAgeP'],-1*preds['sAgeM']], linestyle='', marker='o', color="#B521B2", alpha=0.4)
plt.xlabel(r"age [Gyr], Cannon")
plt.ylabel(r"age [Gyr], Legacy")
#plt.xlim([0, 14])
#plt.ylim([0, 14])
#plt.legend()
plt.savefig(path+'plots/legacy_age_compare.png')
plt.show()
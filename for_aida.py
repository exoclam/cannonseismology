# -*- coding: utf-8 -*-
"""
@author: behmardaida, 4/18/2025

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from astropy.io import fits
from astropy.table import Table

import thecannon as tc
print(tc.__version__)
from process_spectra_gaus import *

import loocv

path = '/Users/chrislam/Desktop/cannon-ages/' 

"""
df = pd.read_csv(path+'data/small.csv',index_col=False)
df = df[df.Teff.notnull()]
df = df[df.mg_h.notnull()]
df = df[df.Age.notnull()]
df = df.reset_index(drop=True)
#df = df.loc[df['sdss_id_sec'] == 114879184]
"""

df = pd.read_csv(path+'data/enriched.csv', sep=',')
df['sdss_id'] = df['sdss_id'].astype(int)
#df = df.iloc[:100]
df = df[df.Teff.notnull()]
df = df[df.mg_h.notnull()]
df = df[df.Age.notnull()]
df = df[df.feh.notnull()]
df = df[df.logg.notnull()]
df = df[df.Dnu.notnull()]
df = df[df.numax.notnull()]
df = df.reset_index(drop=True)

training_names = df['sdss_id'].astype(str)
directory = path+'data/spectra/' # e.g., mwmStar-0.6.0-114879184.fits
spectra_paths = get_files_in_order(directory, training_names)

flux_tr=[]
ivar_tr=[]
for spectra_path in spectra_paths:
	wl,flux_single,ivar_single = process_spectra(spectra_path,10) # 10 is the width of your Gaussian for continuum normalization
	flux_tr.append(flux_single)
	ivar_tr.append(ivar_single)

#flux_tr = np.array(flux_tr)
#ivar_tr = np.array(ivar_tr)

"""
df = pd.read_csv('files/example.csv',index_col=False)
df = df[df.teff_sec.notnull()]
df = df.reset_index(drop=True)

training_names = df['sdss_id_sec'].astype(str)
directory = 'put_your_APOGEE_fits_files_here/' # e.g., mwmStar-0.6.0-114879184.fits
paths = get_files_in_order(directory, training_names)

flux_tr=[]
ivar_tr=[]
for path in paths:
	wl,flux_single,ivar_single = process_spectra(path,10) # 10 is the width of your Gaussian for continuum normalization
	flux_tr.append(flux_single)
	ivar_tr.append(ivar_single)
"""

# constructs training set label matrix
#Teff = df['Teff'].values
#logg = df['logg'].values
#fe_h = df['feh'].values
#mg_h = df['mg_h'].values
#Age = df['Age'].values
#Dnu = df['Dnu'].values
#label_tr = np.vstack((Teff,logg,fe_h,mg_h,Age,Dnu)).T
label_names = ['Teff', 'logg', 'feh', 'mg_h', 'Age', 'Dnu','numax']
#preds = loocv.loocv(df, wl, flux_tr, ivar_tr, label_names)

test_labels_arr, true_labels_arr, model = loocv.loocv(df, wl, flux_tr, ivar_tr, label_names)

model.write(path+"apogee-serenelli.model") # write out model
# new_model = tc.CannonModel.read("apogee-dr14-giants.model") # read in model

preds = pd.DataFrame()
preds['sdss_id'] = df['sdss_id']
#preds['sdss_id'] = df['sdss_id'][:temp_length]
#print(np.array(s2_arr))

preds['Teff_pred'] = np.array(test_labels_arr)[:,0][:,0]
preds['logg_pred'] = np.array(test_labels_arr)[:,0][:,1]
preds['fe_h_pred'] = np.array(test_labels_arr)[:,0][:,2]
preds['mg_h_pred'] = np.array(test_labels_arr)[:,0][:,3]
preds['Age_pred'] = np.array(test_labels_arr)[:,0][:,4]
preds['Dnu_pred'] = np.array(test_labels_arr)[:,0][:,5]
preds['numax_pred'] = np.array(test_labels_arr)[:,0][:,6]

preds['Teff_test'] = np.array(true_labels_arr)[:,0][:,0]
preds['logg_test'] = np.array(true_labels_arr)[:,0][:,1]
preds['fe_h_test'] = np.array(true_labels_arr)[:,0][:,2]
preds['mg_h_test'] = np.array(true_labels_arr)[:,0][:,3]
preds['Age_test'] = np.array(true_labels_arr)[:,0][:,4]
preds['Dnu_test'] = np.array(true_labels_arr)[:,0][:,5]
preds['numax_test'] = np.array(true_labels_arr)[:,0][:,6]
print(preds)
preds.to_csv(path+'data/preds_dnu_numax_full.csv', index=False)

plt.scatter(preds['Teff_pred'], preds['Teff_test'])
plt.plot(preds['Teff_test'], preds['Teff_test'])
plt.xlabel(r"$T_{\rm eff}$ [K], pred")
plt.ylabel(r"$T_{\rm eff}$ [K], test")
plt.xlim([4750, 6750])
plt.ylim([4750, 6750])
#plt.legend()
plt.savefig(path+'plots/teff_check_dnu_numax_full.png')
plt.show()

plt.scatter(preds['logg_pred'], preds['logg_test'])
plt.plot(preds['logg_test'], preds['logg_test'])
plt.xlabel(r"logg, pred")
plt.ylabel(r"logg, test")
plt.xlim([3.3, 4.4])
plt.ylim([3.3, 4.4])
plt.savefig(path+'plots/logg_check_dnu_numax_full.png')
plt.show()

plt.scatter(preds['fe_h_pred'], preds['fe_h_test'])
plt.plot(preds['fe_h_test'], preds['fe_h_test'])
plt.xlabel(r"[Fe/H], pred")
plt.ylabel(r"[Fe/H], test")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
plt.savefig(path+'plots/feh_check_dnu_numax_full.png')
plt.show()

plt.scatter(preds['mg_h_pred'], preds['mg_h_test'])
plt.plot(preds['mg_h_test'], preds['mg_h_test'])
plt.xlabel(r"[Mg/H], pred")
plt.ylabel(r"[Mg/H], test")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
plt.savefig(path+'plots/mg_h_check_dnu_numax_full.png')
plt.show()

plt.scatter(preds['Age_pred'], preds['Age_test'])
plt.plot(preds['Age_test'], preds['Age_test'])
plt.xlabel(r"age [Gyr], pred")
plt.ylabel(r"age [Gyr], test")
plt.xlim([0, 14])
plt.ylim([0, 14])
plt.savefig(path+'plots/age_check_dnu_numax_full.png')
plt.show()

plt.scatter(preds['Dnu_pred'], preds['Dnu_test'])
plt.plot(preds['Dnu_test'], preds['Dnu_test'])
plt.xlabel(r'$\Delta \nu [\mu Hz]$, pred')
plt.ylabel(r'$\Delta \nu [\mu Hz]$, test')
plt.xlim([0, 160])
plt.ylim([0, 160])
plt.savefig(path+'plots/Dnu_check_dnu_numax_full.png')
plt.show()

plt.scatter(preds['numax_pred'], preds['numax_test'])
plt.plot(preds['numax_test'], preds['numax_test'])
plt.xlabel(r'$\nu_{max} [\mu Hz]$, pred')
plt.ylabel(r'$\nu_{max} [\mu Hz]$, test')
plt.xlim([300, 3600])
plt.ylim([300, 3600])
plt.savefig(path+'plots/numax_check_dnu_numax_full.png')
plt.show()
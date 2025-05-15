#### labels from Serenelli+17 https://iopscience.iop.org/article/10.3847/1538-4365/aa97df
#### upstream data products, including Serenelli+17 Tables 3 and 4, ASPCAP abundances, and mwmAllStar from Astra, are in cannon-ages.ipynb.

from astropy.table import Table
from six.moves import cPickle as pickle
from sys import version_info

import thecannon as tc
print(tc.__version__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table

from tqdm import tqdm

import process_spectra_gaus
import loocv

path = '/Users/chrislam/Desktop/cannon-ages/' 

# mise en place
apokasc_sdss_bedell_df = pd.read_csv(path+'data/enriched.csv', sep=',')
#apokasc_sdss_bedell_df = apokasc_sdss_bedell_df.dropna(subset=["Teff", "logg", "feh", "mg_h", "Age", "numax", "Dnu"])
apokasc_sdss_bedell_df['sdss_id'] = apokasc_sdss_bedell_df['sdss_id'].astype(int)
apokasc_sdss_bedell_df = apokasc_sdss_bedell_df.iloc[:100]

"""
# run this one time, ever, in order to copy the same 100 set of test stars to a demo directory
prefix = "mwmStar-0.6.0-"
suffix = ".fits"
source_directory = path+"data/spectra/"  # Replace with the actual source directory path
destination_directory = path+"data/spectra-100/"  # Replace with the actual destination directory path
ids = np.array(apokasc_sdss_bedell_df['sdss_id'])

filenames = loocv.create_filenames_from_ids(ids, prefix, suffix)
loocv.copy_files(filenames, source_directory, destination_directory)
"""

# diagnose numax and Dnu
"""
plt.scatter(apokasc_sdss_bedell_df['numax'], apokasc_sdss_bedell_df['Teff'])
plt.xlabel('numax')
plt.ylabel('Teff')
plt.show()

plt.scatter(apokasc_sdss_bedell_df['Dnu'], apokasc_sdss_bedell_df['Teff'])
plt.xlabel('Dnu')
plt.ylabel('Teff')
plt.show()

plt.scatter(apokasc_sdss_bedell_df['numax'], apokasc_sdss_bedell_df['Dnu'])
plt.xlabel('numax')
plt.ylabel('Dnu')
plt.show()
"""

#test_sdss_list = [67760674, 67760898, 67762082, 67762314, 67763169, 67764481, 67765417, 67766052, 67766853, 67767827, 67768040] # for debugging only!
#apokasc_sdss_bedell_df = apokasc_sdss_bedell_df.loc[apokasc_sdss_bedell_df['sdss_id'].isin(test_sdss_list)].reset_index() # for debugging only!
sdss_ids = apokasc_sdss_bedell_df['sdss_id']

directory = path+'data/spectra-100/' # use /test-spectra/ or /spectra-100/ for debugging and \spectra\ for production
fits_paths = process_spectra_gaus.get_files_in_order(directory, str(sdss_ids)) # courtesy of Aida Behmard

# continuum normalize spectra, using Aida Behmard's process_spectra_gaus module
fluxes=[]
ivars=[]
sdss_ids = []
for fits_path in fits_paths:
	sdss_id = process_spectra_gaus.get_number_between(fits_path, "0.6.0-", ".fits")
	sdss_ids.append(sdss_id)
	
	try:
		wl,flux_single,ivar_single = process_spectra_gaus.process_spectra(fits_path,10) # 10 is the width of your Gaussian for continuum normalization
		fluxes.append(flux_single)
		ivars.append(ivar_single)
		
	except Exception as e:
		print(e)
		print(fits_path)
		break
	
	print(flux_single)
	print(ivar_single)
	print(sdss_id)
	print(fits_path)
	quit()

# for debugging only: keep medium subset of 100 stars
apokasc_sdss_bedell_df = apokasc_sdss_bedell_df.loc[apokasc_sdss_bedell_df['sdss_id'].isin(sdss_ids)].reset_index()

test_labels_arr, true_labels_arr = loocv.loocv(apokasc_sdss_bedell_df, wl, fluxes, ivars, label_names=["Teff", "logg", "feh", "mg_h", "Age"])

preds = pd.DataFrame()
preds['sdss_id'] = apokasc_sdss_bedell_df['sdss_id']
#preds['sdss_id'] = df['sdss_id'][:temp_length]
#print(np.array(s2_arr))

preds['Teff_pred'] = np.array(test_labels_arr)[:,0][:,0]
preds['logg_pred'] = np.array(test_labels_arr)[:,0][:,1]
preds['fe_h_pred'] = np.array(test_labels_arr)[:,0][:,2]
preds['mg_h_pred'] = np.array(test_labels_arr)[:,0][:,3]
preds['Age_pred'] = np.array(test_labels_arr)[:,0][:,4]

preds['Teff_test'] = np.array(true_labels_arr)[:,0][:,0]
preds['logg_test'] = np.array(true_labels_arr)[:,0][:,1]
preds['fe_h_test'] = np.array(true_labels_arr)[:,0][:,2]
preds['mg_h_test'] = np.array(true_labels_arr)[:,0][:,3]
preds['Age_test'] = np.array(true_labels_arr)[:,0][:,4]

#preds['numax_pred'] = np.array(test_labels_arr)[:,0][:,5]
#preds['Dnu_pred'] = np.array(test_labels_arr)[:,0][:,5]
#preds['numax_test'] = np.array(true_labels_arr)[:,0][:,5]
#preds['Dnu_test'] = np.array(true_labels_arr)[:,0][:,5]
preds.to_csv(path+'data/preds_small.csv', index=False)	

plt.scatter(preds['Teff_pred'], preds['Teff_test'])
plt.plot(preds['Teff_test'], preds['Teff_test'])
plt.xlabel(r"$T_{\rm eff}$ [K], pred")
plt.ylabel(r"$T_{\rm eff}$ [K], test")
plt.xlim([4700, 7000])
plt.ylim([4700, 7000])
#plt.legend()
plt.savefig(path+'plots/teff_check_small.png')
plt.show()

plt.scatter(preds['logg_pred'], preds['logg_test'])
plt.plot(preds['logg_test'], preds['logg_test'])
plt.xlabel(r"logg, pred")
plt.ylabel(r"logg, test")
plt.xlim([3.3, 4.5])
plt.ylim([3.3, 4.5])
plt.savefig(path+'plots/logg_check_small.png')
plt.show()

plt.scatter(preds['fe_h_pred'], preds['fe_h_test'])
plt.plot(preds['fe_h_test'], preds['fe_h_test'])
plt.xlabel(r"[Fe/H], pred")
plt.ylabel(r"[Fe/H], test")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
plt.savefig(path+'plots/feh_check_small.png')
plt.show()

plt.scatter(preds['mg_h_pred'], preds['mg_h_test'])
plt.plot(preds['mg_h_test'], preds['mg_h_test'])
plt.xlabel(r"[Mg/H], pred")
plt.ylabel(r"[Mg/H], test")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
plt.savefig(path+'plots/mg_h_check_small.png')
plt.show()

plt.scatter(preds['Age_pred'], preds['Age_test'])
plt.plot(preds['Age_test'], preds['Age_test'])
plt.xlabel(r"age [Gyr], pred")
plt.ylabel(r"age [Gyr], test")
plt.xlim([0, 14])
plt.ylim([0, 14])
plt.savefig(path+'plots/age_check_small.png')
plt.show()

#plt.scatter(preds['numax_pred'], preds['numax_test'])
#plt.plot(preds['numax_test'], preds['numax_test'])
#plt.xlabel(r'$\nu_{max} [\mu Hz]$, pred')
#plt.ylabel(r'$\nu_{max} [\mu Hz]$, test')
#plt.xlim([300, 3600])
#plt.ylim([300, 3600])
#plt.savefig(path+'plots/numax_check_small.png')
#plt.show()

"""
plt.scatter(preds['Dnu_pred'], preds['Dnu_test'])
plt.plot(preds['Dnu_test'], preds['Dnu_test'])
plt.xlabel(r'$\Delta \nu [\mu Hz]$, pred')
plt.ylabel(r'$\Delta \nu [\mu Hz]$, test')
plt.xlim([0, 160])
plt.ylim([0, 160])
plt.savefig(path+'plots/Dnu_check_small.png')
plt.show()
"""
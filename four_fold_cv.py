from sklearn.model_selection import KFold

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

# these are mwmwLite params, which are often from Gaia...need to use ASPCAP, see below
df = pd.read_csv(path+'data/enriched_lite_visits.csv', sep=',') # formerly enriched_lite_visits_chisq_ruwe.csv, enriched_lite_visits.csv
df['sdss_id'] = df['sdss_id'].astype(int)

# use ASPCAP labels, for uniformity's sake
aspcap = pd.read_csv(path+'data/aspcap.csv', sep=',')
df = pd.merge(df, aspcap, on='source_id')
df['Teff'] = df['aspcap_teff']
df['logg'] = df['aspcap_logg']
df['feh'] = df['aspcap_fe_h']
df['mg_h'] = df['aspcap_mg_h']
df = df[df.Teff.notnull()]
df = df[df.mg_h.notnull()]
df = df[df.Age.notnull()]
df = df[df.feh.notnull()]
df = df[df.logg.notnull()]
df = df[df.Dnu.notnull()]
df = df[df.numax.notnull()]

# get rid of SB2s (and one SB3), vetted by manual inspection after an initial round of CV showed some anomalously high chisq
bad_sdss_ids = [67660379, 67478208, 67161877, 67225129, 67114365]
df = df.loc[~df['sdss_id'].isin(bad_sdss_ids)]
df = df.loc[df.Age >= 0] # get rid of -99 Gyr ages (error flag from APOKASC)
df = df.drop_duplicates(subset=['sdss_id'])

### SOFT RUWE CUT, turn off by default!!
#df = df.loc[df['ruwe']<=2]
### TEFF CUT, turn off by default!!
#df = df.loc[(df['Teff']>=5200) & (df['Teff']<=6400)]
### LOGG CUT
#df = df.loc[df['logg']>=4]
df = df.reset_index(drop=True)
#print(df[['sdss_id', 'KIC', 'Teff', 'logg', 'feh', 'mg_h', 'Age', 'Dnu']])

label_names = ['Teff', 'logg', 'feh', 'mg_h', 'Age', 'Dnu'] # 'numax'
training_names = df['sdss_id'].astype(str)
directory = path+'data/spectra/' # e.g., mwmStar-0.6.0-114879184.fits
spectra_paths = get_files_in_order(directory, training_names)

fluxes = np.genfromtxt(path+'data/fluxes_norm.txt', delimiter=',')#[:,:-1]
ivars = np.genfromtxt(path+'data/ivars_norm.txt', delimiter=',')#[:,:-1]

# run this once to grab wavelengths the lazy way
wl,flux_single,ivar_single = process_spectra_chisq(spectra_paths[0],10) 

"""
temp_sdss_ids = []
fluxes=[]
ivars=[]
for spectra_path in spectra_paths:
	wl,flux_single,ivar_single = process_spectra_chisq(spectra_path,10) # 10 is the width of your Gaussian for continuum normalization
	fluxes.append(flux_single)
	ivars.append(ivar_single)
np.savetxt(path+'data/fluxes_norm_logg.txt', fluxes, delimiter=',', newline='\n')
np.savetxt(path+'data/ivars_norm_logg.txt', ivars, delimiter=',', newline='\n')
quit()
"""

def cv(df, wl, fluxes, ivars, label_names=["Teff", "logg", "feh", "mg_h", "Age", "Dnu", "numax"]):
	"""
	Four-fold cross-validation.

	Inputs:
	- df: label DataFrame
	- wl: wavelength support
	- fluxes: normalized fluxes
	- ivars: inverse variances

	Output: 
	- test_labels_arr: literally one row of a label
	- true_labels_arr: corresponding APOKASC/Gaia label
	- model: The Cannon model object
	- s2_arr: s_lambda (Ness+15 Eqn 4) s squared array
	"""

	# Specify the labels that we will use to construct this model.
	#label_names = ["Teff", "logg", "feh"]
	fluxes = np.array(fluxes)
	ivars = np.array(ivars)

	sdss_ids = []
	test_labels_arr = []
	true_labels_arr = []
	theta_arr = []
	cov_arr = []
	s2_arr = []
	spec_fit_chisq_arr = []
	folds = []
	temp_length = 5
	
	kfold = KFold(n_splits=4, shuffle=True, random_state=1)
	for fold, (train, test) in enumerate(kfold.split(df)):
		#print('train: %s, test: %s' % (train, test))
		df_tr = df.iloc[train]
		df_test = df.iloc[test]
		
		fluxes_tr = fluxes[train]
		ivars_tr = ivars[train]
		fluxes_test = fluxes[test]
		ivars_test = ivars[test]
		
		Teff_tr = np.array(df_tr[label_names[0]].values)
		logg_tr = np.array(df_tr[label_names[1]].values)
		fe_h_tr = np.array(df_tr[label_names[2]].values)
		mg_h_tr = np.array(df_tr[label_names[3]].values)
		Age_tr = np.array(df_tr[label_names[4]].values)
		Dnu_tr = np.array(df_tr[label_names[5]].values)
		labels_tr = np.vstack((Teff_tr,logg_tr,fe_h_tr,mg_h_tr,Age_tr,Dnu_tr)).T
		
		Teff_test = np.array(df_test[label_names[0]])
		logg_test = np.array(df_test[label_names[1]])
		fe_h_test = np.array(df_test[label_names[2]])
		mg_h_test = np.array(df_test[label_names[3]])
		Age_test = np.array(df_test[label_names[4]])
		Dnu_test = np.array(df_test[label_names[5]]) 
		labels_test = np.vstack((Teff_test,logg_test,fe_h_test,mg_h_test,Age_test,Dnu_test)).T
		
		model = tc.CannonModel(
			labels_tr, fluxes_tr, ivars_tr, dispersion=wl, # needed to set dispersion explicitly
			vectorizer=tc.vectorizer.PolynomialVectorizer(label_names, 2)) # 1 or 2
		#print(model.vectorizer.human_readable_label_vector)

		# training step
		theta, s2, metadata = model.train(threads=1)
		s2_arr.append(s2)
		theta_arr.append(theta)
		
		def _test_step(fluxes_test, ivars_test):
			
			# test step
			test_labels, cov_val, metadata_val = model.test(fluxes_test, ivars_test)
			#print("test, cov, metadata: ", test_labels, cov_val, metadata_val)
		
			# use cov to propagate per-star, per-visit uncertainty 
			matrix = np.zeros((len(cov_val),len(label_names))) # Pre-allocate matrix
			for i in range(0,len(cov_val)):
				matrix[i,:] = np.sqrt(np.diag(cov_val[i]))

			# get Cannon-derived model spectra
			model_spectrum = model(test_labels) 
			# chisq of model spectral fit
			spec_fit_chisq = np.sum(((model_spectrum-fluxes_test)**2)/(ivars_test**-1 + s2))

			return test_labels, spec_fit_chisq
	
		# test one star at a time
		for test_index in test:
			test_label, chisq = _test_step(fluxes[test_index], ivars[test_index])
			test_labels_arr.append(test_label)
			spec_fit_chisq_arr.append(chisq)
			sdss_ids.append(df.iloc[test_index]['sdss_id'])
			folds.append(fold)
			#print("test index: ", test_index)
			#print("sdss id: ", df.iloc[test_index]['sdss_id'])
			#print("label :", test_label)
			#print("chisq: ", chisq)
			#quit()

	return sdss_ids, test_labels_arr, model, s2_arr, spec_fit_chisq_arr, folds

"""
### run four fold CV
sdss_ids, test_labels_arr, model, s2_arr, spec_fit_chisq_arr, folds = cv(df, wl, fluxes, ivars, label_names)

preds = pd.DataFrame()
preds['sdss_id'] = sdss_ids
preds['Teff_pred'] = np.array(test_labels_arr)[:,0][:,0]
preds['logg_pred'] = np.array(test_labels_arr)[:,0][:,1]
preds['fe_h_pred'] = np.array(test_labels_arr)[:,0][:,2]
preds['mg_h_pred'] = np.array(test_labels_arr)[:,0][:,3]
preds['Age_pred'] = np.array(test_labels_arr)[:,0][:,4]
preds['Dnu_pred'] = np.array(test_labels_arr)[:,0][:,5]
preds['chisq'] = spec_fit_chisq_arr
preds['fold'] = folds
#plt.hist(preds['chisq'])
#plt.show()

preds = pd.merge(df, preds, on='sdss_id')
#preds = preds[['sdss_id', 'Teff']]
#preds['Teff_aspcap'] = df['Teff']
#preds['logg_aspcap'] = df['logg']
#preds['fe_h_aspcap'] = df['fe_h']
#preds['mg_h_aspcap'] = df['mg_h']
#preds['Age_apokasc'] = df['Age']
#preds['Dnu_apokasc'] = df['Dnu']
#print(preds)
preds.to_csv(path+'data/4_fold_cv_logg.csv', index=False) # 4_fold_cv.csv, 4_fold_cv_ruwe.csv, 4_fold_cv_teff.csv
quit()
"""

preds = pd.read_csv(path+'data/4_fold_cv.csv') # 4_fold_cv.csv, 4_fold_cv_ruwe.csv, 4_fold_cv_teff.csv, 4_fold_cv_teff_ruwe.csv
print(preds)
print(np.median(preds.chisq))
print(np.max(preds.chisq))
reduced_chisq_modifier = len(wl) - len(label_names)
print("DoF: ", reduced_chisq_modifier)
print(preds.loc[preds['chisq']/reduced_chisq_modifier>8])
plt.hist(preds['chisq']/reduced_chisq_modifier, bins=10)
plt.xlabel(r'reduced $\chi^2$')
plt.tight_layout()
plt.show()

### rmse per fold
def compute_rmse(preds, cannon, aspcap):
	preds_one = preds.loc[preds['fold']==0]
	preds_two = preds.loc[preds['fold']==1]
	preds_three = preds.loc[preds['fold']==2]
	preds_four = preds.loc[preds['fold']==3]

	diff_one = preds_one[cannon] - preds_one[aspcap]
	rmse_one = np.sqrt(np.sum(diff_one**2)/len(diff_one))
	diff_two = preds_two[cannon] - preds_two[aspcap]
	rmse_two = np.sqrt(np.sum(diff_two**2)/len(diff_two))
	diff_three = preds_three[cannon] - preds_three[aspcap]
	rmse_three = np.sqrt(np.sum(diff_three**2)/len(diff_three))
	diff_four = preds_four[cannon] - preds_four[aspcap]
	rmse_four = np.sqrt(np.sum(diff_four**2)/len(diff_four))
	#print("RMSE: ", rmse_one, rmse_two, rmse_three, rmse_four)
	print("RMSE: ", np.mean([rmse_one, rmse_two, rmse_three, rmse_four]), np.std([rmse_one, rmse_two, rmse_three, rmse_four]))

	mean_one = np.mean(preds_one[cannon])
	mean_two = np.mean(preds_two[cannon])
	mean_three = np.mean(preds_three[cannon])
	mean_four = np.mean(preds_four[cannon])
	#print("mean: ", mean_one, mean_two, mean_three, mean_four)
	print("mean: ", np.mean([mean_one, mean_two, mean_three, mean_four]), np.std([mean_one, mean_two, mean_three, mean_four]))

	std_one = np.std(preds_one[cannon])
	std_two = np.std(preds_two[cannon])
	std_three = np.std(preds_three[cannon])
	std_four = np.std(preds_four[cannon])
	#print("std: ", std_one, std_two, std_three, std_four)
	print("std: ", np.mean([std_one, std_two, std_three, std_four]), np.std([std_one, std_two, std_three, std_four]))

	return 

print("TEFF")
compute_rmse(preds, 'Teff_pred', 'Teff')
print("LOGG")
compute_rmse(preds, 'logg_pred', 'logg')
print("FE/H")
compute_rmse(preds, 'fe_h_pred', 'feh')
print("MG/H")
compute_rmse(preds, 'mg_h_pred', 'mg_h')
print("AGE")
compute_rmse(preds, 'Age_pred', 'Age')
print("DNU")
compute_rmse(preds, 'Dnu_pred', 'Dnu')

modifier = ''
### paper figures
im = plt.scatter(preds['Teff'], preds['Teff_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier)
min_teff = np.min(pd.concat([preds['Teff'], preds['Teff_pred']]))
max_teff = np.max(pd.concat([preds['Teff'], preds['Teff_pred']]))
plt.plot([min_teff,max_teff], [min_teff,max_teff])
plt.ylabel(r"$T_{\rm eff}$ [K], Cannon, CV")
plt.xlabel(r"$T_{\rm eff}$ [K], ASPCAP")
#plt.xlim([min_teff, max_teff])
#plt.ylim([min_teff, max_teff])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model reduced $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_teff'+modifier+'.png')
plt.show()

im = plt.scatter(preds['logg'], preds['logg_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier)
min_logg = np.min(pd.concat([preds['logg'], preds['logg_pred']]))
max_logg = np.max(pd.concat([preds['logg'], preds['logg_pred']]))
plt.plot([min_logg,max_logg], [min_logg,max_logg])
plt.ylabel(r"logg, Cannon, CV")
plt.xlabel(r"logg, ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model reduced $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_logg'+ modifier +'.png')
plt.show()

im = plt.scatter(preds['feh'], preds['fe_h_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier)
min_feh = np.min(pd.concat([preds['feh'], preds['fe_h_pred']]))
max_feh = np.max(pd.concat([preds['feh'], preds['fe_h_pred']]))
plt.plot([min_feh,max_feh], [min_feh,max_feh])
plt.ylabel(r"[Fe/H], Cannon, CV")
plt.xlabel(r"[Fe/H], ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model reduced $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_fe_h'+modifier+'.png')
plt.show()

im = plt.scatter(preds['mg_h'], preds['mg_h_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier)
min_mg_h = np.min(pd.concat([preds['mg_h'], preds['mg_h_pred']]))
max_mg_h = np.max(pd.concat([preds['mg_h'], preds['mg_h_pred']]))
plt.plot([min_mg_h,max_mg_h], [min_mg_h,max_mg_h])
plt.ylabel(r"[Mg/H], Cannon, CV")
plt.xlabel(r"[Mg/H], ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model reduced $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_mg_h'+modifier+'.png')
plt.show()

im = plt.scatter(preds['Age'], preds['Age_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier)
min_Age = np.min(pd.concat([preds['Age'], preds['Age_pred']]))
max_Age = np.max(pd.concat([preds['Age'], preds['Age_pred']]))
plt.plot([min_Age,max_Age], [min_Age,max_Age])
plt.ylabel(r"Age [Gyr], Cannon, CV")
plt.xlabel(r"Age [Gyr], ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model reduced $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_Age'+modifier+'.png')
plt.show()

im = plt.scatter(preds['Dnu'], preds['Dnu_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier)
min_Dnu = np.min(pd.concat([preds['Dnu'], preds['Dnu_pred']]))
max_Dnu = np.max(pd.concat([preds['Dnu'], preds['Dnu_pred']]))
plt.plot([min_Dnu,max_Dnu], [min_Dnu,max_Dnu])
plt.ylabel(r'$\Delta \nu [\mu Hz]$, Cannon, CV')
plt.xlabel(r'$\Delta \nu [\mu Hz]$, ASPCAP')
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model reduced $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_Dnu'+modifier+'.png')
plt.show()
quit()
"""
### plot label space distributions
plt.hist(preds['Teff'],color='k')
plt.xlabel(r"ASPCAP $T_{\rm eff}$ [K]")
plt.savefig(path+'plots/teff.png')
plt.show()

plt.hist(preds['logg'],color='k')
plt.xlabel(r"ASPCAP logg")
plt.savefig(path+'plots/logg.png')
plt.show()

plt.hist(preds['fe_h'],color='k')
plt.xlabel(r"ASPCAP [Fe/H]")
plt.savefig(path+'plots/fe_h.png')
plt.show()

plt.hist(preds['mg_h'],color='k')
plt.xlabel(r"ASPCAP [Mg/H]")
plt.savefig(path+'plots/mg_h.png')
plt.show()

plt.hist(preds['Age'],color='k')
plt.xlabel(r"S17 Age [Gyr]")
plt.savefig(path+'plots/age.png')
plt.show()

plt.hist(preds['Dnu'],color='k')
plt.xlabel(r'S17 $\Delta \nu [\mu Hz]$')
plt.savefig(path+'plots/Dnu.png')
plt.show()
"""

### keep only young, alpha-rich stars
preds_young_alpha_rich = preds.loc[(preds['Age'] <= 6) & (preds['mg_h'] >= 0.2)]

im = plt.scatter(preds_young_alpha_rich['Teff_pred'], preds_young_alpha_rich['Teff'], alpha=0.7, c=preds_young_alpha_rich['chisq'])
min_teff = np.min(pd.concat([preds_young_alpha_rich['Teff'], preds_young_alpha_rich['Teff_pred']]))
max_teff = np.max(pd.concat([preds_young_alpha_rich['Teff'], preds_young_alpha_rich['Teff_pred']]))
plt.plot([min_teff,max_teff], [min_teff,max_teff])
plt.xlabel(r"$T_{\rm eff}$ [K], Cannon")
plt.ylabel(r"$T_{\rm eff}$ [K], ASPCAP")
#plt.xlim([min_teff, max_teff])
#plt.ylim([min_teff, max_teff])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_teff_young_alpha_rich.png')
plt.show()

im = plt.scatter(preds_young_alpha_rich['logg_pred'], preds_young_alpha_rich['logg'], alpha=0.7, c=preds_young_alpha_rich['chisq'])
min_logg = np.min(pd.concat([preds_young_alpha_rich['logg'], preds_young_alpha_rich['logg_pred']]))
max_logg = np.max(pd.concat([preds_young_alpha_rich['logg'], preds_young_alpha_rich['logg_pred']]))
plt.plot([min_logg,max_logg], [min_logg,max_logg])
plt.xlabel(r"logg, Cannon")
plt.ylabel(r"logg, ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_logg_young_alpha_rich.png')
plt.show()

im = plt.scatter(preds_young_alpha_rich['fe_h_pred'], preds_young_alpha_rich['feh'], alpha=0.7, c=preds_young_alpha_rich['chisq'])
min_feh = np.min(pd.concat([preds_young_alpha_rich['feh'], preds_young_alpha_rich['fe_h_pred']]))
max_feh = np.max(pd.concat([preds_young_alpha_rich['feh'], preds_young_alpha_rich['fe_h_pred']]))
plt.plot([min_feh,max_feh], [min_feh,max_feh])
plt.xlabel(r"[Fe/H], Cannon")
plt.ylabel(r"[Fe/H], ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_fe_h_young_alpha_rich.png')
plt.show()

im = plt.scatter(preds_young_alpha_rich['mg_h_pred'], preds_young_alpha_rich['mg_h'], alpha=0.7, c=preds_young_alpha_rich['chisq'])
min_mg_h = np.min(pd.concat([preds_young_alpha_rich['mg_h'], preds_young_alpha_rich['mg_h_pred']]))
max_mg_h = np.max(pd.concat([preds_young_alpha_rich['mg_h'], preds_young_alpha_rich['mg_h_pred']]))
plt.plot([min_mg_h,max_mg_h], [min_mg_h,max_mg_h])
plt.xlabel(r"[Mg/H], Cannon")
plt.ylabel(r"[Mg/H], ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_mg_h_young_alpha_rich.png')
plt.show()

im = plt.scatter(preds_young_alpha_rich['Age_pred'], preds_young_alpha_rich['Age'], alpha=0.7, c=preds_young_alpha_rich['chisq'])
min_Age = np.min(pd.concat([preds_young_alpha_rich['Age'], preds_young_alpha_rich['Age_pred']]))
max_Age = np.max(pd.concat([preds_young_alpha_rich['Age'], preds_young_alpha_rich['Age_pred']]))
plt.plot([min_Age,max_Age], [min_Age,max_Age])
plt.xlabel(r"Age [Gyr], Cannon")
plt.ylabel(r"Age [Gyr], ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_Age_young_alpha_rich.png')
plt.show()

im = plt.scatter(preds_young_alpha_rich['Dnu_pred'], preds_young_alpha_rich['Dnu'], alpha=0.7, c=preds_young_alpha_rich['chisq'])
min_Dnu = np.min(pd.concat([preds_young_alpha_rich['Dnu'], preds_young_alpha_rich['Dnu_pred']]))
max_Dnu = np.max(pd.concat([preds_young_alpha_rich['Dnu'], preds_young_alpha_rich['Dnu_pred']]))
plt.plot([min_Dnu,max_Dnu], [min_Dnu,max_Dnu])
plt.xlabel(r'$\Delta \nu [\mu Hz]$, Cannon')
plt.ylabel(r'$\Delta \nu [\mu Hz]$, ASPCAP')
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/4_fold_cv_Dnu_young_alpha_rich.png')
plt.show()

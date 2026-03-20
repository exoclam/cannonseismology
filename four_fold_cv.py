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
pylab_params = {'legend.fontsize': 'x-large',
		 'axes.labelsize': 'xx-large',
		 'axes.titlesize':'xx-large',
		 'xtick.labelsize':'x-large',
		 'ytick.labelsize':'x-large'}
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

rgb_df = df.loc[(df['logg']<3.8) & (df['Teff']<5250)]
print(rgb_df)

# get rid of SB2s (and one SB3), vetted by manual inspection after an initial round of CV showed some anomalously high chisq
bad_sdss_ids = [67660379, 67478208, 67161877, 67225129, 67114365]
bad_df = df.loc[df['sdss_id'].isin(bad_sdss_ids)]
df = df.loc[~df['sdss_id'].isin(bad_sdss_ids)]
df = df.loc[df.Age >= 0] # get rid of -99 Gyr ages (error flag from APOKASC)
df = df.drop_duplicates(subset=['sdss_id'])
print(df)

### SOFT RUWE CUT, turn off by default!!
#df = df.loc[df['ruwe']<=2]
### TEFF CUT, turn off by default!!
#df = df.loc[(df['Teff']>=5200) & (df['Teff']<=6400)]
### LOGG CUT
#df = df.loc[df['logg']>=4]
### RGB CUT
rgb_df = df.loc[(df['logg']<3.8) & (df['Teff']<5250)]
df = df.loc[~((df['logg']<3.8) & (df['Teff']<5250))]
print(df)
print(rgb_df)
"""
plt.scatter(rgb_df['Teff'], rgb_df['logg'], label='RGB')
plt.scatter(df['Teff'], df['logg'], label='no RGB')
plt.xlabel('Teff')
plt.ylabel('logg')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.legend()
plt.show()
"""

#plt.hist
#plt.hist(rgb_df['Age'], density=True, label='RGB')
#plt.hist(df['Age'], density=True, label='no RGB')
#plt.xlabel('age [Gyr]')
#plt.legend()
#plt.tight_layout()
#plt.show()

df = df.reset_index(drop=True)
#print(df[['sdss_id', 'KIC', 'Teff', 'logg', 'feh', 'mg_h', 'Age', 'Dnu']])

worst = df.loc[(df['KIC']==7976303) | (df['KIC']==10454113)] # this one fails continuum normalization so hard, it is an outlier among outliers
df = df.loc[df['KIC']!=7976303] # this one fails continuum normalization so hard, it is an outlier among outliers
df = df.loc[df['KIC']!=10454113]

label_names = ['Teff', 'logg', 'feh', 'mg_h', 'Age', 'Dnu'] # 'numax'
training_names = df['sdss_id'].astype(str)
directory = path+'data/spectra/' # e.g., mwmStar-0.6.0-114879184.fits
spectra_paths = get_files_in_order(directory, training_names)
fluxes = np.genfromtxt(path+'data/fluxes_norm_no_rgb.txt', delimiter=',')#[:,:-1]
ivars = np.genfromtxt(path+'data/ivars_norm_no_rgb.txt', delimiter=',')#[:,:-1]

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
np.savetxt(path+'data/fluxes_norm_no_rgb.txt', fluxes, delimiter=',', newline='\n')
np.savetxt(path+'data/ivars_norm_no_rgb.txt', ivars, delimiter=',', newline='\n')
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
print(preds)
preds.to_csv(path+'data/4_fold_cv_no_rgb.csv', index=False) # 4_fold_cv.csv, 4_fold_cv_ruwe.csv, 4_fold_cv_teff.csv
quit()
"""

preds = pd.read_csv(path+'data/4_fold_cv_no_rgb.csv')
#plt.hist(preds['Age'], density=True, label='no RGB')
#plt.xlabel('age [Gyr]')
#plt.legend()
#plt.tight_layout()
#plt.show()
#print(np.median(preds.chisq))
#print(np.max(preds.chisq))
reduced_chisq_modifier = len(wl) - len(label_names)
print("DoF: ", reduced_chisq_modifier)
print(preds.loc[preds['chisq']/reduced_chisq_modifier>4])
#preds = preds.loc[preds['chisq']/reduced_chisq_modifier < 8]
# plt.hist(preds['chisq']/reduced_chisq_modifier, bins=10)
# plt.xlabel(r'reduced $\chi^2$')
# plt.tight_layout()
# plt.show()

# introduce LEGACY sample to put everything relevant in the same plot
legacy = pd.read_csv(path+'data/silva-aguirre-legacy.txt',sep='\s+')
bedell = Table.read('/Users/chrislam/Desktop/psps/data/kepler_dr3_good.fits')
bedell_df = bedell.to_pandas()
legacy_bedell = pd.merge(legacy, bedell_df, left_on='KIC', right_on='kepid', how='left')
fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)  
lite_source_ids = hdul_lite[1].data.gaia_dr3_source_id

# use DR3 source_id to get sdss_id from mwmLite
legacy_bedell_apogee = legacy_bedell.loc[legacy_bedell['source_id'].isin(lite_source_ids)]

#"""
### Kiel diagram: Teff vs logg
#plt.scatter(nataf_aspcap_cull['aspcap_teff'], nataf_aspcap_cull['aspcap_logg'], s=5, alpha=0.5, label='Nataf+24', color='pink')
#plt.scatter(berger_aspcap_cull['aspcap_teff'], berger_aspcap_cull['aspcap_logg'], s=5, alpha=0.5, label='Berger+20', color='pink', marker='s')
#plt.scatter(bouma_aspcap_cull['aspcap_teff'], bouma_aspcap_cull['aspcap_logg'], s=5, alpha=0.3, label='Bouma+24', color='purple')
#plt.scatter(lu_aspcap_cull['aspcap_teff'], lu_aspcap_cull['aspcap_logg'], s=5, alpha=0.3, label='Lu+24', color='purple', marker='s')
#preds_aspcap = pd.merge(preds, aspcap_df, on='source_id', how='left')
#im = plt.scatter(preds['Teff'], preds['logg'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier)
plt.scatter(preds['Teff'], preds['logg'], alpha=0.7, c='black', label='final training sample')
plt.scatter(bad_df['Teff'], bad_df['logg'], alpha=0.7, c='red', label='SB2s and SB3')
plt.scatter(worst['Teff'], worst['logg'], alpha=0.7, marker='d', c='red', label='bad normalization')
#plt.scatter(rgb_df['Teff'], rgb_df['logg'], alpha=0.7, c='pink', label='RGB base')
plt.scatter(legacy_bedell_apogee['teff'], legacy_bedell_apogee['logg'], alpha=0.7, label='Silva Aguirre+17', color='purple')
plt.xlabel(r"$T_{\rm eff}$ [K], ASPCAP")
plt.ylabel('logg, ASPCAP')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
#plt.legend(fontsize='medium')
plt.legend(loc='upper left', bbox_to_anchor=(0.02, 1.0), fontsize='medium')
#cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model reduced $\chi^2$ fit')
plt.tight_layout()
plt.savefig(path+'plots/kiel_our_sample_only.png')
plt.show()
#"""

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
print(preds.loc[preds['chisq']>30000][['KIC','Teff','logg','chisq']])

#blah = pd.read_csv(path+'data/aspcap.csv')
#blah = blah.loc[blah.aspcap_sdss_id==67137792]
#print(blah.aspcap_snr)
#quit()
#plt.hist(preds['chisq'])
#plt.show()
#quit()


# there's one more bad continuum normalization star: KIC 10454113
preds = preds.loc[preds['KIC']!=10454113]
modifier = '_no_rgb'
### paper figures
vmax = max(preds['chisq'])/reduced_chisq_modifier
print("vmax: ", vmax)
print("final: ", preds)
#print(preds.loc[preds['KIC']==10070754][['aspcap_snr']]) # snr=223

### keep only young, alpha-rich stars
preds['mg_fe'] = preds['mg_h'] - preds['feh']
print(preds['mg_fe'])
preds_young_alpha_rich = preds.loc[(preds['Age'] <= 6) & (preds['mg_fe'] >= 0.1)]
print(preds_young_alpha_rich)
print(max(preds['chisq']))

min_teff = np.min(pd.concat([preds['Teff'], preds['Teff_pred']]))
max_teff = np.max(pd.concat([preds['Teff'], preds['Teff_pred']]))
plt.plot([min_teff,max_teff], [min_teff,max_teff], color='k', zorder=1)
im = plt.scatter(preds['Teff'], preds['Teff_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
im_young_alpha_rich = plt.scatter(preds_young_alpha_rich['Teff'], preds_young_alpha_rich['Teff_pred'], facecolors='none', edgecolors='magenta', linewidths=2, zorder=2)
plt.ylabel(r"$T_{\rm eff}$ [K], Cannon, CV")
plt.xlabel(r"$T_{\rm eff}$ [K], ASPCAP")
#x_rms = int(np.std(preds['Teff_pred']-preds['Teff']))
x_rms = int(np.std(np.sqrt(0.5 * (preds['Teff_pred'] - preds['Teff'])**2)))
x_std = int(np.std(preds['Teff']))
plt.text(5150, 6400, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms}', fontsize=15)
plt.text(5150, 6300, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std}', fontsize=15)
x_rms_pink = int(np.std(np.sqrt(0.5 * (preds_young_alpha_rich['Teff_pred'] - preds_young_alpha_rich['Teff'])**2)))
x_std_pink = int(np.std(preds_young_alpha_rich['Teff']))
plt.text(6400, 5300, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms_pink}', fontsize=15, c='magenta', horizontalalignment='right')
plt.text(6400, 5200, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std_pink}', fontsize=15, c='magenta', horizontalalignment='right')
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_teff'+modifier+'.png')
plt.show()
#print("teff rms: ", np.std(preds['Teff_pred']-preds['Teff']), "teff APOKASC scatter: ", np.std(preds['Teff']))

min_logg = np.min(pd.concat([preds['logg'], preds['logg_pred']]))
max_logg = np.max(pd.concat([preds['logg'], preds['logg_pred']]))
plt.plot([min_logg,max_logg], [min_logg,max_logg], color='k', zorder=1)
im = plt.scatter(preds['logg'], preds['logg_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
im_young_alpha_rich = plt.scatter(preds_young_alpha_rich['logg'], preds_young_alpha_rich['logg_pred'], facecolors='none', edgecolors='magenta', linewidths=2, zorder=2)
plt.ylabel(r"logg, Cannon, CV")
plt.xlabel(r"logg, ASPCAP")
#x_rms = np.round(np.std(preds['logg_pred']-preds['logg']),2)
x_rms = np.round(np.std(np.sqrt(0.5*(preds['logg_pred']-preds['logg'])**2)),2)
x_std = np.round(np.std(preds['logg']),2)
plt.text(3.45, 4.4, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms}', fontsize=15)
plt.text(3.45, 4.3, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std}', fontsize=15)
x_rms_pink = np.round(np.std(np.sqrt(0.5 * (preds_young_alpha_rich['logg_pred'] - preds_young_alpha_rich['logg'])**2)),2)
x_std_pink = np.round(np.std(preds_young_alpha_rich['logg']),2)
plt.text(4.42, 3.6, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms_pink}', fontsize=15, c='magenta', horizontalalignment='right')
plt.text(4.42, 3.5, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std_pink}', fontsize=15, c='magenta', horizontalalignment='right')
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_logg'+ modifier +'.png')
plt.show()

min_feh = np.min(pd.concat([preds['feh'], preds['fe_h_pred']]))
max_feh = np.max(pd.concat([preds['feh'], preds['fe_h_pred']]))
plt.plot([min_feh,max_feh], [min_feh,max_feh], color='k', zorder=1)
im = plt.scatter(preds['feh'], preds['fe_h_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
im_young_alpha_rich = plt.scatter(preds_young_alpha_rich['feh'], preds_young_alpha_rich['fe_h_pred'], facecolors='none', edgecolors='magenta', linewidths=2, zorder=2)
plt.ylabel(r"[Fe/H], Cannon, CV")
plt.xlabel(r"[Fe/H], ASPCAP")
#x_rms = np.round(np.std(preds['fe_h_pred']-preds['feh']),2)
x_rms = np.round(np.std(np.sqrt(0.5*(preds['fe_h_pred']-preds['feh'])**2)),2)
x_std = np.round(np.std(preds['feh']),2)
plt.text(-0.85, 0.5, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms}', fontsize=15)
plt.text(-0.85, 0.4, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std}', fontsize=15)
#x_rms_pink = np.round(np.std(preds_young_alpha_rich['fe_h_pred']-preds_young_alpha_rich['feh']),2)
x_rms_pink = np.round(np.std(np.sqrt(0.5 * (preds_young_alpha_rich['fe_h_pred'] - preds_young_alpha_rich['feh'])**2)),2)
x_std_pink = np.round(np.std(preds_young_alpha_rich['feh']),2)
plt.text(0.55, -0.7, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms_pink}', fontsize=15, c='magenta', horizontalalignment='right')
plt.text(0.55, -0.8, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std_pink}', fontsize=15, c='magenta', horizontalalignment='right')
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_fe_h'+modifier+'.png')
plt.show()

min_mg_h = np.min(pd.concat([preds['mg_h'], preds['mg_h_pred']]))
max_mg_h = np.max(pd.concat([preds['mg_h'], preds['mg_h_pred']]))
plt.plot([min_mg_h,max_mg_h], [min_mg_h,max_mg_h], color='k', zorder=1)
im = plt.scatter(preds['mg_h'], preds['mg_h_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
im_young_alpha_rich = plt.scatter(preds_young_alpha_rich['mg_h'], preds_young_alpha_rich['mg_h_pred'], facecolors='none', edgecolors='magenta', linewidths=2, zorder=2)
plt.ylabel(r"[Mg/H], Cannon, CV")
plt.xlabel(r"[Mg/H], ASPCAP")
#x_rms = np.round(np.std(preds['mg_h_pred']-preds['mg_h']),2)
x_rms = np.round(np.std(np.sqrt(0.5*(preds['mg_h_pred']-preds['mg_h'])**2)),2)
x_std = np.round(np.std(preds['mg_h']),2)
plt.text(-0.75, 0.45, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms}', fontsize=15)
plt.text(-0.75, 0.35, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std}', fontsize=15)
#x_rms_pink = np.round(np.std(preds_young_alpha_rich['mg_h_pred']-preds_young_alpha_rich['mg_h']),2)
x_rms_pink = np.round(np.std(np.sqrt(0.5 * (preds_young_alpha_rich['mg_h_pred'] - preds_young_alpha_rich['mg_h'])**2)),2)
x_std_pink = np.round(np.std(preds_young_alpha_rich['mg_h']),2)
plt.text(0.5, -0.6, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms_pink}', fontsize=15, c='magenta', horizontalalignment='right')
plt.text(0.5, -0.7, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std_pink}', fontsize=15, c='magenta', horizontalalignment='right')
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_mg_h'+modifier+'.png')
plt.show()

min_Age = np.min(pd.concat([preds['Age'], preds['Age_pred']]))
max_Age = np.max(pd.concat([preds['Age'], preds['Age_pred']]))
plt.plot([min_Age,max_Age], [min_Age,max_Age], color='k', zorder=1)
im = plt.scatter(preds['Age'], preds['Age_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
im_young_alpha_rich = plt.scatter(preds_young_alpha_rich['Age'], preds_young_alpha_rich['Age_pred'], facecolors='none', edgecolors='magenta', linewidths=2, zorder=2)
plt.ylabel(r"Age [Gyr], Cannon, CV")
plt.xlabel(r"Age [Gyr], ASPCAP")
#x_rms = np.round(np.std(preds['Age_pred']-preds['Age']),2)
x_rms = np.round(np.std(np.sqrt(0.5*(preds['Age_pred']-preds['Age'])**2)),2)
x_std = np.round(np.std(preds['Age']),2)
plt.text(0, 13, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms}', fontsize=15)
plt.text(0, 12, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std}', fontsize=15)
#x_rms_pink = np.round(np.std(preds_young_alpha_rich['Age_pred']-preds_young_alpha_rich['Age']),2)
x_rms_pink = np.round(np.std(np.sqrt(0.5 * (preds_young_alpha_rich['Age_pred'] - preds_young_alpha_rich['Age'])**2)),2)
x_std_pink = np.round(np.std(preds_young_alpha_rich['Age']),2)
plt.text(13.5, 1, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms_pink}', fontsize=15, horizontalalignment='right')
plt.text(13.5, 0, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std_pink}', fontsize=15, horizontalalignment='right')
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_Age'+modifier+'.png')
plt.show()

min_Dnu = np.min(pd.concat([preds['Dnu'], preds['Dnu_pred']]))
max_Dnu = np.max(pd.concat([preds['Dnu'], preds['Dnu_pred']]))
plt.plot([min_Dnu,max_Dnu], [min_Dnu,max_Dnu], color='k', zorder=1)
im = plt.scatter(preds['Dnu'], preds['Dnu_pred'], alpha=0.7, c=preds['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
im_young_alpha_rich = plt.scatter(preds_young_alpha_rich['Dnu'], preds_young_alpha_rich['Dnu_pred'], facecolors='none', edgecolors='magenta', linewidths=2, zorder=2)
plt.ylabel(r'$\Delta \nu [\mu Hz]$, Cannon, CV')
plt.xlabel(r'$\Delta \nu [\mu Hz]$, ASPCAP')
#x_rms = np.round(np.std(preds['Dnu_pred']-preds['Dnu']),2)
x_rms = np.round(np.std(np.sqrt(0.5*(preds['Dnu_pred']-preds['Dnu'])**2)),2)
x_std = np.round(np.std(preds['Dnu']),2)
plt.text(25, 150, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms}', fontsize=15)
plt.text(25, 140, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std}', fontsize=15)
#x_rms_pink = np.round(np.std(preds_young_alpha_rich['Dnu_pred']-preds_young_alpha_rich['Dnu']),2)
x_rms_pink = np.round(np.std(np.sqrt(0.5 * (preds_young_alpha_rich['Dnu_pred'] - preds_young_alpha_rich['Dnu'])**2)),2)
x_std_pink = np.round(np.std(preds_young_alpha_rich['Dnu']),2)
plt.text(155, 40, r'$\sigma_{X_{APOKASC}-X_{pred}}$ = ' + f'{x_rms_pink}', fontsize=15, horizontalalignment='right')
plt.text(155, 30, r'$\sigma_{X_{APOKASC}}$ = ' + f'{x_std_pink}', fontsize=15, horizontalalignment='right')
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_Dnu'+modifier+'.png')
plt.show()
quit()
"""
### plot label space distributions
plt.hist(preds['Teff'],color='k')
plt.xlabel(r"ASPCAP $T_{\rm eff}$ [K]")
plt.tight_layout()
plt.savefig(path+'plots/teff'+modifier+'.png')
plt.show()

plt.hist(preds['logg'],color='k')
plt.xlabel(r"ASPCAP logg")
plt.tight_layout()
plt.savefig(path+'plots/logg'+modifier+'.png')
plt.show()

plt.hist(preds['fe_h'],color='k')
plt.xlabel(r"ASPCAP [Fe/H]")
plt.tight_layout()
plt.savefig(path+'plots/fe_h'+modifier+'.png')
plt.show()

plt.hist(preds['mg_h'],color='k')
plt.xlabel(r"ASPCAP [Mg/H]")
plt.tight_layout()
plt.savefig(path+'plots/mg_h'+modifier+'.png')
plt.show()

plt.hist(preds['Age'],color='k')
plt.xlabel(r"S17 Age [Gyr]")
plt.tight_layout()
plt.savefig(path+'plots/age'+modifier+'.png')
plt.show()

plt.hist(preds['Dnu'],color='k')
plt.xlabel(r'S17 $\Delta \nu [\mu Hz]$')
plt.tight_layout()
plt.savefig(path+'plots/Dnu'+modifier+'.png')
plt.show()
"""

### keep only young, alpha-rich stars
preds['mg_fe'] = preds['mg_h'] - preds['feh']
print(preds['mg_fe'])
preds_young_alpha_rich = preds.loc[(preds['Age'] <= 6) & (preds['mg_fe'] >= 0.1)]
print(preds_young_alpha_rich)

#preds_young_alpha_rich = preds.loc[(preds['Age'] <= 1) & (preds['mg_fe'] >= 0.2)]
#print(preds_young_alpha_rich)
#quit()

print(preds_young_alpha_rich.loc[preds_young_alpha_rich['mg_h']<-0.4][['KIC','feh','mg_h','mg_fe']])

plt.plot([min_teff,max_teff], [min_teff,max_teff], color='k', zorder=1)
im = plt.scatter(preds_young_alpha_rich['Teff'], preds_young_alpha_rich['Teff_pred'], alpha=0.7, c=preds_young_alpha_rich['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
#min_teff = np.min(pd.concat([preds_young_alpha_rich['Teff'], preds_young_alpha_rich['Teff_pred']]))
#max_teff = np.max(pd.concat([preds_young_alpha_rich['Teff'], preds_young_alpha_rich['Teff_pred']]))
plt.ylabel(r"$T_{\rm eff}$ [K], Cannon, CV")
plt.xlabel(r"$T_{\rm eff}$ [K], ASPCAP")
#plt.xlim([min_teff, max_teff])
#plt.ylim([min_teff, max_teff])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_teff_young_alpha_rich'+modifier+'.png')
plt.show()

plt.plot([min_logg,max_logg], [min_logg,max_logg], color='k', zorder=1)
im = plt.scatter(preds_young_alpha_rich['logg'], preds_young_alpha_rich['logg_pred'], alpha=0.7, c=preds_young_alpha_rich['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
#min_logg = np.min(pd.concat([preds_young_alpha_rich['logg'], preds_young_alpha_rich['logg_pred']]))
#max_logg = np.max(pd.concat([preds_young_alpha_rich['logg'], preds_young_alpha_rich['logg_pred']]))
plt.ylabel(r"logg, Cannon, CV")
plt.xlabel(r"logg, ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_logg_young_alpha_rich'+modifier+'.png')
plt.show()

plt.plot([min_feh,max_feh], [min_feh,max_feh], color='k', zorder=1)
im = plt.scatter(preds_young_alpha_rich['fe_h'], preds_young_alpha_rich['fe_h_pred'], alpha=0.7, c=preds_young_alpha_rich['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
#min_feh = np.min(pd.concat([preds_young_alpha_rich['feh'], preds_young_alpha_rich['fe_h_pred']]))
#max_feh = np.max(pd.concat([preds_young_alpha_rich['feh'], preds_young_alpha_rich['fe_h_pred']]))
plt.ylabel(r"[Fe/H], Cannon, CV")
plt.xlabel(r"[Fe/H], ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_fe_h_young_alpha_rich'+modifier+'.png')
plt.show()

plt.plot([min_mg_h,max_mg_h], [min_mg_h,max_mg_h], color='k', zorder=1)
im = plt.scatter(preds_young_alpha_rich['mg_h'], preds_young_alpha_rich['mg_h_pred'], alpha=0.7, c=preds_young_alpha_rich['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
#min_mg_h = np.min(pd.concat([preds_young_alpha_rich['mg_h'], preds_young_alpha_rich['mg_h_pred']]))
#max_mg_h = np.max(pd.concat([preds_young_alpha_rich['mg_h'], preds_young_alpha_rich['mg_h_pred']]))
plt.ylabel(r"[Mg/H], Cannon, CV")
plt.xlabel(r"[Mg/H], ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_mg_h_young_alpha_rich'+modifier+'.png')
plt.show()

plt.plot([min_Age,max_Age], [min_Age,max_Age], color='k', zorder=1)
im = plt.scatter(preds_young_alpha_rich['Age'], preds_young_alpha_rich['Age_pred'], alpha=0.7, c=preds_young_alpha_rich['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
#min_Age = np.min(pd.concat([preds_young_alpha_rich['Age'], preds_young_alpha_rich['Age_pred']]))
#max_Age = np.max(pd.concat([preds_young_alpha_rich['Age'], preds_young_alpha_rich['Age_pred']]))
plt.ylabel(r"Age [Gyr], Cannon, CV")
plt.xlabel(r"Age [Gyr], ASPCAP")
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_Age_young_alpha_rich'+modifier+'.png')
plt.show()

plt.plot([min_Dnu,max_Dnu], [min_Dnu,max_Dnu], color='k', zorder=1)
im = plt.scatter(preds_young_alpha_rich['Dnu'], preds_young_alpha_rich['Dnu_pred'], alpha=0.7, c=preds_young_alpha_rich['chisq']/reduced_chisq_modifier, vmax=max(preds['chisq'])/reduced_chisq_modifier, zorder=2)
#min_Dnu = np.min(pd.concat([preds_young_alpha_rich['Dnu'], preds_young_alpha_rich['Dnu_pred']]))
#max_Dnu = np.max(pd.concat([preds_young_alpha_rich['Dnu'], preds_young_alpha_rich['Dnu_pred']]))
plt.ylabel(r'$\Delta \nu [\mu Hz]$, Cannon, CV')
plt.xlabel(r'$\Delta \nu [\mu Hz]$, ASPCAP')
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon $\chi_{red}^2$')
plt.tight_layout()
plt.savefig(path+'plots/cv/4_fold_cv_Dnu_young_alpha_rich'+modifier+'.png')
plt.show()


### mono abundances
modifier = '_no_rgb'
def plot_mono(training_sub, lower, upper, xerr=0, yerr=0):
    """util function for plotting mono abundance age relation

    Args:
        training_sub (Pandas DF): training sample, selected for the mono-abundance
        lower (float): lower abundance
        upper (float): upper abundance
    """

    plt.axis('square')
    im = plt.errorbar(training_sub['Age'], training_sub['Age_pred'], xerr=xerr, yerr=yerr, c='k', linestyle='', fmt='o') # training_sub['chisq']
    plt.plot([min_Age,max_Age], [min_Age,max_Age])
    plt.xlabel('APOKASC age [Gyr]')
    plt.ylabel('Cannon age [Gyr]')
    plt.xlim([0, 14])
    plt.ylim([0, 14])
    #plt.legend(bbox_to_anchor=(1., 1.05))
    plt.text(0.5, 13., f'{np.round(lower,1)} <= [Mg/Fe] < {np.round(upper,1)}: {len(training_sub)} stars', fontsize=15)
    #cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
    plt.tight_layout()
    if lower<0:
        plt.savefig(path+f'plots/cv/cv_mg_fe_mono_abundance_negative_{10*np.round(np.abs(lower),1)}'+modifier+'.png', format='png', bbox_inches='tight')
    else:
        plt.savefig(path+f'plots/cv/cv_mg_fe_mono_abundance_{10*np.round(lower,1)}'+modifier+'.png', format='png', bbox_inches='tight')
    plt.show()

    return

lowers = np.linspace(-1, 3, 9) # mg/fe
for lower in lowers:
    #upper = lower+0.1 # fe/h
    upper = lower+0.5 # mg/fe

    #if lower == 0.4: # fe/h
    if lower == 3: # mg/fe
        break
    else:
        #training_sub = training.loc[(training['fe_h_test'] >= lower) & (training['fe_h_test'] < upper)]
        training_sub = preds.loc[(preds['mg_fe'] >= lower) & (preds['mg_fe'] < upper)]

    #training_sub = pd.merge(training_sub, serenelli, left_on='kepid', right_on='KIC', how='inner')
    #print(list(training_sub.columns))
    #quit()
    #training_sub['Age'] = training_sub['Age_x']
    xerr = [training_sub['E_Age'], np.abs(training_sub['e_Age'])]
    #print(len(training_sub))
    #plt.hist2d(training_sub['Age_test'], training_sub['Age_pred'])
    plot_mono(training_sub, lower, upper, xerr)
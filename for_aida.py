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


"""
df = pd.read_csv(path+'data/small.csv',index_col=False)
df = df[df.Teff.notnull()]
df = df[df.mg_h.notnull()]
df = df[df.Age.notnull()]
df = df.reset_index(drop=True)
#df = df.loc[df['sdss_id_sec'] == 114879184]
"""

df = pd.read_csv(path+'data/enriched_lite_visits_chisq_ruwe.csv', sep=',') # formerly enriched_lite_visits.csv
df['sdss_id'] = df['sdss_id'].astype(int)
#df = df.iloc[:100]
df = df[df.Teff.notnull()]
df = df[df.mg_h.notnull()]
df = df[df.Age.notnull()]
df = df[df.feh.notnull()]
df = df[df.logg.notnull()]
df = df[df.Dnu.notnull()]
df = df[df.numax.notnull()]

df = df.loc[df.Age >= 0] # get rid of -99 Gyr ages (error flag from APOKASC)
df = df.reset_index(drop=True)
print(df)

#print(np.min(df.Teff), np.max(df.Teff))
#print(np.min(df.logg), np.max(df.logg))
#print(np.min(df.feh), np.max(df.feh))
#print(np.min(df.mg_h), np.max(df.mg_h))
#print(np.min(df.Age), np.max(df.Age))
#print(np.min(df.Dnu), np.max(df.Dnu))
"""
color='black'
plt.hist(df.Teff, color=color)
plt.xlabel(r"$T_{\rm eff}$ [K]")
plt.savefig(path+'plots/teff.png')
plt.show()

plt.hist(df.logg, color=color)
plt.xlabel("log(g)")
plt.savefig(path+'plots/logg.png')
plt.show()

plt.hist(df.feh, color=color)
plt.xlabel("[Fe/H]")
plt.savefig(path+'plots/feh.png')
plt.show()

plt.hist(df.mg_h, color=color)
plt.xlabel("[Mg/H]")
plt.savefig(path+'plots/mgh.png')
plt.show()

plt.hist(df.Age, color=color)
plt.xlabel("Age [Gyr]")
plt.savefig(path+'plots/age.png')
plt.show()

plt.hist(df.Dnu, color=color)
plt.xlabel(r'$\Delta \nu [\mu Hz]$')
plt.savefig(path+'plots/Dnu.png')
plt.show()
quit()
"""

training_names = df['sdss_id'].astype(str)
directory = path+'data/spectra/' # e.g., mwmStar-0.6.0-114879184.fits
spectra_paths = get_files_in_order(directory, training_names)
print(len(spectra_paths))

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
label_names = ['Teff', 'logg', 'feh', 'mg_h', 'Age', 'Dnu'] # 'numax'
#preds = loocv.loocv(df, wl, flux_tr, ivar_tr, label_names)

"""
# train model on everyone
labels = np.vstack((np.array(df.Teff),np.array(df.logg),np.array(df.feh),np.array(df.mg_h),np.array(df.Age),np.array(df.Dnu))).T
model = tc.CannonModel(
	labels, flux_tr, ivar_tr, dispersion=wl, # needed to set dispersion explicitly
	vectorizer=tc.vectorizer.PolynomialVectorizer(label_names, 2)) 
#print(model.vectorizer.human_readable_label_vector)

# training step
theta, s2, metadata = model.train(threads=1)
model.write(path+"apogee-serenelli-lite-ruwe.model") # write out model
quit()
"""

#"""
# LOOCV
test_labels_arr, true_labels_arr, model, s2_arr, spec_fit_chisq_arr = loocv.loocv(df, wl, flux_tr, ivar_tr, label_names)
s2_arr = np.array(s2_arr)
print("s2: ", s2_arr, s2_arr.shape)
print(np.nanmean(s2_arr))
print(np.nanstd(s2_arr))
spec_fit_chisq_arr = np.array(spec_fit_chisq_arr)
print("spec_fit_chisq_arr: ", spec_fit_chisq_arr, spec_fit_chisq_arr.shape)
#"""

# model.write(path+"apogee-serenelli-lite.model") # write out model
# new_model = tc.CannonModel.read("apogee-dr14-giants.model") # read in model

preds = pd.DataFrame()
preds['sdss_id'] = df['sdss_id']
#preds['s2'] = s2_arr
#preds['sdss_id'] = df['sdss_id'][:temp_length]
#print(np.array(s2_arr))

preds['Teff_pred'] = np.array(test_labels_arr)[:,0][:,0]
preds['logg_pred'] = np.array(test_labels_arr)[:,0][:,1]
preds['fe_h_pred'] = np.array(test_labels_arr)[:,0][:,2]
preds['mg_h_pred'] = np.array(test_labels_arr)[:,0][:,3]
preds['Age_pred'] = np.array(test_labels_arr)[:,0][:,4]
preds['Dnu_pred'] = np.array(test_labels_arr)[:,0][:,5]
#preds['numax_pred'] = np.array(test_labels_arr)[:,0][:,6]

preds['Teff_test'] = np.array(true_labels_arr)[:,0][:,0]
preds['logg_test'] = np.array(true_labels_arr)[:,0][:,1]
preds['fe_h_test'] = np.array(true_labels_arr)[:,0][:,2]
preds['mg_h_test'] = np.array(true_labels_arr)[:,0][:,3]
preds['Age_test'] = np.array(true_labels_arr)[:,0][:,4]
preds['Dnu_test'] = np.array(true_labels_arr)[:,0][:,5]
#preds['numax_test'] = np.array(true_labels_arr)[:,0][:,6]

preds['chisq'] = spec_fit_chisq_arr
print(preds)
preds.to_csv(path+'data/preds_dnu_full_ruwe.csv', index=False)

np.savetxt(path+'data/s2_lite.txt', s2_arr, fmt='%d', delimiter=',', newline='\n')
quit()

plt.scatter(preds['Teff_pred'], preds['Teff_test'])
plt.plot(preds['Teff_test'], preds['Teff_test'])
plt.xlabel(r"$T_{\rm eff}$ [K], pred")
plt.ylabel(r"$T_{\rm eff}$ [K], test")
plt.xlim([4750, 6750])
plt.ylim([4750, 6750])
#plt.legend()
plt.savefig(path+'plots/teff_check_dnu_full.png')
plt.show()

plt.scatter(preds['logg_pred'], preds['logg_test'])
plt.plot(preds['logg_test'], preds['logg_test'])
plt.xlabel(r"logg, pred")
plt.ylabel(r"logg, test")
plt.xlim([3.3, 4.4])
plt.ylim([3.3, 4.4])
plt.savefig(path+'plots/logg_check_dnu_full.png')
plt.show()

plt.scatter(preds['fe_h_pred'], preds['fe_h_test'])
plt.plot(preds['fe_h_test'], preds['fe_h_test'])
plt.xlabel(r"[Fe/H], pred")
plt.ylabel(r"[Fe/H], test")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
plt.savefig(path+'plots/feh_check_dnu_full.png')
plt.show()

plt.scatter(preds['mg_h_pred'], preds['mg_h_test'])
plt.plot(preds['mg_h_test'], preds['mg_h_test'])
plt.xlabel(r"[Mg/H], pred")
plt.ylabel(r"[Mg/H], test")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
plt.savefig(path+'plots/mg_h_check_dnu_full.png')
plt.show()

plt.scatter(preds['Age_pred'], preds['Age_test'])
plt.plot(preds['Age_test'], preds['Age_test'])
plt.xlabel(r"age [Gyr], pred")
plt.ylabel(r"age [Gyr], test")
plt.xlim([0, 14])
plt.ylim([0, 14])
plt.savefig(path+'plots/age_check_dnu_full.png')
plt.show()

plt.scatter(preds['Dnu_pred'], preds['Dnu_test'])
plt.plot(preds['Dnu_test'], preds['Dnu_test'])
plt.xlabel(r'$\Delta \nu [\mu Hz]$, pred')
plt.ylabel(r'$\Delta \nu [\mu Hz]$, test')
plt.xlim([0, 160])
plt.ylim([0, 160])
plt.savefig(path+'plots/Dnu_check_dnu_full.png')
plt.show()

"""
plt.scatter(preds['numax_pred'], preds['numax_test'])
plt.plot(preds['numax_test'], preds['numax_test'])
plt.xlabel(r'$\nu_{max} [\mu Hz]$, pred')
plt.ylabel(r'$\nu_{max} [\mu Hz]$, test')
plt.xlim([300, 3600])
plt.ylim([300, 3600])
plt.savefig(path+'plots/numax_check_dnu_numax_full.png')
plt.show()
"""

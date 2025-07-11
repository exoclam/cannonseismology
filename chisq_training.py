import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
from process_spectra_gaus import *

import thecannon as tc

from sdss_access import Access

path = '/Users/chrislam/Desktop/cannon-ages/' 
#path = '/home/c.lam/blue/cannon-ages/'

import matplotlib
import matplotlib.pylab as pylab
matplotlib.rcParams.update({'errorbar.capsize': 1})
pylab_params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(pylab_params)

"""
### is The Bad Star a fast rotator? a binary? why is its chisq higher than everyone else's? (see crossmatch.ipynb for the conclusion of this saga.)
vsini = 2* np.pi * 695700 * np.sin(1.19) /2005344
print(vsini)

fits_image_filename_aspcap = path+'data/astraAllStarASPCAP-0.6.0.fits'
hdul_aspcap = fits.open(fits_image_filename_aspcap)
vsini = hdul_aspcap[2].data[hdul_aspcap[2].data.sdss_id==80419035].v_sini[0] # 67660379 bad, 66668317 good
print(vsini)

fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)  
vsini = hdul_lite[1].data[hdul_lite[1].data.sdss_id==67660379].v_sini[0]
print(vsini)

p_da_ms = hdul_lite[1].data[hdul_lite[1].data.sdss_id==67660379].p_da_ms[0]
print(p_da_ms)
"""

#"""
df = pd.read_csv(path+'data/enriched_lite_visits_chisq_ruwe.csv', sep=',')
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

df = df.loc[df['sdss_id']!=67766853] # drop indices where snr<50 (see test below)

model = tc.CannonModel.read(path+"apogee-serenelli-lite.model")
s2 = np.loadtxt(path+'data/s2.txt',delimiter=',',dtype=float)
spec_fit_chisq_arr = []

def cov_matrix(cov):
	# model-assigned label scatter: this is for sigma_A, B, as well as errorbars at the individual star level 
	matrix = np.zeros((len(cov),6)) # Pre-allocate matrix
	for i in range(0,len(cov)):
		matrix[i,:] = np.sqrt(np.diag(cov[i]))

	df_sigma = pd.DataFrame(matrix)
	return df_sigma

sigma_stars = []
for sdss_id in tqdm(df['sdss_id']):
    sdss_id = 67660379 #this one is the bad chisq one from the training set
    #print(sdss_id)
    access = Access(release='ipl-3', verbose=False)
    access.remote()
    access.add('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)

    access.set_stream()
    access.commit()
    mwm_filenameStar = access.full('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)
    mwmStar = fits.open(mwm_filenameStar)

    wl_star = mwmStar[3].data['wavelength'][0]
    flux = mwmStar[3].data['flux'][0]
    ivar = mwmStar[3].data['ivar'][0]
    wl, norm_flux, ivar = process_spectra_gaus_chris_version(flux, ivar, wl_star, L=10)

    test_labels, cov_val, metadata_val = model.test(norm_flux, ivar)

    # get Cannon-derived model spectra
    model_spectrum = model(test_labels)
    plt.plot(wl, model_spectrum, label='Cannon')
    plt.plot(wl, norm_flux, label='data')
    plt.xlabel('wavelength')
    plt.ylabel('flux')
    plt.show()
    quit()

    # chisq of model spectral fit
    spec_fit_chisq = np.sum(((model_spectrum-norm_flux)**2)/(ivar**-1 + s2))
    spec_fit_chisq_arr.append(spec_fit_chisq)
	
    # get individual sigma per label per star
    sigma_star = np.array(cov_matrix(cov_val))
    sigma_stars.append(sigma_star)

df['chisq'] = np.array(spec_fit_chisq_arr)
df['sigma_star_Teff'] = np.array(sigma_stars)[:,0][:,0]
df['sigma_star_logg'] = np.array(sigma_stars)[:,0][:,1]
df['sigma_star_fe_h'] = np.array(sigma_stars)[:,0][:,2]
df['sigma_star_mg_h'] = np.array(sigma_stars)[:,0][:,3]
df['sigma_star_age'] = np.array(sigma_stars)[:,0][:,4]
df['sigma_star_Dnu'] = np.array(sigma_stars)[:,0][:,5]
df.to_csv(path+'data/enriched_lite_visits_chisq_ruwe.csv', index=False)
quit()
#"""

df = pd.read_csv(path+'data/enriched_lite_visits_chisq_ruwe.csv', sep=',')
preds = pd.read_csv(path+'data/preds_dnu_full_ruwe.csv', sep=',')
preds = pd.merge(preds, df, on='sdss_id', how='left')
#bad = preds.loc[preds['chisq']>50000]
#print(bad[['sdss_id', 'Teff_test', 'logg_test', 'fe_h_test', 'mg_h_test', 'Age_test', 'Dnu_test', 'KIC', 'source_id', 'snr']])

im = plt.scatter(preds['Teff_pred'], preds['Teff_test'], c=preds['chisq'])
plt.plot(preds['Teff_test'], preds['Teff_test'])
plt.xlabel(r"Cannon $T_{\rm eff}$ [K]")
plt.ylabel(r"ASPCAP $T_{\rm eff}$ [K]")
plt.xlim([4750, 6750])
plt.ylim([4750, 6750])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/teff_check_dnu_full_ruwe.png')
plt.show()

im = plt.scatter(preds['logg_pred'], preds['logg_test'], c=preds['chisq'])
plt.plot(preds['logg_test'], preds['logg_test'])
plt.xlabel(r"Cannon logg")
plt.ylabel(r"ASPCAP logg")
plt.xlim([3.3, 4.4])
plt.ylim([3.3, 4.4])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/logg_check_dnu_full_ruwe.png')
plt.show()

im = plt.scatter(preds['fe_h_pred'], preds['fe_h_test'], c=preds['chisq'])
plt.plot(preds['fe_h_test'], preds['fe_h_test'])
plt.xlabel(r"Cannon [Fe/H]")
plt.ylabel(r"ASPCAP [Fe/H]")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/feh_check_dnu_full_ruwe.png')
plt.show()

im = plt.scatter(preds['mg_h_pred'], preds['mg_h_test'], c=preds['chisq'])
plt.plot(preds['mg_h_test'], preds['mg_h_test'])
plt.xlabel(r"Cannon [Mg/H]")
plt.ylabel(r"ASPCAP [Mg/H]")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/mg_h_check_dnu_full_ruwe.png')
plt.show()

im = plt.scatter(preds['Age_pred'], preds['Age_test'], c=preds['chisq'])
plt.plot(preds['Age_test'], preds['Age_test'])
plt.xlabel(r"Cannon age [Gyr]")
plt.ylabel(r"S17 age [Gyr]")
plt.xlim([0, 14])
plt.ylim([0, 14])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/age_check_dnu_full_ruwe.png')
plt.show()

im = plt.scatter(preds['Dnu_pred'], preds['Dnu_test'], c=preds['chisq'])
plt.plot(preds['Dnu_test'], preds['Dnu_test'])
plt.xlabel(r'Cannon $\Delta \nu [\mu Hz]$')
plt.ylabel(r'S17 $\Delta \nu [\mu Hz]$')
plt.xlim([0, 160])
plt.ylim([0, 160])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/Dnu_check_dnu_full_ruwe.png')
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

preds_young_rich = preds.loc[(preds['Age_test'] <= 6) & (preds['mg_h_test'] >= 0.2)]
print(preds_young_rich)

im = plt.scatter(preds_young_rich['Teff_pred'], preds_young_rich['Teff_test'], c=preds_young_rich['chisq'])
plt.plot(preds['Teff_test'], preds['Teff_test'])
plt.xlabel(r"Cannon $T_{\rm eff}$ [K]")
plt.ylabel(r"ASPCAP $T_{\rm eff}$ [K]")
plt.xlim([4750, 6750])
plt.ylim([4750, 6750])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/teff_check_dnu_young_alpha_rich_ruwe.png')
plt.show()

im = plt.scatter(preds_young_rich['logg_pred'], preds_young_rich['logg_test'], c=preds_young_rich['chisq'])
plt.plot(preds['logg_test'], preds['logg_test'])
plt.xlabel(r"Cannon logg")
plt.ylabel(r"ASPCAP logg")
plt.xlim([3.3, 4.4])
plt.ylim([3.3, 4.4])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/logg_check_dnu_young_alpha_rich_ruwe.png')
plt.show()

im = plt.scatter(preds_young_rich['fe_h_pred'], preds_young_rich['fe_h_test'], c=preds_young_rich['chisq'])
plt.plot(preds['fe_h_test'], preds['fe_h_test'])
plt.xlabel(r"Cannon [Fe/H]")
plt.ylabel(r"ASPCAP [Fe/H]")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/feh_check_dnu_young_alpha_rich_ruwe.png')
plt.show()

im = plt.scatter(preds_young_rich['mg_h_pred'], preds_young_rich['mg_h_test'], c=preds_young_rich['chisq'])
plt.plot(preds['mg_h_test'], preds['mg_h_test'])
plt.xlabel(r"Cannon [Mg/H]")
plt.ylabel(r"ASPCAP [Mg/H]")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/mg_h_check_dnu_young_alpha_rich_ruwe.png')
plt.show()

im = plt.scatter(preds_young_rich['Age_pred'], preds_young_rich['Age_test'], c=preds_young_rich['chisq'])
plt.plot(preds['Age_test'], preds['Age_test'])
plt.xlabel(r"Cannon age [Gyr]")
plt.ylabel(r"S17 age [Gyr]")
plt.xlim([0, 14])
plt.ylim([0, 14])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/age_check_dnu_young_alpha_rich_ruwe.png')
plt.show()

im = plt.scatter(preds_young_rich['Dnu_pred'], preds_young_rich['Dnu_test'], c=preds_young_rich['chisq'])
plt.plot(preds['Dnu_test'], preds['Dnu_test'])
plt.xlabel(r'Cannon $\Delta \nu [\mu Hz]$')
plt.ylabel(r'S17 $\Delta \nu [\mu Hz]$')
plt.xlim([0, 160])
plt.ylim([0, 160])
cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
plt.savefig(path+'plots/Dnu_check_dnu_young_alpha_rich_ruwe.png')
plt.show()
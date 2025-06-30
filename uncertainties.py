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

df = pd.read_csv(path+'data/enriched_lite_visits.csv', sep=',')
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
print(df)
print(list(df.columns))

fits_image_filename = path+'data/mwmAllStar-0.6.0.fits'
hdul = fits.open(fits_image_filename)

fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)  
lite_source_ids = hdul_lite[1].data.gaia_dr3_source_id

preds = pd.read_csv(path+'data/preds_dnu_full.csv', sep=',')

aspcap_df = pd.read_csv(path+'data/aspcap.csv')
print(aspcap_df)
print(list(aspcap_df.columns))

# I painstakingly looked through 10% of this file and got 100 valid stars this way. Don't let that go to waste.
chisq_dones = pd.read_csv(path+'data/chisq_dones.csv')
aspcap_dones = aspcap_df.loc[aspcap_df['aspcap_sdss_id'].isin(chisq_dones['sdss_id'])]
print(aspcap_dones)

# get row index of last done
last_iloc = aspcap_df['aspcap_sdss_id'].eq(57958968).idxmax()
aspcap_df = aspcap_df.loc[last_iloc:]
aspcap_df = pd.concat([aspcap_dones, aspcap_df])
print(aspcap_df) 

inference_kic = pd.read_csv(path+'data/inferences_kic.csv')

"""
### I can further narrow down aspcap_df by grabbing number of visits via a one-time astraAllStarASPCAP crossmatch
fits_image_filename_aspcap = path+'data/astraAllStarASPCAP-0.6.0.fits'
hdul_aspcap = fits.open(fits_image_filename_aspcap)
aspcap_sdss_id = np.array(hdul_aspcap[2].data['sdss_id']).byteswap().newbyteorder()
aspcap_n_vists = np.array(hdul_aspcap[2].data['n_visits']).byteswap().newbyteorder()
aspcap_full_df = pd.DataFrame({'sdss_id': aspcap_sdss_id, 'n_visit': aspcap_n_vists})
aspcap_full_df['n_visit'] = aspcap_full_df['n_visit'].astype(int)
aspcap_full_df = aspcap_full_df.loc[aspcap_full_df['n_visit']==2]
aspcap_full_df.to_csv(path+'data/aspcap_df_visits.csv', index=False)
quit()
"""
aspcap_df_visits = pd.read_csv(path+'data/aspcap_df_visits.csv')
aspcap_df = pd.merge(aspcap_df, aspcap_df_visits, left_on='aspcap_sdss_id', right_on='sdss_id', how='left')
aspcap_df = aspcap_df.loc[aspcap_df['n_visit']==2]

"""
### look back in astraMWMLite to get number of APOGEE visits per star
n_visits = []
source_ids = []
source_id_dr2s = []
teffs = []
e_teffs = []
loggs = []
e_loggs = []
fe_hs = []
e_fe_hs = []
mg_hs = []
e_mg_hs = []
snrs = []
sdss_ids = []
for source_id in tqdm(df['source_id']):
    try:
        n_visit = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].n_apogee_visits[0]
        source_id_dr2 = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].gaia_dr2_source_id[0]
        #teff = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].teff[0]
        #e_teff = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].e_teff[0]
        #logg = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].logg[0]
        #e_logg = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].e_logg[0]
        #fe_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].fe_h[0]
        #e_fe_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].e_fe_h[0]
        #mg_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].mg_h[0]
        #e_mg_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].e_mg_h[0]
        snr = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].snr[0]
        sdss_id = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].sdss_id[0]

    except Exception as e:
        n_visit = np.nan
        source_id_dr2 = np.nan
        #teff = np.nan
        #e_teff = np.nan
        #logg = np.nan
        #e_logg = np.nan
        #fe_h = np.nan
        #e_fe_h = np.nan
        #mg_h = np.nan
        #e_mg_h = np.nan
        snr = np.nan
        sdss_id = np.nan
        print(e)

    n_visits.append(n_visit)
    source_id_dr2s.append(source_id_dr2)
    #teffs.append(teff)
    #e_teffs.append(e_teff)
    #loggs.append(logg)
    #e_loggs.append(e_logg)
    #fe_hs.append(fe_h)
    #e_fe_hs.append(e_fe_h)
    #mg_hs.append(mg_h)
    #e_mg_hs.append(e_mg_h)
    snrs.append(snr)
    source_ids.append(source_id)
    sdss_ids.append(sdss_id)

new_df = pd.DataFrame()
new_df['source_id'] = source_ids
new_df['n_apogee_visits'] = n_visits
new_df['source_id_dr2'] = source_id_dr2s
#df['teff'] = teffs
#df['e_teff'] = e_teffs
#df['logg'] = loggs
#df['e_logg'] = e_loggs
#df['fe_h'] = fe_hs
#df['e_fe_h'] = e_fe_hs
#df['mg_h'] = mg_hs
#df['e_mg_h'] = e_mg_hs
new_df['snr'] = snrs
new_df['sdss_id'] = sdss_id

new_aspcap_df = pd.merge(new_df, aspcap_df, on='source_id', how='left')
new_aspcap_df.to_csv(path+'data/enriched_lite_aspcap.csv', index=False)

new_aspcap_df = pd.read_csv(path+'data/enriched_lite_aspcap.csv', sep=',')
training = pd.merge(preds, new_aspcap_df, on='sdss_id', how='left')
print(training.loc[training['snr']<50]) # sdss_id = 67766853 is bad SNR 

### plot histogram of snrs for training set spectra (make sure none are <50, which would be bad bc they're stacked)
plt.hist(training['snr'],bins=50)
plt.xlabel('stacked spectrum SNR')
plt.savefig(path+'plots/training_snr.png')
plt.show()

plt.hist(training['aspcap_e_teff'],bins=20, color='k', alpha=0.8)
plt.xlabel('Teff uncertainty [K]')
plt.savefig(path+'plots/training_e_teff.png')
plt.show()

plt.hist(training['aspcap_e_logg'],bins=20, color='k', alpha=0.8)
plt.xlabel('logg uncertainty [dex]')
plt.savefig(path+'plots/training_e_logg.png')
plt.show()

plt.hist(training['aspcap_e_fe_h'],bins=20, color='k', alpha=0.8)
plt.xlabel('[Fe/H] uncertainty [dex]')
plt.savefig(path+'plots/training_e_fe_h.png')
plt.show()

plt.hist(training['aspcap_e_mg_h'],bins=20, color='k', alpha=0.8)
plt.xlabel('[Mg/H] uncertainty [dex]')
plt.savefig(path+'plots/training_e_mg_h.png')
plt.show()

quit()
"""

"""
### Acquire training set fluxes and ivars
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

### turn training set DataFrame into ingestable label array. also, declare label names
label_names=["Teff", "logg", "feh", "mg_h", "Age", "Dnu"]
Teff_tr = np.array(df[label_names[0]].values)
logg_tr = np.array(df[label_names[1]].values)
fe_h_tr = np.array(df[label_names[2]].values)
mg_h_tr = np.array(df[label_names[3]].values)
Age_tr = np.array(df[label_names[4]].values)
Dnu_tr = np.array(df[label_names[5]].values)
labels_tr = np.vstack((Teff_tr,logg_tr,fe_h_tr,mg_h_tr,Age_tr,Dnu_tr)).T

### Construct a CannonModel object using a quadratic (O=2) polynomial vectorizer
print(labels_tr.shape)
print(len(flux_tr))
print(len(ivar_tr))
model = tc.CannonModel(
    labels_tr, flux_tr, ivar_tr, dispersion=wl, # needed to set dispersion explicitly
    vectorizer=tc.vectorizer.PolynomialVectorizer(label_names, 2)) 

### training step
theta, s2, metadata = model.train(threads=1)
np.savetxt(path+'data/s2.txt', s2.reshape(1, -1), fmt='%.4f', delimiter=',')
#print(model.vectorizer.human_readable_label_vector)
model.write(path+"apogee-serenelli-lite.model") # write out model
quit()
"""

### for a star with 2 spectra, look for 500 such stars where n_apogee_visits==2, snr<=600, snr>=200, spec_chisq<100000
# read in trained model
model = tc.CannonModel.read(path+"apogee-serenelli-lite.model")
s2 = np.loadtxt(path+'data/s2.txt',delimiter=',',dtype=float)
count = 0
spec_fit_chisq_arr = []
# ivar specifically for chisq calculation
ivar_chisq=[]
sdss_ids = []
source_ids = []
pre_fit_z_teffs = []
pre_fit_z_loggs = []
pre_fit_z_fe_hs = []
pre_fit_z_mg_hs = []
pre_fit_z_ages = []
pre_fit_z_Dnus = []
l_A_teffs = []
l_B_teffs = []
l_A_loggs = []
l_B_loggs = []
l_A_fe_hs = []
l_B_fe_hs = []
l_A_mg_hs = []
l_B_mg_hs = []
l_A_ages = []
l_B_ages = []
l_A_Dnus = []
l_B_Dnus = []
sigma_A_teffs = []
sigma_B_teffs = []
sigma_A_loggs = []
sigma_B_loggs = []
sigma_A_fe_hs = []
sigma_B_fe_hs = []
sigma_A_mg_hs = []
sigma_B_mg_hs = []
sigma_A_ages = []
sigma_B_ages = []
sigma_A_Dnus = []
sigma_B_Dnus = []

print(len(aspcap_df))
aspcap_df_label_space = aspcap_df.loc[(aspcap_df['aspcap_teff'] <= np.max(preds['Teff_pred'])) & (aspcap_df['aspcap_teff'] >= np.min(preds['Teff_pred'])) & (aspcap_df['aspcap_logg'] <= np.max(preds['logg_pred'])) & (aspcap_df['aspcap_logg'] >= np.min(preds['logg_pred'])) & (aspcap_df['aspcap_fe_h'] <= np.max(preds['fe_h_pred'])) & (aspcap_df['aspcap_fe_h'] >= np.min(preds['fe_h_pred'])) & (aspcap_df['aspcap_mg_h'] <= np.max(preds['mg_h_pred'])) & (aspcap_df['aspcap_mg_h'] >= np.min(preds['mg_h_pred']))]
print(len(aspcap_df_label_space))

sdss_id_skips = []
sdss_id_dones = []
for sdss_id in tqdm(aspcap_df_label_space['aspcap_sdss_id']):
    #sdss_id = 55502474 #67401798 #66668317
    #print(sdss_id)
    access = Access(release='ipl-3', verbose=False)
    access.remote()
    access.add('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)
    
    try:
        access.set_stream()
        access.commit()
        mwm_filenameStar = access.full('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)
        mwmStar = fits.open(mwm_filenameStar)

        """
        n_visit = mwmStar[0].header['N_APOGEE']
        if n_visit==2:
            pass
        else:
            print("skip ", sdss_id, ", total: ", count)
            sdss_id_skips.append(sdss_id)
            pd.DataFrame({'sdss_id': sdss_id_skips}).to_csv(path+'data/chisq_skips.csv', index=False)
            continue
        """
        """
        # only use label space stars to calculate error
        aspcap_teff = np.array(aspcap_df.loc[aspcap_df['aspcap_sdss_id']==sdss_id]['aspcap_teff'])[0]
        aspcap_logg = np.array(aspcap_df.loc[aspcap_df['aspcap_sdss_id']==sdss_id]['aspcap_logg'])[0]
        aspcap_fe_h = np.array(aspcap_df.loc[aspcap_df['aspcap_sdss_id']==sdss_id]['aspcap_fe_h'])[0]
        aspcap_mg_h = np.array(aspcap_df.loc[aspcap_df['aspcap_sdss_id']==sdss_id]['aspcap_mg_h'])[0]
        if (aspcap_teff <= np.max(preds['Teff_pred'])) & (aspcap_teff >= np.min(preds['Teff_pred'])) & (aspcap_logg <= np.max(preds['logg_pred'])) & (aspcap_logg >= np.min(preds['logg_pred'])) & (aspcap_fe_h <= np.max(preds['fe_h_pred'])) & (aspcap_fe_h >= np.min(preds['fe_h_pred'])) & (aspcap_mg_h <= np.max(preds['mg_h_pred'])) & (aspcap_mg_h >= np.min(preds['mg_h_pred'])):
            pass
        else:
            print("skip ", sdss_id, ", total: ", count)
            continue
        """

        wl_star = mwmStar[3].data['wavelength'][0]
        snr = mwmStar[3].data['snr'][0]
        if snr>=200: # if stacked snr doesn't pass this, then individual visit snrs won't either
            pass
        else:
            print("skip ", sdss_id, ", total: ", count, ", snr: ", snr)
            sdss_id_skips.append(sdss_id)
            pd.DataFrame({'sdss_id': sdss_id_skips}).to_csv(path+'data/chisq_skips.csv', index=False)
            continue

        access = Access(release='ipl-3', verbose=False)
        access.remote()
        access.add('mwmVisit', v_astra='0.6.0', component='', sdss_id=sdss_id)
        access.set_stream()
        access.commit()
        mwm_filename = access.full('mwmVisit', v_astra='0.6.0', component='', sdss_id=sdss_id)
        mwmVisit = fits.open(mwm_filename)
        snr1 = mwmVisit[3].data['snr'][0]
        snr2 = mwmVisit[3].data['snr'][1]

        if (snr1 <= 600) and (snr1 >= 200) and (snr2 <= 600) and (snr2 >= 200):
            pass
        else:
            print("skip ", sdss_id, ", total: ", count, ", snr1: " , snr1, ", snr2: ", snr2)
            sdss_id_skips.append(sdss_id)
            pd.DataFrame({'sdss_id': sdss_id_skips}).to_csv(path+'data/chisq_skips.csv', index=False)
            continue
        
        flux1 = mwmVisit[3].data['flux'][0]
        flux2 = mwmVisit[3].data['flux'][1]
        ivar1 = mwmVisit[3].data['ivar'][0]
        ivar2 = mwmVisit[3].data['ivar'][1]

        wl1, norm_flux1, ivar1 = process_spectra_gaus_chris_version(flux1, ivar1, wl_star, L=10)
        wl2, norm_flux2, ivar2 = process_spectra_gaus_chris_version(flux2, ivar2, wl_star, L=10)

        # apply trained model to this random inference set star's spectrum
        test_labels1, cov_val1, metadata_val1 = model.test(norm_flux1, ivar1)
        test_labels2, cov_val2, metadata_val2 = model.test(norm_flux2, ivar2)

        # get Cannon-derived model spectra
        model_spectrum1 = model(test_labels1)
        model_spectrum2 = model(test_labels2)

        # chisq of model spectral fit
        spec_fit_chisq1 = np.sum(((model_spectrum1-norm_flux1)**2)/(ivar1**-1 + s2))
        spec_fit_chisq2 = np.sum(((model_spectrum2-norm_flux2)**2)/(ivar2**-1 + s2))

        # calculate Z_A,B
        def _calculate_z(l_a, l_b, spec_fit_chisq1, spec_fit_chisq2):
            """Eq11 from Behmard+25, for calculating error per label

            Args:
                l_a (float): inferred label from first visit spectrum
                l_b (float): inferred label from second visit spectrum
                spec_fit_chisq1 (float): chisq of inferred spectrum vs first visit spectrum
                spec_fit_chisq2 (float): chisq of inferred spectrum vs second visit spectrum

            Returns:
                z (float): Z_A,B
            """
            sigma_inflate = np.log10(0.016) # range from 0.016-0.025 dex
            numerator = l_a - l_b
            denominator = np.sqrt(spec_fit_chisq1**2 + spec_fit_chisq2**2 + 2*sigma_inflate**2)
            z = numerator/denominator

            return z

        def cov_matrix(cov):
            # model-assigned label scatter: this is for sigma_A, B, as well as errorbars at the individual star level 
            matrix = np.zeros((len(cov),6)) # Pre-allocate matrix
            for i in range(0,len(cov)):
                matrix[i,:] = np.sqrt(np.diag(cov[i]))

            df_sigma = pd.DataFrame(matrix)
            return df_sigma
        
        sigma_A = np.array(cov_matrix(cov_val1))
        sigma_B = np.array(cov_matrix(cov_val2))

        """
        pre_fit_z_teff = _calculate_z(test_labels1[0][0], test_labels2[0][0], spec_fit_chisq1, spec_fit_chisq2)
        pre_fit_z_logg = _calculate_z(test_labels1[0][1], test_labels2[0][1], spec_fit_chisq1, spec_fit_chisq2)
        pre_fit_z_fe_h = _calculate_z(test_labels1[0][2], test_labels2[0][2], spec_fit_chisq1, spec_fit_chisq2)
        pre_fit_z_mg_h = _calculate_z(test_labels1[0][3], test_labels2[0][3], spec_fit_chisq1, spec_fit_chisq2)
        pre_fit_z_age = _calculate_z(test_labels1[0][4], test_labels2[0][4], spec_fit_chisq1, spec_fit_chisq2)
        pre_fit_z_Dnu = _calculate_z(test_labels1[0][5], test_labels2[0][5], spec_fit_chisq1, spec_fit_chisq2)
        print(pre_fit_z_teff)
        print(pre_fit_z_logg)
        print(pre_fit_z_fe_h)
        print(pre_fit_z_mg_h)
        print(pre_fit_z_age)
        print(pre_fit_z_Dnu)
        """
        sdss_ids.append(sdss_id)
        l_A_teffs.append(test_labels1[0][0])
        l_B_teffs.append(test_labels2[0][0])
        l_A_loggs.append(test_labels1[0][1])
        l_B_loggs.append(test_labels2[0][1])
        l_A_fe_hs.append(test_labels1[0][2])
        l_B_fe_hs.append(test_labels2[0][2])
        l_A_mg_hs.append(test_labels1[0][3])
        l_B_mg_hs.append(test_labels2[0][3])
        l_A_ages.append(test_labels1[0][4])
        l_B_ages.append(test_labels2[0][4])
        l_A_Dnus.append(test_labels1[0][5])
        l_B_Dnus.append(test_labels2[0][5])

        sigma_A_teffs.append(sigma_A[0][0])
        sigma_B_teffs.append(sigma_B[0][0])
        sigma_A_loggs.append(sigma_A[0][1])
        sigma_B_loggs.append(sigma_B[0][1])
        sigma_A_fe_hs.append(sigma_A[0][2])
        sigma_B_fe_hs.append(sigma_B[0][2])
        sigma_A_mg_hs.append(sigma_A[0][3])
        sigma_B_mg_hs.append(sigma_B[0][3])
        sigma_A_ages.append(sigma_A[0][4])
        sigma_B_ages.append(sigma_B[0][4])
        sigma_A_Dnus.append(sigma_A[0][5])
        sigma_B_Dnus.append(sigma_B[0][5])

        count += 1
        print("keep: ", sdss_id)
        sdss_id_dones.append(sdss_id)
        pd.DataFrame({'sdss_id': sdss_id_dones}).to_csv(path+'data/chisq_dones.csv', index=False)

        if count == 500:
            break

    except Exception as e:
        print("AHHHHHHH: ", e)
    
    print("count: ", count)

chisq_df = pd.DataFrame({'sdss_id': sdss_ids, 'l_A_Teffs': l_A_teffs, 'l_B_Teffs': l_B_teffs, 'l_A_loggs': l_A_loggs, 'l_B_loggs': l_B_loggs,
                         'l_A_fe_hs': l_A_fe_hs, 'l_B_fe_hs': l_B_fe_hs, 'l_A_mg_hs': l_A_mg_hs, 'l_B_mg_hs': l_B_mg_hs, 'l_A_ages': l_A_ages,
                         'l_B_ages': l_B_ages, 'l_A_Dnus': l_A_Dnus, 'l_B_Dnus': l_B_Dnus, 'sigma_A_teff': sigma_A_teffs, 'sigma_B_teff': sigma_B_teffs,
                         'sigma_A_logg': sigma_A_loggs, 'sigma_B_logg': sigma_B_loggs, 'sigma_A_fe_h': sigma_A_fe_hs, 'sigma_B_fe_h': sigma_B_fe_h,
                         'sigma_A_mg_h': sigma_A_mg_hs, 'sigma_B_mg_h': sigma_B_mg_hs, 'sigma_A_age': sigma_A_ages, 'sigma_B_age': sigma_B_ages,
                         'sigma_A_Dnu': sigma_A_Dnus, 'sigma_B_Dnu': sigma_B_Dnus})
chisq_df.to_csv(path+'data/chisq_pre_fit_aspcap.csv', index=False)
#np.savetxt(path+'data/chisq.txt', spec_fit_chisq_arr.reshape(1, -1), fmt='%.4f', delimiter=',')
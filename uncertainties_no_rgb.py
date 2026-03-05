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

"""
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
"""

aspcap_df = pd.read_csv(path+'data/aspcap.csv') # from comparison.csv
print(aspcap_df)
print(list(aspcap_df.columns))
print(len(np.unique(aspcap_df['aspcap_sdss_id'])))

# inferences_df = pd.read_csv(path+'data/inferences_kic_no_rgb.csv')
# print(inferences_df)

# inferences_df_matched = pd.read_csv(path+'data/inferences_df_matched.csv')
# print(inferences_df_matched)

training = pd.read_csv(path+'data/4_fold_cv_no_gb.csv') # 4_fold_cv.csv, 4_fold_cv_ruwe.csv, 4_fold_cv_teff.csv, 4_fold_cv_teff_ruwe.csv, 4_fold_cv_no_rgb.csv
training = training.loc[training['KIC']!=7976303]
training = training.loc[training['KIC']!=10454113]

"""
# ~5000 inference stars. Get those with at least 2 visits and SNR between 200 and 600. Hopefully there's well more than 500.
# There wasn't. There was 380 stars. Even if there were 500, I'd need all to have chisq <100000, which is unrealistic. 
# Maybe it's unreasonable to use a match sampled set for uncertainty calculations. Here, we just need to cull by Teff and logg bounds.

keep = []
for source_id in tqdm(inferences_df_matched['source_id']):
    n_visit = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].n_apogee_visits[0]
    snr = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].snr[0]
    
    if (n_visit >= 2) & (snr < 600) & (snr > 200):
        keep.append(source_id)
print(keep)

inferences_df_keep = inferences_df.loc[inferences_df['source_id'].isin(keep)]
print(inferences_df_keep)

inferences_df_keep.to_csv(path+'data/inferences_df_keep_for_chisq.csv', index=False)

# now calculate spectra chisq, which needs to be <100000

### I got my 500 stars! 
chisq_pre_fit_aspcap = pd.read_csv(path+'data/chisq_pre_fit_aspcap.csv', sep=',')
print(chisq_pre_fit_aspcap)
quit()
"""

### I got my 500 stars!
chisq_pre_fit_aspcap = pd.read_csv(path+'data/chisq_no_rgb.csv', sep=',') # as of Dec 5, 2025

# calculate Z_A,B
def _calculate_z(l_a, l_b, sigma_A, sigma_B, sigma_inflate_dex):
    """Eq11 from Behmard+25, for calculating error per label

    Args:
        l_a (float): inferred label from first visit spectrum
        l_b (float): inferred label from second visit spectrum
        sigma_A (float): chisq of inferred spectrum vs first visit spectrum
        sigma_B (float): chisq of inferred spectrum vs second visit spectrum

    Returns:
        z (float): Z_A,B
    """

    # Aida's ranges were from 0.016-0.025 dex 
    sigma_inflate_regular = 10**(sigma_inflate_dex) # this is in regular space
    numerator = l_a - l_b
    denominator = np.sqrt(sigma_A**2 + sigma_B**2 + 2*sigma_inflate_regular**2)
    z = numerator/denominator

    #print(sigma_inflate_dex, sigma_inflate_regular, np.median(z), np.mean(z), np.nanstd(z))

    return z

def tighter_cull(reference_df, sample_df):

    print("tighter cull: min Teff ", min(reference_df['teff']), ", max Teff ", max(reference_df['teff']), ", min logg ", min(reference_df['logg']), ", max logg ", max(reference_df['logg']))
    ### limit Teff and logg of sample (inference set) to bounds from reference (training) set
    sample_df = sample_df.loc[((sample_df['aspcap_logg']<max(reference_df['logg'])) & (sample_df['aspcap_logg']>min(reference_df['logg'])))]
    sample_df = sample_df.loc[((sample_df['aspcap_teff']<max(reference_df['teff'])) & (sample_df['aspcap_teff']>min(reference_df['teff'])))]
    #sample_df = sample_df.loc[((sample_df['aspcap_fe_h']<max(reference_df['feh'])) & (sample_df['aspcap_fe_h']>min(reference_df['feh'])))]
    #sample_df = sample_df.loc[((sample_df['aspcap_mg_h']<max(reference_df['mg_h'])) & (sample_df['aspcap_mg_h']>min(reference_df['mg_h'])))]
    
    return sample_df

# ad hoc visual check if Z is Gaussian
# z_Dnu = _calculate_z(chisq_pre_fit_aspcap['l_A_Dnus'], chisq_pre_fit_aspcap['l_B_Dnus'], chisq_pre_fit_aspcap['sigma_A_Dnu'], chisq_pre_fit_aspcap['sigma_B_Dnu'],np.log10(5.69))
# plt.hist(z_Dnu, bins=np.linspace(-1, 1, 10))
# plt.show()
# quit()

#""" Turn this on when I've run the 500 stars and want to start calculating Z
for sigma_inflate in np.linspace(0.6, 0.8, 50): # this is in dex space
    #z_Teff = _calculate_z(chisq_pre_fit_aspcap['l_A_Teffs'], chisq_pre_fit_aspcap['l_B_Teffs'], chisq_pre_fit_aspcap['sigma_A_teff'], chisq_pre_fit_aspcap['sigma_B_teff'], sigma_inflate)
    #z_logg = _calculate_z(chisq_pre_fit_aspcap['l_A_loggs'], chisq_pre_fit_aspcap['l_B_loggs'], chisq_pre_fit_aspcap['sigma_A_logg'], chisq_pre_fit_aspcap['sigma_B_logg'], sigma_inflate)
    #z_fe_h = _calculate_z(chisq_pre_fit_aspcap['l_A_fe_hs'], chisq_pre_fit_aspcap['l_B_fe_hs'], chisq_pre_fit_aspcap['sigma_A_fe_h'], chisq_pre_fit_aspcap['sigma_B_fe_h'], sigma_inflate)    
    #z_mg_h = _calculate_z(chisq_pre_fit_aspcap['l_A_mg_hs'], chisq_pre_fit_aspcap['l_B_mg_hs'], chisq_pre_fit_aspcap['sigma_A_mg_h'], chisq_pre_fit_aspcap['sigma_B_mg_h'], sigma_inflate)    
    #z_age = _calculate_z(chisq_pre_fit_aspcap['l_A_ages'], chisq_pre_fit_aspcap['l_B_ages'], chisq_pre_fit_aspcap['sigma_A_age'], chisq_pre_fit_aspcap['sigma_B_age'], sigma_inflate)    
    z_Dnu = _calculate_z(chisq_pre_fit_aspcap['l_A_Dnus'], chisq_pre_fit_aspcap['l_B_Dnus'], chisq_pre_fit_aspcap['sigma_A_Dnu'], chisq_pre_fit_aspcap['sigma_B_Dnu'], sigma_inflate)    
    #print(sigma_inflate, 10**sigma_inflate, np.median(z_Dnu), np.mean(z_Dnu), np.nanstd(z_Dnu))
#"""

### are the sigma columns in inferences_kic_no_rgb.csv including sigma_inflate or nah?
inferences_kic = pd.read_csv(path+'data/inferences_kic_no_rgb.csv')
print(list(inferences_kic.columns))
print(inferences_kic[['kepid','Teff_pred','logg_pred','Age_pred','sigma_star_Teff','sigma_star_logg','sigma_star_age']])
quit()

# sigma_inflate_teff = 31.0 K (1.491 dex --> -0.014, -0.038, 1.000) (10**sigma_inflate --> median, mean, std)
# sigma_inflate_logg = 0.043 (-1.371 dex --> -0.000, -0.013, 1.000)
# sigma_inflate_fe_h = 0.027 (-1.565 dex --> 0.005, -0.087, 0.999)
# sigma_inflate_mg_h = 0.026 (-1.586 dex --> -0.037, -0.101 1.000)
# sigma_inflate_age = 0.417 (-0.380 dex --> -0.008, -0.045, 0.998)
# sigma_inflate_Dnu = 0.751 (5.637 dex --> -0.005, 0.038, 1.000)

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

#chisq_dones_one = pd.read_csv(path+'data/chisq_dones_one.csv')

### for a star with 2 spectra, look for 500 such stars where n_apogee_visits==2, snr<=600, snr>=200, spec_chisq<100000
# read in trained model
model = tc.CannonModel.read(path+"no-rgb.model") # apogee-serenelli-lite.model
s2 = np.loadtxt(path+'data/s2-no-rgb.txt',delimiter=',',dtype=float) # s2.txt
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

#aspcap_df_label_space = aspcap_df.loc[(aspcap_df['aspcap_teff'] <= np.max(preds['Teff_pred'])) & (aspcap_df['aspcap_teff'] >= np.min(preds['Teff_pred'])) & (aspcap_df['aspcap_logg'] <= np.max(preds['logg_pred'])) & (aspcap_df['aspcap_logg'] >= np.min(preds['logg_pred'])) & (aspcap_df['aspcap_fe_h'] <= np.max(preds['fe_h_pred'])) & (aspcap_df['aspcap_fe_h'] >= np.min(preds['fe_h_pred'])) & (aspcap_df['aspcap_mg_h'] <= np.max(preds['mg_h_pred'])) & (aspcap_df['aspcap_mg_h'] >= np.min(preds['mg_h_pred']))]
#print(len(aspcap_df_label_space.drop_duplicates(subset=['aspcap_sdss_id'])))

### first cull ASPCAP on label space so that it's within bounds of training set parameter space
print("training: ", list(training.columns))
print("aspcap: ", list(aspcap_df.columns))
aspcap_df_culled = tighter_cull(training, aspcap_df)
print("ASPCAP within training set label space: ", aspcap_df_culled)

aspcap_df_visits = aspcap_df_culled.loc[aspcap_df_culled['aspcap_visits']==2]
print("sample size with 2 visits: ", len(aspcap_df_visits.drop_duplicates(subset=['aspcap_sdss_id'])))

aspcap_df_snr = aspcap_df_visits.loc[(aspcap_df_visits['aspcap_snr']>=200)] # this is stacked. the snr<600 requirement is for individual spectra
print("sample size with enough SNR: ", len(aspcap_df_snr.drop_duplicates(subset=['aspcap_sdss_id'])))

#plt.hist(aspcap_df_snr.drop_duplicates(subset=['aspcap_sdss_id'])['aspcap_snr'], bins=40)
#plt.xlabel('snr')
#plt.savefig(path+'plots/aspcap_snr.png')
#plt.show()
#print(aspcap_dones)
#print(aspcap_dones.drop_duplicates(subset=['aspcap_sdss_id']))

sdss_id_skips = []
sdss_id_dones = []
snr1s = []
snr2s = []
lite_snrs = []
#for sdss_id in tqdm(aspcap_df_label_space['aspcap_sdss_id']):
#for sdss_id in tqdm(aspcap_dones['aspcap_sdss_id']):
#for sdss_id in tqdm(aspcap_df_snr.drop_duplicates(subset=['aspcap_sdss_id'])['aspcap_sdss_id']):
for sdss_id in tqdm(chisq_pre_fit_aspcap['sdss_id']): # aspcap_df_snr or aspcap_dones. But now that we've identified 500 good ones, we can do this again w/o the hard part: chisq_pre_fit_aspcap
    #sdss_id = 55502474 #67401798 #66668317
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
        lite_snrs.append(snr)
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
        snr1s.append(snr1)
        snr2s.append(snr2)

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
        print("keep: ", sdss_id, ", SNRs: ", snr1, snr, ", total: ", count)
        sdss_id_dones.append(sdss_id)
        pd.DataFrame({'sdss_id': sdss_id_dones}).to_csv(path+'data/chisq_dones_ruwe.csv', index=False)

        if count == 500:
            break

    except Exception as e:
        print("AHHHHHHH: ", e)
    
    print("count: ", count)

print(len(sdss_ids))
print(len(l_A_teffs))
print(len(lite_snrs))
print(len(snr1s))
print(len(snr2s))
chisq_df = pd.DataFrame({'sdss_id': sdss_ids, 'l_A_Teffs': l_A_teffs, 'l_B_Teffs': l_B_teffs, 'l_A_loggs': l_A_loggs, 'l_B_loggs': l_B_loggs,
                         'l_A_fe_hs': l_A_fe_hs, 'l_B_fe_hs': l_B_fe_hs, 'l_A_mg_hs': l_A_mg_hs, 'l_B_mg_hs': l_B_mg_hs, 'l_A_ages': l_A_ages,
                         'l_B_ages': l_B_ages, 'l_A_Dnus': l_A_Dnus, 'l_B_Dnus': l_B_Dnus, 'sigma_A_teff': sigma_A_teffs, 'sigma_B_teff': sigma_B_teffs,
                         'sigma_A_logg': sigma_A_loggs, 'sigma_B_logg': sigma_B_loggs, 'sigma_A_fe_h': sigma_A_fe_hs, 'sigma_B_fe_h': sigma_B_fe_hs,
                         'sigma_A_mg_h': sigma_A_mg_hs, 'sigma_B_mg_h': sigma_B_mg_hs, 'sigma_A_age': sigma_A_ages, 'sigma_B_age': sigma_B_ages,
                         'sigma_A_Dnu': sigma_A_Dnus, 'sigma_B_Dnu': sigma_B_Dnus, 'lite_snr': lite_snrs, 'snr1': snr1s, 'snr2': snr2s})
chisq_df.to_csv(path+'data/chisq_no_rgb.csv', index=False) # chisq_pre_fit_aspcap_ruwe.csv
#np.savetxt(path+'data/chisq.txt', spec_fit_chisq_arr.reshape(1, -1), fmt='%.4f', delimiter=',')
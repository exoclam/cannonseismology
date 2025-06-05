import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

import thecannon as tc

path = '/Users/chrislam/Desktop/cannon-ages/' 
#path = '/home/c.lam/blue/cannon-ages/'

df = pd.read_csv(path+'data/enriched_lite.csv', sep=',')
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

fits_image_filename = path+'data/mwmAllStar-0.6.0.fits'
hdul = fits.open(fits_image_filename)

fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)  

lite_source_ids = hdul_lite[1].data.gaia_dr3_source_id

"""
### look back in astraMWMLite to get number of APOGEE visits per star
n_visits = []
teffs = []
e_teffs = []
loggs = []
e_loggs = []
fe_hs = []
e_fe_hs = []
mg_hs = []
e_mg_hs = []
for source_id in tqdm(df['source_id']):
    try:
        n_visit = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].n_apogee_visits[0]
        teff = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].teff[0]
        e_teff = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].e_teff[0]
        logg = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].logg[0]
        e_logg = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].e_logg[0]
        fe_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].fe_h[0]
        e_fe_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].e_fe_h[0]
        mg_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].mg_h[0]
        e_mg_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].e_mg_h[0]

    except Exception as e:
        n_visit = np.nan
        teff = np.nan
        e_teff = np.nan
        logg = np.nan
        e_logg = np.nan
        fe_h = np.nan
        e_fe_h = np.nan
        mg_h = np.nan
        e_mg_h = np.nan
        print(e)

    n_visits.append(n_visit)
    teffs.append(teff)
    e_teffs.append(e_teff)
    loggs.append(logg)
    e_loggs.append(e_logg)
    fe_hs.append(fe_h)
    e_fe_hs.append(e_fe_h)
    mg_hs.append(mg_h)
    e_mg_hs.append(e_mg_h)

df['n_apogee_visits'] = n_visits
df['teff'] = teffs
df['e_teff'] = e_teffs
df['logg'] = loggs
df['e_logg'] = e_loggs
df['fe_h'] = fe_hs
df['e_fe_h'] = e_fe_hs
df['mg_h'] = mg_hs
df['e_mg_h'] = e_mg_hs

df.to_csv(path+'data/enriched_lite_visits.csv', index=False)
quit()
"""

### for a star with 2 spectra, look for 500 such stars where n_apogee_visits==2, snr<=600, snr>=200, spec_chisq<100000
# read in trained model
model = tc.CannonModel.read(path+"apogee-dr14-giants.model")
count_target = 0
for source_id in lite_source_ids:
    if count_target <= 500:
        n_visit = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].n_apogee_visits[0]
        snr = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].snr[0]

        test_labels, cov_val, metadata_val = model.test(test_flux, test_ivar)
        # get Cannon-derived model spectra
        model_spectrum = model(test_labels)

        # chisq of model spectral fit
        spec_fit_chisq = np.sum(((model_spectrum-test_flux)**2)/(test_ivar_chisq**-1 + s2))
        spec_fit_chisq_arr.append(spec_fit_chisq)
        
        if (n_visit==2) and (snr <= 600) and (snr >= 200):
            pass


        count_target += 1

    else: 
        break


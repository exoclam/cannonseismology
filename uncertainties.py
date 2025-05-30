import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

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

# look back in astraMWMLite to get number of APOGEE visits per star
fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)  

lite_source_ids = hdul_lite[1].data.gaia_dr3_source_id

n_visits = []
for source_id in tqdm(df['source_id']):
    try:
        n_visit = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].n_apogee_visits[0]

    except Exception as e:
        n_visit = np.nan
        print(e)

    n_visits.append(n_visit)

df['n_apogee_visits'] = n_visits
df.to_csv(path+'data/enriched_lite_visits.csv', index=False)


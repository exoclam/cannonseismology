"""
Get spectra and labels for Serenelli+17 stars: https://iopscience.iop.org/article/10.3847/1538-4365/aa97df
"""

import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import sdss_access
import matplotlib.pyplot as plt
from sdss_access import Access
access = Access(release='ipl-3', verbose=False)
access.remote()

path = '/home/c.lam/blue/cannon-ages/'
#path = '/Users/chrislam/Desktop/cannon-ages/' 

# Serenelli+17 stars
apokasc_sdss = pd.read_csv(path+'data/apokasc-sdss-teff.txt', sep='\s+')
serenelli_table3 = pd.read_csv(path+'data/serenelli_table3.txt',sep='\s+')

# Bedell cross-match has the Gaia DR3 source_id we need 
bedell = Table.read(path+'data/kepler_dr3_good.fits')
bedell_df = bedell.to_pandas()
apokasc_sdss_bedell_df = pd.merge(apokasc_sdss, bedell_df, left_on='KIC', right_on='kepid')
apokasc_sdss_bedell_df = apokasc_sdss_bedell_df.rename(columns={"logg_x": "logg"})
apokasc_sdss_bedell_df = pd.merge(apokasc_sdss_bedell_df, serenelli_table3, on='KIC')
print(apokasc_sdss_bedell_df)

# get udpated labels from astraMWMLite
fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)  
lite_source_ids = hdul_lite[1].data.gaia_dr3_source_id

# build intersection sdss_id list, to grab updated labels from astraMWMLite
apokasc_sdss_bedell_df = apokasc_sdss_bedell_df.loc[apokasc_sdss_bedell_df['source_id'].isin(lite_source_ids)]
source_ids = apokasc_sdss_bedell_df['source_id']
sdss_ids = []
teffs = []
loggs = []
fe_hs = []
mg_hs = []
for source_id in tqdm(source_ids):
    sdss_id = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].sdss_id[0]
    teff = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].teff[0]
    logg = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].logg[0]
    fe_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].fe_h[0]
    mg_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].mg_h[0]

    sdss_ids.append(sdss_id)
    teffs.append(teff)
    loggs.append(logg)
    fe_hs.append(fe_h)
    mg_hs.append(mg_h)

apokasc_sdss_bedell_df['sdss_id'] = sdss_ids
apokasc_sdss_bedell_df['Teff'] = teffs
apokasc_sdss_bedell_df['logg'] = loggs
apokasc_sdss_bedell_df['feh'] = fe_hs
apokasc_sdss_bedell_df['mg_h'] = mg_hs

print(apokasc_sdss_bedell_df)
apokasc_sdss_bedell_df.dropna(subset=['sdss_id','mg_h','Teff','feh','logg','Age','Dnu']).to_csv(path+'data/enriched_lite.csv', index=False)

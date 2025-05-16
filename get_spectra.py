"""
Get KIC field star APOGEE spectra
"""

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

fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)  
lite_source_ids = hdul_lite[1].data.gaia_dr3_source_id

# Bedell cross-match has the Gaia DR3 source_id we need 
bedell = Table.read(path+'data/kepler_dr3_good.fits')
bedell_df = bedell.to_pandas()

# build intersection sdss_id list
bedell_kic_apogee = bedell_df.loc[bedell_df['source_id'].isin(lite_source_ids)]
source_ids = bedell_kic_apogee['source_id']
sdss_ids = []
for source_id in source_ids:
    sdss_id = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].sdss_id[0]
    sdss_ids.append(sdss_id)
bedell_kic_apogee['sdss_id'] = sdss_ids
print(bedell_kic_apogee)

for sdss_id in tqdm(sdss_ids):
    try:
        access.add('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)
        access.set_stream()
        access.commit()
        
        mwmStar_filename = access.full('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)

        # read to fits, bc actually it'll be easier to handle columns of lists this way
        mwmStar = fits.open(mwmStar_filename)
        try:
            mwmStar.writeto(path+'data/kic_spectra/mwmStar-0.6.0-'+str(sdss_id)+'.fits', overwrite=False)
        except:
            print("already have it!")
            pass

    except Exception as e:
        print(e)
        pass
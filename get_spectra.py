"""
Get KIC field star APOGEE spectra
"""

import pandas as pd
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

def build_inference_set_labels():

    """We are not training on these labels, right? 
    But then...what is the point of this project if we already have labels? 
    Are these not spectroscopically derived parameters? 
    I guess these are from Gaia, which we...don't trust as much? 
    I must certainly plot my predicted labels against these...

    Returns:
        _type_: _description_
    """
    #path = '/home/c.lam/blue/cannon-ages/'
    path = '/Users/chrislam/Desktop/cannon-ages/' 

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
    for source_id in tqdm(source_ids):
        sdss_id = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].sdss_id[0]
        sdss_ids.append(sdss_id)
    bedell_kic_apogee['sdss_id'] = sdss_ids
    print(bedell_kic_apogee)
    bedell_kic_apogee.to_csv(path+'data/bedell_kic_apogee.csv',index=False)
    
    return bedell_kic_apogee


def build_inference_set_spectra(df):

    sdss_ids = df['sdss_id']

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

    return


def get_spectra(sdss_id, path, folder):
    access.add('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)
    access.set_stream()
    access.commit()
    
    mwmStar_filename = access.full('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)

    # read to fits, bc actually it'll be easier to handle columns of lists this way
    mwmStar = fits.open(mwmStar_filename)
    try:
        mwmStar.writeto(path+'data/'+folder+'/mwmStar-0.6.0-'+str(sdss_id)+'.fits', overwrite=False)
    except:
        print("already have it!")
        pass

#bedell_kic_apogee = build_inference_set_labels() # I did this in HPG already, and rsynced the product back to local
df = pd.read_csv(path+'data/bedell_kic_apogee.csv')
print(list(df.columns))
build_inference_set_spectra(df)

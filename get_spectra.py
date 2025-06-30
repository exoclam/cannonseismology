"""
Get KIC field star APOGEE spectra
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import sdss_access
import matplotlib.pyplot as plt
import process_spectra_gaus
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
    mg_hs = []
    for source_id in tqdm(source_ids):
        sdss_id = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].sdss_id[0]
        sdss_ids.append(sdss_id)

        mg_h = hdul_lite[1].data[hdul_lite[1].data.gaia_dr3_source_id==source_id].mg_h[0]
        mg_hs.append(mg_h)
    bedell_kic_apogee['sdss_id'] = sdss_ids
    bedell_kic_apogee['mg_h'] = mg_hs
    print(bedell_kic_apogee)

    # cull based on label space of Serenelli+17 training set
    bedell_kic_apogee = bedell_kic_apogee.loc[(bedell_kic_apogee['teff'] <= 6540) & (bedell_kic_apogee['teff'] >= 4740)]
    bedell_kic_apogee = bedell_kic_apogee.loc[(bedell_kic_apogee['logg'] <= 4.5) & (bedell_kic_apogee['logg'] >= 3.2)]
    bedell_kic_apogee = bedell_kic_apogee.loc[(bedell_kic_apogee['feh'] <= 0.5) & (bedell_kic_apogee['feh'] >= -0.75)]
    bedell_kic_apogee = bedell_kic_apogee.loc[(bedell_kic_apogee['mg_h'] <= 0.45) & (bedell_kic_apogee['mg_h'] >= -0.70)]
    print(bedell_kic_apogee)
    bedell_kic_apogee.to_csv(path+'data/bedell_kic_apogee.csv',index=False)
    
    return bedell_kic_apogee


def build_inference_set_spectra(df, sdss_id_dones):

    sdss_ids = df['sdss_id']
    print(len(sdss_ids))
    sdss_ids = sdss_ids[~np.isin(np.array(sdss_ids), sdss_id_dones)]
    print(len(sdss_ids))
    
    for sdss_id in tqdm(sdss_ids):
        print("sdss_id: ", sdss_id)
        try:
            access = Access(release='ipl-3', verbose=False)
            access.remote()
            access.add('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)
            access.set_stream()
            access.commit()
            
            mwmStar_filename = access.full('mwmStar', v_astra='0.6.0', component='', sdss_id=sdss_id)
            print("SDSS ID: ", sdss_id, mwmStar_filename)

            # read to fits, bc actually it'll be easier to handle columns of lists this way
            mwmStar = fits.open(mwmStar_filename)
            try:
                mwmStar.writeto(path+'data/kic_spectra/mwmStar-0.6.0-'+str(sdss_id)+'.fits', overwrite=False)
            except Exception as e:
                print("problem with writing: ", e)
                pass

        except Exception as e:
            print("problem with accessing: ", e)
            pass

    return


def get_spectra(sdss_id, path, folder, visit_flag=False):

    # do I get the squashed version or the visit-by-visit version?
    if visit_flag==False:
        visit_or_star = 'Star'
    elif visit_flag==True:
        visit_or_star = 'Visit'

    access.add('mwm'+visit_or_star, v_astra='0.6.0', component='', sdss_id=sdss_id)
    access.set_stream()
    access.commit()
    
    mwm_filename = access.full('mwm'+visit_or_star, v_astra='0.6.0', component='', sdss_id=sdss_id)
    print(mwm_filename)
    
    # read to fits, bc actually it'll be easier to handle columns of lists this way
    mwm = fits.open(mwm_filename)
    try:
        mwm.writeto(path+'data/'+folder+'/mwm'+visit_or_star+'-0.6.0-'+str(sdss_id)+'.fits', overwrite=True)
        return path+'data/'+folder+'/mwm'+visit_or_star+'-0.6.0-'+str(sdss_id)+'.fits'
    
    except Exception as e:
        print(e)
        pass

    

#bedell_kic_apogee = build_inference_set_labels() # I did this in HPG already, and rsynced the product back to local
df = pd.read_csv(path+'data/bedell_kic_apogee.csv')
#print(list(df.columns))

"""
### query inference set spectra (KIC-APOGEE)
# build no-query list
start_string = 'mwmStar-0.6.0-'
end_string = '.fits'
sdss_id_dones = []
for filename in os.listdir(path+'data/kic_spectra/'):
    sdss_id_done = process_spectra_gaus.get_number_between(filename, start_string, end_string)
    sdss_id_dones.append(sdss_id_done)

build_inference_set_spectra(df, sdss_id_dones)
"""

### for a star with 2 spectra, look for 500 such stars where n_apogee_visits==2, snr<=600, snr>=200, spec_chisq<100000
enriched_lite_visits = pd.read_csv(path+'data/enriched_lite_visits.csv')
print(list(enriched_lite_visits.columns))
twos = enriched_lite_visits.loc[enriched_lite_visits['n_apogee_visits']==2]
print(twos) # 147 stars
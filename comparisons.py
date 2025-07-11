""" 
Compare predicted ages with gyrochronology (Lu+24, Bouma+24), co-moving group, kinematic (zoomies; Sagear+submitted) ages
"""

import pandas as pd
import numpy as np
import os
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
#from plot_for_aida import plot_heatmaps

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

def plot_heatmaps(label1, label2, color='Blues'):
    """Plot 2D histogram of Cannon vs APOKASC/Gaia stellar param

    Args:
        label1 (_type_): "older truth" label
        label2 (_type_): Cannon-predicted label

    Returns:
        ax: plt colormesh object
    """

    norm = 10
    bins2d = [np.linspace(np.nanmin(label1), np.nanmax(label1), 20), np.linspace(np.nanmin(label2), np.nanmax(label2), 20)]

    hist, xedges, yedges = np.histogram2d(label1, label2, bins=bins2d)
    hist = hist.T
    #with np.errstate(divide='ignore', invalid='ignore'):  # suppress division by zero warnings
        #hist *= norm / hist.sum(axis=0, keepdims=True)
        #hist *= norm / hist.sum(axis=1, keepdims=True)
    ax = plt.pcolormesh(xedges, yedges, hist, cmap=color)

    #ax.set_xlim([xedges[0], xedges[-1]])
    #ax.set_ylim([yedges[0], yedges[-1]])

    return ax

def cull(preds_df, comparison_df, comparison_teff, comparison_logg, comparison_fe_h=None, comparison_mg_h=None):

    comparison_df = comparison_df.loc[(comparison_teff <= np.max(preds_df['Teff_pred'])) & (comparison_teff >= np.min(preds_df['Teff_pred']))]
    comparison_df = comparison_df.loc[(comparison_logg <= np.max(preds_df['logg_pred'])) & (comparison_logg >= np.min(preds_df['logg_pred']))]
    
    # not all datasets, eg. trilegal, have abundances. Treat these as optional. 
    try:
        comparison_df = comparison_df.loc[(comparison_fe_h <= np.max(preds_df['fe_h_pred'])) & (comparison_fe_h >= np.min(preds_df['fe_h_pred']))]
    except Exception as e:
        print(e)
    try:
        comparison_df = comparison_df.loc[(comparison_mg_h <= np.max(preds_df['mg_h_pred'])) & (comparison_mg_h >= np.min(preds_df['mg_h_pred']))]
    except Exception as e:
        print(e)

    return comparison_df

### field stars, for Teff vs logg plots
#hdul_dr2 = fits.open(path+'data/kepler_dr2_1arcsec.fits')
#gaia_kepler_dr2 = Table(hdul_dr2[1].data).to_pandas()
#hdul_dr2.close()

### training set
preds = pd.read_csv(path+'data/preds_dnu_full.csv')
#plt.scatter(preds['Teff_test'], preds['logg_test'])
#plt.xlabel(r"$T_{\rm eff}$ [K], ASPCAP")
#plt.ylabel('logg')
#plt.show()
#quit()

# add KIC column to preds DF
df = pd.read_csv(path+'data/enriched_lite_visits.csv', sep=',')
kics = df.loc[df['sdss_id'].isin(preds['sdss_id'])]['KIC']
source_ids = df.loc[df['sdss_id'].isin(preds['sdss_id'])]['source_id']
source_id_dr2s = df.loc[df['sdss_id'].isin(preds['sdss_id'])]['source_id_dr2']

preds['KIC'] = kics
preds['source_id'] = source_ids
preds['source_id_dr2'] = source_id_dr2s
preds = preds.dropna(subset=['KIC', 'source_id'])
preds['KIC'] = preds['KIC'].astype(int)
preds['source_id'] = preds['source_id'].astype(int)
preds['source_id_dr2'] = preds['source_id_dr2'].astype(int)

### inference set
inferences_df = pd.read_csv(path+'data/inferences_kic.csv', sep=',')
inferences_df['KIC'] = inferences_df['kepid']

### combine both age prediction sets for comparison 
preds['kepid'] = preds['KIC']
columns = ['KIC','kepid','source_id','source_id_dr2','sdss_id','Teff_pred','logg_pred','fe_h_pred','mg_h_pred','Age_pred','Dnu_pred']
preds = pd.concat([preds[columns],inferences_df[columns]])
print("preds: ", preds)

"""
# open clusters
### Cantat-Gaudin+18,20 and Kounkel+20, referenced in Bouma+24 as cluster membership references
cantat_gaudin2020 = pd.read_csv(path+'data/cantat-gaudin2020.txt', sep='[\\s|]+')
cantat_gaudin2018 = pd.read_csv(path+'data/cantat-gaudin2018.txt', sep='[\\s|]+')
kounkel_table1 = pd.read_csv(path+'data/kounkel_table1.txt',sep='[\\s|]+')
kounkel_tablea1 = pd.read_csv(path+'data/kounkel_tablea1.txt',sep='|')
kounkel_table1['Group'] = kounkel_table1['Group'].astype(str)
kounkel_tablea1['Group'] = kounkel_tablea1['Group'].astype(str)
kounkel2020 = pd.merge(kounkel_table1, kounkel_tablea1, on='Group', how='left')

preds_cantat_gaudin2020 = pd.merge(preds, cantat_gaudin2020, left_on='source_id_dr2', right_on='GaiaDR2', how='left').dropna(subset=['Cluster'])
preds_cantat_gaudin2018 = pd.merge(preds, cantat_gaudin2018, left_on='source_id_dr2', right_on='Source', how='left').dropna(subset=['Cluster'])
preds_kounkel2020 = pd.merge(preds, kounkel2020, left_on='source_id_dr2', right_on='Gaia', how='left').dropna(subset=['Cluster'])
print("KOUNKEL PREDS: ", preds_kounkel2020)
print(preds_cantat_gaudin2020['Cluster'])
print(preds_cantat_gaudin2018['Cluster'])
print(preds_kounkel2020['Cluster'])

### enrich these with Gaia DR2 source_id; EDIT: not needed anymore after pulling these from ASPCAP directly
#hdul_dr2 = fits.open(path+'data/kepler_dr2_1arcsec.fits')
#gaia_kepler_dr2 = Table(hdul_dr2[1].data).to_pandas()
#hdul_dr2.close()
#gaia_kepler_dr2['source_id_dr2'] = gaia_kepler_dr2['source_id'].astype(str)
#gaia_kepler_dr2 = gaia_kepler_dr2[['source_id_dr2', 'kepid']]
#preds2 = pd.merge(preds, gaia_kepler_dr2, on='kepid', how='left')
#preds2 = preds2.dropna(subset=['kepid','source_id','source_id_dr2'])
#print(preds2)

ngc6811 = pd.read_csv(path+'data/ngc6811.csv', sep=',') 
ngc6811_cannon = pd.merge(ngc6811, preds, on='source_id')
ngc6819 = pd.read_csv(path+'data/ngc6819.csv', sep=',') 
ngc6819_cannon = pd.merge(ngc6819, preds, on='source_id')
print(ngc6811_cannon)
print(ngc6819_cannon)

plt.scatter(1*np.ones(len(ngc6811_cannon)), ngc6811_cannon['Age_pred']/1., label='NGC 6811')
plt.scatter(2.5*np.ones(len(ngc6819_cannon)), ngc6819_cannon['Age_pred']/2.5, label='NGC 6819')
plt.ylabel('Star Age/Cluster Age')
plt.xlabel('Cluster Age [Gyr]')
plt.legend()
plt.tight_layout()
#plt.savefig(path+'plots/cluster_comparison.png')
plt.show()
quit()
"""

# when plotting Kiel diagram, params should be homogeneous...should they be ASPCAP or Gaia phot or Gaia spec? 
fits_image_filename_lite = path+'data/astraMWMLite-0.6.0.fits'
hdul_lite = fits.open(fits_image_filename_lite)    
lite_sdss_ids = np.array(hdul_lite[1].data.sdss_id).byteswap().newbyteorder()
lite_snr = np.array(hdul_lite[1].data.snr).byteswap().newbyteorder()
lite_df = pd.DataFrame({'sdss_id': lite_sdss_ids, 'snr': lite_snr})

#lite = Table(hdul_lite[1].data).to_pandas()
#hdul_lite.close()
#print(lite.head())

"""
# run one time to get only relevant columns from million-row ASPCAP fits file
fits_image_filename_aspcap = path+'data/astraAllStarASPCAP-0.6.0.fits'
hdul_aspcap = fits.open(fits_image_filename_aspcap)

aspcap_source_ids = hdul_aspcap[2].data.gaia_dr3_source_id
aspcap_source_ids_dr2 = hdul_aspcap[2].data.gaia_dr2_source_id
aspcap_sdss_ids = np.array(hdul_aspcap[2].data.sdss_id).byteswap().newbyteorder()
aspcap_visits = np.array(hdul_aspcap[2].data.n_apogee_visits).byteswap().newbyteorder()
aspcap_snr = hdul_aspcap[2].data.snr
aspcap_teffs = hdul_aspcap[2].data.teff
aspcap_e_teffs = hdul_aspcap[2].data.e_teff
aspcap_loggs = hdul_aspcap[2].data.logg
aspcap_e_loggs = hdul_aspcap[2].data.e_logg
aspcap_fe_hs = hdul_aspcap[2].data.fe_h
aspcap_e_fe_hs = hdul_aspcap[2].data.e_fe_h
aspcap_mg_hs = hdul_aspcap[2].data.mg_h
aspcap_e_mg_hs = hdul_aspcap[2].data.e_mg_h
aspcap_df = pd.DataFrame({'source_id': aspcap_source_ids, 'aspcap_source_id_dr2': aspcap_source_ids_dr2, 'aspcap_sdss_id': aspcap_sdss_ids, 'aspcap_visits': aspcap_visits, 'aspcap_teff': aspcap_teffs,
                           'aspcap_snr': aspcap_snr, 'aspcap_e_teff': aspcap_e_teffs, 'aspcap_logg': aspcap_loggs, 'aspcap_e_logg': aspcap_e_loggs, 'aspcap_fe_h': aspcap_fe_hs, 'aspcap_e_fe_h': aspcap_e_fe_hs,
                           'aspcap_mg_h': aspcap_mg_hs, 'aspcap_e_mg_h': aspcap_e_mg_hs})

aspcap_lite_df = pd.merge(aspcap_df, lite_df, left_on='aspcap_sdss_id', right_on='sdss_id')
print(aspcap_lite_df)
print(aspcap_lite_df[['sdss_id', 'snr', 'aspcap_snr']])

aspcap_df.to_csv(path+'data/aspcap.csv', index=False)
quit()
"""
aspcap_df = pd.read_csv(path+'data/aspcap.csv')
print(aspcap_df)

# read in gyrochronological ages 
bouma = pd.read_csv(path+'data/bouma_gyro_ages.txt', sep='\s+')
# many of these stars are >4000 Myr old; drop these, which have no age estimate
try:
    bouma['e_tGyro'] = bouma['e_tGyro'].astype(int)
except ValueError:
    # Find rows that cannot be converted to int
    problematic_rows = bouma[~bouma['e_tGyro'].str.isnumeric()].index
    bouma = bouma.drop(problematic_rows)
bouma['e_tGyro'] = bouma['e_tGyro'].astype(int)
# enrich Bouma+24 with ASPCAP stellar parameter labels
bouma_aspcap = pd.merge(bouma, aspcap_df, left_on='Gaia', right_on='source_id', how='left')

lu = pd.read_csv(path+'data/lu_gyro_ages.txt', sep='\s+')
lu_aspcap = pd.merge(lu, aspcap_df, on='source_id', how='left')

bouma_cannon = pd.merge(preds, bouma, on='KIC')
lu_cannon = pd.merge(preds, lu, left_on='KIC', right_on='Kic')

yerr_bouma = np.array([bouma_cannon['E_tGyro'], bouma_cannon['e_tGyro']])
yerr_lu = np.array([lu_cannon['E_Age'], -1*lu_cannon['e_Age']])

# compare bouma vs lu
#bouma_lu = pd.merge(bouma, lu, left_on='KIC', right_on='Kic')
#print(bouma_lu)
#bouma_lu_xerr = np.array([bouma_lu['E_Age'], -1*bouma_lu['e_Age']])
#bouma_lu_yerr = np.array([bouma_lu['E_tGyro'], bouma_lu['e_tGyro']])
#plt.plot(np.arange(0, 7), np.arange(0, 7), color='k', alpha=0.5)
#plt.errorbar(bouma_lu['Age'], bouma_lu['tGyro']/1000, xerr=bouma_lu_xerr, yerr=bouma_lu_yerr/1000, linestyle='', marker='o', color='k', alpha=0.1)
#plt.xlabel('age [Gyr], Lu+24')
#plt.ylabel('age [Gyr], Bouma+24')
#plt.savefig(path+'plots/gyro_lu_vs_bouma.png')
#plt.show()

"""
plt.plot(np.arange(0, 7), np.arange(0, 7), color='k', alpha=0.5)
plt.errorbar(bouma_cannon['Age_pred'], bouma_cannon['tGyro']/1000, yerr=yerr_bouma/1000, label='Bouma+24', linestyle='', marker='o', color='steelblue')
plt.errorbar(lu_cannon['Age_pred'], lu_cannon['Age'], yerr=yerr_lu, label='Lu+24', linestyle='', marker='o', color='powderblue')
plt.xlabel(r"age [Gyr], Cannon")
plt.ylabel(r"age [Gyr], comparison")
#plt.xlim([0, 14])
#plt.ylim([0, 14])
plt.legend()
plt.savefig(path+'plots/gyro_age_compare.png')
plt.show()

residual_bouma = np.abs(bouma_cannon['Age_pred']-bouma_cannon['tGyro']/1000)
residual_lu = np.abs(lu_cannon['Age_pred']-lu_cannon['Age'])
plt.errorbar(bouma_cannon['fe_h_pred'], residual_bouma, yerr=yerr_bouma/1000, linestyle='', marker='o', color='pink', label='Bouma+24')
plt.errorbar(lu_cannon['fe_h_pred'], residual_lu, yerr=yerr_lu, linestyle='', marker='o', color='steelblue', label='Lu+24')
plt.xlabel("Fe/H")
plt.ylabel(r"|Cannon age - gyro age| [Gyr]")
#plt.xlim([0, 14])
#plt.ylim([0, 14])
plt.legend()
plt.savefig(path+'plots/gyro_age_compare_by_feh.png')
plt.show()

ax_age = plot_heatmaps(bouma_cannon['fe_h_pred'], residual_bouma, color='Oranges')
plt.xlabel("Fe/H")
plt.ylabel(r"|Cannon age - Bouma gyro age| [Gyr]")
plt.legend(bbox_to_anchor=(1., 1.05))
plt.tight_layout()
plt.savefig(path+'plots/gyro_age_compare_by_feh_heatmap_bouma.png', format='png', bbox_inches='tight')
plt.show()

ax_age = plot_heatmaps(lu_cannon['fe_h_pred'], residual_lu)
plt.xlabel("Fe/H")
plt.ylabel(r"|Cannon age - Lu gyro age| [Gyr]")
plt.legend(bbox_to_anchor=(1., 1.05))
plt.tight_layout()
plt.savefig(path+'plots/gyro_age_compare_by_feh_heatmap_lu.png', format='png', bbox_inches='tight')
plt.show()

plt.errorbar(bouma_cannon['Teff_pred'], residual_bouma, yerr=yerr_bouma/1000, linestyle='', marker='o', color='pink', label='Bouma+24')
plt.errorbar(lu_cannon['Teff_pred'], residual_lu, yerr=yerr_lu, linestyle='', marker='o', color='steelblue', label='Lu+24')
plt.xlabel("Teff [K]")
plt.ylabel(r"|Cannon age - gyro age| [Gyr]")
#plt.xlim([0, 14])
#plt.ylim([0, 14])
plt.legend()
plt.savefig(path+'plots/gyro_age_compare_by_teff.png')
plt.show()

ax_age = plot_heatmaps(bouma_cannon['Teff_pred'], residual_bouma, color='Oranges')
plt.xlabel("Teff [K]")
plt.ylabel(r"|Cannon age - Bouma gyro age| [Gyr]")
plt.legend(bbox_to_anchor=(1., 1.05))
plt.tight_layout()
plt.savefig(path+'plots/gyro_age_compare_by_teff_heatmap_bouma.png', format='png', bbox_inches='tight')
plt.show()

ax_age = plot_heatmaps(lu_cannon['Teff_pred'], residual_lu)
plt.xlabel("Teff [K]")
plt.ylabel(r"|Cannon age - Lu gyro age| [Gyr]")
plt.legend(bbox_to_anchor=(1., 1.05))
plt.tight_layout()
plt.savefig(path+'plots/gyro_age_compare_by_teff_heatmap_lu.png', format='png', bbox_inches='tight')
plt.show()

plt.errorbar(bouma_cannon['logg_pred'], residual_bouma, yerr=yerr_bouma/1000, linestyle='', marker='o', color='pink', label='Bouma+24')
plt.errorbar(lu_cannon['logg_pred'], residual_lu, yerr=yerr_lu, linestyle='', marker='o', color='steelblue', label='Lu+24')
plt.xlabel("logg")
plt.ylabel(r"|Cannon age - gyro age| [Gyr]")
#plt.xlim([0, 14])
#plt.ylim([0, 14])
plt.legend()
plt.savefig(path+'plots/gyro_age_compare_by_logg.png')
plt.show()

ax_age = plot_heatmaps(bouma_cannon['logg_pred'], residual_bouma, color='Oranges')
plt.xlabel("logg")
plt.ylabel(r"|Cannon age - Bouma gyro age| [Gyr]")
plt.legend(bbox_to_anchor=(1., 1.05))
plt.tight_layout()
plt.savefig(path+'plots/gyro_age_compare_by_logg_heatmap_bouma.png', format='png', bbox_inches='tight')
plt.show()

ax_age = plot_heatmaps(lu_cannon['logg_pred'], residual_lu)
plt.xlabel("logg")
plt.ylabel(r"|Cannon age - Lu gyro age| [Gyr]")
plt.legend(bbox_to_anchor=(1., 1.05))
plt.tight_layout()
plt.savefig(path+'plots/gyro_age_compare_by_logg_heatmap_lu.png', format='png', bbox_inches='tight')
plt.show()

preds_young = preds.loc[preds['Age_pred']<= 4]
lu_young = lu.loc[lu['Age'] <= 4]
#print(bouma.loc[bouma['tGyro']/1000 < 0.4]['tGyro']/1000)
plt.hist(preds_young['Age_pred'], bins=20, fill=False, density=True, edgecolor='black', lw=1.5, label='this work')
plt.hist(bouma['tGyro']/1000, bins=20, fill=False, density=True, edgecolor='pink', lw=1.5, label='Bouma+24')
plt.hist(lu_young['Age'], bins=20, fill=False, density=True, edgecolor='steelblue', lw=1.5, label='Lu+24')
plt.xlabel('Age [Gyr]')
#plt.xlim([0,4])
plt.legend()
plt.tight_layout()
plt.savefig(path+'plots/age_inference_kic_young.png')
plt.show()
"""

# read in Berger+20 isochrone ages 
berger = pd.read_csv(path+'data/GKSPCPapTable2_cleaned.txt', sep='&', header=0)
berger_cannon = pd.merge(preds, berger, on='KIC')
print(berger_cannon)
yerr_berger = np.array([berger_cannon['iso_age_err1'], -1*berger_cannon['iso_age_err2']])

# Bedell cross-match has the Gaia DR3 source_id we need 
bedell = pd.read_csv(path+'data/bedell_kic_apogee.csv', sep=',')
berger_bedell = pd.merge(berger, bedell, left_on='KIC', right_on='kepid', how='left')
berger_aspcap = pd.merge(berger_bedell, aspcap_df, on='source_id', how='left')

# read in Nataf+24 isochrone ages
nataf = pd.read_csv(path+'data/nataf24.txt', sep='\s+')
nataf['age'] = 10**nataf['age']/1e9
nataf_cannon = pd.merge(preds, nataf, left_on='source_id', right_on='dr3_source_id')
nataf_aspcap = pd.merge(nataf, aspcap_df, left_on='dr3_source_id', right_on='source_id', how='left')

"""
plt.plot(np.arange(0, 14), np.arange(0, 14), color='k', alpha=0.5)
plt.errorbar(berger_cannon['Age_pred'], berger_cannon['iso_age'], yerr=yerr_berger, label='Berger+20', linestyle='', marker='o', color='steelblue', alpha=0.4)
plt.xlabel(r"age [Gyr], Cannon")
plt.ylabel(r"age [Gyr], comparison")
#plt.xlim([0, 14])
#plt.ylim([0, 14])
plt.legend()
plt.savefig(path+'plots/isochrone_age_compare.png')
plt.show()

plt.plot(np.arange(0, 14), np.arange(0, 14), color='k', alpha=0.5)
plt.errorbar(berger_cannon['Age_pred'], berger_cannon['iso_age'], yerr=yerr_berger, label='Berger+20', linestyle='', marker='o', color='steelblue', alpha=0.4)
plt.xlabel(r"age [Gyr], Cannon")
plt.ylabel(r"age [Gyr], comparison")
#plt.xlim([0, 14])
#plt.ylim([0, 14])
plt.legend()
#plt.savefig(path+'plots/isochrone_age_compare.png')
plt.show()
"""

def process_trilegal(trilegal_dir_str):
    ### If I'm going to do this for several TRILEGAL models, I should functionalize this boring part

    trilegal_dir = path+'data/'+trilegal_dir_str+'/'
    trilegal_files = os.listdir(trilegal_dir)
    trilegal = pd.concat([pd.read_csv(trilegal_dir+trilegal_file, sep='\s+') for trilegal_file in trilegal_files], ignore_index=True) 

    # remove binaries
    trilegal = trilegal.loc[trilegal['m2/m1']==0.].reset_index()
    trilegal['logAge'] = trilegal['logAge'].astype(float)
    trilegal['Age'] = 10**trilegal['logAge']/1e9
    trilegal['Teff'] = 10**trilegal['logTe']

    return trilegal

# introduce TRILEGAL
trilegal = process_trilegal('trilegal')
trilegal_constant_sfr = process_trilegal('trilegal_constant_sfr')
trilegal_no_heating = process_trilegal('trilegal_no_heating')
trilegal_constant_sfr_no_heating = process_trilegal('trilegal_constant_sfr_no_heating')
print(trilegal_no_heating)
print(trilegal_constant_sfr_no_heating)
print(trilegal)

# cull comparison samples to the label space
trilegal_cull = cull(preds, trilegal, trilegal['Teff'], trilegal['logg'], trilegal['[M/H]'])
trilegal_constant_sfr_cull = cull(preds, trilegal_constant_sfr, trilegal_constant_sfr['Teff'], trilegal_constant_sfr['logg'], trilegal_constant_sfr['[M/H]'])
trilegal_no_heating_cull = cull(preds, trilegal_no_heating, trilegal_no_heating['Teff'], trilegal_no_heating['logg'], trilegal_no_heating['[M/H]'])
trilegal_constant_sfr_no_heating_cull = cull(preds, trilegal_constant_sfr_no_heating, trilegal_constant_sfr_no_heating['Teff'], trilegal_constant_sfr_no_heating['logg'], trilegal_constant_sfr_no_heating['[M/H]'])

print(nataf)
nataf_aspcap_cull = cull(preds, nataf_aspcap, nataf_aspcap['aspcap_teff'], nataf_aspcap['aspcap_logg'], nataf_aspcap['aspcap_fe_h'], nataf_aspcap['aspcap_mg_h'])
berger_aspcap_cull = cull(preds, berger_aspcap, berger_aspcap['aspcap_teff'], berger_aspcap['aspcap_logg'], berger_aspcap['aspcap_fe_h'], berger_aspcap['aspcap_mg_h'])
lu_aspcap_cull = cull(preds, lu_aspcap, lu_aspcap['aspcap_teff'], lu_aspcap['aspcap_logg'], lu_aspcap['aspcap_fe_h'], lu_aspcap['aspcap_mg_h'])
bouma_aspcap_cull = cull(preds, bouma_aspcap, bouma_aspcap['aspcap_teff'], bouma_aspcap['aspcap_logg'], bouma_aspcap['aspcap_fe_h'], bouma_aspcap['aspcap_mg_h'])
print(bouma_aspcap_cull)
print(bouma_aspcap)

#plt.hist(trilegal_no_heating_cull['Age'], fill=True, color="#C46914", edgecolor="#C46914", alpha=0.5, lw=1.5, label='TRI no heating')
#plt.show()
#quit()

bins = np.linspace(1, 8, 12) #np.linspace(0, 14, 20)
plt.hist(trilegal_cull['Age'], bins=bins, fill=True, density=True, color="#309433", edgecolor="#309433", alpha=0.5, lw=1.5, label='TRI 2-step SFR')
#plt.hist(trilegal_constant_sfr_cull['Age'], bins=bins, fill=True, density=True, color="#EB72DF", edgecolor="#EB72DF", alpha=0.5, lw=1.5, label='TRI constant SFR')
#plt.hist(trilegal_no_heating_cull['Age'], bins=bins, fill=True, density=True, color="#C46914", edgecolor="#C46914", alpha=0.5, lw=1.5, label='TRI no heating')
#plt.hist(trilegal_constant_sfr_no_heating_cull['Age'], bins=bins, fill=True, density=True, color="#8C72EB", edgecolor="#8C72EB", alpha=0.5, lw=1.5, label='TRI constant SFR, no heating')
#plt.hist(berger_aspcap_cull['iso_age'], bins=bins, fill=False, density=True, edgecolor="#EB72DF", alpha=0.5, lw=1.5, label='Berger+20')
#plt.hist(nataf_aspcap_cull['age'], bins=bins, fill=False, density=True, edgecolor="#729CEB", alpha=0.5, lw=1.5, label='Nataf+24')
plt.hist(preds['Age_pred'], bins=bins, fill=True, density=True, color='black', edgecolor='black', alpha=0.7, lw=1.5, label='this work')
plt.xlabel('Age [Gyr]')
plt.legend()
plt.tight_layout()
#plt.savefig(path+'plots/age_inference_kic.png')
plt.savefig(path+'plots/trilegal_comparison.png')
plt.show()
quit()

preds_young = preds.loc[preds['Age_pred']<= 4]
lu_young = lu_aspcap_cull.loc[lu_aspcap_cull['Age'] <= 4]
#print(bouma.loc[bouma['tGyro']/1000 < 0.4]['tGyro']/1000)
plt.hist(preds_young['Age_pred'], bins=20, fill=False, density=True, edgecolor='black', lw=1.5, label='this work')
plt.hist(bouma_aspcap_cull['tGyro']/1000, bins=20, fill=False, density=True, edgecolor='pink', lw=1.5, label='Bouma+24')
plt.hist(lu_young['Age'], bins=20, fill=False, density=True, edgecolor='steelblue', lw=1.5, label='Lu+24')
plt.xlabel('Age [Gyr]')
#plt.xlim([0,4])
plt.legend()
plt.tight_layout()
plt.savefig(path+'plots/age_inference_kic_young.png')
plt.show()
#"""

### Kiel diagram: Teff vs logg
plt.scatter(nataf_aspcap_cull['aspcap_teff'], nataf_aspcap_cull['aspcap_logg'], s=5, alpha=0.5, label='Nataf+24', color='pink')
plt.scatter(berger_aspcap_cull['aspcap_teff'], berger_aspcap_cull['aspcap_logg'], s=5, alpha=0.5, label='Berger+20', color='pink', marker='s')
plt.scatter(bouma_aspcap_cull['aspcap_teff'], bouma_aspcap_cull['aspcap_logg'], s=5, alpha=0.3, label='Bouma+24', color='purple')
plt.scatter(lu_aspcap_cull['aspcap_teff'], lu_aspcap_cull['aspcap_logg'], s=5, alpha=0.3, label='Lu+24', color='purple', marker='s')
preds_aspcap = pd.merge(preds, aspcap_df, on='source_id', how='left')
plt.scatter(preds_aspcap['aspcap_teff'], preds_aspcap['aspcap_logg'], s=5, alpha=0.3, label='this training sample', color='black')
plt.xlabel(r"$T_{\rm eff}$ [K], ASPCAP")
plt.ylabel('logg, ASPCAP')
plt.legend()
plt.savefig(path+'plots/kiel.png')
plt.show()

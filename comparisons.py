""" 
Compare predicted ages with gyrochronology (Lu+24, Bouma+24), co-moving group, kinematic (zoomies; Sagear+submitted) ages
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/chrislam/Desktop/cannon-ages/' 

preds = pd.read_csv(path+'data/preds_dnu_full.csv')

# add KIC column to preds DF
df = pd.read_csv(path+'data/enriched_lite.csv', sep=',')
kics = df.loc[df['sdss_id'].isin(preds['sdss_id'])]['KIC']
source_ids = df.loc[df['sdss_id'].isin(preds['sdss_id'])]['source_id']

preds['KIC'] = kics
preds['source_id'] = source_ids
preds = preds.dropna(subset=['KIC', 'source_id'])
preds['KIC'] = preds['KIC'].astype(int)
preds['source_id'] = preds['source_id'].astype(int)
print(preds)

"""
# open clusters
clusters = pd.read_csv(path+'data/long23-kepler-gaia-open-clusters.txt', sep='\s+')
print(clusters)
clusters_training = pd.merge(preds, clusters, left_on='source_id', right_on='Gaia')
print(clusters_training)
quit()
"""

#"""
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

lu = pd.read_csv(path+'data/lu_gyro_ages.txt', sep='\s+')
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

plt.errorbar(bouma_cannon['Teff_pred'], residual_bouma, yerr=yerr_bouma/1000, linestyle='', marker='o', color='pink', label='Bouma+24')
plt.errorbar(lu_cannon['Teff_pred'], residual_lu, yerr=yerr_lu, linestyle='', marker='o', color='steelblue', label='Lu+24')
plt.xlabel("Teff [K]")
plt.ylabel(r"|Cannon age - gyro age| [Gyr]")
#plt.xlim([0, 14])
#plt.ylim([0, 14])
plt.legend()
plt.savefig(path+'plots/gyro_age_compare_by_teff.png')
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
quit()
#"""

# read in Berger+20 isochrone ages 
berger = pd.read_csv(path+'data/GKSPCPapTable2_cleaned.txt', sep='&', header=0)

berger_cannon = pd.merge(preds, berger, on='KIC')
print(berger_cannon)

yerr_berger = np.array([berger_cannon['iso_age_err1'], -1*berger_cannon['iso_age_err2']])

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
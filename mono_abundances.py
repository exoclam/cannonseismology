""" 
Perform a test similar to Ness+15 (not the Cannon paper, the other one).
For a given mono-abundance slice (Fe/H, Mg/H), does the Cannon age follow the asteroseismic age from APOKASC? 
If yes, it shows that there is age information in your spectra beyond what you might get from correlations with Fe/H and Mg/H
"""

import pandas as pd
import numpy as np
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

serenelli = pd.read_csv(path+'data/apokasc-sdss-teff-valid.txt')
#print(serenelli)

training = pd.read_csv(path+'data/preds_dnu_full.csv') 

preds = pd.read_csv(path+'data/4_fold_cv_no_rgb.csv') # 4_fold_cv.csv, 4_fold_cv_ruwe.csv, 4_fold_cv_teff.csv, 4_fold_cv_teff_ruwe.csv, 4_fold_cv_no_rgb.csv
preds = pd.read_csv(path+'data/4_fold_cv.csv') # 4_fold_cv.csv, 4_fold_cv_ruwe.csv, 4_fold_cv_teff.csv, 4_fold_cv_teff_ruwe.csv, 4_fold_cv_no_rgb.csv
preds = preds.loc[preds['KIC']!=7976303]
preds = preds.loc[preds['KIC']!=10454113]
print(preds)

inferences = pd.read_csv(path+'data/inferences_kic_no_rgb.csv') # these are inferences from the fully-trained Cannon model on the Serenelli+17 sample, minus RGB stars
#inferences = pd.read_csv(path+'data/inferences_kic.csv') # these are inferences from the fully-trained Cannon model on the Serenelli+17 sample
preds_inferences = pd.merge(preds, inferences, on='sdss_id', how='left') # see how inferences did for the training set itself
preds_inferences['Age_pred'] = preds_inferences['Age_pred_y']
preds_inferences['mg_h'] = preds_inferences['mg_h_y']
preds_inferences['fe_h'] = preds_inferences['fe_h_y']
preds_inferences['kepid'] = preds_inferences['kepid_y']
preds_inferences['Teff_pred'] = preds_inferences['Teff_pred_y']
preds_inferences['mg_h_pred'] = preds_inferences['mg_h_pred_y']
preds_inferences['e_mg_h'] = preds_inferences['e_mg_h_y']
preds_inferences['e_fe_h'] = preds_inferences['e_fe_h_y']
preds_inferences['fe_h_pred'] = preds_inferences['fe_h_pred_y']
preds_inferences['Dnu_pred'] = preds_inferences['Dnu_pred_y']
preds_inferences['logg_pred'] = preds_inferences['logg_pred_y']
preds_inferences = preds_inferences[['sdss_id', 'kepid', 'Age', 'Age_pred', 'E_Age','e_Age','logg_aspcap','e_logg_aspcap','logg_pred','mg_h', 'mg_h_pred','e_mg_h','fe_h','e_fe_h','fe_h_pred','Teff','e_Teff','Teff_pred','Dnu','e_Dnu','Dnu_pred']]
preds_inferences = preds_inferences.dropna().reset_index()
print(preds_inferences)

#"""
### plot alpha-rich/young stars
preds_young_alpha_rich = preds_inferences.loc[(preds_inferences['Age'] <= 6) & (preds_inferences['mg_h'] >= 0.2)]
print(preds_young_alpha_rich)
modifier = '_rgb_full'

im = plt.errorbar(preds_young_alpha_rich['Teff'], preds_young_alpha_rich['Teff_pred'], xerr=preds_young_alpha_rich['e_Teff'], c='k', linestyle='', fmt='o')
min_teff = np.min(pd.concat([preds_young_alpha_rich['Teff'], preds_young_alpha_rich['Teff_pred']]))
max_teff = np.max(pd.concat([preds_young_alpha_rich['Teff'], preds_young_alpha_rich['Teff_pred']]))
plt.plot([min_teff,max_teff], [min_teff,max_teff])
plt.ylabel(r"$T_{\rm eff}$ [K], Cannon, CV")
plt.xlabel(r"$T_{\rm eff}$ [K], ASPCAP")
buffer = 25
plt.xlim([min_teff-buffer, max_teff+buffer])
plt.ylim([min_teff-buffer, max_teff+buffer])
plt.tight_layout()
plt.savefig(path+'plots/teff_young_alpha_rich'+modifier+'.png')
plt.show()

im = plt.errorbar(preds_young_alpha_rich['logg_aspcap'], preds_young_alpha_rich['logg_pred'], xerr=preds_young_alpha_rich['e_logg_aspcap'], c='k', linestyle='', fmt='o')
min_logg = np.min(pd.concat([preds_young_alpha_rich['logg_aspcap'], preds_young_alpha_rich['logg_pred']]))
max_logg = np.max(pd.concat([preds_young_alpha_rich['logg_aspcap'], preds_young_alpha_rich['logg_pred']]))
plt.plot([min_logg,max_logg], [min_logg,max_logg])
plt.ylabel(r"logg, Cannon, CV")
plt.xlabel(r"logg, ASPCAP")
buffer = 0.02
plt.xlim([min_logg-buffer, max_logg+buffer])
plt.ylim([min_logg-buffer, max_logg+buffer])
plt.tight_layout()
plt.savefig(path+'plots/logg_young_alpha_rich'+modifier+'.png')
plt.show()

im = plt.errorbar(preds_young_alpha_rich['fe_h'], preds_young_alpha_rich['fe_h_pred'], xerr=preds_young_alpha_rich['e_fe_h'], c='k', linestyle='', fmt='o')
min_feh = np.min(pd.concat([preds_young_alpha_rich['fe_h'], preds_young_alpha_rich['fe_h_pred']]))
max_feh = np.max(pd.concat([preds_young_alpha_rich['fe_h'], preds_young_alpha_rich['fe_h_pred']]))
plt.plot([min_feh,max_feh], [min_feh,max_feh])
plt.ylabel(r"[Fe/H], Cannon, CV")
plt.xlabel(r"[Fe/H], ASPCAP")
buffer = 0.01
plt.xlim([min_feh-buffer, max_feh+buffer])
plt.ylim([min_feh-buffer, max_feh+buffer])
plt.tight_layout()
plt.savefig(path+'plots/fe_h_young_alpha_rich'+modifier+'.png')
plt.show()

im = plt.errorbar(preds_young_alpha_rich['mg_h'], preds_young_alpha_rich['mg_h_pred'], xerr=preds_young_alpha_rich['e_mg_h'], c='k', linestyle='', fmt='o')
min_mg_h = np.min(pd.concat([preds_young_alpha_rich['mg_h'], preds_young_alpha_rich['mg_h_pred']]))
max_mg_h = np.max(pd.concat([preds_young_alpha_rich['mg_h'], preds_young_alpha_rich['mg_h_pred']]))
plt.plot([min_mg_h,max_mg_h], [min_mg_h,max_mg_h])
plt.ylabel(r"[Mg/H], Cannon, CV")
plt.xlabel(r"[Mg/H], ASPCAP")
buffer = 0.01
plt.xlim([min_mg_h-buffer, max_mg_h+buffer])
plt.ylim([min_mg_h-buffer, max_mg_h+buffer])
plt.tight_layout()
plt.savefig(path+'plots/mg_h_young_alpha_rich'+modifier+'.png')
plt.show()

im = plt.errorbar(preds_young_alpha_rich['Age'], preds_young_alpha_rich['Age_pred'], xerr=[preds_young_alpha_rich['E_Age'], -1*preds_young_alpha_rich['e_Age']], c='k', linestyle='', fmt='o')
min_Age = np.min(pd.concat([preds_young_alpha_rich['Age'], preds_young_alpha_rich['Age_pred']]))
max_Age = np.max(pd.concat([preds_young_alpha_rich['Age'], preds_young_alpha_rich['Age_pred']]))
plt.plot([min_Age,max_Age], [min_Age,max_Age])
plt.ylabel(r"Age [Gyr], Cannon, CV")
plt.xlabel(r"Age [Gyr], ASPCAP")
buffer = 0.1
plt.tight_layout()
plt.savefig(path+'plots/Age_young_alpha_rich'+modifier+'.png')
plt.show()

im = plt.errorbar(preds_young_alpha_rich['Dnu'], preds_young_alpha_rich['Dnu_pred'], xerr=preds_young_alpha_rich['e_Dnu'], c='k', linestyle='', fmt='o')
min_Dnu = np.min(pd.concat([preds_young_alpha_rich['Dnu'], preds_young_alpha_rich['Dnu_pred']]))
max_Dnu = np.max(pd.concat([preds_young_alpha_rich['Dnu'], preds_young_alpha_rich['Dnu_pred']]))
plt.plot([min_Dnu,max_Dnu], [min_Dnu,max_Dnu])
plt.ylabel(r'$\Delta \nu [\mu Hz]$, Cannon, CV')
plt.xlabel(r'$\Delta \nu [\mu Hz]$, ASPCAP')
buffer = 0.1
plt.tight_layout()
plt.savefig(path+'plots/Dnu_young_alpha_rich'+modifier+'.png')
plt.show()
#"""

### resume mono-abundances
#training = pd.merge(training, inferences, on='sdss_id', how='left')
#training['Age_pred'] = training['Age_pred_y']
#training = training[['sdss_id', 'kepid', 'Age_test', 'Age_pred', 'mg_h_test', 'fe_h_test']]
#training = training.dropna().reset_index()

# add KIC column to preds DF
#df = pd.read_csv(path+'data/enriched_lite_visits_chisq.csv', sep=',')
#kics = df.loc[df['sdss_id'].isin(training['sdss_id'])]['KIC']
#source_ids = df.loc[df['sdss_id'].isin(training['sdss_id'])]['source_id']
#source_id_dr2s = df.loc[df['sdss_id'].isin(training['sdss_id'])]['source_id_dr2']

#training['KIC'] = kics
#training['source_id'] = source_ids
#training['source_id_dr2'] = source_id_dr2s
#training = training.dropna(subset=['KIC', 'source_id'])
#training['KIC'] = training['KIC'].astype(int)
#training['source_id'] = training['source_id'].astype(int)
#training['source_id_dr2'] = training['source_id_dr2'].astype(int)

# crossmatch to enrich with chisq column
#training_merge = pd.merge(training, df, on='sdss_id', how='left')
#training['chisq'] = training_merge['chisq']

def plot_heatmaps(label1, label2):
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
    ax = plt.pcolormesh(xedges, yedges, hist, cmap='Blues')

    #ax.set_xlim([xedges[0], xedges[-1]])
    #ax.set_ylim([yedges[0], yedges[-1]])

    return ax

#plt.hist2d(training['fe_h_test'], training['mg_h_test'], bins=20)
#plt.xlabel('ASPCAP [Fe/H]')
#plt.ylabel('ASPCAP [Mg/H]')
#plt.legend(bbox_to_anchor=(1., 1.05))
#plt.tight_layout()
#plt.savefig(path+'plots/training_age_heatmap.png', format='png', bbox_inches='tight')
#plt.show()

#plt.hist(training['fe_h_test'])
#plt.xlabel('ASPCAP [Mg/Fe]')
#plt.show()

modifier = '_no_rgb'
def plot_mono(training_sub, lower, upper, xerr=0):
    """util function for plotting mono abundance age relation

    Args:
        training_sub (Pandas DF): training sample, selected for the mono-abundance
        lower (float): lower abundance
        upper (float): upper abundance
    """

    plt.axis('square')
    im = plt.errorbar(training_sub['Age'], training_sub['Age_pred'], xerr=xerr, yerr=training_sub['age_error'], c='k', linestyle='', fmt='o') # training_sub['chisq']
    plt.xlabel('APOKASC age [Gyr]')
    plt.ylabel('Cannon age [Gyr]')
    plt.xlim([0, 14])
    plt.ylim([0, 14])
    #plt.legend(bbox_to_anchor=(1., 1.05))
    plt.text(0.5, 13., f'{np.round(lower,1)} <= [Mg/Fe] < {np.round(upper,1)}: {len(training_sub)} stars', fontsize=15)
    #cbar = plt.colorbar(im, cmap='viridis', label=r'Cannon model $\chi^2$ fit')
    plt.tight_layout()
    if lower<0:
        plt.savefig(path+f'plots/mg_fe_mono_abundance_negative_{10*np.round(np.abs(lower),1)}'+modifier+'.png', format='png', bbox_inches='tight')
    else:
        plt.savefig(path+f'plots/mg_fe_mono_abundance_{10*np.round(lower,1)}'+modifier+'.png', format='png', bbox_inches='tight')
    plt.show()

    return

lowers = np.linspace(-0.4, 0.4, 9) # fe/h
lowers = np.linspace(-1, 3, 9) # mg/fe
preds_inferences['mg_fe'] = preds_inferences['mg_h']/preds_inferences['fe_h']
#plt.hist(preds_inferences['mg_fe'], bins=np.linspace(-1, 5, 20))
#plt.xlabel('ASPCAP [Mg/Fe]')
#plt.savefig(path+'plots/mg_fe.png')
#plt.show()

cannon_preds = pd.read_csv(path+'data/enriched_lite_visits_chisq.csv', sep=',')
cannon_preds['age_error'] = np.sqrt(cannon_preds['sigma_star_age']**2 + 0.398**2)
print(cannon_preds['age_error'])
#training['age_error'] = cannon_preds['age_error']
preds_inferences['age_error'] = np.nanmean(cannon_preds['age_error']) # this is just until I can calculate the new error for real
print(preds_inferences)

for lower in lowers:
    #upper = lower+0.1 # fe/h
    upper = lower+0.5 # mg/fe

    #if lower == 0.4: # fe/h
    if lower == 3: # mg/fe
        break
    else:
        #training_sub = training.loc[(training['fe_h_test'] >= lower) & (training['fe_h_test'] < upper)]
        training_sub = preds_inferences.loc[(preds_inferences['mg_fe'] >= lower) & (preds_inferences['mg_fe'] < upper)]

    #training_sub = pd.merge(training_sub, serenelli, left_on='kepid', right_on='KIC', how='inner')
    #print(list(training_sub.columns))
    #quit()
    #training_sub['Age'] = training_sub['Age_x']
    xerr = [training_sub['E_Age'], np.abs(training_sub['e_Age'])]
    #print(len(training_sub))
    #plt.hist2d(training_sub['Age_test'], training_sub['Age_pred'])
    plot_mono(training_sub, lower, upper, xerr)

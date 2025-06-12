# -*- coding: utf-8 -*-
"""
@author: behmardaida, 4/18/2025

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from astropy.io import fits
from astropy.table import Table

import thecannon as tc
print(tc.__version__)
from process_spectra_gaus import *
from loocv import *

path = '/Users/chrislam/Desktop/cannon-ages/' 

#"""
import matplotlib
import matplotlib.pylab as pylab
matplotlib.rcParams.update({'errorbar.capsize': 1})
pylab_params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(pylab_params)
#"""

preds = pd.read_csv(path+'data/preds_dnu_full.csv')
print(preds)

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

def compute_rms_scatter(label1, label2):

    """Compute LOOCV rms scatter

    Input:
    - label1: predicted label
    - label2: "ground truth" label

    Returns:
        float: rms
    """

    errors = np.abs(label2 - label1)
    fractional_errors = errors/label1
    squared_errors = errors**2
    mean_squared_errors = np.nanmean(squared_errors)
    rms = np.sqrt(mean_squared_errors)

    return rms, errors, fractional_errors

### inference plots: KIC
#"""
inferences_kic = pd.read_csv(path+'data/inferences_kic.csv')
print(inferences_kic)

enriched_lite_visits = pd.read_csv(path+'data/enriched_lite_visits.csv')
print(enriched_lite_visits)

color='black'
plt.errorbar(inferences_kic['Teff_pred'], inferences_kic['teff'], yerr=[inferences_kic['teff_err1'], -1*inferences_kic['teff_err2']], 
             color=color, marker='o', linestyle='', alpha=0.4)
plt.plot(inferences_kic['teff'], inferences_kic['teff'], color=color)
plt.xlabel(r"$T_{\rm eff}$ [K], Cannon")
plt.ylabel(r"$T_{\rm eff}$ [K], Gaia")
#plt.xlim([4500, 6750])
#plt.ylim([4500, 6750])
plt.tight_layout()
plt.savefig(path+'plots/teff_inference_kic.png')
plt.show()

plt.errorbar(inferences_kic['logg_pred'], inferences_kic['logg'], yerr=[inferences_kic['logg_err1'], -1*inferences_kic['logg_err2']], 
             color=color, marker='o', linestyle='', alpha=0.4)
plt.plot(inferences_kic['logg'], inferences_kic['logg'], color=color)
plt.xlabel(r"logg, Cannon")
plt.ylabel(r"logg, Gaia")
#plt.xlim([3.1, 4.4])
#plt.ylim([3.1, 4.4])
plt.tight_layout()
plt.savefig(path+'plots/logg_inference_kic.png')
plt.show()

plt.errorbar(inferences_kic['fe_h_pred'], inferences_kic['feh'], yerr=[inferences_kic['feh_err1'], -1*inferences_kic['feh_err2']], 
             color=color, marker='o', linestyle='', alpha=0.4)
plt.plot(inferences_kic['feh'], inferences_kic['feh'], color=color)
plt.xlabel(r"Fe/H, Cannon")
plt.ylabel(r"Fe/H, Gaia")
#plt.xlim([3.1, 4.4])
#plt.ylim([3.1, 4.4])
plt.tight_layout()
plt.savefig(path+'plots/feh_inference_kic.png')
plt.show()

plt.scatter(inferences_kic['mg_h_pred'], inferences_kic['mg_h'], color=color, alpha=0.4)
plt.plot(inferences_kic['mg_h'], inferences_kic['mg_h'], color=color)
plt.xlabel(r"Mg/H, Cannon")
plt.ylabel(r"Mg/H, Gaia")
#plt.xlim([3.1, 4.4])
#plt.ylim([3.1, 4.4])
plt.tight_layout()
plt.savefig(path+'plots/mg_h_inference_kic.png')
plt.show()

inferences_kic_young = inferences_kic.loc[inferences_kic['Age_pred']<=4]
plt.hist(inferences_kic['Age_pred'], density=True, alpha=0.4, color=color)
plt.xlabel('stellar age [Gyr]')
plt.ylabel('probability density')
plt.tight_layout()
plt.savefig(path+'plots/age_inference_kic.png')
plt.show()

# compare with Bouma gyro ages
bouma = pd.read_csv(path+'data/bouma_gyro_ages.txt', sep='\s+')
print(inferences_kic_young['Age_pred'])
plt.hist(inferences_kic_young['Age_pred'], density=True, alpha=0.4, color=color, label='Cannon')
#plt.hist(bouma['tGyro'], density=True, alpha=0.4, color='orange', label='Bouma+24')
plt.xlabel('stellar age [Gyr]')
plt.ylabel('probability density')
plt.legend()
plt.tight_layout()
plt.savefig(path+'plots/age_inference_kic_young.png')
plt.show()
quit()
#"""

### keep only young, alpha-rich stars
preds_young_alpha_rich = preds.loc[(preds['Age_test'] <= 6) & (preds['mg_h_test'] >= 0.2)]

### feature importance
"""
theta = pd.read_csv(path+'data/theta_arr_sum.csv')
theta = np.array(theta)
label_names=["Teff", "logg", "feh", "mg_h", "Age", "Dnu"]
wl = pd.read_csv(path+'data/wl.csv')
wl = np.array(wl['0'])
feature_importance(wl, theta, label_names)
quit()
"""

### core plots
rms_age, errors_age, fractional_errors_age = compute_rms_scatter(preds['Age_pred'], preds['Age_test'])
print("age: ", rms_age, errors_age, fractional_errors_age)
rms_teff, errors_teff, fractional_errors_teff = compute_rms_scatter(preds['Teff_pred'], preds['Teff_test'])
print("teff: ", rms_teff)
rms_logg, _, _ = compute_rms_scatter(preds['logg_pred'], preds['logg_test'])
print("logg: ", rms_logg)
rms_feh, _, _ = compute_rms_scatter(preds['fe_h_pred'], preds['fe_h_test'])
print("feH: ", rms_feh)
rms_mgh, _, _ = compute_rms_scatter(preds['mg_h_pred'], preds['mg_h_test'])
print("mg_h: ", rms_mgh)
rms_dnu, _, _ = compute_rms_scatter(preds['Dnu_pred'], preds['Dnu_test'])
print("Dnu: ", rms_dnu)

"""
plt.plot(preds['Age_pred'], fractional_errors_age)
plt.xlabel('Cannon age [Gyr]')
plt.ylabel('fractional age error')
plt.show()

plt.plot(preds['Teff_pred'], fractional_errors_teff)
plt.xlabel(r'Cannon Teff [K]')
plt.ylabel('fractional Teff error')
plt.show()
"""

#ax_age = plot_heatmaps(preds['Age_test'], preds['Age_pred'])
#plt.xlabel('APOKASC age [Gyr]')
#plt.ylabel('Cannon age [Gyr]')
#plt.legend(bbox_to_anchor=(1., 1.05))
#plt.tight_layout()
#plt.savefig(path+'plots/training_age_heatmap.png', format='png', bbox_inches='tight')
#plt.show()

color='black'

plt.scatter(preds['Teff_pred'], preds['Teff_test'], color=color)
plt.scatter(preds_young_alpha_rich['Teff_pred'], preds_young_alpha_rich['Teff_test'], color='pink', label=r'young, $\alpha$-rich')
plt.plot(preds['Teff_test'], preds['Teff_test'], color=color)
plt.xlabel(r"$T_{\rm eff}$ [K], Cannon")
plt.ylabel(r"$T_{\rm eff}$ [K], APOKASC")
plt.xlim([4500, 6750])
plt.ylim([4500, 6750])
plt.legend(loc='upper left')
#plt.savefig(path+'plots/teff_check_dnu_full.png')
plt.savefig(path+'plots/teff_check_dnu_young_alpha_rich.png')
plt.show()

plt.scatter(preds['logg_pred'], preds['logg_test'], color=color)
plt.scatter(preds_young_alpha_rich['logg_pred'], preds_young_alpha_rich['logg_test'], color='pink', label=r'young, $\alpha$-rich')
plt.plot(preds['logg_test'], preds['logg_test'], color=color)
plt.xlabel(r"logg, Cannon")
plt.ylabel(r"logg, APOKASC")
plt.xlim([3.1, 4.4])
plt.ylim([3.1, 4.4])
plt.legend(loc='upper left')
#plt.savefig(path+'plots/logg_check_dnu_full.png')
plt.savefig(path+'plots/logg_check_dnu_young_alpha_rich.png')
plt.show()

plt.scatter(preds['fe_h_pred'], preds['fe_h_test'], color=color)
plt.scatter(preds_young_alpha_rich['fe_h_pred'], preds_young_alpha_rich['fe_h_test'], color='pink', label=r'young, $\alpha$-rich')
plt.plot(preds['fe_h_test'], preds['fe_h_test'], color=color)
plt.xlabel(r"[Fe/H], Cannon")
plt.ylabel(r"[Fe/H], APOKASC")
plt.xlim([-0.7, 0.7])
plt.ylim([-0.7, 0.7])
plt.legend(loc='upper left')
#plt.savefig(path+'plots/feh_check_dnu_full.png')
plt.savefig(path+'plots/feh_check_dnu_young_alpha_rich.png')
plt.show()

plt.scatter(preds['mg_h_pred'], preds['mg_h_test'], color=color)
plt.scatter(preds_young_alpha_rich['mg_h_pred'], preds_young_alpha_rich['mg_h_test'], color='pink', label=r'young, $\alpha$-rich')
plt.plot(preds['mg_h_test'], preds['mg_h_test'], color=color)
plt.xlabel(r"[Mg/H], Cannon")
plt.ylabel(r"[Mg/H], APOKASC")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
plt.legend(loc='upper left')
#plt.savefig(path+'plots/mg_h_check_dnu_full.png')
plt.savefig(path+'plots/mg_h_check_dnu_young_alpha_rich.png')
plt.show()

plt.scatter(preds['Age_pred'], preds['Age_test'], color=color)
plt.scatter(preds_young_alpha_rich['Age_pred'], preds_young_alpha_rich['Age_test'], color='pink', label=r'young, $\alpha$-rich')
plt.plot(preds['Age_test'], preds['Age_test'], color=color)
plt.xlabel(r"age [Gyr], Cannon")
plt.ylabel(r"age [Gyr], APOKASC")
plt.xlim([0, 14])
plt.ylim([0, 14])
plt.legend(loc='upper left')
#plt.savefig(path+'plots/age_check_dnu_full.png')
plt.savefig(path+'plots/age_check_dnu_young_alpha_rich.png')
plt.show()

plt.scatter(preds['Dnu_pred'], preds['Dnu_test'], color=color)
plt.scatter(preds_young_alpha_rich['Dnu_pred'], preds_young_alpha_rich['Dnu_test'], color='pink', label=r'young, $\alpha$-rich')
plt.plot(preds['Dnu_test'], preds['Dnu_test'], color=color)
plt.xlabel(r'$\Delta \nu [\mu Hz]$, Cannon')
plt.ylabel(r'$\Delta \nu [\mu Hz]$, APOKASC')
plt.xlim([0, 160])
plt.ylim([0, 160])
plt.legend(loc='upper left')
#plt.savefig(path+'plots/Dnu_check_dnu_full.png')
plt.savefig(path+'plots/Dnu_check_dnu_young_alpha_rich.png')
plt.show()

"""
plt.scatter(preds['numax_pred'], preds['numax_test'])
plt.plot(preds['numax_test'], preds['numax_test'])
plt.xlabel(r'$\nu_{max} [\mu Hz]$, Cannon')
plt.ylabel(r'$\nu_{max} [\mu Hz]$, APOKASC')
plt.xlim([300, 3600])
plt.ylim([300, 3600])
plt.savefig(path+'plots/numax_check_dnu_numax_full.png')
plt.show()
"""
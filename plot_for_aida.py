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

import loocv

path = '/Users/chrislam/Desktop/cannon-ages/' 

preds = pd.read_csv(path+'data/preds_dnu_full.csv')
print(preds)

def compute_rms_scatter(label1, label2):

    """Compute rms scatter

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

rms_age, errors_age, fractional_errors_age = compute_rms_scatter(preds['Age_pred'], preds['Age_test'])
print(rms_age, errors_age, fractional_errors_age)
rms_teff, errors_teff, fractional_errors_teff = compute_rms_scatter(preds['Teff_pred'], preds['Teff_test'])
print(rms_teff)
rms_logg, _, _ = compute_rms_scatter(preds['logg_pred'], preds['logg_test'])
print(rms_logg)
rms_feh, _, _ = compute_rms_scatter(preds['fe_h_pred'], preds['fe_h_test'])
print(rms_feh)
rms_mgh, _, _ = compute_rms_scatter(preds['mg_h_pred'], preds['mg_h_test'])
print(rms_mgh)
rms_dnu, _, _ = compute_rms_scatter(preds['Dnu_pred'], preds['Dnu_test'])
print(rms_dnu)

plt.plot(preds['Age_pred'], fractional_errors_age)
plt.xlabel('Cannon age [Gyr]')
plt.ylabel('fractional age error')
plt.show()

plt.plot(preds['Teff_pred'], fractional_errors_teff)
plt.xlabel(r'Cannon Teff [K]')
plt.ylabel('fractional Teff error')
plt.show()

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

ax_age = plot_heatmaps(preds['Age_test'], preds['Age_pred'])
plt.xlabel('APOKASC age [Gyr]')
plt.ylabel('Cannon age [Gyr]')
#plt.legend(bbox_to_anchor=(1., 1.05))
plt.tight_layout()
#plt.savefig(path+'plots/trilegal/kepmag_vs_cdpp.png', format='png', bbox_inches='tight')
plt.show()

plt.scatter(preds['Age_pred'], preds['Age_test'])
plt.plot(preds['Age_test'], preds['Age_test'])
plt.xlabel(r"age [Gyr], pred")
plt.ylabel(r"age [Gyr], test")
#plt.xlim([0, 14])
#plt.ylim([0, 14])
plt.show()
quit()

plt.scatter(preds['Teff_pred'], preds['Teff_test'])
plt.plot(preds['Teff_test'], preds['Teff_test'])
plt.xlabel(r"$T_{\rm eff}$ [K], pred")
plt.ylabel(r"$T_{\rm eff}$ [K], test")
plt.xlim([4500, 6750])
plt.ylim([4500, 6750])
#plt.legend()
plt.savefig(path+'plots/teff_check_dnu_full.png')
plt.show()

plt.scatter(preds['logg_pred'], preds['logg_test'])
plt.plot(preds['logg_test'], preds['logg_test'])
plt.xlabel(r"logg, pred")
plt.ylabel(r"logg, test")
plt.xlim([3.1, 4.4])
plt.ylim([3.1, 4.4])
plt.savefig(path+'plots/logg_check_dnu_full.png')
plt.show()

plt.scatter(preds['fe_h_pred'], preds['fe_h_test'])
plt.plot(preds['fe_h_test'], preds['fe_h_test'])
plt.xlabel(r"[Fe/H], pred")
plt.ylabel(r"[Fe/H], test")
plt.xlim([-0.7, 0.7])
plt.ylim([-0.7, 0.7])
plt.savefig(path+'plots/feh_check_dnu_full.png')
plt.show()

plt.scatter(preds['mg_h_pred'], preds['mg_h_test'])
plt.plot(preds['mg_h_test'], preds['mg_h_test'])
plt.xlabel(r"[Mg/H], pred")
plt.ylabel(r"[Mg/H], test")
plt.xlim([-0.6, 0.5])
plt.ylim([-0.6, 0.5])
plt.savefig(path+'plots/mg_h_check_dnu_full.png')
plt.show()

plt.scatter(preds['Age_pred'], preds['Age_test'])
plt.plot(preds['Age_test'], preds['Age_test'])
plt.xlabel(r"age [Gyr], pred")
plt.ylabel(r"age [Gyr], test")
plt.xlim([0, 14])
plt.ylim([0, 14])
plt.savefig(path+'plots/age_check_dnu_full.png')
plt.show()

plt.scatter(preds['Dnu_pred'], preds['Dnu_test'])
plt.plot(preds['Dnu_test'], preds['Dnu_test'])
plt.xlabel(r'$\Delta \nu [\mu Hz]$, pred')
plt.ylabel(r'$\Delta \nu [\mu Hz]$, test')
plt.xlim([0, 160])
plt.ylim([0, 160])
plt.savefig(path+'plots/Dnu_check_dnu_full.png')
plt.show()

"""
plt.scatter(preds['numax_pred'], preds['numax_test'])
plt.plot(preds['numax_test'], preds['numax_test'])
plt.xlabel(r'$\nu_{max} [\mu Hz]$, pred')
plt.ylabel(r'$\nu_{max} [\mu Hz]$, test')
plt.xlim([300, 3600])
plt.ylim([300, 3600])
plt.savefig(path+'plots/numax_check_dnu_numax_full.png')
plt.show()
"""
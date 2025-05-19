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
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
print(training)

# add KIC column to preds DF
df = pd.read_csv(path+'data/enriched_lite_visits.csv', sep=',')
kics = df.loc[df['sdss_id'].isin(training['sdss_id'])]['KIC']
source_ids = df.loc[df['sdss_id'].isin(training['sdss_id'])]['source_id']
source_id_dr2s = df.loc[df['sdss_id'].isin(training['sdss_id'])]['source_id_dr2']

training['KIC'] = kics
training['source_id'] = source_ids
training['source_id_dr2'] = source_id_dr2s
training = training.dropna(subset=['KIC', 'source_id'])
training['KIC'] = training['KIC'].astype(int)
training['source_id'] = training['source_id'].astype(int)
training['source_id_dr2'] = training['source_id_dr2'].astype(int)

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

def plot_mono(training_sub, lower, upper, xerr=0):
    """util function for plotting mono abundance age relation

    Args:
        training_sub (Pandas DF): training sample, selected for the mono-abundance
        lower (float): lower abundance
        upper (float): upper abundance
    """

    plt.errorbar(training_sub['Age_test'], training_sub['Age_pred'], xerr=xerr, c='k', linestyle='', fmt='o')
    plt.xlabel('APOKASC age [Gyr]')
    plt.ylabel('Cannon age [Gyr]')
    plt.xlim([0, 14])
    plt.ylim([0, 14])
    #plt.legend(bbox_to_anchor=(1., 1.05))
    plt.text(0.5, 13.5, f'{np.round(lower,1)} <= [Mg/Fe] < {np.round(upper,1)}: {len(training_sub)} stars')
    plt.tight_layout()
    if lower<0:
        plt.savefig(path+f'plots/mg_fe_mono_abundance_negative_{10*np.round(np.abs(lower),1)}.png', format='png', bbox_inches='tight')
    else:
        plt.savefig(path+f'plots/mg_femono_abundance_{10*np.round(lower,1)}.png', format='png', bbox_inches='tight')
    plt.show()

    return

lowers = np.linspace(-0.4, 0.4, 9) # fe/h
lowers = np.linspace(-1, 3, 9) # mg/fe
training['mg_fe_test'] = training['mg_h_test']/training['fe_h_test']
plt.hist(training['mg_fe_test'], bins=np.linspace(-1, 5, 20))
plt.xlabel('ASPCAP [Mg/Fe]')
plt.savefig(path+'plots/mg_fe_test.png')
plt.show()

for lower in lowers:
    #upper = lower+0.1 # fe/h
    upper = lower+0.5 # mg/fe

    #if lower == 0.4: # fe/h
    if lower == 3: # mg/fe
        break
    else:
        #training_sub = training.loc[(training['fe_h_test'] >= lower) & (training['fe_h_test'] < upper)]
        training_sub = training.loc[(training['mg_fe_test'] >= lower) & (training['mg_fe_test'] < upper)]

    training_sub = pd.merge(training_sub, serenelli, on='KIC', how='inner')
    xerr = [training_sub['E_Age'], np.abs(training_sub['e_Age'])]

    #print(len(training_sub))
    #plt.hist2d(training_sub['Age_test'], training_sub['Age_pred'])
    plot_mono(training_sub, lower, upper, xerr)

import numpy as np 
import pandas as pd 
import thecannon as tc
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil

path = '/Users/chrislam/Desktop/cannon-ages/' 
path = '/home/c.lam/blue/cannon-ages/'

import matplotlib.pylab as pylab
import matplotlib
"""
matplotlib.rcParams.update({'errorbar.capsize': 1})
pylab_params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(pylab_params)
"""

def loocv(df, wl, fluxes, ivars, label_names=["Teff", "logg", "feh", "mg_h", "Age", "Dnu", "numax"]):
    """
    Leave one out cross-validation:
    For each data point, make it the test and rest into training. 

    Inputs:
    - df: label DataFrame
    - wl: wavelength support
    - fluxes: normalized fluxes
    - ivars: inverse variances

    Output: 
    - test_labels_arr: literally one row of a label
    - true_labels_arr: corresponding APOKASC/Gaia label
    - model: The Cannon model object
    - s2_arr: s_lambda (Ness+15 Eqn 4) s squared array
    """

    # Specify the labels that we will use to construct this model.
    #label_names = ["Teff", "logg", "feh"]
    
    fluxes = np.array(fluxes)
    ivars = np.array(ivars)

    test_labels_arr = []
    true_labels_arr = []
    theta_arr = []
    cov_arr = []
    s2_arr = []
    spec_fit_chisq_arr = []
    temp_length = 5
    #count = 0
    for i in tqdm(range(len(df))):
    #for i in tqdm(range(temp_length)):
        # split between training and test label sets. Train on all but one data point
        df_test = df.iloc[i]
        df_tr = df.drop(i)

        # training set
        try:
            #flux_tr = np.concatenate((fluxes[:i+1], fluxes[i+1:]))
            flux_tr = np.delete(fluxes,i,axis=0) 
        except:
            flux_tr = fluxes[:-1]
        try:
            #ivar_tr = np.concatenate((ivars[:i+1], ivars[i+1:]))
            ivar_tr = np.delete(ivars,i,axis=0) 
        except:
            ivar_tr = ivars[:-1]

        flux_tr = np.array(flux_tr)
        ivar_tr = np.array(ivar_tr)

        Teff_tr = np.array(df_tr[label_names[0]].values)
        logg_tr = np.array(df_tr[label_names[1]].values)
        fe_h_tr = np.array(df_tr[label_names[2]].values)
        mg_h_tr = np.array(df_tr[label_names[3]].values)
        Age_tr = np.array(df_tr[label_names[4]].values)
        if len(label_names) == 7:
            Dnu_tr = np.array(df_tr[label_names[5]].values)
            numax_tr = np.array(df_tr[label_names[6]].values)
            labels_tr = np.vstack((Teff_tr,logg_tr,fe_h_tr,mg_h_tr,Age_tr,Dnu_tr,numax_tr)).T
        elif len(label_names) == 6:
            Dnu_tr = np.array(df_tr[label_names[5]].values)
            labels_tr = np.vstack((Teff_tr,logg_tr,fe_h_tr,mg_h_tr,Age_tr,Dnu_tr)).T
        elif len(label_names) == 5:
            labels_tr = np.vstack((Teff_tr,logg_tr,fe_h_tr,mg_h_tr,Age_tr)).T

        # test set
        flux_test = fluxes[i]
        ivar_test = ivars[i]
        flux_test = np.array(flux_test)
        ivar_test = np.array(ivar_test)

        Teff_test = np.array(df_test[label_names[0]])
        logg_test = np.array(df_test[label_names[1]])
        fe_h_test = np.array(df_test[label_names[2]])
        mg_h_test = np.array(df_test[label_names[3]])
        Age_test = np.array(df_test[label_names[4]])
        if len(label_names) == 7:
            Dnu_test = np.array(df_test[label_names[5]]) # or 6
            numax_test = np.array(df_test[label_names[6]])
            labels_test = np.vstack((Teff_test,logg_test,fe_h_test,mg_h_test,Age_test,Dnu_test,numax_test)).T
        elif len(label_names) == 6:
            Dnu_test = np.array(df_test[label_names[5]]) 
            labels_test = np.vstack((Teff_test,logg_test,fe_h_test,mg_h_test,Age_test,Dnu_test)).T
        elif len(label_names) == 5:
            labels_test = np.vstack((Teff_test,logg_test,fe_h_test,mg_h_test,Age_test)).T
        #print(labels_test)
        true_labels_arr.append(labels_test)

        """
        vectorizer=tc.vectorizer.PolynomialVectorizer(label_names, 2)
        print(vectorizer.label_names)
        print(len(vectorizer.label_names))
        print(labels_tr.shape[1])
        if isinstance(labels_tr, np.ndarray):
            print("aaaaa")
        if labels_tr.shape[0] == flux_tr.shape[0]:
            print("bbbbbbb")
        if labels_tr.shape[1] == len(vectorizer.label_names):
            print("wheeeeeeee")
        else:
            print("booo")
        quit()
        """
        print(flux_tr.shape)
        print(ivar_tr.shape)
        print(labels_tr.shape)
        # Construct a CannonModel object using a quadratic (O=2) polynomial vectorizer. No wait, linear should be much faster. But it was bad.
        model = tc.CannonModel(
            labels_tr, flux_tr, ivar_tr, dispersion=wl, # needed to set dispersion explicitly
            vectorizer=tc.vectorizer.PolynomialVectorizer(label_names, 2)) 
        #print(model.vectorizer.human_readable_label_vector)

        # training step
        theta, s2, metadata = model.train(threads=1)
        s2_arr.append(s2)
        theta_arr.append(theta)

        # inspect coefficients and run feature importance
        print(theta.shape)
        print(len(theta[:,0]))
        #feature_importance(theta, label_names)
        #fig_theta = tc.plot.theta(model)

        # test step
        test_labels, cov_val, metadata_val = model.test(flux_test, ivar_test)
        print("test, cov, metadata: ", test_labels, cov_val, metadata_val)
        test_labels_arr.append(test_labels)

        # use cov to propagate per-star, per-visit uncertainty 
        matrix = np.zeros((len(cov_val),len(label_names))) # Pre-allocate matrix
        for i in range(0,len(cov_val)):
            matrix[i,:] = np.sqrt(np.diag(cov_val[i]))
        cov_arr.append(matrix)

        df_sigma = pd.DataFrame(matrix)
        #df_sigma.to_csv(path+'data/test_output_sigmaA.csv',index=False)

        # get Cannon-derived model spectra
        model_spectrum = model(test_labels) 
        # chisq of model spectral fit
        spec_fit_chisq = np.sum(((model_spectrum-flux_test)**2)/(ivar_test**-1 + s2))
        spec_fit_chisq_arr.append(spec_fit_chisq)

        #Teff_pred = test_labels[:,0]
        #logg_pred = test_labels[:,1]
        #fe_h_pred = test_labels[:,2]
        #mg_h_pred = test_labels[:,3]
        #Age_pred = test_labels[:,4]
        #numax_pred = test_labels[:,5]
        #Dnu_pred = test_labels[:,6]

    theta_arr_sum = np.sum(theta_arr, axis=0)
    theta_arr_sum = pd.DataFrame(theta_arr_sum)
    print(theta_arr_sum)
    #theta_arr_sum.to_csv(path+'data/theta_arr_sum.csv', index=False)

    print(cov_arr)
    cov_arr = np.array(cov_arr).reshape(len(cov_arr), 6)
    print(cov_arr)
    cov_df = pd.DataFrame(cov_arr)
    #cov_df.to_csv(path+'data/sigma_A.csv', index=False)

    return test_labels_arr, true_labels_arr, model, s2_arr, spec_fit_chisq_arr


def create_filenames_from_ids(ids, prefix, suffix):
    """Creates a list of filenames from a list of IDs. This is a demo from Google AI. Scary.

    Args:
        ids: A list of IDs.
        prefix: A string to prepend to each ID.
        suffix: A string to append to each ID.

    Returns:
        A list of filenames.
    """
    return [f"{prefix}{id}{suffix}" for id in ids]

def copy_files(filenames, source_dir, destination_dir):
    """Copies files from a source directory to a destination directory. This is a demo from Google AI. Scary.

    Args:
        filenames: A list of filenames to copy.
        source_dir: The path to the source directory.
        destination_dir: The path to the destination directory.
    """
    for filename in filenames:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        try:
            shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
            print(f"Copied {filename} to {destination_dir}")
        except FileNotFoundError:
            print(f"File not found: {filename}")

def feature_importance(wl, theta, label_names):
    """Plot feature importance

    Args:
        wl (1D np array of floats): wavelength range
        theta (1D np array of floats): theta summed over all LOOCV stars
        label_names (list of strings): determines number of subplots to generate for a quadratic model
    
    Terms allow us to split a giant subplot sensibly. 
    eg. if len(label_names)==6:
    then we need
    - 1 scalar
    - 6 linear
    - 6 squared
    - n-1 triangular number, ie. 5+4+3+2+1=15 cross terms
    For a total of 28 features. That's gonna need more than one plot. 
    """

    indices = []
    # terms (string): 'linear' means scalar and linear terms; 'squared' means squared terms; 'cross' means cross terms
    # linear terms
    fig, axes = plt.subplots(len(label_names)+1, layout="constrained")
    axes = np.array([axes]).flatten()
    names = ['1','$T_{eff}$', 'logg', 'Fe/H', 'Mg/H', 'Age', '$\Delta {\\nu}$']

    for index in range(len(label_names)+1):
        y = theta[:,index]
        scale = np.max(np.abs(y))
        axes[index].plot(wl, y/scale, color='k')
        axes[index].set_ylabel(names[index])

        indices.append(index)

    plt.xlabel('Pixel')
    plt.tight_layout()
    plt.savefig(path+'plots/theta_linear.pdf')
    plt.show()

    # squared terms
    fig, axes = plt.subplots(len(label_names), layout="constrained")
    axes = np.array([axes]).flatten()
    names = ['$T_{eff}^2$', '$logg^2$', '$Fe/H^2$', '$Mg/H^2$', '$Age^2$', '$\Delta {\\nu}^2$']
    
    cross_term_gap = len(label_names) # adjust to the way model.vectorizer.human_readable_label_vector outputs features
    index = len(label_names)+1 # pattern: squared, plus 6, squared, plus 5, squared, plus 4, etc.
    for i in range(len(label_names)):
        print(index)
        y = theta[:,index]
        scale = np.max(np.abs(y))
        axes[i].plot(wl, y/scale, color='k')
        axes[i].set_ylabel(names[i])

        indices.append(index)
        index += cross_term_gap
        cross_term_gap += -1
    
    plt.xlabel('Pixel')
    plt.tight_layout()
    plt.savefig(path+'plots/theta_squared.pdf')
    plt.show()

    # cross terms
    def triangle(n):
        return int(n*(n+1)/2)
    
    fig, axes = plt.subplots(triangle(len(label_names)-1), layout="constrained", figsize=(10, 8))
    axes = np.array([axes]).flatten()
    names = ['$T_{eff}$*logg', '$T_{eff}$*Fe/H', '$T_{eff}$*Mg/H', '$T_{eff}$*Age', '$T_{eff}$*$\Delta {\\nu}^2$',
                'Fe/H*logg','logg*Mg/H','Age*logg','$\Delta {\\nu}^2$*logg',
                'Fe/H*Mg/H', 'Age*Fe/H', '$\Delta {\\nu}^2$*Fe/H',
                'Age*Mg/H', '$\Delta {\\nu}^2$*Mg/H',
                'Age*$\Delta {\\nu}^2$']
    
    cross_term_gap = len(label_names) - 2
    index = len(label_names)+2
    total_indices = np.arange(len(label_names)+1+len(label_names)+triangle(len(label_names)-1)-1)
    indices_remaining = np.setdiff1d(total_indices, indices)
    for enum_i, i in enumerate(indices_remaining):
        if cross_term_gap != 0:
            y = theta[:,i]
            scale = np.max(np.abs(y))
            axes[enum_i].plot(wl, y/scale, color='k')
            axes[enum_i].set_ylabel(names[enum_i], rotation=0)
    
    plt.xlabel('Pixel')
    plt.tight_layout()
    plt.savefig(path+'plots/theta_cross.pdf')
    plt.show()
    
    return

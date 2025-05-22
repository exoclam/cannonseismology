import numpy as np 
import pandas as pd 
import thecannon as tc
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil

path = '/Users/chrislam/Desktop/cannon-ages/' 

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
    - s2_arr: s_lambda (Ness+15 Eqn 4) squared array
    """

    # Specify the labels that we will use to construct this model.
    #label_names = ["Teff", "logg", "feh"]
    
    fluxes = np.array(fluxes)
    ivars = np.array(ivars)

    test_labels_arr = []
    true_labels_arr = []
    s2_arr = []
    temp_length = 5
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

        # training step
        theta, s2, metadata = model.train(threads=1)
        s2_arr.append(s2)

        # inspect coefficients
        #print(theta.shape)
        #print(len(theta[:,0]))
        """
        fig, axes = plt.subplots(len(label_names)+1)
        axes = np.array([axes]).flatten()
        y = theta[:,0]
        scale = np.max(np.abs(y))
        axes[0].plot(wl, y/scale, color='k')
        axes[0].set_ylabel('1')

        y = theta[:,1]
        scale = np.max(np.abs(y))
        axes[1].plot(wl, y/scale, color='k')
        axes[1].set_ylabel(r'$T_{eff}$')

        y = theta[:,2]
        scale = np.max(np.abs(y))
        axes[2].plot(wl, y/scale, color='k')
        axes[2].set_ylabel(r'logg')

        y = theta[:,3]
        scale = np.max(np.abs(y))
        axes[3].plot(wl, y/scale, color='k')
        axes[3].set_ylabel(r'Fe/H')

        y = theta[:,4]
        scale = np.max(np.abs(y))
        axes[4].plot(wl, y/scale, color='k')
        axes[4].set_ylabel(r'Mg/H')

        y = theta[:,5]
        scale = np.max(np.abs(y))
        axes[5].plot(wl, y/scale, color='k')
        axes[5].set_ylabel(r'Age')

        y = theta[:,6]
        scale = np.max(np.abs(y))
        axes[6].plot(wl, y/scale, color='k')
        axes[6].set_ylabel(r'$\nu_{max}$')

        y = theta[:,7]
        scale = np.max(np.abs(y))
        axes[7].plot(wl, y/scale, color='k')
        axes[7].set_ylabel(r'$\Delta_{\nu}$')

        plt.xlabel('Pixel')
        plt.tight_layout()
        plt.savefig(path+'plots/theta_small.png')
        plt.show()
        """
        # test step
        test_labels, cov_val, metadata_val = model.test(flux_test, ivar_test)
        #print(test_labels, cov_val, metadata_val)
        test_labels_arr.append(test_labels)

        # get Cannon-derived model spectra
        model_spectra = model(test_labels) ## this errors out? 

        #Teff_pred = test_labels[:,0]
        #logg_pred = test_labels[:,1]
        #fe_h_pred = test_labels[:,2]
        #mg_h_pred = test_labels[:,3]
        #Age_pred = test_labels[:,4]
        #numax_pred = test_labels[:,5]
        #Dnu_pred = test_labels[:,6]

    return test_labels_arr, true_labels_arr, model, s2_arr


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
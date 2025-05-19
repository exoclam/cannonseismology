# -*- coding: utf-8 -*-
"""
@author: behmardaida, 8/14/2024

- library of functions for normalizing SDSS-V spectra
- Only applies Gaussian normalization to non-pseudo-continuum normalized spectra
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import os
import re

from astropy.io import fits
#from sortedcontainers import SortedList
from scipy import interpolate
#from heapq import nsmallest

import thecannon as tc
from thecannon import continuum



def _get_pixmask(flux,flux_errs):
    """ From Anna H:
    Create and return a bad pixel mask for an APOGEE spectrum

    Bad pixels are defined as follows: fluxes or errors are not finite, or 
    reported errors are <= 0, or fluxes are 0

    Parameters
    ----------
    fluxes: ndarray
        Flux data values 

    flux_errs: ndarray
        Flux uncertainty data values 

    Returns
    -------
    mask: ndarray
        Bad pixel mask, value of True corresponds to bad pixels
    """
    bad_flux = np.logical_or(~np.isfinite(flux), flux == 0)
    bad_err = (~np.isfinite(flux_errs)) | (flux_errs <= 0)
    bad_pix = bad_err | bad_flux
    return bad_pix


def _weight(wav0,wav,L):
    """Gaussian Weights as a Func. of Wavelength
    Args:
        wav0 (float): starting wavelength point
        wav (array): wavelength array
        L (int): width of Gaussian - should be broader than typical absorption lines
    Returns:
        array: Gaussian weight
    """ 
    return np.exp(-0.5*(wav0-wav)**2/L**2)



def _gauss_norm(flux,flux_err,ivar,wl,L):
    """Normalized Flux via Gaussian Filter
    Args:
        flux (array): flux array
        flux_err (array): flux err array
        wl (array): wavelength array
        L (int): width of Gaussian
    Returns:
        array: Gaussian weight
    """ 

    # removes gaps between APOGEE chips
    take1 = np.logical_and(wl > 15150, wl < 15790)
    take2 = np.logical_and(wl > 15870, wl < 16420)
    take3 = np.logical_and(wl > 16490, wl < 16940)
    takeit = np.logical_or(np.logical_or(take1, take2), take3)

    flux_no_gaps = flux[takeit]
    flux_err_no_gaps = flux_err[takeit]
    ivar_no_gaps = ivar[takeit]
    wl_no_gaps = wl[takeit]

    # wavelength range of HIRES spectra
    length = len(flux_no_gaps)
    
    # initialize flux normalization array
    norms = np.zeros(length)

    # ith pixel index 
    for i in range(length):
        w = _weight(wl_no_gaps[i],wl_no_gaps,L)

        num = (flux_no_gaps*w*ivar_no_gaps).sum()
        denom = (ivar_no_gaps*w).sum()
        norm = num/denom

        # fill flux normalization array
        norms[i] = norm

    norm_flux = flux_no_gaps/norms
    norm_flux_err = flux_err_no_gaps/norms

    # make flux_err very large at bad pixels
    npixels = len(norm_flux)
    badpix = _get_pixmask(norm_flux, norm_flux_err)
    ivar = np.full(npixels, 1/999**2)
    ivar[~badpix] = 1. / norm_flux_err[~badpix]**2

    print("spectrum normalized!")

    return wl_no_gaps,norm_flux,ivar



def _gauss_norm_chisq(flux,flux_err,ivar,wl,L):
    """Normalized Flux via Gaussian Filter
    Args:
        flux (array): flux array
        flux_err (array): flux err array
        wl (array): wavelength array
        L (int): width of Gaussian
    Returns:
        array: Gaussian weight
    """ 

    take1 = np.logical_and(wl > 15150, wl < 15790)
    take2 = np.logical_and(wl > 15870, wl < 16420)
    take3 = np.logical_and(wl > 16490, wl < 16940)
    takeit = np.logical_or(np.logical_or(take1, take2), take3)

    flux_no_gaps = flux[takeit]
    flux_err_no_gaps = flux_err[takeit]
    ivar_no_gaps = ivar[takeit]
    wl_no_gaps = wl[takeit]

    # wavelength range of HIRES spectra
    length = len(flux_no_gaps)
    
    # initialize flux normalization array
    norms = np.zeros(length)

    # ith pixel index 
    for i in range(length):
        w = _weight(wl_no_gaps[i],wl_no_gaps,L)

        num = (flux_no_gaps*w*ivar_no_gaps).sum()
        denom = (ivar_no_gaps*w).sum()
        norm = num/denom

        # fill flux normalization array
        norms[i] = norm

    norm_flux = flux_no_gaps/norms
    norm_flux_err = flux_err_no_gaps/norms

    # adjust unrealistically low flux errs (mainly an issue with SNR>200 targets), specifically for chisq
    norm_flux_err[norm_flux_err<=0.005] = 0.005

    # make flux_err very large at bad pixels
    npixels = len(norm_flux)
    badpix = _get_pixmask(norm_flux, norm_flux_err)
    ivar = np.full(npixels, 1/999**2)
    ivar[~badpix] = 1. / norm_flux_err[~badpix]**2

    print("spectrum normalized!")

    return wl_no_gaps,norm_flux,ivar



def process_spectra(file_path,L):
    """Combine all functions to process SDSS-V spectra 
    Args:
        file_path (string): path to SDSS spectrum .fits file
        L (int): width of Gaussian - should be broader than typical absorption lines
    Returns:
        array: Gaussian weight
    """ 

    # read in file
    hdu_list = fits.open(file_path)

    if hdu_list[3].data:
        flux = np.array(hdu_list[3].data['flux'][0])
        wl = np.array(hdu_list[3].data['wavelength'][0])
        ivar = np.array(hdu_list[3].data['ivar'][0])
        flags = np.array(hdu_list[3].data['pixel_flags'][0])

    if hdu_list[4].data:
        flux = np.array(hdu_list[4].data['flux'][0])
        wl = np.array(hdu_list[4].data['wavelength'][0])
        ivar = np.array(hdu_list[4].data['ivar'][0])
        flags = np.array(hdu_list[4].data['pixel_flags'][0])

    flux_err = 1/np.sqrt(ivar)

    # divide spectrum by Gaussian-smoothed version of itself to remove large-scale shape
    wl,norm_flux,ivar = _gauss_norm(flux,flux_err,ivar,wl,L)

    # mask out where sky subtraction failed
    sky_mask = np.logical_or(norm_flux<0.4, norm_flux>1.2)
    norm_flux[sky_mask] = np.median(norm_flux)

    return wl,norm_flux,ivar



def process_spectra_chisq(file_path,L):
    """Combine all functions to process spectra specifically for chisq calculation
    Args:
        file_path (string): path to SDSS spectrum .fits file
        L (int): width of Gaussian - should be broader than typical absorption lines
    Returns:
        array: Gaussian weight
    """ 

    # read in file
    hdu_list = fits.open(file_path)

    if hdu_list[3].data:
        flux = np.array(hdu_list[3].data['flux'][0])
        wl = np.array(hdu_list[3].data['wavelength'][0])
        ivar = np.array(hdu_list[3].data['ivar'][0])
        continuum = np.array(hdu_list[3].data['continuum'][0])
        flags = np.array(hdu_list[3].data['pixel_flags'][0])

    if hdu_list[4].data:
        flux = np.array(hdu_list[4].data['flux'][0])
        wl = np.array(hdu_list[4].data['wavelength'][0])
        ivar = np.array(hdu_list[4].data['ivar'][0])
        continuum = np.array(hdu_list[4].data['continuum'][0])
        flags = np.array(hdu_list[4].data['pixel_flags'][0])

    flux_err = 1/np.sqrt(ivar)

    # divide spectrum by Gaussian-smoothed version of itself to remove large-scale shape
    wl,norm_flux,ivar = _gauss_norm_chisq(flux,flux_err,ivar,wl,L)

    # mask out where sky subtraction failed
    sky_mask = np.logical_or(norm_flux<0.4, norm_flux>1.2)
    norm_flux[sky_mask] = np.median(norm_flux)

    return wl,norm_flux,ivar


def get_files_in_order(directory,search_strings):
    # Initialize an empty list to store the matched files
    matched_files = []

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the filename contains any of the search strings
        for search_string in search_strings:
            if search_string in filename:
                matched_files.append(filename)
                break  # Move to the next file once a match is found

    # Sort the matched files by the order of the search strings
    ordered_files = sorted(matched_files, key=lambda x: next((i for i, search_string in enumerate(search_strings) if search_string in x), len(search_strings)))

    paths=[]
    for file in ordered_files:
        path = directory+file
        paths.append(path)

    return paths


def get_number_between(text, start_string, end_string):
    pattern = re.escape(start_string) + r"(\d+)" + re.escape(end_string)
    match = re.search(pattern, text)
    if match:
        return int(match.group(1))
    return None
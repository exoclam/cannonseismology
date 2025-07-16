### Cannonseismology

Train The Cannon on stars with asteroseismic ages to infer more precise ages for other Kepler (and, eventually, PLATO) stars.

- for_aida.py: aka main; this is where I run loocv.py and train The Cannon; this is also where I will run inference on KIC and PIC
- plot_for_aida.py: after reading out results from HiPerGator (HPG), make plots here
- loocv.py: find loocv.py() here
- cannon-ages.ipynb: the original notebook, where I do my mise en place (prepping and crossmatching Serenelli, Bedell, APOKASC, etc); see build_training_set.py for minimum viable product 
- process_spectra_gaus.py: utils from Aida Behmard, eg. to continuum normalize and otherwise process spectra
- get_spectra.py: functions for building inference set and then querying inference spectra from this APOGEE-KIC crossmatch
- build_training_set.py: streamlined, barebones script to do the main job of cannon-ages.ipynb w/o all the checks and plots
- train_serenelli.py: small test script for troubleshooting LOOCV and model training on a smaller subset of stars
- comparisons.py: compare predicted Cannon ages with gyrochronology, isochrone, etc ages for those same stars
- uncertainties.py: calculate per-star errorbars
- inference_kic.py: conduct inference using the trained model on KIC-APOGEE stars in the training label space
- chisq_training.py: calculate spectra chisq for training sample. 

Here's the order of operations for reproducing our results and figures in Lam, Behmard, et al., in prep. 
- cannon-ages.ipynb: prep training sample.
- crossmatch.ipynb: enrich with RUWE to omit binaries
- for_aida.py: I never changed this file's name and now it's too late lol. train model. output LOOCV predictions and s2 scatter array. 
- plot_for_aida.py: plot Fig 1 (label space histograms)
- inference_kic.py: infer ages for KIC-APOGEE-label space cross match. 
- inference_legacy.py: infer ages for the Silva Aguirre+17 APOKASC Legacy sample.
- mono_abundances.py: 
- chisq_training.py: compute spectra chisq for training sample. plot Figs 3 & 4 (LOOCV results, with and w/o old, alpha-poor stars) 
- uncertainties_chisq.py: this has been repurposed several times. first it was used to identify 500 appropriate stars. that took a while. then to calculate chisq and sigmas for those 500 stars. that took a smaller while. 
- 
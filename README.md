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

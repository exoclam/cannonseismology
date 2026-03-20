### Cannonseismology

Train The Cannon on stars with asteroseismic ages to infer more precise ages for a much larger sample of Kepler (and, eventually, PLATO) dwarfs with APOGEE spectra.

Here's the order of operations for reproducing our results and figures in Lam, Behmard, et al., in prep. 
- cannon-ages.ipynb: explore the training set and their APOGEE spectra. More of a learning document than anything else.
- crossmatch.ipynb: enrich with RUWE to omit binaries
- four_fold_cv.py: do 4-fold cross-validation. Also plot results in Fig 1, including young, alpha-rich pink circles.
- mono_abundances.py: do mono-abundance test (Fig 2)
- silver_and_gold.ipynb: select gold and silver inference samples from Kepler-APOGEE cross-match
- training.py: actually train the final model on inference sample
- uncertainties_no_rgb.py: identify 500 appropriate stars with 2 visits and SNR between 200 and 600 (see Behmard+25a). Then calculate chisq and sigma_inflates for each label per star.  
- inference_kic.py: infer ages using trained Cannon model. Enrich inference dataset with columns for label errors, using sigma_inflates from previous step and each star's model uncertainty. Compare with LEGACY. 

Other potentially relevant files:
- process_spectra_gaus.py: helper functions for continuum normalization and chisq calculations here, courtesy of Aida Behmard
- for_aida.py: previous main file, in which I re-purposed Aida Behmard's code used for Behmard+25a for this project. 

Computation time: all codes can be run on a laptop on ~hour timescales except for two steps:
- Initial cross-match of training and inference Kepler-APOGEE samples and querying of APOGEE spectra (~1 day on 8 cores).
- SDSS MWM search for APOGEE spectra of 500 Kepler stars that fulfill our chisq sigma_inflate caclulation requirements (~2 days on 8 cores). 
These were done on UF's HiPerGator via array jobs.


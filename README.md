# PA-MI-LSD
This is code for photoacoustic oximetry by multiple illumination learned spectral decoloring.
TODO give link to published paper

## General notes
Always set the correct path in each .ipynb or .py script were ``drive = "/SETDRIVEPATH/data_lsd_trip/"`` or `"SETPATH"` is specified.
The paper data is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4549631.svg)](https://doi.org/10.5281/zenodo.4549631)

## Reproducing results and figures
8GB RAM should suffice.
To reproduce figure "Absorption coefficient spectra." run all cells in ``absorbtion_figures.ipynb``
To reproduce figure "Optical properties of the uncolored phantom background medium." run all cells in ``tcspc_figures.ipynb``
To reproduce figure "Example illustrations of L1 normalized spectra [...]" and "Comparison between a validation phantom (for svf = 0) and its digital twin [...]" run all cells in ``plot_qualitative_example_spectra.ipynb``

### Reproducing from trained models and preprocessed test data
Warning: more than 20GB RAM is needed to load the MI-LSD random forest model.
To reproduce all other result figures, table 1 and results from trained models run all cells in ``test_decoloring_all_rCu.ipynb``.
### Train yourself
Warning: a CUDA capable GPU and an up and running CUDA is needed to train these FFNN models
The code for training RF and NN estimators is provided in ``train_decoloring_models_rCu.ipynb``.
### Processing raw PA data yourself
Warning: will take long to compute on CPU and you have to build your own MITK (see mitk.org)
Use ``batch_preprocess_raw_PA_data.ipynb`` to generate beamforming batch script for MITK. Beamforming and bandpass options are set within the ``beamforming.options.xml``, and ``bandpass.bmode.options.xml`` config files.
### MCX simulations
Warning: full resimulation will take very long (around 500 days) to compute on a single SOA GPU. Better implement on HPC. The simulations also use the ippai libary for data management. The library is available upon reasonable request.
The in silico data sets were simulated with ``python SET03_mc_sim.py id`` on a HPC cluster. The settings for preliminary simulations are given in ``SET01_mc_sim.py`` and ``SET02_mc_sim.py``. The training data arrays (and in silico test data arrays) were generated from the simulation using ``generate_training_data_array.py``.

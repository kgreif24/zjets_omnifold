
# Z + jets Omnifold: A package for applying Omnifold to ATLAS data

This package implements the Omnifold algorithm for the specific use case of unfolding $Z(\rightarrow \mu\mu)$ + jets events in the ATLAS experiment's full run 2 dataset.
Compute resources are assumed to be the perlmutter supercomputer at NERSC.
Extensions to accommodate other sorts of data and resources certainly possible!

## Quick start

Assuming you have all of the required software installed (see below), activate your python environment then do the following to run Omnifold:

```
# First make a default yaml config file
cd cli
python of_config.py
mv default_of_template.yml my_of_config.yml

# Then run Omnifold
cd ..
python run_omnifold.py --config_path ./cli/my_of_config.yml
```

This will run Omnifold interactively with the default settings.
Warning this will take a **very** long time without decent resources.
A submit script for running the procedure via SLURM on the perlmutter supercomputer is at `run_of.sh`.
Make sure you set the correct project code and point the script to your config before submitting!

## Environment setup on Perlmutter

First set up a conda release on Perlmutter.
The NERSC documentation has instructions on how to do this.
Once you have a base environment, the recommended environment setup method is to install all remaining software via pip.
See `requirements.txt`.
Once these packages are installed, you can run the unfolding procedure.
Note the C++ fastjet routines require installation of fastjet and ROOT, which are not included in the python environment.
See the README in the `fastjet` sub-directory for instructions on running this code.

## I just want to plot things

Fair enough! A repository of the weights derived with Omnifold is available on Perlmutter at `$CFS/m3246/ZjetOmnifold/weights/`. In this directory you'll find one directory per generation of Omnifold results. Inside these directories are .npz files which are the output of `ensemble_weights.py`. These files contain many sets of weights, which can all be applied to the MC test dataset (see below). These weights are:

* Central (name contains tag `-central`) and ensemble (name contains a number) weights for each weight group produced by Omnifold
* Weight groups will at a minimum contain the nominal results, but may also contain results for a one of the many systematic variations that produce the uncertainties on the final results.

The parent directory also contains a convenient link `best_weights` which will always point to the current SOTA weights.

These weights can be used as inputs to the plotting routines. For example `run_standalone_plots.py` uses the weights `nominal-ensemble-central` to make plots of the method bias.
`run_uncert_plots.py` will use all weights in the file to also make plots of systematic uncertainties.
Running these scripts is the right place to start when analyzing Omnifold results.

## Omnifold datasets

The most up-to-date datasets for use in Omnifold are listed below. They are all in the location `$CFS/m3246/ZjetOmnifold/data/slimmed_files_v4/` on Perlmutter. These are also the defaults in `of_config.py`.

| Dataset | Location |
|----------|----------|
|   Pretrain source   | ZjetOmnifold_May19_MGPy8FxFx_WithTracks_slim_Systematics_Pretrain_shuffled.root |
|   Pretrain target   | ZjetOmnifold_Mar10_Sherpa2211_WithTracks_slim_Systematics_Pretrain_shuffled.root |
|   Train MC   | ZjetOmnifold_5Jul2025_MGPy8FxFxPlusNonStrong_syst_Train_shuffled.root |
|   Test MC   |   ZjetOmnifold_5Jul2025_MGPy8FxFxPlusNonStrong_syst_Test_shuffled.root |
|   Pseudodata | Pseudodata_SherpaDY_PowhegPythiaTop_June2025_shuffled.root |
|   Truth Pseudodata | TruthPseudodata_Sherpa2211DY_Dibo_EW_PowhegPythiaTop_PosWeights_WithTracks_shuffled.root |
|   Data  | ZjetOmnifold_Nov11_data_WithTracks_slim_Systematics_shuffled.root |


Note the pseudodata is not separated into training and testing sets. This is because we do not use the network outputs over the pseudodata, so it is not essential to completely avoid overtraining on the pseudodata. The code does construct a held-out validation set for tracking overtraining, but the pseudodata used in the testing set is the concatenation of the validation and training sets. For this reason the loss values / performance metrics calculated on the testing set should be taken with a grain of salt. This of course has no impact on the performance of Omnifold.

## Omnifold config files

Omnifold has many hyperparameters. All of these, and other options for runnign the procedure, are controlled via yaml config files. Instructions for generating a template config are in the quickstart above. **Important**: Any time you add an option to anywhere in the code, please remember to update the script `cli/of_config.py` so that future templates have the correct options.

An overview of the important options in the config files is as follows (last updated 22/05/2024):

* `debug`: This option runs Omnifold only on the muon kinematics with a very simple NN architecture. Useful for testing.
* `interactive`: Prints progress bars during NN training
* `s1_pretrain_directory` and `s2_pretrain_directory`: Directories containing the pre-trained checkpoints for step 1 and step 2 trainings. Can leave one or both as `null` to train from scratch.
* Data paths: Use these fields to point code to the correct ROOT files.
* `split_seed`: Set this to a positive integer to fix the seeds used to produce the train / val splits in the code. If -1, a random seed will be chosen for each training.
* `max_tracks`: The maximum # of tracks to consider per event. The default value is 264 to cut out a few very high multiplicity events which exhaust CUDA memory when included in a batch.
* `syst_kw`: A keyword for activation a given systematic uncertainty. See `./utils/data_utils.py` for more.
* LR scheduler arguments: The code uses a cosine annealing with warmup learning rate scheduler for all trainings. These settings adjust the LR scheduler for the pretraining, step 1, and step 2 trainings. In practice these are the most important hyper-parameters to tune by far!
* `wandb`: Whether to use weights and biases for logging information about the trainings
* `project_name`: The name of the project this run will be logged under. Can use one project name for many runs of Omnifold
* `group_name`: The name of the group this run will be logged under. If ensembling, the ensemble number will be added on the end of this group name to distinguish different runs of Omnifold within an ensemble.
* `checkpoint_dir`: This is the directory in which all outputs of Omnifold will be placed. The code will build the repository structure described below within this directory. On perlmutter it's recommended to make this a symlink to a large volume drive, given the NN checkpoints produced can be large.
* Network hyperparameters: Everything below this is a hyperparameter to be used for each Omnifold Transformer. Default values are what I've found tend to work best on the Z+jets data.

## Ensembling

We will need to perform a lot of ensembling to get good results from Omnifold.
Currently the code only supports parallel ensembling, where many independent runs of Omnifold are run in parallel to each other, and then their results are aggregated only at the end.
Running an ensemble is as simple as adjusting the job array settings in `run_of.sh` and passing the job array index to the `run_omnifold.py` script.

## Weights and biases logging

Training many neural networks is complicated. Luckily there's some great software available that lets you visualize what is happening with each training run. For this repository we use Weights and Biases. You can create an academic account at https://wandb.ai/site, then you can either create your own team or join my team and spy on my models :).

Weights and biases does the following things for you:

* Tracks train / val losses, learning rate, etc. with iteration. This helps you spot issues quickly.
* Logs performance metrics (e.g. the wasserstein distance between reweighted truth MC and truth pseudodata) for each run so you can compare various runs
* Logs plots so you can easily visualize the quality of a reweighting and run of Omnifold as a whole

To turn this on, set `wandb: True` in the config file and adjust the project and group names accordingly.



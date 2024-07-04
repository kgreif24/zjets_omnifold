
# Z + jets Omnifold: A package for applying Omnifold to ATLAS data

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

This will run Omnifold interactively with the default settings. Warning this takes awhile! In practice, it's best to run it using slurm. Assuming you have a computing account on Perlmutter, first make a yaml config file as above and modify the `run_of.sh` script. Make sure you modify the last line to point to your config file, versus `default_of_template.yml`. Then just do `sbatch run_of.sh`.

The `run_of.sh` script requires you to set a NERSC project code. We have access to the following two project codes:

* Generic ATLAS: m2616
* Ben's project: m3246

All of our files are stored in Ben's project space (`$CFS/m3246`) but in general it's best to use the ATLAS code unless there is some limitation.

## Environment setup on Perlmutter

First set up a conda release on Perlmutter. The NERSC documentation has instructions on how to do this. Once you have a base environment, you can build an environment for running Omnifold with the command `conda env create --name envname --file=environment.yml` run from this directory.

This will install all of the software you need save one github repository that we use for learning rate scheduling during tagger training. If you want to run a training, you'll also need to run this command after activating your environment:

```
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```

## I just want to plot things

Fair enough! A repository of the weights derived with Omnifold is available on Perlmutter at `$CFS/m3246/ZjetOmnifold/weights/`. In this directory you'll find one directory per run of Omnifold. Inside these directories are .npz files which store the following information for each iteration and step:

* `train`: The current weight for each event in the MC training set. This includes fakes and inefficiencies.
* `test`: As above, but for the MC testing set.
* `network_train`: The network weights for all events that pass selection in the MC training set. Selection could be either the reco or truth level selections, depending on the step of Omnifold (step 1 == reco, step 2 == truth).
* `network_test`: As above, but for the MC testing set.

Note we don't store weights for the pseudodata. The parent directory also contains a convenient link `best_weights` which will always point to the best **nominal** weights derived thus far.

All of these weights can be used as inputs to the plotting scripts I have developed thus far. For example `run_comp_plots.py` produces plots comparing the result of any step 2 to the truth level pseudodata. MRs improving the number of things we can plot are very welcome!

## Omnifold datasets

The most up-to-date datasets for use in Omnifold are listed below. They are all in the location `/eos/user/m/mbsmith/Omnifold_Data/slimmedSamples/TestTrainSplits/syst/sliced/`. They are also copied to `$CFS/m3246/ZjetOmnifold/data/slimmed_files/` for convenience.

| Dataset | Location |
|----------|----------|
|   Train MC   | WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train_Mar1023.root |
|   Test MC   |   WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test_Mar0723.root |
|   Pseudodata | WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Apr8_1_All.root |
|   Truth Pseudodata | WithTracks_TruthPseudodata_Combined_1-18.root |

Note the pseudodata is not separated into training and testing sets. This is because we do not use the network outputs over the pseudodata, so it is not essential to completely avoid overtraining on the pseudodata. The code does construct a held-out validation set for tracking overtraining, but the pseudodata used in the testing set is the concatenation of the validation and training sets. For this reason the loss values / performance metrics calculated on the testing set should be taken with a grain of salt. This of course has no impact on the performance of Omnifold.

## Omnifold config files

Omnifold has many hyperparameters. All of these, and other options for runnign the procedure, are controlled via yaml config files. Instructions for generating a template config are in the quickstart above. **Important**: Any time you add an option to anywhere in the code, please remember to update the script `cli/of_config.py` so that future templates have the correct options.

An overview of the important options in the config files is as follows (last updated 22/05/2024):

* debug: This option runs Omnifold only on the muon kinematics with a very simple NN architecture. Useful for testing.
* num_iterations: Sets the number of iterations to run
* Data paths: Should be set to the datasets listed above
* split_seed: This random seed controls how the train / validation sets are created. If continuing a run (see below), make sure this matches what was used previously!
* max_tracks: The maximum # of tracks to consider per event. Currently all of the data with all of the tracks does not fit in memory :(
* Max train events: Sets the number of events to include in the training sets for step one / step two
* wandb: Whether to use weights and biases for logging information about the trainings (see below)
* project_name: The name of the project this run will be logged under. Can use one project name for many runs of Omnifold
* group_name: The name of the group this run will be logged under. This should be unique for each run of Omnifold, but really only matters if you are logging to W&B.
* checkpoint_dir: This is the directory in which all weights, models, and plots will be stored
* plot_val: If set to true, make reweighting plots using the validation set
* Network hyperparameters: Everything below this is a hyperparameter to be used for each Omnifold Transformer. I (Kevin) will try to keep these defaults updated as I learn more about what parameters work well.

## Weights and biases logging

Training many neural networks is complicated. Luckily there's some great software available that lets you visualize what is happening with each training run. For this repository we use Weights and Biases. You can create an academic account at https://wandb.ai/site, then you can either create your own team or join my team and spy on my models :).

Weights and biases does the following things for you:

* Tracks train / val losses, learning rate, etc. with iteration. This helps you spot issues quickly.
* Logs performance metrics (e.g. the wasserstein distance between reweighted truth MC and truth pseudodata) for each run so you can compare various runs
* Logs plots so you can easily visualize the quality of a reweighting and run of Omnifold as a whole

To turn this on, set `wandb: True` in the config file and adjust the project and group names accordingly.

## Code overview

In development. If there is interest in me filling this in from the analysis team I can do so!



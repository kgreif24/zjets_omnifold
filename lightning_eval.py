""" lightning_eval.py - This program defines the OfEval class, which is responsible for
running evaluation and prediciton for Omnifold classifiers. It produces the weights
derived from each training, which are written to disk.

Meant to be run as a subprocesss from the Omnifolder class.

Author: Kevin Greif
Last updated 03/08/2024
python3
"""

import os
import argparse
import numpy as np
import uproot
import awkward as ak
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from lightning_module import *
import plotting_utils as pu
import data_utils as du
from cli.of_config import OfConfig
from wasserstein_metric import WassersteinOne


class OfEval:
    """ OfEval - This class handles the evaluation and prediction for an
    Omnifold classifier. It is run by the driver function below, which
    is meant to be called as a subprocess from the Omnifolder class.
    """

    def __init__(self, check_path, run_id, config_path, iteration, step):
        """ __init__ - The init function for this class. It takes the OfConfig object
        used for this run of Omnifold, plus the iteration and step of this evaluation.

        Arguments:
        check_path - The path to the checkpoint to evaluate
        run_id - The ID of the run for evaluation
        config_path - The path of the of config file
        iteration - The iteration number for this training
        step - The step number for this training

        Returns:
        None
        """

        # Store the configuration
        self.config = OfConfig(config_name=config_path)
        self.run_id = run_id
        self.iteration = iteration
        self.step = step

        # Make directories for storing plots and weights
        self.test_dir = f'./{self.config.checkpoint_dir}/{self.config.project_name}/{self.run_id}/test_plots'
        os.makedirs(self.test_dir, exist_ok=True)
        self.comp_dir = f'./{self.config.checkpoint_dir}/{self.config.project_name}/{self.run_id}/comp_plots'
        os.makedirs(self.comp_dir, exist_ok=True)
        self.weight_dir = f'./{self.config.checkpoint_dir}/{self.config.project_name}/{self.config.group_name}/weights'
        os.makedirs(self.weight_dir, exist_ok=True)

        # Find the data and weight files to use for this iteration and step
        # For step one:
        if self.step == 1:
            use_truth = False
            train_source_file = self.config.mc_train_path
            test_source_file = self.config.mc_test_path
            train_target_file = self.config.data_path
            test_target_file = self.config.data_path
            # If this is the first iteration, use the weights from the root file for source 
            # and no weights for the target
            if self.iteration == 0:
                source_weight_file = 'root'
                target_weight_file = None
            # Otherwise use the weights from the previous step two for the source, and no
            # weights for the target
            else:
                source_weight_file = f"{self.weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                target_weight_file = None
        # For step two:
        if self.step == 2:
            use_truth = True
            train_source_file = self.config.mc_train_path
            test_source_file = self.config.mc_test_path
            train_target_file = self.config.mc_train_path
            test_target_file = self.config.mc_test_path
            # If this is the first iteration, use the weights from step one for target, and the
            # weights from the root file as source.
            if self.iteration == 0:
                source_weight_file = 'root'
                target_weight_file = f"{self.weight_dir}/iteration_{self.iteration}_step_1.npz"
            # Otherwise use the weights from the previous step two for source, and the weights
            # from the previous step one for target
            else:
                source_weight_file = f"{self.weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                target_weight_file = f"{self.weight_dir}/iteration_{self.iteration}_step_1.npz"


        # Build a data module. We want to run prediction on every event
        # we have, so need to define two data modules, one for the training / val
        # set and one for the test set. Both of these will be in testing mode
        self.d_module_train = LOfData(
            source_file=train_source_file,
            target_file=train_target_file,
            source_weight_path=source_weight_file,
            target_weight_path=target_weight_file,
            muon_only=self.config.debug,
            batch_size=self.config.test_batch_size,
            split_seed=self.config.split_seed,
            testing=True,
            max_tracks=self.config.max_tracks,
            use_truth=use_truth
        )
        self.d_module_test = LOfData(
            source_file=test_source_file,
            target_file=test_target_file,
            source_weight_path=source_weight_file,
            target_weight_path=target_weight_file,
            muon_only=self.config.debug,
            batch_size=self.config.test_batch_size,
            split_seed=self.config.split_seed,
            testing=True,
            max_tracks=self.config.max_tracks,
            use_truth=use_truth
        )

        # Initialise the wandb logger
        if self.config.wandb:

            run_name = f"iteration_{self.iteration}_step_{self.step}"
            self.wandb_logger = WandbLogger(
                project=self.config.project_name, 
                group=self.config.group_name,
                name=run_name,
                save_dir=self.config.checkpoint_dir,
                id=self.run_id,
                resume="must"
            )

            # Get run ID
            self.run_id = self.wandb_logger.experiment.id

        # Else we use no logger
        else:
            self.wandb_logger = None
            # Set a dummy run ID
            self.run_id = "test_run"

        # Load model checkpoint
        self.model = LOfTransformer.load_from_checkpoint(
            check_path,
            test_plots=self.test_dir,
            debug=self.config.debug
        )

        # Make lightning trainer for testing
        self.trainer = L.Trainer(
            accelerator='gpu', 
            devices=1,
            logger=self.wandb_logger,
            enable_progress_bar=False
        )

        # Make wasserstein metric object for comparing derived reweighting
        # to truth level pseudo data
        self.wasserstein = WassersteinOne(draw_plots=True, save_location=self.comp_dir)

    
    def run_testing(self):
        """ run_testing - Run testing over the test data module.
        The point here is to get performance metrics (AUC and test loss)

        No arguments or returns
        """

        self.trainer.test(self.model, self.d_module_test)


    def run_prediction(self):
        """ run_prediction - Run predictions over every data point.
        Then we calculate the derived weights, and write weights to a
        .npz file.

        No arguments or returns
        """

        # Run predictions
        predictions_train = self.trainer.predict(self.model, self.d_module_train)
        predictions_test = self.trainer.predict(self.model, self.d_module_test)

        # Send predictions to CPU, convert to numpy, and concatenate
        predictions_train = np.concatenate([pred.cpu().numpy().flatten() for pred in predictions_train])
        predictions_test = np.concatenate([pred.cpu().numpy().flatten() for pred in predictions_test])

        # Get labels from data loaders
        labels_train = self.d_module_train.get_labels()
        labels_test = self.d_module_test.get_labels()

        # Drop data predictions
        mc_predictions_train = predictions_train[labels_train == 0]
        mc_predictions_test = predictions_test[labels_test == 0]

        # Calculate derived weights
        probs_train = 1 / (1 + np.exp(-mc_predictions_train))
        probs_test = 1 / (1 + np.exp(-mc_predictions_test))
        derived_weights_train = probs_train / (1 - probs_train)
        derived_weights_test = probs_test / (1 - probs_test)

        # Get starting weights from the data modules
        start_weights_train = self.d_module_train.get_source_weights()
        start_weights_test = self.d_module_test.get_source_weights()

        # Update weights
        self.new_weights_train = start_weights_train * derived_weights_train
        self.new_weights_test = start_weights_test * derived_weights_test

        # Save new weights for future use
        np.savez(
            f"{self.weight_dir}/iteration_{self.iteration}_step_{self.step}.npz",
            train=self.new_weights_train,
            test=self.new_weights_test
        )

        # Evaluate difference between reweighted truth MC and truth data if this is step 2
        if self.step == 2:
            self.compare()


    def compare(self):
        """ compare - Compare the reweighted truth MC to the truth pseudodata. Will draw all relevant plots,
        calculate the wasserstein metric, and upload all results to wandb using the WassersteinOne class.

        No arguments or returns
        """

        # Since this is step 2, we already have truth level MC in train data loader.
        # We still need to load the truth level pseudodata
        f_pd = uproot.open(self.config.data_path)
        tree_pd = f_pd["OmniTree"]

        # Get the truth level pseudodata cuts
        filter_pd = ak.to_numpy(tree_pd["pass190"].array())

        # Get the plot data
        plotting_pd = du.get_plotting(tree_pd, vars=pu.default_settings.keys(), filter=filter_pd, get_truth=True)

        # Get the track kinematics
        kinematics_pd = du.get_kinematics(tree_pd, filter=filter_pd, get_mask=False, one_hot=False, get_truth=True, max_tracks=self.config.max_tracks)
        
        # Get the same quantities for the truth level MC from data module
        plotting_train = self.d_module_train.get_plotting()
        kinematics_train = self.d_module_train.get_track_kinematics()
        labels_train = self.d_module_train.get_labels()
        plotting_mc = plotting_train[labels_train == 0,...]
        kinematics_mc = kinematics_train[labels_train == 0,...]

        # Concatenate the truth level MC and truth level pseudodata
        plotting = np.concatenate([plotting_mc, plotting_pd], axis=0)
        kinematics = np.concatenate([kinematics_mc, kinematics_pd], axis=0)

        # Make labels
        labels_mc = np.zeros(plotting_mc.shape[0])
        labels_pd = np.ones(plotting_pd.shape[0])
        labels = np.concatenate([labels_mc, labels_pd], axis=0)

        # Make start weights
        root_weights_mc = self.d_module_train.get_source_root_weights()
        weights_pd = np.ones(plotting_pd.shape[0])
        start_weights = np.concatenate([root_weights_mc, weights_pd], axis=0)

        # Make end weights
        end_weights = np.concatenate([self.new_weights_train, weights_pd], axis=0)

        # Update and compute metric
        self.wasserstein.update(plotting, start_weights, end_weights, labels)
        comp_wass, plot_dict = self.wasserstein.compute()
        print("Reweighted truth MC to truth PD Wasserstein metric:", comp_wass)

        # Log wasserstein metric and plots if we are using wandb
        if self.config.wandb:
            self.wandb_logger.experiment.log({"comp_wasserstein": comp_wass})
            for key, histpath in plot_dict.items():
                log_name = f"comp_{key}"
                self.wandb_logger.experiment.log({log_name: wandb.Image(histpath)})

        # Reset metric
        self.wasserstein.reset()



    def run(self):
        """ run - This function runs the evaluation routine for an omnifold classifier

        No Arguments or Returns
        """

        print("Run testing")
        self.run_testing()
        print("Run predictions")
        self.run_prediction()


############## MAIN FUNCTION ##############
        
# This function will be called as a subprocess from the Omnifolder class
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the omnifold evaluation')
    parser.add_argument('--check_path', type=str, default=None, help='Path to the checkpoint to evaluate')
    parser.add_argument('--run_id', type=str, default=None, help='ID of the run for evaluation')
    parser.add_argument('--config_path', type=str, default=None, help='Path to the configuration file')
    parser.add_argument('--iteration', type=int, default=None, help='The iteration number for this training run')
    parser.add_argument('--step', type=int, default=None, help='The step number for this training run')
    args, _ = parser.parse_known_args()

    # Run the evaluation
    evaluator = OfEval(
        check_path=args.check_path, 
        run_id=args.run_id, 
        config_path=args.config_path, 
        iteration=args.iteration, 
        step=args.step
    )
    evaluator.run()
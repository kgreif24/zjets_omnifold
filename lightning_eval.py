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
import lightning as L
from pytorch_lightning.loggers import WandbLogger

from lightning_module import *
import plotting_utils as pu
from cli.of_config import OfConfig


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

        # Make test plot and weight directories
        self.test_dir = f'./{self.config.checkpoint_dir}/{self.config.project_name}/{self.run_id}/test_plots'
        os.makedirs(self.test_dir, exist_ok=True)
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
            mc_file=test_source_file,
            data_file=test_target_file,
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
        start_weights_train = self.d_module_train.get_mc_weights()
        start_weights_test = self.d_module_test.get_mc_weights()

        # Update weights
        new_weights_train = start_weights_train * derived_weights_train
        new_weights_test = start_weights_test * derived_weights_test

        # Save new weights for future use
        np.savez(
            f"{self.weight_dir}/iteration_{self.iteration}_step_{self.step}.npz",
            train=new_weights_train,
            test=new_weights_test
        )


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
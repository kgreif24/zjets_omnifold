""" lightning_eval.py - This program defines the OfEval class, which is responsible for
running evaluation and prediciton for Omnifold classifiers. It produces the weights
derived from each training, which are written to disk.

Meant to be run as a subprocesss from the Omnifolder class.

Author: Kevin Greif
Last updated 11/20/2024
python3
"""

import os
import time
import argparse
import signal

import numpy as np
import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment
import wandb
from pytorch_lightning.loggers import WandbLogger

from lightning_module import LOfTransformer
from lightning_data_module import LOfData
from plotter import Plotter
from cli.of_config import OfConfig
from wasserstein_metric import WassersteinOne


class OfEval:
    """OfEval - This class handles the evaluation and prediction for an
    Omnifold classifier. It is run by the driver function below, which
    is meant to be called as a subprocess from the Omnifolder class.
    """

    def __init__(
        self,
        config_path,
        iteration,
        step,
        check_path=None,
        verify=False,
        store=None,
        index=-1,
        unit_test=False,
    ):
        """__init__ - The init function for this class. It takes the OfConfig object
        used for this run of Omnifold, plus the iteration and step of this evaluation.

        Arguments:
        config_path - The path of the of config file
        iteration - The iteration number for this training
        step - The step number for this training
        check_path - Defaults None, the path to the checkpoint to evaluate
            If left none, use the best model symlink
        verify - Defaults False, if set to true forget about testing and just run
            prediction.
        store - Defaults None, if set, store weights here instead of in the default
            location
        index - Defaults -1, the index of the ensemble to run. Add this number to the
            end of the group ID if it is not -1
        unit_test - If true, trainer will just run a few steps of evaluation and exit

        Returns:
        None
        """

        # ------------------ Config and directory setup ------------------

        # Store the configuration
        self.config = OfConfig(config_name=config_path)
        self.iteration = iteration
        self.step = step
        self.verify = verify
        self.unit_test = unit_test

        # Modify the group name of an index is provided
        if index != -1:
            self.config.group_name = f"{self.config.group_name}_{index}"

        # Get the run name
        if self.iteration == 0:
            # Typically we don't run evaluation for pre-trainings
            self.run_name = f"pretrain_step_{self.step}"
        else:
            self.run_name = f"iteration_{self.iteration}_step_{self.step}"

        # Hard code the number of tracks to use in building inclusive track plots
        self.max_tracks = 150

        # Make directories for storing plots and weights
        root_dir = (
            f"{self.config.checkpoint_dir}/{self.config.project_name}/"
            f"{self.config.group_name}"
        )
        self.checkpoint_dir = f"{root_dir}/{self.run_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.test_dir = f"{root_dir}/{self.run_name}/test_plots"
        os.makedirs(self.test_dir, exist_ok=True)
        self.comp_dir = f"{root_dir}/{self.run_name}/comp_plots"
        os.makedirs(self.comp_dir, exist_ok=True)
        self.weight_dir = f"{root_dir}/weights"
        os.makedirs(self.weight_dir, exist_ok=True)

        # Set the checkpoint path if not provided
        if check_path is None:
            check_path = f"{self.checkpoint_dir}/best_model.ckpt"
        print(f"Evaluating with checkpoint: {check_path}")

        # Change the save location if the store argument is set
        if store is not None:
            self.save_dir = store
        else:
            self.save_dir = self.weight_dir

        # ----------------- Lightning setup -----------------

        # Load model checkpoint
        self.model = LOfTransformer.load_from_checkpoint(
            check_path,
            debug=self.config.debug,
            step=self.step,
        )

        # Initialise the wandb logger
        if self.config.wandb:
            self.wandb_logger = WandbLogger(
                project=self.config.project_name,
                group=self.config.group_name,
                name=self.run_name,
                save_dir=self.checkpoint_dir,
                id=self.model.run_id,
                resume="allow",
                settings=wandb.Settings(init_timeout=90),
            )
        else:
            self.wandb_logger = None

        # Make lightning trainer for testing
        self.trainer = L.Trainer(
            accelerator="auto" if (self.config.debug or unit_test) else "gpu",
            num_nodes=1,
            devices=1,
            logger=self.wandb_logger,
            plugins=[SLURMEnvironment(auto_requeue=False)],
            enable_progress_bar=self.config.interactive,
            fast_dev_run=unit_test,
            use_distributed_sampler=False,
        )

        # Make wasserstein metric object for comparing derived reweighting
        # to truth level pseudo data
        self.wasserstein = WassersteinOne()

        # ---------------------- Data setup ----------------------

        # Find the data and weight files to use for this iteration and step
        # For step one:
        if self.step == 1:
            use_truth = False
            # If this is pre-training (iteration 0), use the MC train file and
            # Sherpa file
            if self.iteration == 0:
                train_source_file = None
                test_source_file = self.config.pretrain_source_path
                train_target_file = None
                test_target_file = self.config.pretrain_target_path
                source_weight_file = "root"
                target_weight_file = "root"
            # If this is the first iteration, use the weights from the root file
            # for source and no weights for the target
            elif self.iteration == 1:
                train_source_file = self.config.mc_train_path
                test_source_file = self.config.mc_test_path
                train_target_file = None
                test_target_file = self.config.data_path
                source_weight_file = "root"
                target_weight_file = None
            # Otherwise use the weights from the previous step two for the source,
            # and no weights for the target
            else:
                train_source_file = self.config.mc_train_path
                test_source_file = self.config.mc_test_path
                train_target_file = None
                test_target_file = self.config.data_path
                source_weight_file = (
                    f"{self.weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                )
                target_weight_file = None
        # For step two:
        if self.step == 2:
            use_truth = True
            # If this is pre-training (iteration 0), use the MC train file and
            # Sherpa file
            if self.iteration == 0:
                train_source_file = self.config.mc_train_path
                test_source_file = self.config.mc_test_path
                train_target_file = None
                test_target_file = self.config.pretrain_path
                source_weight_file = "root"
                target_weight_file = "root"
            # If this is the first iteration, use the weights from step one for target,
            # and the weights from the root file as source.
            elif self.iteration == 1:
                train_source_file = self.config.mc_train_path
                test_source_file = self.config.mc_test_path
                train_target_file = None
                test_target_file = self.config.mc_test_path
                source_weight_file = "root"
                target_weight_file = (
                    f"{self.weight_dir}/iteration_{self.iteration}_step_1.npz"
                )
            # Otherwise use weights from previous step 2 for source, and the weights
            # from the previous step one for target
            else:
                train_source_file = self.config.mc_train_path
                test_source_file = self.config.mc_test_path
                train_target_file = None
                test_target_file = self.config.mc_test_path
                source_weight_file = (
                    f"{self.weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                )
                target_weight_file = (
                    f"{self.weight_dir}/iteration_{self.iteration}_step_1.npz"
                )

        # Build a data module. In Omnifold iterations, we want to run prediction on
        # every event we have, so need to define two data modules, one for the
        # training / val set and one for the test set. Both of these will be
        # in testing mode. Note the data modules filter data by the relevant
        # pass 190 flag. Need to add in weights for events which fail these
        # flags after prediction.
        self.d_module_train = LOfData(
            source_file=train_source_file,
            target_file=train_target_file,
            source_weight_path=source_weight_file,
            target_weight_path=target_weight_file,
            max_events_target=self.config.max_events_target,
            data_divisor=1,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.test_batch_size,
            split_seed=self.config.split_seed,
            dataloader_workers=30,
            testing=False,
            use_truth=use_truth,
        )
        self.d_module_test = LOfData(
            source_file=test_source_file,
            target_file=test_target_file,
            source_weight_path=source_weight_file,
            target_weight_path=target_weight_file,
            max_events_target=self.config.max_events_target,
            data_divisor=1,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.test_batch_size,
            split_seed=self.config.split_seed,
            dataloader_workers=30,
            persistent_workers=True,
            testing=True,
            use_truth=use_truth,
        )

        # Make plotter objects
        if step == 1:
            labels = ("RecoMC", "RecoPD")
        else:
            labels = ("TruthMC", "PulledWeightsMC")

        self.test_plotter = Plotter(
            test_source_file,
            test_target_file,
            self.test_dir,
            use_truth=use_truth,
            labels=labels,
            verbosity=1,
            max_events=self.config.max_events_target,
        )

        if step == 2:
            self.comp_plotter = Plotter(
                self.config.mc_test_path,
                self.config.truth_data_path,
                self.comp_dir,
                use_truth=use_truth,
                labels=("TruthMC", "TruthPD"),
                verbosity=2,
                max_events=self.config.max_events_target,
            )

    def run_testing(self):
        """run_testing - Run testing over the test data module.
        The point here is to get performance metrics (AUC and test loss)

        No arguments or returns
        """

        self.trainer.test(self.model, self.d_module_test)

    def run_prediction(self):
        """run_prediction - Run predictions over every data point in the
        train / test datamodules. Then calculate the updated weights.

        Then save the updated weights as .npz files

        No arguments or returns
        """

        # Run predictions, note this only produces predictions for the source events
        predictions_train = self.trainer.predict(self.model, self.d_module_train)
        predictions_test = self.trainer.predict(self.model, self.d_module_test)

        # Send predictions to CPU, convert to numpy, and concatenate
        predictions_train = np.concatenate(
            [pred.cpu().numpy().flatten() for pred in predictions_train]
        )
        predictions_test = np.concatenate(
            [pred.cpu().numpy().flatten() for pred in predictions_test]
        )

        # Calculate network weights, can just take the exponential
        network_weights_train = np.exp(predictions_train)
        network_weights_test = np.exp(predictions_test)

        # Get source weights from the data modules, note this is before normalization
        start_weights_train = self.d_module_train.get_source_all_weights()
        start_weights_test = self.d_module_test.get_source_all_weights()

        # Get the filters
        pass190_train = self.d_module_train.get_source_pass190()
        pass190_test = self.d_module_test.get_source_pass190()

        # Calculate updated weights
        source_weights_train = start_weights_train.copy()
        source_weights_train[pass190_train == 1] *= network_weights_train
        source_weights_test = start_weights_test.copy()
        source_weights_test[pass190_test == 1] *= network_weights_test
        self.all_updated_weights_train = source_weights_train
        self.all_updated_weights_test = source_weights_test

        # Get all pass190s
        source_pass190_train = self.d_module_train.get_source_reco_pass190()
        source_pass190_test = self.d_module_test.get_source_reco_pass190()
        source_truth_pass190_train = self.d_module_train.get_source_truth_pass190()
        source_truth_pass190_test = self.d_module_test.get_source_truth_pass190()
        target_pass190_train = self.d_module_train.get_target_reco_pass190()
        target_pass190_test = self.d_module_test.get_target_reco_pass190()
        target_truth_pass190_train = self.d_module_train.get_target_truth_pass190()
        target_truth_pass190_test = self.d_module_test.get_target_truth_pass190()

        # Save new weights for future use
        np.savez(
            f"{self.save_dir}/iteration_{self.iteration}_step_{self.step}.npz",
            raw_train_output=predictions_train,
            raw_test_output=predictions_test,
            network_train=network_weights_train,
            network_test=network_weights_test,
            train=self.all_updated_weights_train,
            test=self.all_updated_weights_test,
            source_pass190_train=source_pass190_train,
            source_pass190_test=source_pass190_test,
            source_truth_pass190_train=source_truth_pass190_train,
            source_truth_pass190_test=source_truth_pass190_test,
            target_pass190_train=target_pass190_train,
            target_pass190_test=target_pass190_test,
            target_truth_pass190_train=target_truth_pass190_train,
            target_truth_pass190_test=target_truth_pass190_test,
        )

        # Make and log test plots
        test_plot_dict = self.test_plotter.plot(
            start_weights_test,
            self.all_updated_weights_test,
            self.d_module_test.get_target_all_weights(),
        )
        if self.config.wandb:
            for key, histpath in test_plot_dict.items():
                log_name = f"test_{key}"
                self.wandb_logger.experiment.log({log_name: wandb.Image(str(histpath))})

        # Evaluate difference between reweighted truth MC and truth data
        # if this is step 2
        if self.step == 2:
            self.compare()

    def compare(self):
        """compare - Compare the reweighted truth MC to the truth pseudodata.

        No arguments or returns
        """

        # Compute and log wasserstein metric with plotter class
        _, w1_end = self.comp_plotter.wasserstein_distance(
            "weight",
            self.all_updated_weights_test,
            "weight_mc",
        )
        print("Reweighted truth MC to truth PD Wasserstein metric:", w1_end)
        if self.config.wandb:
            self.wandb_logger.experiment.log({"comp_wasserstein": w1_end})

        # Generate and log plots with plotter class
        plot_dict = self.comp_plotter.plot(
            "weight",
            self.all_updated_weights_test,
            "weight_mc",
        )
        if self.config.wandb:
            for key, histpath in plot_dict.items():
                log_name = f"comp_{key}"
                self.wandb_logger.experiment.log({log_name: wandb.Image(str(histpath))})

    def run(self):
        """run - This function runs the evaluation routine for an omnifold classifier

        No Arguments or Returns
        """

        # Run testing
        if not self.verify:
            print("Run testing")
            self.run_testing()
            if self.unit_test:
                return

        # Run prediction, unless this is pretraining
        if self.iteration > 0:
            print("Run predictions")
            self.run_prediction()

        # Call wandb finish to set run status to finished
        if self.config.wandb:
            wandb.finish()


# ------------------------- MAIN FUNCTION -------------------------

# This function will be called as a subprocess from the Omnifolder class
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the omnifold evaluation")
    parser.add_argument(
        "--check_path",
        type=str,
        default=None,
        help="Path to the checkpoint to evaluate",
    )
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="The iteration number for this training run",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="The step number for this training run"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="If set, do not run testing, just run prediction.",
    )
    parser.add_argument(
        "--store",
        type=str,
        default=None,
        help="If set, store weights here instead of in the default location",
    )
    parser.add_argument(
        "--index", type=int, default=-1, help="The index of the ensemble to run"
    )
    args, _ = parser.parse_known_args()

    # Signal handler just requeues the job on SIGUSR1
    # This is done automatically on SIGTERM
    def handle_signal(signum, frame):
        """Signal handler for the program."""
        print(f"Received signal {signum}, requeue job.")
        time.sleep(10)
        os.system("sync")
        time.sleep(10)
        os.system("scontrol requeue $SLURM_JOB_ID")

    signal.signal(signal.SIGUSR1, handle_signal)

    # Run the evaluation
    evaluator = OfEval(
        args.config_path,
        args.iteration,
        args.step,
        check_path=args.check_path,
        verify=args.verify,
        store=args.store,
        index=args.index,
    )
    evaluator.run()

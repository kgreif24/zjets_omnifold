""" lightning_eval.py - This program defines the OfEval class, which is responsible for
running evaluation and prediciton for Omnifold classifiers. It produces the weights
derived from each training, which are written to disk.

Meant to be run as a subprocesss from the Omnifolder class.

Author: Kevin Greif
Last updated 11/20/2024
python3
"""

import os
import argparse
import numpy as np
import uproot
import awkward as ak
import lightning as L
import wandb
from pytorch_lightning.loggers import WandbLogger

from lightning_module import *
from lightning_data_module import *
import utils.plotting_utils as pu
import utils.data_utils as du
from cli.of_config import OfConfig
from wasserstein_metric import WassersteinOne


class OfEval:
    """OfEval - This class handles the evaluation and prediction for an
    Omnifold classifier. It is run by the driver function below, which
    is meant to be called as a subprocess from the Omnifolder class.
    """

    def __init__(
        self,
        check_path,
        run_id,
        config_path,
        iteration,
        step,
        verify=False,
        store=None,
        index=-1,
        unit_test=False,
    ):
        """__init__ - The init function for this class. It takes the OfConfig object
        used for this run of Omnifold, plus the iteration and step of this evaluation.

        Arguments:
        check_path - The path to the checkpoint to evaluate
        run_id - The ID of the run for evaluation
        config_path - The path of the of config file
        iteration - The iteration number for this training
        step - The step number for this training
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

        #################### Config and directory setup ####################

        # Store the configuration
        self.config = OfConfig(config_name=config_path)
        self.run_id = run_id
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

        # Hard code the number of truth pseudodata events to use in step 2 comparison
        self.n_compare_events = 1000000

        # Hard code the number of tracks to use in building inclusive track plots
        self.max_tracks = 150

        # Make directories for storing plots and weights
        root_dir = f"{self.config.checkpoint_dir}/{self.config.project_name}/{self.config.group_name}"
        self.checkpoint_dir = f"{root_dir}/{self.run_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.test_dir = f"{root_dir}/{self.run_name}/test_plots"
        os.makedirs(self.test_dir, exist_ok=True)
        self.comp_dir = f"{root_dir}/{self.run_name}/comp_plots"
        os.makedirs(self.comp_dir, exist_ok=True)
        self.weight_dir = f"{root_dir}/weights"
        os.makedirs(self.weight_dir, exist_ok=True)

        # Change the save location if the store argument is set
        if store is not None:
            self.save_dir = store
        else:
            self.save_dir = self.weight_dir

        #################### Lightning setup ####################

        # Initialise the wandb logger
        if self.config.wandb:

            self.wandb_logger = WandbLogger(
                project=self.config.project_name,
                group=self.config.group_name,
                name=self.run_name,
                save_dir=self.checkpoint_dir,
                id=self.run_id,
                resume="must",
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
            log=self.config.wandb,
            debug=self.config.debug,
            step=self.step,
        )

        # Make lightning trainer for testing
        self.trainer = L.Trainer(
            accelerator="auto" if (self.config.debug or unit_test) else "gpu",
            num_nodes=1,
            devices=1,
            logger=self.wandb_logger,
            enable_progress_bar=self.config.interactive,
            fast_dev_run=unit_test,
            use_distributed_sampler=False,
        )

        # Make wasserstein metric object for comparing derived reweighting
        # to truth level pseudo data
        self.wasserstein = WassersteinOne(
            pu.default_settings, draw_plots=True, save_location=self.comp_dir
        )

        #################### Data setup ####################

        # Find the data and weight files to use for this iteration and step
        # For step one:
        if self.step == 1:
            use_truth = False
            # If this is pre-training (iteration 0), use the MC train file and Sherpa file
            if self.iteration == 0:
                train_source_file = self.config.mc_train_path
                test_source_file = self.config.mc_test_path
                train_target_file = None
                test_target_file = self.config.pretrain_path
                source_weight_file = "root"
                target_weight_file = "root"
            # If this is the first iteration, use the weights from the root file for source
            # and no weights for the target
            elif self.iteration == 1:
                train_source_file = self.config.mc_train_path
                test_source_file = self.config.mc_test_path
                train_target_file = None
                test_target_file = self.config.data_path
                source_weight_file = "root"
                target_weight_file = None
            # Otherwise use the weights from the previous step two for the source, and no
            # weights for the target
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
            # If this is pre-training (iteration 0), use the MC train file and Sherpa file
            if self.iteration == 0:
                train_source_file = self.config.mc_train_path
                test_source_file = self.config.mc_test_path
                train_target_file = None
                test_target_file = self.config.pretrain_path
                source_weight_file = "root"
                target_weight_file = "root"
            # If this is the first iteration, use the weights from step one for target, and the
            # weights from the root file as source.
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

        # Build a data module. In Omnifold iterations, we want to run prediction on every event
        # we have, so need to define two data modules, one for the training / val
        # set and one for the test set. Both of these will be in testing mode.
        # Note the data modules filter data by the relevant pass 190 flag. Need to add in weights for
        # events which fail these flags after prediction.
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
            load_all=True,
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
            load_all=True,
            testing=True,
            use_truth=use_truth,
        )

    def run_testing(self):
        """run_testing - Run testing over the test data module.
        The point here is to get performance metrics (AUC and test loss)

        No arguments or returns
        """

        self.trainer.test(self.model, self.d_module_test)

    def run_prediction(self):
        """run_prediction - Run predictions over every data point in the train / test datamodules.
        Then calculate the updated weights. Also need to think about how to handle the events
        which do not pass the pass190 flags. For now just assign the starting weight to these events,
        using the "get_source_all_weights" method of the data modules.

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
        source_weights_train = self.d_module_train.get_source_all_weights()
        source_weights_test = self.d_module_test.get_source_all_weights()

        # Get the filters
        pass190_train = self.d_module_train.get_source_pass190()
        pass190_test = self.d_module_test.get_source_pass190()

        # Calculate updated weights
        source_weights_train[pass190_train == 1] *= network_weights_train
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

        # Evaluate difference between reweighted truth MC and truth data if this is step 2
        if self.step == 2:
            self.compare()

    def compare(self):
        """compare - Compare the reweighted truth MC to the truth pseudodata. Will draw all relevant plots,
        calculate the wasserstein metric, and upload all results to wandb using the WassersteinOne class.

        No arguments or returns
        """

        # Load truth level MC and pseudodata from scratch
        f_mc = uproot.open(self.config.mc_test_path)  # Compare testing set MC to data
        tree_mc = f_mc["OmniTree"]
        f_pd = uproot.open(self.config.truth_data_path)
        tree_pd = f_pd["OmniTree"]

        # Get the truth level pass190 filters
        filter_mc = ak.to_numpy(
            tree_mc["truth_pass190"].array(entry_stop=self.n_compare_events)
        )
        filter_pd = ak.to_numpy(
            tree_pd["truth_pass190"].array(entry_stop=self.n_compare_events)
        )

        # Get the plot data
        plotting_mc = du.get_plotting(
            tree_mc,
            vars=pu.default_settings.keys(),
            get_truth=True,
            stop=self.n_compare_events,
        )
        plotting_pd = du.get_plotting(
            tree_pd,
            vars=pu.default_settings.keys(),
            get_truth=True,
            stop=self.n_compare_events,
        )

        # Get the track kinematics
        kinematics_mc, _ = du.get_kinematics(
            tree_mc, get_truth=True, stop=self.n_compare_events
        )
        kinematics_pd, _ = du.get_kinematics(
            tree_pd, get_truth=True, stop=self.n_compare_events
        )

        # Pad kinematics
        kinematics_mc = du.pad_kinematics(kinematics_mc, max_tracks=self.max_tracks)
        kinematics_pd = du.pad_kinematics(kinematics_pd, max_tracks=self.max_tracks)

        # Slice the track kinematics (log pT, eta, cos(phi), sin(phi))
        kinematics_mc = kinematics_mc[:, :3, 2:]
        kinematics_pd = kinematics_pd[:, :3, 2:]

        # Concatenate the truth level MC and truth level pseudodata
        plotting = np.concatenate([plotting_mc, plotting_pd], axis=0)
        kinematics = np.concatenate([kinematics_mc, kinematics_pd], axis=0)

        # Make labels
        labels_mc = np.zeros(plotting_mc.shape[0])
        labels_pd = np.ones(plotting_pd.shape[0])
        labels = np.concatenate([labels_mc, labels_pd], axis=0)

        # Get start weights for MC and truth pseudodata
        root_weights_mc = ak.to_numpy(
            tree_mc["weight"].array(entry_stop=self.n_compare_events)
        )
        root_weights_mc = root_weights_mc[filter_mc == 1]
        root_weights_pd = ak.to_numpy(
            tree_pd["weight"].array(entry_stop=self.n_compare_events)
        )
        root_weights_pd = root_weights_pd[filter_pd == 1]
        start_weights = np.concatenate([root_weights_mc, root_weights_pd], axis=0)

        # Make end weights
        mc_end_weights = self.all_updated_weights_test
        if len(self.all_updated_weights_test) > self.n_compare_events:
            mc_end_weights = self.all_updated_weights_test[: self.n_compare_events]
        mc_end_weights = mc_end_weights[filter_mc == 1]
        end_weights = np.concatenate([mc_end_weights, root_weights_pd], axis=0)

        # Update and compute metrics, generate plots
        self.wasserstein.update(plotting, start_weights, end_weights, labels)
        comp_wass, plot_dict = self.wasserstein.compute(
            from_torch=False, names=("TruthMC", "TruthPD"), is_comp=True
        )
        track_dict = pu.make_inclusive_track_plots(
            kinematics, labels, start_weights, end_weights, save_location=self.comp_dir
        )
        plot_dict = {**plot_dict, **track_dict}
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
        """run - This function runs the evaluation routine for an omnifold classifier

        No Arguments or Returns
        """

        # Run testing
        if not self.verify:
            print("Run testing")
            self.run_testing()
            if self.unit_test or (not self.trainer.is_global_zero):
                return

        # Run prediction, unless this is pretraining
        if self.iteration > 0:
            print("Run predictions")
            self.run_prediction()

        # Call wandb finish to set run status to finished
        if self.config.wandb:
            wandb.finish()


############## MAIN FUNCTION ##############

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
        "--run_id", type=str, default=None, help="ID of the run for evaluation"
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

    # Run the evaluation
    evaluator = OfEval(
        check_path=args.check_path,
        run_id=args.run_id,
        config_path=args.config_path,
        iteration=args.iteration,
        step=args.step,
        verify=args.verify,
        store=args.store,
        index=args.index,
    )
    evaluator.run()

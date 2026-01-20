"""lightning_eval.py - This program defines the OfEval class, which is responsible for
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
import re

import numpy as np
import torch
import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment
import wandb
from pytorch_lightning.loggers import WandbLogger

from lightning_module import LOfTransformer
from lightning_data_module import LOfData
from plotter import Plotter
from cli.of_config import OfConfig
from wasserstein_metric import WassersteinOne
import utils.subprocess_utils as su


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
        world_size=1,
        rank=0,
        check_path=None,
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
        world_size - Defaults 1, the number of GPUs to use for PREDICTION
            (not testing). Testing, handled by lightning, will only use one GPU
        rank - Defaults 0, the rank of this process in the distributed
            environment.
        check_path - Defaults None, the path to the checkpoint to evaluate
            If left none, use the best model symlink
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
        self.world_size = world_size
        self.rank = rank
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
        if self.config.wandb and self.rank == 0:
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
            enable_progress_bar=self.config.interactive if self.rank == 0 else False,
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
            self.use_truth = False
            # If this is pre-training (iteration 0), use the MC train file and
            # Sherpa file
            if self.iteration == 0:
                self.train_source_file = None
                self.test_source_file = self.config.pretrain_source_path
                self.train_target_file = None
                self.test_target_file = self.config.pretrain_target_path
                self.source_weight_file = None
                self.target_weight_file = None
            # If this is the first iteration, use the weights from the root file
            # for source and target
            elif self.iteration == 1:
                self.train_source_file = self.config.mc_train_path
                self.test_source_file = self.config.mc_test_path
                self.train_target_file = None
                self.test_target_file = self.config.data_path
                self.source_weight_file = None
                self.target_weight_file = self.config.top_sub_weights
            # Otherwise use the weights from the previous step two for the source,
            # and weights from ROOT file for the target
            else:
                self.train_source_file = self.config.mc_train_path
                self.test_source_file = self.config.mc_test_path
                self.train_target_file = None
                self.test_target_file = self.config.data_path
                self.source_weight_file = (
                    f"{self.weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                )
                self.target_weight_file = self.config.top_sub_weights
        # For step two:
        if self.step == 2:
            self.use_truth = True
            # If this is pre-training (iteration 0), use the MC train file and
            # Sherpa file
            if self.iteration == 0:
                self.train_source_file = self.config.mc_train_path
                self.test_source_file = self.config.mc_test_path
                self.train_target_file = None
                self.test_target_file = self.config.pretrain_path
                self.source_weight_file = None
                self.target_weight_file = None
            # If this is the first iteration, use the weights from step one for target,
            # and the weights from the ROOT file as source.
            elif self.iteration == 1:
                self.train_source_file = self.config.mc_train_path
                self.test_source_file = self.config.mc_test_path
                self.train_target_file = None
                self.test_target_file = self.config.mc_test_path
                self.source_weight_file = None
                self.target_weight_file = (
                    f"{self.weight_dir}/iteration_{self.iteration}_step_1.npz"
                )
            # Otherwise use weights from previous step 2 for source, and the weights
            # from the previous step one for target
            else:
                self.train_source_file = self.config.mc_train_path
                self.test_source_file = self.config.mc_test_path
                self.train_target_file = None
                self.test_target_file = self.config.mc_test_path
                self.source_weight_file = (
                    f"{self.weight_dir}/iteration_{self.iteration-1}_step_2.npz"
                )
                self.target_weight_file = (
                    f"{self.weight_dir}/iteration_{self.iteration}_step_1.npz"
                )

        # Data module are now built in the testing / prediction methods.
        # This means the testing module is built twice, but it also allows the
        # use of multiple GPUs in prediction so it saves time.

        # Make plotter objects on rank zero process
        # Kinematic region is -1 to use all events with pass190 == 1
        if self.rank == 0:
            if step == 1:
                labels = ("RecoMC", "RecoPD")
            else:
                labels = ("TruthMC", "PulledWeightsMC")

            self.test_plotter = Plotter(
                self.test_source_file,
                self.test_target_file,
                self.test_dir,
                use_truth=self.use_truth,
                labels=labels,
                verbosity=1,
                max_events=self.config.max_events_target,
                syst_kw=self.config.syst_kw if self.step == 1 else None,
                kinematic_region=-1,
            )

            if step == 2 and self.config.truth_data_path is not None:
                self.comp_plotter = Plotter(
                    self.config.mc_test_path,
                    self.config.truth_data_path,
                    self.comp_dir,
                    use_truth=True,
                    labels=("TruthMC", "TruthPD"),
                    verbosity=2,
                    max_events=self.config.max_events_target,
                    kinematic_region=-1,
                )

    def run_testing(self):
        """run_testing - Run testing over the test data module.
        The point here is to get performance metrics (AUC and test loss)

        No arguments or returns
        """

        # Build data module, note we do not split the data into pieces
        # and use distibuted evaluation for testing
        is_theory_syst = (
            self.config.syst_kw is not None and "theory" in self.config.syst_kw
        )
        use_syst = self.config.syst_kw if (self.step == 1 or is_theory_syst) else None
        d_module_test = LOfData(
            source_file=self.test_source_file,
            target_file=self.test_target_file,
            source_weight_path=self.source_weight_file,
            target_weight_path=self.target_weight_file,
            # Limit the number of events for calculating AUC and Wasserstein
            max_events_source=500000,
            max_events_target=500000,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.test_batch_size,
            split_seed=self.config.split_seed,
            dataloader_workers=30,
            testing=True,
            use_truth=self.use_truth,
            syst_kw=use_syst,
            theory_weight_mode=is_theory_syst and self.step == 2,
        )

        self.trainer.test(self.model, d_module_test)

    def run_prediction(self):
        """run_prediction - Run predictions over every data point in the
        train / test datamodules. Then calculate the updated weights.

        Then save the updated weights as .npz files

        No arguments or returns
        """

        # Build data modules
        is_theory_syst = (
            self.config.syst_kw is not None and "theory" in self.config.syst_kw
        )
        use_syst = self.config.syst_kw if (self.step == 1 or is_theory_syst) else None
        d_module_train = LOfData(
            source_file=self.train_source_file,
            target_file=self.train_target_file,
            source_weight_path=self.source_weight_file,
            target_weight_path=self.target_weight_file,
            max_events_target=self.config.max_events_target,
            data_divisor=self.world_size,
            piece=self.rank,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.test_batch_size,
            split_seed=self.config.split_seed,
            dataloader_workers=20,
            testing=False,
            use_truth=self.use_truth,
            syst_kw=use_syst,
            theory_weight_mode=is_theory_syst and self.step == 2,
        )
        d_module_test = LOfData(
            source_file=self.test_source_file,
            target_file=self.test_target_file,
            source_weight_path=self.source_weight_file,
            target_weight_path=self.target_weight_file,
            max_events_target=self.config.max_events_target,
            data_divisor=self.world_size,
            piece=self.rank,
            max_tracks=self.config.max_tracks,
            muon_only=self.config.debug,
            batch_size=self.config.test_batch_size,
            split_seed=self.config.split_seed,
            dataloader_workers=20,
            testing=True,
            use_truth=self.use_truth,
            syst_kw=use_syst,
            theory_weight_mode=is_theory_syst and self.step == 2,
        )

        # Run predictions, note this only produces predictions for the source events
        rank_predictions_train = self.trainer.predict(self.model, d_module_train)
        rank_predictions_train = torch.cat(
            [pred.cpu().flatten() for pred in rank_predictions_train]
        )
        rank_predictions_test = self.trainer.predict(self.model, d_module_test)
        rank_predictions_test = torch.cat(
            [pred.cpu().flatten() for pred in rank_predictions_test]
        )
        if self.world_size > 1:
            torch.distributed.barrier()
            predictions_train = su.all_gather(rank_predictions_train, self.world_size)
            predictions_train = torch.cat(predictions_train)
            predictions_test = su.all_gather(rank_predictions_test, self.world_size)
            predictions_test = torch.cat(predictions_test)
        else:
            predictions_train = rank_predictions_train
            predictions_test = rank_predictions_test

        # The rest of the code is run on rank 0
        if self.rank == 0:

            # Send predictions to numpy
            predictions_train = predictions_train.numpy()
            predictions_test = predictions_test.numpy()

            # Calculate network weights, can just take the exponential
            network_weights_train = np.exp(predictions_train)
            network_weights_test = np.exp(predictions_test)

            # Get the filters
            pass190_train = d_module_train.get_source_pass190()
            pass190_test = d_module_test.get_source_pass190()

            # Get start weights.
            # For iteration 1, these are vectors of 1s
            # For all subsequent iterations, use the source weights from the data
            # modules
            if self.iteration == 1:
                start_weights_train = np.ones_like(pass190_train, dtype=np.float32)
                start_weights_test = np.ones_like(pass190_test, dtype=np.float32)
            else:
                start_weights_train = d_module_train.get_source_network_weights()
                start_weights_test = d_module_test.get_source_network_weights()

            # Calculate updated weights
            self.updated_weights_train = start_weights_train.copy()
            self.updated_weights_train[pass190_train == 1] *= network_weights_train
            self.updated_weights_test = start_weights_test.copy()
            self.updated_weights_test[pass190_test == 1] *= network_weights_test

            # Make and log test plots
            root_weights_test = d_module_test.get_source_root_weights()
            plot_weights_test = self.updated_weights_test * root_weights_test
            test_plot_dict = self.test_plotter.plot(
                d_module_test.get_source_all_weights(),
                plot_weights_test,
                d_module_test.get_target_all_weights(),
            )
            if self.config.wandb:
                for key, histpath in test_plot_dict.items():
                    log_name = f"test_{key}"
                    self.wandb_logger.experiment.log(
                        {log_name: wandb.Image(str(histpath))}
                    )

            # Evaluate difference between reweighted truth MC and truth data
            # if this is step 2
            if self.step == 2 and self.config.truth_data_path is not None:
                self.compare(
                    root_weights_test,
                    plot_weights_test,
                    "weight_mc",
                )

            # Save new weights for future use
            np.savez(
                f"{self.save_dir}/iteration_{self.iteration}_step_{self.step}.npz",
                raw_train_output=predictions_train,
                raw_test_output=predictions_test,
                network_train=network_weights_train,
                network_test=network_weights_test,
                train=self.updated_weights_train,
                test=self.updated_weights_test,
            )

        # Wait for rank 0 process to finish saving weights and plotting
        if self.world_size > 1:
            torch.distributed.barrier()

    def compare(self, start_weights, end_weights, target_weights):
        """compare - Compare the reweighted truth MC to the truth pseudodata.

        Arguments:
        plot_weights (np.ndarray) - Weights to apply to the truth level testing
            MC that should produce the truth level data distribution.

        Returns:
        None
        """

        # Compute and log wasserstein metric with plotter class
        _, w1_end = self.comp_plotter.wasserstein_distance(
            start_weights,
            end_weights,
            target_weights,
        )
        print("Reweighted truth MC to truth PD Wasserstein metric:", w1_end)
        if self.config.wandb:
            self.wandb_logger.experiment.log({"comp_wasserstein": w1_end})

        # Generate and log plots with plotter class
        plot_dict = self.comp_plotter.plot(
            start_weights,
            end_weights,
            target_weights,
        )
        if self.config.wandb:
            for key, histpath in plot_dict.items():
                log_name = f"comp_{key}"
                self.wandb_logger.experiment.log({log_name: wandb.Image(str(histpath))})


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
        "--run_test",
        action="store_true",
        help="If set, run testing instead of prediction",
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

    # Infer whether we are using multiple GPUs, and if so the world size and rank
    # from SLURM environment variables
    if "SLURM_JOB_ID" in os.environ and "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
    else:
        world_size = 1
        rank = 0

    # Set distributed environment variables
    if world_size > 1:
        nodelist = re.split(
            r"[\[\],-]", os.environ.get("SLURM_JOB_NODELIST", "localhost")
        )
        os.environ["MASTER_ADDR"] = nodelist[0] + nodelist[1]
        os.environ["MASTER_PORT"] = "29500"

    # Initialize distributed processes if we are using multiple GPUs
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    # Run the evaluation
    evaluator = OfEval(
        args.config_path,
        args.iteration,
        args.step,
        world_size=world_size,
        rank=rank,
        check_path=args.check_path,
        store=args.store,
        index=args.index,
    )
    if args.run_test:
        evaluator.run_testing()
    else:
        evaluator.run_prediction()

    # Clean up distributed processes
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

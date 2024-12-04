""" of_config.py - This class implements a CLI for the Omnifold algorithm.
All of the needed hyper-parameters for the algorithm can be controlled with this class.

Author: Kevin Greif
Last updated 03/05/2024
python3
"""

import argparse
import yaml
import sys
from datetime import datetime
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class OfConfig:

    def add_default_arguments(self):
        """add_default_arguments - Add default arguments to the parser. This sets all of the defaults
        needed to run the Omnifold algorithm succesfully.

        Arguments: None
        Returns: None
        """

        # Dummy argument to prevent error on parsing the command line arg "config_path"
        ## This is a hack, but I don't want to carry the argparser from the top level "run_omnifold.py" to the "OfConfig" class
        self.parser.add_argument(
            "--config_path",
            type=str,
            default=None,
            help="Path to the configuration file",
        )

        # General
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="Run in debug mode (single device, muons only)",
        )
        self.parser.add_argument(
            "--interactive",
            action="store_true",
            help="Run in interactive mode with progress bars",
        )

        # Omnifold
        self.parser.add_argument(
            "--num_iterations",
            type=int,
            default=6,
            help="Number of iterations to run the Omnifold algorithm",
        )

        # Data
        self.parser.add_argument(
            "--mc_train_path",
            type=str,
            default="/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train_Mar1023.root",
            help="Path for the MC training data",
        )
        self.parser.add_argument(
            "--mc_test_path",
            type=str,
            default="/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test_Mar0723.root",
            help="Path for the MC testing data",
        )
        self.parser.add_argument(
            "--data_path",
            type=str,
            default="/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Apr8_1_All.root",
            help="Path for the data",
        )
        self.parser.add_argument(
            "--truth_data_path",
            type=str,
            default="/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_TruthPseudodata_Mar12_Combined_1_50_Top_shuffled.root",
            help="Path to the truth pseudodata",
        )
        self.parser.add_argument(
            "--pretrain_path",
            type=str,
            default="/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Mar10_Sherpa2211_slim_Pretrain_all.root",
            help="Path to the file to use in pretraining",
        )
        self.parser.add_argument(
            "--split_seed",
            type=int,
            default=-1,
            help="Seed for the train / validation split, set to -1 to produce random seed at train time",
        )
        self.parser.add_argument(
            "--max_tracks",
            type=int,
            default=264,
            help="Maximum number of tracks to use in the data",
        )
        self.parser.add_argument(
            "--num_pretrain_pieces",
            type=int,
            default=3,
            help="Number of pieces to split the pretraining data into",
        )
        self.parser.add_argument(
            "--max_events_target",
            type=int,
            default=500000,
            help="Maximum number of events to use in the target data",
        )

        # Pretraining checkpoints, only needed if picking up a run from a checkpoint
        self.parser.add_argument(
            "--pt_step_one_checkpoint",
            type=str,
            default=None,
            help="Path to the pretraining step one checkpoint",
        )
        self.parser.add_argument(
            "--pt_step_two_checkpoint",
            type=str,
            default=None,
            help="Path to the pretraining step two checkpoint",
        )

        # Training
        self.parser.add_argument(
            "--batch_size", type=int, default=256, help="Batch size for training"
        )
        self.parser.add_argument(
            "--test_batch_size", type=int, default=256, help="Batch size for testing"
        )
        self.parser.add_argument(
            "--max_epochs",
            type=int,
            default=70,
            help="Maximum number of epochs to train for",
        )

        self.parser.add_argument(
            "--top_k_checkpoints",
            type=int,
            default=5,
            help="Number of top checkpoints to save",
        )
        self.parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=12,
            help="Number of epochs to wait before stopping training",
        )
        self.parser.add_argument(
            "--num_gpus", type=int, default=4, help="Number of GPUs to use for training"
        )
        self.parser.add_argument(
            "--num_nodes",
            type=int,
            default=1,
            help="Number of nodes to use for training",
        )

        # Learning rate schedule
        self.parser.add_argument(
            "--pt_min_lr",
            type=float,
            default=0.00001,
            help="Minimum learning rate for pre-training",
        )
        self.parser.add_argument(
            "--pt_max_lr",
            type=float,
            default=0.0001,
            help="Maximum learning rate for pre-training",
        )
        self.parser.add_argument(
            "--s1_min_lr",
            type=float,
            default=0.000008,
            help="Minimum learning rate for step one training",
        )
        self.parser.add_argument(
            "--s1_max_lr",
            type=float,
            default=0.00005,
            help="Maximum learning rate for step one training",
        )
        self.parser.add_argument(
            "--s2_min_lr",
            type=float,
            default=0.00005,
            help="Minimum learning rate for step two training",
        )
        self.parser.add_argument(
            "--s2_max_lr",
            type=float,
            default=0.0001,
            help="Maximum learning rate for step two training",
        )
        self.parser.add_argument(
            "--s1_max_decay",
            type=float,
            default=1.0,
            help="Decay with iteration for step one trainings maximum learning rate",
        )
        self.parser.add_argument(
            "--s2_max_decay",
            type=float,
            default=1.0,
            help="Decay with iteration for step two trainings maximum learning rate",
        )
        self.parser.add_argument(
            "--cycle_steps", type=int, default=50000, help="Number of steps in a cycle"
        )
        self.parser.add_argument(
            "--warmup_steps",
            type=int,
            default=10000,
            help="Number of steps to warm up the learning rate",
        )
        self.parser.add_argument(
            "--gamma",
            type=float,
            default=0.85,
            help="Gamma for the learning rate scheduler",
        )

        # Logging
        self.parser.add_argument(
            "--wandb", action="store_true", help="Use wandb for logging"
        )
        self.parser.add_argument(
            "--project_name",
            type=str,
            default="test-of-project",
            help="Name of the wandb project",
        )
        self.parser.add_argument(
            "--group_name",
            type=str,
            default="test-of-run",
            help="Name of the wandb group for all training runs",
        )
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoints",
            help="Directory in which save model checkpoints",
        )

        # Model
        self.parser.add_argument(
            "--input_dim", type=int, default=10, help="Input dimension for the model"
        )
        self.parser.add_argument(
            "--pair_input_dim",
            type=int,
            default=4,
            help="Pair input dimension for the model",
        )
        self.parser.add_argument(
            "--remove_self_pair",
            action="store_true",
            help="Remove self pair from the model",
        )
        self.parser.add_argument(
            "--run_trimmer",
            action="store_true",
            help="Use the sequence trimmer in training the model",
        )
        self.parser.add_argument(
            "--embed_dims",
            type=int,
            nargs="+",
            default=[128, 512, 128],
            help="Embedding dimensions for the model",
        )
        self.parser.add_argument(
            "--pair_embed_dims",
            type=int,
            nargs="+",
            default=[64, 64, 64],
            help="Pair embedding dimensions for the model",
        )
        self.parser.add_argument(
            "--num_heads", type=int, default=8, help="Number of heads for the model"
        )

        self.parser.add_argument(
            "--num_layers", type=int, default=8, help="Number of layers for the model"
        )
        self.parser.add_argument(
            "--block_dropout",
            type=float,
            default=0.0,
            help="Dropout rate to use in regular attention blocks",
        )
        self.parser.add_argument(
            "--block_attn_dropout",
            type=float,
            default=0.0,
            help="Attention dropout rate to use in regular attention blocks",
        )
        self.parser.add_argument(
            "--block_activation_dropout",
            type=float,
            default=0.0,
            help="Activation dropout rate to use in regular attention blocks",
        )

        self.parser.add_argument(
            "--num_cls_layers",
            type=int,
            default=2,
            help="Number of classification layers for the model",
        )
        self.parser.add_argument(
            "--cls_block_dropout",
            type=float,
            default=0.0,
            help="Dropout rate to use in classification attention blocks",
        )
        self.parser.add_argument(
            "--cls_block_attn_dropout",
            type=float,
            default=0.0,
            help="Attention dropout rate to use in classification attention blocks",
        )
        self.parser.add_argument(
            "--cls_block_activation_dropout",
            type=float,
            default=0.0,
            help="Activation dropout rate to use in classification attention blocks",
        )

        self.parser.add_argument(
            "--fc_nodes",
            type=int,
            nargs="+",
            default=[256, 256],
            help="Fully connected nodes for the model",
        )
        self.parser.add_argument(
            "--fc_dropout",
            type=float,
            default=0.0,
            help="Dropout rate to use in fully connected layers",
        )

    def __init__(self, config_name=None):
        """init function for the OfConfig class.

        Arguments:
        config_name - The name of the configuration file to load. If this is None, then the default settings are used

        Returns:
        None
        """

        # Make a parser
        self.parser = argparse.ArgumentParser(
            description="Omnifold algorithm hyperparameters"
        )

        # Load and parse command line args
        self.add_default_arguments()
        self.args, unknown = self.parser.parse_known_args()

        # If a configuration file is provided, load it
        if config_name is not None:
            rank_zero_info(f"Loading configuration from file: {config_name}")
            with open(config_name, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            rank_zero_info("No config given, using defaults and command line arguments")
            self.config = None

    def __getattr__(self, name):
        """__getattr__ - This function allows the OfConfig object to access the arguments
        as instance variables. Function first looks for value in a loaded config file.
        If none exists, then we take the default

        If the instance variable is not found in any of these places, then the function raises an exception
        and exits.

        Arguments:
        name - The name of the instance variable to access

        Returns:
        The value of the instance variable
        """

        # Return config object if asked for
        if name == "config":
            return self.config

        # Look for the name in the configuration file
        if self.config is not None:
            try:
                val = self.config[name]
            except KeyError:
                raise AttributeError(f"Config has no field {name}")
            return val

        # If this fails take the default value
        try:
            return getattr(self.args, name)
        # If the instance variable is not found in the parsed args or the configuration file, raise exception
        # and exit
        except Exception:
            print(f"Instance variable {name} not found in args or config. Exiting!")
            sys.exit(1)

    def mod_config(self, name: str, value) -> None:
        """mod_config - This function modifies the config loaded from a yaml file at a given name and value.
        If no yaml file has been loaded, then this function raises and exception.

        Arguments:
        name - The name of the field to modify in the config
        value - The new value of the field

        Returns:
        None
        """

        # Make sure we have a config loaded
        try:
            assert self.config is not None
        except AssertionError:
            raise AttributeError("Config object has no yaml file loaded to modify!")

        # If so set the field to the new value
        self.config[name] = value

    def create_template(
        self, template_path="./default_of_template.yml", arg_blacklist=None
    ):
        """create_template - Create a template for the configuration file. This template
        is a YAML file with all of the default arguments set to default values

        Arguments:
        name - The name of the template file
        arg_blacklist - A list of arguments to exclude from the template

        Returns:
        None
        """

        # Get a list of all the default arguments
        base_args = vars(self.args).keys()

        # Write the template to the file
        with open(template_path, "w") as f:
            f.write(
                "# This is a config file for the Omnifold algorithm\n"
                + f"# It was created on {datetime.now()}\n\n"
            )

            # Look through all of the arguments in the PARSER
            for arg in base_args:

                # If the argument is in the blacklist, skip it
                if arg_blacklist is not None and arg in arg_blacklist:
                    continue

                # Get the comment for the argument and write
                comment = self.parser._option_string_actions["--" + arg].help
                if comment is not None:
                    f.write(f"\n# {comment}\n")

                # Write the argument to the file, pulling first from the config
                # and second from the default settings
                this_attr = self.__getattr__(arg)
                if this_attr is not None:
                    # Also format floating point numbers to avoid scientific notation
                    if type(this_attr) is float:
                        this_attr = f"{this_attr:.9f}"
                    f.write(f"{arg}: {this_attr}\n")
                else:
                    f.write(f"{arg}: null\n")

        print("Created template file at: ", template_path)


# Main function for minting a new configuration file
if __name__ == "__main__":

    config = OfConfig()
    config.create_template()
    config = OfConfig(config_name="default_of_template.yml")

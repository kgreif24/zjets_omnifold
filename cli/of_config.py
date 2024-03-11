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
from pytorch_lightning.utilities.rank_zero import *


class OfConfig:

    def add_default_arguments(self):
        """ add_default_arguments - Add default arguments to the parser. This sets all of the defaults
        needed to run the Omnifold algorithm succesfully.

        Arguments: None
        Returns: None
        """

        # Dummy argument to prevent error on parsing the command line arg "config_path"
        ## This is a hack, but I don't want to carry the argparser from the top level "run_omnifold.py" to the "OfConfig" class
        self.parser.add_argument('--config_path', type=str, default=None, help='Path to the configuration file')

        # General
        self.parser.add_argument('--debug', action='store_true', help='Run in debug mode (single device, muons only)')

        # Omnifold
        self.parser.add_argument('--num_iterations', type=int, default=6, help='Number of iterations to run the Omnifold algorithm')

        # Data
        self.parser.add_argument(
            '--mc_train_path', 
            type=str, 
            default='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train.root', 
            help='Path for the MC training data'
        )
        self.parser.add_argument(
            '--mc_test_path',
            type=str,
            default='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test.root',
            help='Path for the MC testing data'
        )
        self.parser.add_argument(
            '--data_path',
            type=str,
            default='/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Jan30_Combined_All.root',
            help='Path for the data'
        )
        self.parser.add_argument('--split_seed', type=int, default=420, help='Seed for the train / validation split')
        self.parser.add_argument('--max_tracks', type=int, default=150, help='Maximum number of tracks to use in the data')

        # Training
        self.parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
        self.parser.add_argument('--max_epochs', type=int, default=70, help='Maximum number of epochs to train for')

        self.parser.add_argument('--top_k_checkpoints', type=int, default=5, help='Number of top checkpoints to save')
        self.parser.add_argument('--early_stopping_patience', type=int, default=8, help='Number of epochs to wait before stopping training')

        self.parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use for training')

        # Logging
        self.parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
        self.parser.add_argument('--project_name', type=str, default='test-of-project', help='Name of the wandb project')
        self.parser.add_argument('--group_name', type=str, default='test-of-run', help='Name of the wandb group for all training runs')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory in which save model checkpoints')

        # Model
        self.parser.add_argument('--input_dim', type=int, default=10, help='Input dimension for the model')
        self.parser.add_argument('--pair_input_dim', type=int, default=4, help='Pair input dimension for the model')
        self.parser.add_argument('--remove_self_pair', action='store_true', help='Remove self pair from the model')
        self.parser.add_argument('--run_trimmer', action='store_true', help='Use the sequence trimmer in training the model')
        self.parser.add_argument('--embed_dims', type=int, nargs='+', default=[128, 512, 128], help='Embedding dimensions for the model')
        self.parser.add_argument('--pair_embed_dims', type=int, nargs='+', default=[64, 64, 64], help='Pair embedding dimensions for the model')
        self.parser.add_argument('--num_heads', type=int, default=8, help='Number of heads for the model')

        self.parser.add_argument('--num_layers', type=int, default=8, help='Number of layers for the model')
        self.parser.add_argument('--block_dropout', type=float, default=0.0, help='Dropout rate to use in regular attention blocks')
        self.parser.add_argument('--block_attn_dropout', type=float, default=0.0, help='Attention dropout rate to use in regular attention blocks')
        self.parser.add_argument('--block_activation_dropout', type=float, default=0.0, help='Activation dropout rate to use in regular attention blocks')

        self.parser.add_argument('--num_cls_layers', type=int, default=2, help='Number of classification layers for the model')
        self.parser.add_argument('--cls_block_dropout', type=float, default=0.0, help='Dropout rate to use in classification attention blocks')
        self.parser.add_argument('--cls_block_attn_dropout', type=float, default=0.0, help='Attention dropout rate to use in classification attention blocks')
        self.parser.add_argument('--cls_block_activation_dropout', type=float, default=0.0, help='Activation dropout rate to use in classification attention blocks')

        self.parser.add_argument('--fc_nodes', type=int, nargs='+', default=[256, 256], help='Fully connected nodes for the model')
        self.parser.add_argument('--fc_dropout', type=float, default=0.0, help='Dropout rate to use in fully connected layers')



    def __init__(self, existing_parser=None, config_name=None):
        """ init function for the OfConfig class.

        Arguments:
        existing_parser - An existing parser to add arguments to. If this is None, then a new parser is created.
        config_name - The name of the configuration file to load. If this is None, then the default settings are used

        Returns:
        None
        """

        # Create a new parser if one is not provided
        if existing_parser is None:
            self.parser = argparse.ArgumentParser(description='Omnifold algorithm hyperparameters')
        else:
            self.parser = existing_parser

        # Load and parse command line args
        self.add_default_arguments()
        self.args = self.parser.parse_args()

        # Pull config name from the existing args if it exists
        if (config_name is None) and (hasattr(self.args, 'config')):
            config_name = self.args.config

        # If a configuration file is provided, load it
        if config_name is not None:
            rank_zero_info(f"Loading configuration from file: {config_name}")
            with open(config_name, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            rank_zero_info("No config given, using defaults and command line arguments")
            self.config = None


    def __getattr__(self, name):
        """ __getattr__ - This function allows the OfConfig object to access the arguments
        as instance variables. Function first looks for name in command line args that are not default values.
        Then it looks for name in yaml config. Then it takes the default value. 
        
        If the instance variable is not found in any of these places, then the function raises an exception
        and exits.

        Arguments:
        name - The name of the instance variable to access

        Returns:
        The value of the instance variable
        """

        # First look for the name in parsed args that are not default values
        if (name in vars(self.args).keys()) and (getattr(self.args, name) != self.parser.get_default(name)):
            return getattr(self.args, name)

        # If there are no non-default values, look for the name in the configuration file
        if self.config is not None and name in self.config:
            return self.config[name]
            
        # If both of these fail take the default value
        try:
            return getattr(self.args, name)
        # If the instance variable is not found in the parsed args or the configuration file, raise exception
        # and exit
        except:
            print(f"Instance variable {name} not found in args or config. Exiting!")
            sys.exit(1)


    def create_template(self, arg_blacklist=None):
        """ create_template - Create a template for the configuration file. This template
        is a YAML file with all of the default arguments set to default values

        Arguments:
        arg_blacklist - A list of arguments to exclude from the template

        Returns:
        None
        """

        # Get a list of all the default arguments
        base_args = vars(self.args).keys()

        # Create a yaml file to store the template
        template_path = "default_of_template.yml"

        # Write the template to the file
        with open(template_path, 'w') as f:
            f.write("# This is a config file for the Omnifold algorithm\n" + \
                    f"# It was created on {datetime.now()}\n\n")

            # Add all of the default arguments to the template, with the help as a comment
            # if we have it
            for arg in base_args:
                if arg_blacklist is not None and arg in arg_blacklist:
                    continue
                comment = self.parser._option_string_actions["--"+arg].help
                if comment is not None:
                    f.write(f"\n# {comment}\n")
                f.write(f"{arg}: {getattr(self.args, arg)}\n")

        print("Created template file at: ", template_path)


# Testing code
if __name__ == '__main__':

    # Create a new OfConfig object
    config = OfConfig()
    config.create_template()
    config = OfConfig(config_name="default_of_template.yml")
    print(config.mc_train_path)
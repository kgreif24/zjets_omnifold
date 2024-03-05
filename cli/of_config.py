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
from collections import OrderedDict


class OfConfig:

    def add_default_arguments(self):
        """ add_default_arguments - Add default arguments to the parser. This sets all of the defaults
        needed to run the Omnifold algorithm succesfully.

        Arguments: None
        Returns: None
        """

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
        self.parser.add_argument('--muon_only', action='store_true', help='Use only muons in the data')
        self.parser.add_argument('--split_seed', type=int, default=420, help='Seed for the train / validation split')

        # Training
        self.parser.add_argument('--max_tracks', type=int, default=150, help='Maximum number of tracks to use in the data')
        self.parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')


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
            print("Loading configuration from file: ", config_name)
            with open(config_name, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            print("No config given, using defaults and command line arguments")
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
                    f"# It was created on {datetime.now()}\n\n\n")

            # Add all of the default arguments to the template, with the help as a comment
            # if we have it
            for arg in base_args:
                if arg_blacklist is not None and arg in arg_blacklist:
                    continue
                comment = self.parser._option_string_actions["--"+arg].help
                if comment is not None:
                    f.write(f"# {comment}\n")
                f.write(f"{arg}: {getattr(self.args, arg)}\n")

        print("Created template file at: ", template_path)


# Testing code
if __name__ == '__main__':

    # Create a new OfConfig object
    config = OfConfig()
    config.create_template()
    config = OfConfig(config_name="default_of_template.yml")
    print(config.mc_train_path)
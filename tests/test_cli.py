""" test_cli.py - Test suite for the CLI """

import sys
sys.path.append('./cli')
from of_config import OfConfig

def test_config(tmp_path):

    config = OfConfig()

    # Trying to set an attribute without an existing config file should raise an error
    try:
        config.checkpoint_dir = "test"
        assert False
    except AttributeError:
        assert True

    # Create a template
    config_name = tmp_path / 'config.yaml'
    config.create_template(template_path=config_name)

    # Load the config and set an attribute
    config = OfConfig(config_name=config_name)
    config.checkpoint_dir = "test"
    assert config.checkpoint_dir == "test"

    # Write another template
    config.create_template(template_path=config_name)

    # Load this template and check that we have the correct attribute passed down
    config = OfConfig(config_name=config_name)
    assert config.checkpoint_dir == "test"

    # Check for some other common mistakes
    for key in vars(config).keys():
        assert getattr(config, key) is not 'None'
    
    assert type(config.s1_min_lr) is float
    assert type(config.s1_max_lr) is float
    assert type(config.s2_min_lr) is float
    assert type(config.s2_max_lr) is float
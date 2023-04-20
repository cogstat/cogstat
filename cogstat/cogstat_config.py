""" Settings for CogStat.

The module includes the elements for configuring the behavior of CogStat.

All settings that can be set from the Preferences in the GUI, are included in the .ini file.
"""

import os
import shutil
import configparser

import appdirs  # The module handles the OS-specific user config dirs

# 0. General settings
output_type = 'ipnb'  # if run from GUI, this is switched to 'gui' any other
versions = {}  # To be modified from cogstat.py
test_functions = False  # features can be switched on and off with this

# 1. Settings from the .ini file
# Handle cogstat.ini file in user config dirs
dirs = appdirs.AppDirs('cogstat')

# If there is no cogstat.ini file for the user, create one with the default values
if not os.path.isfile(dirs.user_config_dir + '/cogstat.ini'):
    if not os.path.exists(dirs.user_config_dir):
        os.makedirs(dirs.user_config_dir)
    shutil.copyfile(os.path.dirname(os.path.abspath(__file__)) + '/cogstat.ini', dirs.user_config_dir + '/cogstat.ini')

# config is the user-specific file, default_config includes the default values
config = configparser.ConfigParser(inline_comment_prefixes='#')
try:
    config.read(dirs.user_config_dir + '/cogstat.ini')
except configparser.MissingSectionHeaderError:  # ini file was created before CS version 2.4, and we overwrite it
    # create new ini file based on the default ini
    shutil.copyfile(os.path.dirname(os.path.abspath(__file__)) + '/cogstat.ini', dirs.user_config_dir + '/cogstat.ini')
    config.read(dirs.user_config_dir + '/cogstat.ini')
default_config = configparser.ConfigParser(inline_comment_prefixes='#')
default_config.read(os.path.dirname(os.path.abspath(__file__)) + '/cogstat.ini')

# If new key was added to the default ini file, add it to the user ini file
for key in default_config['Preferences'].keys():
    if not(key in config['Preferences'].keys()):
        config['Preferences'][key] = default_config['Preferences'][key]
        with open(dirs.user_config_dir + '/cogstat.ini', 'w') as configfile:
            config.write(configfile)

# Read the setting values from cogstat.ini
language = config['Preferences']['language']
try:
    # because configparser cannot handle multiple values for a single key, split the values
    theme = config['Preferences']['theme'].split(',')
except KeyError:
    theme = ''
image_format = config['Preferences']['image_format']
detailed_error_message = False if config['Preferences']['detailed_error_message'] == 'False' else True


# 2. Other settings
# Output settings
default_font = 'arial'
default_font_size = 9.5
# Define cs specific tags as html tags
cs_tags = {'<cs_h1>': '<h2>',
           '</cs_h1>': '</h2>',
           '<cs_h2>': '<h3>',
           '</cs_h2>': '</h3>',
           '<cs_h3>': '<h4>',
           '</cs_h3>': '</h4>',
           '<cs_h4>': '<h5>',
           '</cs_h4>': '</h5>',
           '<cs_decision>': '<font style="color: green">',
           '</cs_decision>': '</font>',
           '<cs_warning>': '<font style="color: orange">',
           '</cs_warning>': '</font>',
           '<cs_fix_width_font>': '<font style="font-family: courier">',
           '</cs_fix_width_font>': '</font>'}
# Other chart parameters
fig_size_x = 8  # in inch
fig_size_y = 6  # in inch
# graph size will not give nice graphs with too small values - it is a matplotlib issue
graph_font_size = 'medium'
graph_title_size = 'medium'


def save(config_key, value):
    """
    Save the settings to cogstat.ini file. This should be called when Settings are changed in the Preferences.

    Parameters
    ==========
    config_key : str
        key of the settings
    value : str
        value for the key
    """
    config['Preferences'][config_key] = value
    with open(dirs.user_config_dir + '/cogstat.ini', 'w') as configfile:
        config.write(configfile)

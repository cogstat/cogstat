# -*- coding: utf-8 -*-
"""
Settings
"""

import os

import configobj  # Would it be better to use the standard configparser module?
import appdirs  # The module handles the OS-specific user config dirs

# Settings not handled in cogstat.ini
output_type = 'ipnb'  # if run from GUI, this is switched to 'gui' any other
# code will leave the output (e.g., for testing)
# All other settings values are stored in the cogstat.ini file

# Handle cogstat.ini file in user config dirs
dirs = appdirs.AppDirs('cogstat')

# If there is no cogstat-ini file for the user, create one with the default values
if not os.path.isfile(dirs.user_config_dir + '/cogstat.ini'):
    import shutil
    if not os.path.exists(dirs.user_config_dir):
        os.makedirs(dirs.user_config_dir)
    shutil.copyfile(os.path.dirname(os.path.abspath(__file__)) + '/cogstat.ini', dirs.user_config_dir + '/cogstat.ini')

# config is the user-specific file, default_config includes the default values
config = configobj.ConfigObj(dirs.user_config_dir + '/cogstat.ini')
default_config = configobj.ConfigObj(os.path.dirname(os.path.abspath(__file__)) + '/cogstat.ini')
#old_config = dict(config)

# If new key was added to the default ini file, add it to the user ini file
for key in default_config.keys():
    # TODO if new section is added, this code cannot handle it
    if isinstance(default_config[key], str):
        if not(key in config.keys()):
            config[key] = default_config[key]
            config.write()
    else:
        for key2 in default_config[key].keys():
            if not(key2 in config[key].keys()):
                config[key][key2] = default_config[key][key2]
                config.write()
"""
# This will not only add new keys from default ini to existing custom ini,
but  will change values to default, too
config.merge(default_config)
if old_config != dict(config):
    config.write()
"""

# Read the setting values from cogstat.ini

# UI language
language = config['language']

# Output styles
default_font = config['style']['default output font']
default_font_size = float(config['style']['default output font size'])

styles = config['other styles']  # it reads the params as dictionary

# Graph parameters
try:
    theme = config['graph']['theme']
except KeyError:
    theme = ''
ind_line_col = str(config['graph']['individual line color'])
fig_size_x = int(config['graph']['graph x size'])
fig_size_y = int(config['graph']['graph y size'])
graph_font_size = config['graph']['graph font size']
graph_title_size = config['graph']['graph title size']
image_format = config['graph']['format']
versions = {}  # To be modified from cogstat.py


def save(keys, value):
    """
    Save the settings to cogstat.ini file. This should be called when Settings are changed in the Preferences.

    Parameters
    ==========
    keys : list of str (1 or 2 items)
        key of the settings
    value : str
        value for the key
    """
    if len(keys) == 2:
        config[keys[0]][keys[1]] = value
    else:
        config[keys[0]] = value
    config.write()

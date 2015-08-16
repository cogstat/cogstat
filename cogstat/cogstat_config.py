# -*- coding: utf-8 -*-
"""
Settings
"""

import configobj  # Would it be better to use the standard configparser module?
import appdirs
import os

dirs = appdirs.AppDirs('cogstat')

if not os.path.isfile(dirs.user_config_dir+'/cogstat.ini'):
    import shutil
    if not os.path.exists(dirs.user_config_dir):
        os.makedirs(dirs.user_config_dir)
    shutil.copyfile(os.path.dirname(os.path.abspath(__file__))+'/cogstat.ini', dirs.user_config_dir+'/cogstat.ini')
config = configobj.ConfigObj(dirs.user_config_dir+'/cogstat.ini')

# UI language
language = config['language']

# Output styles
heading_style_begin = config['style']['heading begin']
heading_style_end = config['style']['heading end']
default_font = config['style']['default output font']
default_font_size = float(config['style']['default output font size'])
graph_font_size = float(config['style']['graph font size'])

styles = config['other styles']  # it reads the params as dictionary

# Graph parameters
bg_col = config['graph']['background color']
fig_col = config['graph']['figure color']
fig_col_bold = config['graph']['figure color']
ind_line_col = str(config['graph']['individual line color'])
fig_size_x = int(config['graph']['graph x size'])
fig_size_y = int(config['graph']['graph y size'])

versions = {}  # To be modified from cogstat.py


def save(key, value):
    config[key] = value
    config.write()

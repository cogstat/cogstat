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
test_functions = False  # features can be switched on and off with this
# TODO do we need this anymore?
app_devicePixelRatio = 1.0  # this will be overwritten from cogstat_gui; this is needed for high dpi screens

# 1. versions
# Find the versions of the different components.
# Used for diagnostic and for version specific codes.

versions = {}  # To be modified from cogstat.py

import platform
versions['platform'] = platform.platform()

# Python components
import sys
versions['python'] = sys.version
try:
    import numpy
    versions['numpy'] = numpy.__version__
except (ModuleNotFoundError, NameError):
    versions['numpy'] = None
try:
    import pandas
    versions['pandas'] = pandas.__version__
    # csc.versions['pandas'] = pandas.version.version
except (ModuleNotFoundError, NameError):
    versions['pandas'] = None
try:
    import scipy.stats
    versions['scipy'] = scipy.version.version
except (ModuleNotFoundError, NameError):
    versions['scipy'] = None
try:
    import statsmodels
    versions['statsmodels'] = statsmodels.version.version
except (ModuleNotFoundError, NameError, AttributeError):
    try:
        versions['statsmodels'] = statsmodels.__version__
    except NameError:
        versions['statsmodels'] = None
try:
    import pingouin
    versions['pingouin'] = pingouin.__version__
except (ModuleNotFoundError, NameError):
    versions['pingouin'] = None
try:
    import matplotlib
    versions['matplotlib'] = matplotlib.__version__
    versions['matplotlib_backend'] = matplotlib.get_backend()
except (ModuleNotFoundError, NameError):
    versions['matplotlib'] = None
    versions['matplotlib_backend'] = None
try:
    from PyQt6.QtCore import PYQT_VERSION_STR
    versions['pyqt'] = PYQT_VERSION_STR
    # PyQt style can be checked only if the window is open and the object
    # is available
    # It is GUI specific
    # csc.versions['pyqtstyle'] =
    # main_window.style().metaObject().className()
except (ModuleNotFoundError, NameError):
    versions['pyqt'] = None
    # csc.versions['pyqtstyle'] = None

# R components
try:
    # rpy2 tries to find R in PATH or in R_HOME https://rpy2.github.io/doc/v3.5.x/html/rinterface.html
    # "If calling initr() returns an error stating that R_HOME is not defined, you should either have the R executable
    # in your path (PATH on unix-alikes, or Path on Microsoft Windows) or have the environment variable R_HOME defined."
    #
    # First, let's check if CS is used from an application version where R is also copied next to Python and CS, and
    # R can be reached with the appropriate relative path. If this is the case, set R_HOME to that path.
    if os.path.exists('R'):  # in an installed/application version, R is included in this relative path
        os.environ['R_HOME'] = 'R'
    # Otherwise, use PATH or R_HOME which was set either by the R installation, or by the user
    # (e.g., if they want to specify the version to use among several installed versions).
    # Note that R_HOME may not be set in Windows after installation. https://github.com/rpy2/rpy2/issues/796

    # It is not trivial to check if R is available. We want to avoid Fatal error, and use exception instead. Also, what
    # works on Linux, does not work on Windows, and the other way around. Specifically:
    # - In Linux, robjects causes Fatal error, but rpy2.rinterface.initr() raises an exception.
    # - In Windows, rpy2.rinterface.initr() raises _csv.Error even if the path is correct. So we can't use it in
    # Windows. However, when the R_HOME is incorrect, robjects raises OSError exception, so it can be used.
    # Note that initr() cannot be run multiple times, so this cannot be used repeatedly to check R availability.
    if sys.platform.startswith('linux'):
        import rpy2.rinterface
        rpy2.rinterface.initr()
        import rpy2.robjects as robjects
        versions['r'] = robjects.r('version')[-2][0]
    elif sys.platform == 'win32':
        import rpy2.robjects as robjects
        versions['r'] = robjects.r('version')[-2][0]
    # TODO solution for Mac
except (ModuleNotFoundError, NameError, FileNotFoundError, OSError):
    versions['r'] = None
try:
    import rpy2
    versions['rpy2'] = rpy2.__version__
except (ModuleNotFoundError, NameError):
    versions['rpy2'] = None
'''
try:
    from rpy2.robjects.packages import importr
    importr('car')
    versions['car'] = True
except:
    versions['car'] = None
'''


# 2. Settings from the .ini file
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


# 3. Other text and chart formatting settings
# Text formatting settings
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

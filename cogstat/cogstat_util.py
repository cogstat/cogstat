# -*- coding: utf-8 -*-
"""
Various functions for CogStat.
"""

import os
import sys
import gettext

import numpy as np

from . import cogstat_config as csc

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext


def print_versions(main_window):
    text_output = '<cs_h1>' + _('System components') + '</cs_h1>'
    text_output += 'CogStat: %s\n' % csc.versions['cogstat']
    text_output += 'CogStat path: %s\n' % os.path.dirname(os.path.abspath(__file__))
    text_output += 'Platform: %s\n' % csc.versions['platform']
    text_output += 'Python: %s\n' % csc.versions['python']
    text_output += 'Python interpreter path: %s\n' % sys.executable
    try:
        text_output += 'Stdout encoding: %s\n' % str(sys.stdout.encoding)
    except:  # TODO add exception type
        pass
        # with pythonw stdout is not available
    text_output += 'Filesystem encoding: %s\n' % str(sys.getfilesystemencoding())
    text_output += 'Language: %s\n' % csc.language
    text_output += 'Numpy: %s\n' % csc.versions['numpy']
    text_output += 'Scipy: %s\n' % csc.versions['scipy']
    text_output += 'Pandas: %s\n' % csc.versions['pandas']
    text_output += 'Statsmodels: %s\n' % csc.versions['statsmodels']
    text_output += 'Pingouin: %s\n' % csc.versions['pingouin']
    text_output += 'Matplotlib: %s\n' % csc.versions['matplotlib']
    text_output += 'Matplotlib backend: %s\n' % csc.versions['matplotlib_backend']
    text_output += 'PyQt: %s\n' % csc.versions['pyqt']
    text_output += 'PyQt QStyle:%s\n' % main_window.style().metaObject().className()
    text_output += 'R: %s\n' % csc.versions['r']
    text_output += 'Rpy2: %s\n' % csc.versions['rpy2']

#    text_output += '\n'
#    for param in os.environ.keys():
#        text_output += u'%s %s' % (param,os.environ[param]) + '\n'

    return convert_output({'info': text_output})


def precision(data):
    """Compute the maximal decimal precision in the data.

    Parameters
    ----------
    data: pandas series
        Data series

    Returns
    -------
    int or None
        Maximum number of decimals in data
        None if data was not numerical or empty series was given
    """
    data = data.dropna()  # TODO remove dropna() when all callers already drop it
    if len(data) == 0:
        return None

    # Check if data includes numbers (only the first item is checked here).
    if isinstance(data.iloc[0], (int, float, complex, np.integer)):
        # Use round() to avoid floating-point representation error.
        # It is unlikely that the user uses a scale with higher precision.
        # (Default solutions (dtoa, dragon4, etc.?) did not always seem to work correctly
        # https://stackoverflow.com/questions/55727214/inconsistent-printing-of-floats-why-does-it-work-sometimes,
        # that is why this workaround)
        data = data.round(14)
        # '%s' returns 'x.0' for integers, so use '%d' for integers which returns 'x'
        return max([len(('%d' % x if float(x).is_integer() else '%s' % x).partition('.')[2]) for x in data])
    else:
        return None


def change_color(color, saturation=1.0, brightness=1.0):
    """Modify a color.

    Parameters
    ----------
    color : color format recognized by matplotlib
        color to change
    saturation : float
        multiply original saturation value of HSV with this number
    brightness : float
        multiply original brightness (or value in HSV terms) value of HSV with this number

    Returns
    -------
    str in hex color '#rrggbb'
        modified color
    """
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_hex

    color = to_hex(color)
    hsv_color = rgb_to_hsv(list(int(color[i:i + 2], 16) / 256 for i in (1, 3, 5)))
    #print(color, hsv_color)
    hsv_color[1] = min(1, hsv_color[1] * saturation)  # change the saturation, which cannot be larger than 1
    hsv_color[2] = min(1, hsv_color[2] * brightness)  # change the brightness, which cannot be larger than 1
    #print(matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb(hsv_color)))

    return to_hex(hsv_to_rgb(hsv_color))


def convert_output(outputs):
    """Convert dict-based output.
    - remove Nones
    - convert strings
    - check for invalid items

    Parameters
    ----------
    outputs : dict with values of str or matplotlib figures or None or list of these
        dict of the output items

    Returns
    -------
    dict with values of str or matplotlib figures or None or list of these
        converted output, dict of output items
    """

    import logging
    from matplotlib.figure import Figure
    from pandas.io.formats.style import Styler

    if csc.output_type in ['ipnb', 'gui']:
        def convert_item(item):
            """Convert items, when needed

            Parameters
            ----------
            item : item to be converted

            Returns
            -------
            converted item
            """
            if isinstance(item, (Figure, Styler)):  # keep the matplotlib figure and pandas styler
                return item
            elif isinstance(item, str):
                return _reformat_string(item)
            elif item is None:  # keep None (which will be removed)
                return None
            else:  # No other types are expected
                logging.error('Output includes wrong type: %s' % type(item))

        new_output = {}
        for key, value in outputs.items():
            if isinstance(value, list):
                new_output[key] = [convert_item(item) for item in value]
                new_output[key] = [item for item in new_output[key] if not (item is None)]  # remove Nones
            else:  # value is not a list
                if value is not None:
                    new_output[key] = convert_item(value)
        return new_output
    else:
        return outputs


def _reformat_string(string):
    """Reformat the string to display
    1. Change CogStat-specific tags to html tags.
    2. Change various non-html compatible pieces to html pieces.

    Parameters
    ----------
    string : str
        text to reformat

    Returns
    -------
    str
        reformatted output
    """
    # Change Python '\n' to html <br>
    string = string.replace('\n', '<br>')

    # Change custom cogstat tags to html tags as defined in the csc file
    for cs_tag_key in csc.cs_tags.keys():
        string = string.replace(cs_tag_key, csc.cs_tags[cs_tag_key])

    # In the R output the '< ' (which is non-breaking space here (\xa0) )
    # would be handled as html tag in cogstat, so we change it to '&lt; '
    string = string.replace('<\xa0', '&lt; ')

    return string

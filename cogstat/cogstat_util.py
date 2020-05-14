# -*- coding: utf-8 -*-
"""
Various functions for CogStat.
"""

import os
import sys

import numpy as np

from . import cogstat_config as csc

app_devicePixelRatio = 1.0  # this will be overwritten from cogstat_gui; this is needed for high dpi screens


def get_versions():
    """
    Find the versions of the different components.
    Used for diagnostic and for version specific codes.
    """
    import platform

    csc.versions['platform'] = platform.platform()

    # Python components
    csc.versions['python'] = sys.version
    try:
        import numpy
        csc.versions['numpy'] = numpy.__version__
    except (ModuleNotFoundError, NameError):
        csc.versions['numpy'] = None
    try:
        import pandas
        csc.versions['pandas'] = pandas.__version__
        # csc.versions['pandas'] = pandas.version.version
    except (ModuleNotFoundError, NameError):
        csc.versions['pandas'] = None
    try:
        import scipy.stats
        csc.versions['scipy'] = scipy.version.version
    except (ModuleNotFoundError, NameError):
        csc.versions['scipy'] = None
    try:
        import statsmodels
        csc.versions['statsmodels'] = statsmodels.version.version
    except (ModuleNotFoundError, NameError, AttributeError):
        try:
            csc.versions['statsmodels'] = statsmodels.__version__
        except NameError:
            csc.versions['statsmodels'] = None
    try:
        import matplotlib
        csc.versions['matplotlib'] = matplotlib.__version__
        csc.versions['matplotlib_backend'] = matplotlib.get_backend()
    except (ModuleNotFoundError, NameError):
        csc.versions['matplotlib'] = None
        csc.versions['matplotlib_backend'] = None
    try:
        from PyQt5.Qt import PYQT_VERSION_STR
        csc.versions['pyqt'] = PYQT_VERSION_STR
        # PyQt style can be checked only if the window is open and the object
        # is available
        # It is GUI specific
        # csc.versions['pyqtstyle'] =
        # main_window.style().metaObject().className()
    except (ModuleNotFoundError, NameError):
        csc.versions['pyqt'] = None
        # csc.versions['pyqtstyle'] = None

    # R components
    '''
    try:
        import rpy2.robjects as robjects
        csc.versions['r'] = robjects.r('version')[12][0]
    except:
        csc.versions['r'] = None
    try:
        import rpy2
        csc.versions['rpy2'] = rpy2.__version__
    except:
        csc.versions['rpy2'] = None
    try:
        from rpy2.robjects.packages import importr
        importr('car')
        csc.versions['car'] = True
    except:
        csc.versions['car'] = None
    '''


def print_versions(main_window):
    text_output = ''
    text_output += 'CogStat: %s\n' % csc.versions['cogstat']
    text_output += 'CogStat path: %s\n' % \
                   os.path.dirname(os.path.abspath(__file__))
    text_output += 'Platform: %s\n' % csc.versions['platform']
    text_output += 'Python: %s\n' % csc.versions['python']
    text_output += 'Python interpreter path: %s\n' % sys.executable
    try:
        text_output += 'Stdout encoding: %s\n' % str(sys.stdout.encoding)
    except:  # TODO add exception type
        pass
        # with pythonw stdout is not available
    text_output += 'Filesystem encoding: %s\n' % \
                   str(sys.getfilesystemencoding())
    text_output += 'Language: %s\n' % csc.language
    text_output += 'Numpy: %s\n' % csc.versions['numpy']
    text_output += 'Scipy: %s\n' % csc.versions['scipy']
    text_output += 'Pandas: %s\n' % csc.versions['pandas']
    text_output += 'Statsmodels: %s\n' % csc.versions['statsmodels']
    text_output += 'Matplotlib: %s\n' % csc.versions['matplotlib']
    text_output += 'Matplotlib backend: %s\n' % \
                   csc.versions['matplotlib_backend']
    text_output += 'PyQt: %s\n' % csc.versions['pyqt']
    text_output += 'PyQt QStyle:%s\n' % \
                   main_window.style().metaObject().className()
    # text_output += 'R: %s\n' % csc.versions['r']
    # text_output += 'Rpy2: %s\n' % csc.versions['rpy2']

#    import os
#    text_output += '\n'
#    for param in os.environ.keys():
#        text_output += u'%s %s' % (param,os.environ[param]) + '\n'

    return text_output


def print_p(p):
    """
    Makes an output according to the APA rule:
    if p < 0.001, then print 'p < 0.001'
    otherwise 'p = value'
    """
    return '<i>p</i> &lt; 0.001' if p < 0.001 else '<i>p</i> = %0.3f' % p


def precision(data):
    """Compute the maximal decimal precision in the data.
    data: pandas series

    returns:
        maximum number of decimals in list, or None if data are not numerical
        or empty list was given
    """
    data = data.dropna()
    if len(data) == 0:
        return None

    # Check if data includes numbers (actually only the first item is checked)
    # np.integer should also be included, because in some systems it is not
    # recognised as int or
    # http://stackoverflow.com/questions/4187185/how-can-i-check-if-my-python-object-is-a-number
    if isinstance(data.iloc[0], (int, float, complex, np.integer)):
        return max([len(('%d' % x if int(x) == x else '%s' % x).
                        partition('.')[2]) for x in data])
    else:
        return None


def convert_output(outputs):
    """
    Convert output either to the GUI or to the IPython Notebook
    :param outputs: list of the output items
    :return: converted output, list of items
    """

    import logging
    from matplotlib.figure import Figure
    from matplotlib import rcParams
    from PyQt5 import QtGui

    rcParams['figure.figsize'] = csc.fig_size_x, csc.fig_size_y

    def _figure_to_qimage(figure):
        """Convert matplotlib figure to pyqt qImage.
        """
        figure.canvas.draw()
        size_x, size_y = figure.get_size_inches()*rcParams['figure.dpi']
        # TODO is it better to use figure.canvas.width(), figure.canvas.height()
        string_buffer = figure.canvas.buffer_rgba()
        qimage = QtGui.QImage(string_buffer, size_x*app_devicePixelRatio,
                              size_y*app_devicePixelRatio, QtGui.QImage.Format_ARGB32).rgbSwapped().copy()
        QtGui.QImage.setDevicePixelRatio(qimage, app_devicePixelRatio)
        return qimage
            # I couldn't see it documented, but seemingly the figure uses BGR, not RGB coding
            # this should be a copy, otherwise closing the matplotlib figures would damage the qImages on the GUI

    if csc.output_type in ['ipnb', 'gui']:
        # convert custom notation to html
        new_output = []
        for i, output in enumerate(outputs):
            if isinstance(output, Figure):
                # For gui convert matplotlib to qImage
                new_output.append(output if csc.output_type == 'ipnb' else _figure_to_qimage(output))
            elif isinstance(output, str):
                new_output.append(reformat_output(output))
            elif isinstance(output, list):  # flat list
                new_output.extend(convert_output(output))
            elif output is None:
                pass  # drop None-s from outputs
            else:  # No other types are expected
                logging.error('Output includes wrong type: %s' % type(output))
        return new_output
    else:
        return outputs


def reformat_output(output):
    """Reformat the output to display
    :param output: str - text to reformat
    :return: reformatted str
    """
    if isinstance(output, str):  # TODO in Python3 not needed anymore?
        output = str(output)
    output = output.replace('\n', '<br>')
    for style_element in list(csc.styles.keys()):
        output = output.replace(style_element, csc.styles[style_element])
        output = output.replace(str(style_element),
                                str(csc.styles[style_element]))
    # In the R output the '< ' (which is non breaking space here (\xa0) )
    # would be handled as html tag
    output = output.replace('<\xa0', '&lt; ')
    return output

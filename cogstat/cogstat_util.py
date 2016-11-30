# -*- coding: utf-8 -*-
"""
Various functions for CogStat.
"""

import sys
import os

import numpy as np

import cogstat_config as csc


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
    except:
        csc.versions['numpy'] = None
    try:
        import pandas
        csc.versions['pandas'] = pandas.__version__
        #csc.versions['pandas'] = pandas.version.version
    except:
        csc.versions['pandas'] = None
    try:
        import scipy.stats
        csc.versions['scipy'] = scipy.version.version
    except:
        csc.versions['scipy'] = None
    try:
        import statsmodels
        csc.versions['statsmodels'] = statsmodels.version.version
    except:
        csc.versions['statsmodels'] = None
    try:
        import matplotlib
        csc.versions['matplotlib'] = matplotlib.__version__
    except:
        csc.versions['matplotlib'] = None
    try:
        import PyQt4.pyqtconfig
        csc.versions['pyqt'] = PyQt4.pyqtconfig.Configuration().pyqt_version_str
        # PyQt style can be checked only if the window is open and the object is available
        # It is GUI specific
        #csc.versions['pyqtstyle'] = main_window.style().metaObject().className()
    except:
        csc.versions['pyqt'] = None
        #csc.versions['pyqtstyle'] = None

    # R components
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


def print_versions():
    text_output = u''
    text_output += 'CogStat: %s\n' % csc.versions['cogstat']
    text_output += 'Platform: %s\n' % csc.versions['platform']
    text_output += 'Python: %s\n' % csc.versions['python']
    text_output += 'Default encoding: %s\n' % str(sys.getdefaultencoding())
    text_output += 'Stdout encoding: %s\n' % str(sys.stdout.encoding)
    text_output += 'Filesystem encoding: %s\n' % str(sys.getfilesystemencoding())
    text_output += 'Numpy: %s\n' % csc.versions['numpy']
    text_output += 'Scipy: %s\n' % csc.versions['scipy']
    text_output += 'Pandas: %s\n' % csc.versions['pandas']
    text_output += 'Statsmodels: %s\n' % csc.versions['statsmodels']
    text_output += 'Matplotlib: %s\n' % csc.versions['matplotlib']
    text_output += 'PyQt: %s\n' % csc.versions['pyqt']
    #text_output += 'PyQt QStyle:%s\n' % csc.versions['pyqtstyle']
    text_output += 'R: %s\n' % csc.versions['r']
    text_output += 'Rpy2: %s\n' % csc.versions['rpy2']
    text_output += 'CogStat path: %s\n' % os.path.dirname(os.path.abspath(__file__))

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
    # np.integer should also be included, because in some systems it is not recognised as int
    # or http://stackoverflow.com/questions/4187185/how-can-i-check-if-my-python-object-is-a-number
    if isinstance(data.iloc[0], (int, long, float, complex, np.integer)):
        return max([len(('%d' % x if int(x) == x else '%s' % x).partition('.')[2]) for x in data])
    else:
        return None


def reformat_output(output):
    """Reformat the output to display
    :param output: str - text to reformat
    :return: reformatted str
    """
    if type(output) is str:
        output = unicode(output, encoding='utf-8')
    output = output.replace('\n', '<br>')
    for style_element in csc.styles.keys():
        output = output.replace(style_element, csc.styles[style_element])
        output = output.replace(unicode(style_element), unicode(csc.styles[style_element]))
    output = output.replace(u'<\xa0', '&lt; ')  # In the R output the '< ' (which is non breaking space here (\xa0) ) would be handled as html tag
    return output

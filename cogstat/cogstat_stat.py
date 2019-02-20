# -*- coding: utf-8 -*-

"""
This module contains functions for statistical analysis.
The functions will calculate the text and graphical results that are
compiled in cogstat methods.

Arguments are the pandas data frame (pdf), and parameters (among others they
are usually variable names).
Output is text (html and some custom notations), images (matplotlib)
and list of images.

Mostly scipy.stats, statsmodels and matplotlib is used to generate the results.
"""

import gettext
import os
import numpy as np
import statsmodels.api as sm
import string
import sys
import textwrap
from io import StringIO
from distutils.version import LooseVersion
from scipy import stats
import itertools

from . import cogstat_config as csc
from . import cogstat_util as cs_util
from . import cogstat_stat_num as cs_stat_num

try:
    from statsmodels.graphics.mosaicplot import mosaic
except:
    pass
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.sandbox.stats.runs import mcnemar
from statsmodels.sandbox.stats.runs import cochrans_q
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr  # requires at least rpy 2.1.0; use instead:
    #from rpy2.robjects import r # http://stackoverflow.com/questions/2128806/python-rpy2-cant-import-a-bunch-of-packages
except:
    pass

matplotlib.pylab.rcParams['figure.figsize'] = csc.fig_size_x, csc.fig_size_y

### Set matplotlib styles ###
# Set the styles
if csc.theme not in plt.style.available:
    csc.theme = sorted(plt.style.available)[0]
    csc.save(['graph', 'theme'], csc.theme)
plt.style.use(csc.theme)

#print plt.style.available
#style_num = 15
#print plt.style.available[style_num]
#plt.style.use(plt.style.available[style_num])
theme_colors = [col['color'] for col in list(plt.rcParams['axes.prop_cycle'])]
#print theme_colors
# this is a workaround, as 'C0' notation does not seem to work

# Overwrite style parameters when needed
# https://matplotlib.org/tutorials/introductory/customizing.html
# Some dashed and dotted axes styles (which are simply line styles) are hard to differentiate, so we overwrite the style
#print matplotlib.rcParams['lines.dashed_pattern'], matplotlib.rcParams['lines.dotted_pattern']
matplotlib.rcParams['lines.dashed_pattern'] = [6.0, 6.0]
matplotlib.rcParams['lines.dotted_pattern'] = [1.0, 3.0]
#print matplotlib.rcParams['axes.spines.left']
#print matplotlib.rcParams['font.size'], matplotlib.rcParams['font.serif'], matplotlib.rcParams['font.sans-serif']
#print matplotlib.rcParams['axes.titlesize'], matplotlib.rcParams['axes.labelsize']
matplotlib.rcParams['axes.titlesize'] = csc.graph_font_size # title of the charts
matplotlib.rcParams['axes.labelsize'] = csc.graph_font_size # labels of the axis
#print matplotlib.rcParams['xtick.labelsize'], matplotlib.rcParams['ytick.labelsize']
#print matplotlib.rcParams['figure.facecolor']
#matplotlib.rcParams['figure.facecolor'] = csc.bg_col
# Make sure that the axes are visible
#print matplotlib.rcParams['axes.facecolor'], matplotlib.rcParams['axes.edgecolor']
if matplotlib.colors.to_rgba(matplotlib.rcParams['figure.facecolor']) == matplotlib.colors.to_rgba(matplotlib.rcParams['axes.edgecolor']):
    #print matplotlib.colors.to_rgba(matplotlib.rcParams['axes.edgecolor'])
    matplotlib.rcParams['axes.edgecolor'] = 'w' if matplotlib.colors.to_rgba(matplotlib.rcParams['axes.edgecolor'])==(0, 0, 0, 0) else 'k'

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.ugettext

# matplotlib does not support rtl Unicode yet (http://matplotlib.org/devel/MEP/MEP14.html),
# so we have to handle rtl text on matplotlib plots
rtl_lang = True if csc.language in ['he', 'fa', 'ar'] else False
if rtl_lang:
    from bidi.algorithm import get_display
    _plt = lambda text: get_display(t.ugettext(text))
else:
    _plt = t.ugettext

warn_unknown_variable = '<warning>'+_('The properties of the variables are not set. Set them in your data source.') \
                        + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                        % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                        + '\n<default>' # XXX ezt talán elég az importnál nézni, az elemzéseknél lehet már másként.

### Various things ###

def _get_R_output(obj):
    """
    Returns the output of R, printing obj.
    """
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    robjects.r('options')(width=200)  # Controls wrapping the output
    #print obj
    sys.stdout = old_stdout
    return mystdout.getvalue()


def _split_into_groups(pdf, var_name, grouping_name):
    """
    arguments:
    var_name (str): name of the dependent var
    grouping_name (list of str): name of the grouping var(s)
    
    return:
    level_combinations (list of str or list of tuples of str): list of group levels (for one grouping variable)
        or list of tuples of group levels (for more than one grouping variable)
    grouped data: list of pandas series
    """

    if isinstance(grouping_name, (str)):  # TODO list is required, fix the calls sending string
        grouping_name = [grouping_name]
    # create a list of sets with the levels of all grouping variables
    levels = [set(pdf[group].dropna()) for group in grouping_name]
    # create all level combinations for the grouping variables
    level_combinations = list(itertools.product(*levels))
    grouped_data = [pdf[var_name][(pdf[grouping_name] == pd.Series({group: level for group, level in zip(grouping_name, group_level)})).all(axis=1)].dropna() for group_level in
                 level_combinations]
    return level_combinations, grouped_data


def _wrap_labels(labels):
    """
    labels: list of strings
            or list of lists of single strings
    """
    label_n = len(labels)
    max_chars_in_row = 55
        # TODO need a more precise method; should depend on font size and graph size;
        # but cannot be a very precise method unless the font is fixed width
    if isinstance(labels[0], (list, tuple)):
        wrapped_labels = [textwrap.fill(' : '.join(map(str, label)), max(5, max_chars_in_row/label_n)) for label in
                          labels]
    else:
        wrapped_labels = [textwrap.fill(str(label), max(5, max_chars_in_row / label_n)) for label in
                          labels]
        # the width should not be smaller than a min value, here 5
        # use the unicode() to convert potentially numerical labels
        # TODO maybe for many lables use rotation, e.g., http://stackoverflow.com/questions/3464359/is-it-possible-to-wrap-the-text-of-xticks-in-matplotlib-in-python
    return wrapped_labels

def _set_axis_measurement_level (ax, x_measurement_level, y_measurement_level):
    """
    Set the axes types of the graph acording to the measurement levels of the variables.
    :param ax: ax object
    :param x_measurement_type: str 'nom', 'ord' or 'int'
    :param y_measurement_type: str 'nom', 'ord' or 'int'
    :return: nothing, the ax object is modified in place
    """

    # Switch off top and right axes
    ax.tick_params(top=False, right=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set the style of the bottom and left spines according to the measurement levels
    measurement_level_to_line_styles = {'int': 'solid', 'ord': 'dashed', 'nom': 'dotted'}
    ax.spines['bottom'].set_linestyle(measurement_level_to_line_styles[x_measurement_level])
    ax.spines['left'].set_linestyle(measurement_level_to_line_styles[y_measurement_level])


def _format_html_table(html_table, add_style=False):
    """Format html table

    :return: str html
    """
    if add_style:
        # Because qt does not support table borders, use padding to have a more reviewable table
        return '<style> th, td {padding-right: 5px; padding-left: 5px} </style>' + html_table.replace('\n', '').replace('border="1"', 'style="border:1px solid black;"')
    else:
        return html_table.replace('\n', '').replace('border="1"', 'style="border:1px solid black;"')

def pivot(pdf, row_names, col_names, page_names, depend_name, function):
    """
    Build pivot table
    all parameters are lists # TODO doc
    """
            
    if len(depend_name) != 1:
        return _('Sorry, only one dependent variable can be used.')
    if pdf[depend_name[0]].dtype == 'object':
        return _('Sorry, string variables cannot be used in Pivot table.')
    function_code = {'N': 'len', 'Sum': 'np.sum', 'Mean': 'np.mean', 'Median': 'median', 'Lower quartile': 'perc25',
                     'Upper quartile': 'perc75', 'Standard deviation': 'np.std', 'Variance': 'np.var'}
    result = ''
    if page_names:
        result += _('Independent variable(s) - Pages: ') + ', '.join(x for x in page_names) + '\n'
    if col_names:
        result += _('Independent variable(s) - Columns: ') + ', '.join(x for x in col_names) + '\n'
    if row_names:
        result += _('Independent variable(s) - Rows: ') + ', '.join(x for x in row_names) + '\n'
    result += _('Dependent variable: ') + depend_name[0] + '\n' + _('Function: ') + function + '\n'

    if function == 'N':
        prec = 0
    else:
        prec = cs_util.precision(pdf[depend_name[0]])+1

    def format_output(x):
        return '%0.*f' % (prec, x)
    
    def print_pivot_page(df, page_vars, page_value_list='', ptable_result=''):
        """
        Specify the pages recursively.
        """

        # A. These should be used to handle missing data correctly
        # 1. pandas.pivot_table cannot drop na-s
        # 2. np.percentile cannot drop na-s
        # B. These are used to wrap parameters

        def perc25(data):
            return np.percentile(data.dropna(), 25)

        def median(data):
            # TODO ??? Neither np.median nor stats.nanmedian worked corrrectly ??? or Calc computes percentile in a different way?
            return np.percentile(data.dropna(), 50)

        def perc75(data):
            return np.percentile(data.dropna(), 75)

        if page_vars:
            page_values = set(df[page_vars[0]])
            for page_value in page_values:
                if page_value == page_value:  # skip nan
                    ptable_result = print_pivot_page(df[df[page_vars[0]] == page_value], page_vars[1:], '%s%s = %s%s' % (page_value_list, page_vars[0], page_value, ', ' if page_vars[1:] else ''), ptable_result)
        else:  # base case
            if page_value_list:
                ptable_result = '%s\n\n%s' % (ptable_result, page_value_list)
            if row_names or col_names:
                
                if LooseVersion(csc.versions['pandas']) < LooseVersion('0.14'):
                    ptable = pd.pivot_table(df, values=depend_name, rows=row_names, cols=col_names,
                                            aggfunc=eval(function_code[function]))
                else:
                    ptable = pd.pivot_table(df, values=depend_name, index=row_names, columns=col_names,
                                            aggfunc=eval(function_code[function]))
                ptable_result = '%s\n%s' % (ptable_result, _format_html_table(ptable.
                                            to_html(bold_rows=False, sparsify=False, float_format=format_output),
                                                                              add_style=False))
            else:
                temp_result = eval(function_code[function])(np.array(df[depend_name]))
                # TODO convert to html output; when ready stop using fix_width_font
                # np.array conversion needed, np.mean requires axis name for pandas dataframe
                ptable_result = '%s\n%s' % (ptable_result, temp_result)
        return ptable_result
    result += '<fix_width_font>%s\n<default>' % print_pivot_page(pdf, page_names)
    return result


def safe_var_names(names):  # TODO not used at the moment. maybe could be deleted.
    """Change the variable names for R."""
    # TODO exclude unicode characters
    for i in range(len(names)):
        names[i] = str(names[i]).translate(string.maketrans(' -', '__'))  # use underscore instead of space or dash
        if names[i][0].isdigit():  # do not start with number
            names[i]='_'+names[i]
        name_changed = False
        for j in range(i):
            while names[i]==names[j]:
                if not name_changed:
                    names[i] = names[i]+'_1'
                    name_changed = True
                else:
                    underscore_pos = names[i].rfind('_')
                    names[i] = names[i][:underscore_pos]+'_'+str(int(names[i][underscore_pos+1:])+1)
    return names    
# test safe_var_names
#names = ['something', '123asd', 's o m e t h i n g'] + ['something']*20
#print names
#print safe_var_names(names)

### Single variables ###


def display_variable_raw_data(pdf, data_measlevs, var_name):
    """Display n of valid valid and display raw data on a chart
    """
    data = pdf[var_name].dropna()

    text_result=''
    text_result += _('N of valid cases: %g') % len(data) + '\n'
    missing_cases = len(pdf[var_name])-len(data)
    text_result += _('N of missing cases: %g') % missing_cases + '\n'

    if data_measlevs[var_name] == 'ord':
        data_value = pdf[var_name].dropna()
        data = pd.Series(stats.rankdata(data_value))
    if data_measlevs[var_name] in ['int', 'ord', 'unk']:
        # Upper part with histogram and individual data
        fig = plt.figure(figsize=(csc.fig_size_x, csc.fig_size_y * 0.25))
        ax = plt.gca()
        # Add individual data
        plt.scatter(data, np.random.random(size=len(data)), color=theme_colors[0], marker='o')
        ax.axes.set_ylim([-1.5, 2.5])
        fig.subplots_adjust(top=0.85, bottom=0.3)
        # Add labels
        if data_measlevs[var_name] == 'ord':
            plt.title(_plt('Rank of the raw data'))
            plt.xlabel(_('Rank of %s') % var_name)
        else:
            plt.title(_plt('Raw data'))
            plt.xlabel(var_name)
        ax.axes.get_yaxis().set_visible(False)
        if data_measlevs[var_name] == 'ord':
            ax.tick_params(top=False, right=False)
            # Create new tick labels, with the rank and the value of the corresponding rank
            ax.set_xticklabels(['%i\n(%s)' % (i, sorted(data_value)[int(i)-1])
                                if i-1 in range(len(data_value)) else '%i' % i for i in ax.get_xticks()])
            _set_axis_measurement_level(ax, 'ord', 'nom')
    elif data_measlevs[var_name] in ['nom']:
        # For nominal variables the histogram is a frequency graph
        plt.figure()
        values = list(set(pdf[var_name]))
        freqs = [list(pdf[var_name]).count(i) for i in values]
        locs = np.arange(len(values))
        plt.title(_plt('Histogram'))
        plt.bar(locs, freqs, 0.9, color=theme_colors[0])
        plt.xticks(locs+0.9/2., _wrap_labels(values))
        plt.ylabel(_plt('Frequency'))
        ax = plt.gca()
        _set_axis_measurement_level(ax, 'nom', 'int')
    return text_result, plt.gcf()

def frequencies(pdf, var_name, meas_level):
    """Frequencies
    
    arguments:
    var_name (str): name of the variable
    meas_level: measurement level of the variable
    """

    def as_percent(v, precision='0.1'):
        """Convert number to percentage string."""
        # http://blog.henryhhammond.com/pandas-formatting-snippets/
        from numbers import Number
        if isinstance(v, Number):
            return "{{:{}%}}".format(precision).format(v)
        else:
            raise TypeError("Numeric type required")

    # TODO rewrite this; can we use pandas here?
    freq1 = [[i, list(pdf[var_name]).count(i)] for i in set(pdf[var_name])]
    # Remove and count nans - otherwise sort() does not function correctly
    freq = [x for x in freq1 if x[0] == x[0]]
    nan_n = len(freq1) - len(freq)
    freq.sort()
    if nan_n:
        freq.append(['nan', nan_n])
    total_count = float(len(pdf[var_name]))
    running_total = 0
    running_rel_total = 0.0
    for i in range(len(freq)):
        rel_freq = freq[i][1]/total_count
        running_total += freq[i][1]
        running_rel_total += rel_freq
        if meas_level == 'nom':
            freq[i].extend([rel_freq])
        else:
            freq[i].extend([rel_freq, running_total, running_rel_total])
    if meas_level == 'nom':
        column_names = [_('Value'), _('Freq'), _('Rel freq')]
    else:
        column_names = [_('Value'), _('Freq'), _('Rel freq'), _('Cum freq'), _('Cum rel freq')]
    text_result = _format_html_table(pd.DataFrame(freq, columns=column_names).\
        to_html(formatters={_('Rel freq'): as_percent, _('Cum rel freq'): as_percent}, bold_rows=False, index=False))
    return text_result


def histogram(pdf, data_measlevs, var_name):
    """Histogram with individual data and boxplot
    
    arguments:
    var_name (str): name of the variable
    """
    chart_result = ''
    suptitle_text = None
    max_length = 10  # maximum printing length of an item # TODO print ... if it's exceeded
    data = pdf[var_name].dropna()
    if data_measlevs[var_name] == 'ord':
        data_value = pdf[var_name].dropna()  # The original values of the data
        data = pd.Series(stats.rankdata(data_value))  # The ranks of the data
    if data_measlevs[var_name] in ['int', 'ord', 'unk']:
        categories_n = len(set(data))
        if categories_n < 10:
            freq, edge = np.histogram(data, bins=categories_n)
        else:
            freq, edge = np.histogram(data)
#        text_result = _(u'Edge\tFreq\n')
#        text_result += u''.join([u'%.2f\t%s\n'%(edge[j], freq[j]) for j in range(len(freq))])
        
        # Prepare the frequencies for the plot
        val_count = data.value_counts()
        if max(val_count)>1:
            suptitle_text = _plt('Largest tick on the x axes displays %d cases.') % max(val_count)
        val_count = (val_count*(max(freq)/max(val_count)))/20.0

        # Upper part with histogram and individual data
        plt.figure()
        ax_up = plt.axes([0.1, 0.3, 0.8, 0.6])
        plt.hist(data.values, bins=len(edge)-1, color=theme_colors[0])
            # .values needed, otherwise it gives error if the first case is missing data
        # Add individual data
        plt.errorbar(np.array(val_count.index), np.zeros(val_count.shape),
                     yerr=[np.zeros(val_count.shape), val_count.values],
                     fmt='k|', capsize=0, linewidth=2)
        #plt.plot(np.array(val_count.index), np.zeros(val_count.shape), 'k|', markersize=10, markeredgewidth=1.5)
        # Add labels
        if data_measlevs[var_name] == 'ord':
            plt.title(_plt('Histogram of rank data with individual data and boxplot'))
        else:
            plt.title(_plt('Histogram with individual data and boxplot'))
        if suptitle_text:
            plt.suptitle(suptitle_text, x=0.9, y=0.025, horizontalalignment='right', fontsize=10)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.ylabel(_plt('Frequency'))
        # Lower part showing the boxplot
        ax_low = plt.axes([0.1, 0.1, 0.8, 0.2], sharex = ax_up)
        box1 = plt.boxplot(data.values, vert=0, whis='range')  # .values needed, otherwise error when the first case is missing data
        plt.gca().axes.get_yaxis().set_visible(False)
        if data_measlevs[var_name] == 'ord':
            plt.xlabel(_('Rank of %s') % var_name)
        else:
            plt.xlabel(var_name)
        plt.setp(box1['boxes'], color=theme_colors[0])
        plt.setp(box1['whiskers'], color=theme_colors[0])
        plt.setp(box1['caps'], color=theme_colors[0])
        plt.setp(box1['medians'], color=theme_colors[0])
        plt.setp(box1['fliers'], color=theme_colors[0])
        if data_measlevs[var_name] == 'ord':
            ax_low.tick_params(top=False, right=False)
            # Create new tick labels, with the rank and the value of the corresponding rank
            ax_low.set_xticklabels(['%i\n(%s)' % (i, sorted(data_value)[int(i-1)])
                                if i-1 in range(len(data_value)) else '%i' % i for i in ax_low.get_xticks()])
            _set_axis_measurement_level(ax_low, 'ord', 'int')
        chart_result = plt.gcf()
    # For nominal variables the histogram is a frequency graph, which has already been displayed in the Raw data, so it
    # is not repeated here
    return chart_result


def normality_test(pdf, data_measlevs, var_name, group_name='', group_value='', alt_data=None):
    """Check normality
    
    arguments:
    var_name (str):
        Name of the variable to be checked.
    group_name (str):
        Name of the grouping variable if part of var_name should be
        checked. Otherwise ''.
    group_value (str):
        Name of the group in group_name, if grouping is used.
    alt_data (data frame):
        if alt_data is specified, this one is used 
        instead of self.data_frame. This could be useful if some other data
        should be dropped, e.g., in variable comparison, where cases are 
        dropped based on missing cases in other variables.
    
    return:
    norm (bool): is the variable normal (False if normality is violated)
    text_result (html text): APA format
    image (matplotlib): histogram with normal distribution
    image2 (matplotlib): QQ plot
    """
    text_result = ''
    suptitle_text = None
    if repr(alt_data) == 'None':
    # bool(pd.data_frame) would stop on pandas 0.11
    # that's why this weird alt_data check
        temp_data = pdf
    else:
        temp_data = alt_data

    if group_name:
        data = temp_data[temp_data[group_name]==group_value][var_name].dropna()
    else:
        data = temp_data[var_name].dropna()

    if data_measlevs[var_name] in ['nom', 'ord']:
        return False, '<decision>'+_('Normality can be checked only for interval variables.')+'\n<default>', None, None
    if len(set(data)) == 1:
        return False, _('Normality cannot be checked for constant variable in %s%s.\n' % (var_name, ' (%s: %s)' % (group_name, group_value) if group_name else '')), None, None
    # TODO do we need this?
#        if len(data)<7:
#            return False, _(u'Sample size must be greater than 7 for normality test.\n'), None, None

    # http://statsmodels.sourceforge.net/stable/generated/statsmodels.stats.diagnostic.kstest_normal.html#statsmodels.stats.diagnostic.kstest_normal
    # text_result += _('Testing normality with the Kolmogorov-Smirnov test:')+': <i>D</i> = %0.3g, <i>p</i> = %0.3f \n' %sm.stats.kstest_normal(data)
    #text_result += _('Testing normality with the Lillifors test')+': <i>D</i> = %0.3g, <i>p</i> = %0.3f \n' %sm.stats.lillifors(data)
    #A, p = sm.stats.normal_ad(data)
    #text_result += _('Anderson-Darling normality test in variable %s%s') %(var_name, ' (%s: %s)'%(group_name, group_value) if group_name else '') + ': <i>A<sup>2</sup></i> = %0.3g, %s\n' %(A, cs_util.print_p(p))
    #text_result += _('Testing normality with the Anderson-Darling test: <i>A<sup>2</sup></i> = %0.3g, critical values: %r, sig_levels: %r \n') %stats.anderson(data, dist='norm')
    #text_result += _("Testing normality with the D'Agostin and Pearson method")+': <i>k2</i> = %0.3g, <i>p</i> = %0.3f \n' %stats.normaltest(data)
    #text_result += _('Testing normality with the Kolmogorov-Smirnov test')+': <i>D</i> = %0.3g, <i>p</i> = %0.3f \n' %stats.kstest(data, 'norm')
    if len(data) < 3:
        return False, _('Too small sample to test normality in variable %s%s.\n' % (var_name, ' (%s: %s)' % (group_name, group_value) if group_name else '')), None, None
    else:
        W, p = stats.shapiro(data)
        text_result += _('Shapiro-Wilk normality test in variable %s%s') % (var_name, ' (%s: %s)' % (group_name, group_value) if group_name else '') +': <i>W</i> = %0.3g, %s\n' %(W, cs_util.print_p(p))

    # Prepare the frequencies for the plot
    val_count = data.value_counts()
    plt.figure()  # Otherwise the next plt.hist will modify the actual (previously created) graph
    n, bins, patches = plt.hist(data.values, normed=True, color=theme_colors[0])
    if max(val_count) > 1:
        suptitle_text = _plt('Largest tick on the x axes displays %d cases.') % max(val_count)
    val_count = (val_count*(max(n)/max(val_count)))/20.0

    # Graph
    plt.figure()
    n, bins, patches = plt.hist(data.values, normed=True, color=theme_colors[0])
    plt.plot(bins, matplotlib.pylab.normpdf(bins, np.mean(data), np.std(data)), 'g--', linewidth=3)
    plt.title(_plt('Histogram with individual data and normal distribution'))
    if suptitle_text:
        plt.suptitle(suptitle_text, x=0.9, y=0.025, horizontalalignment='right', fontsize=10)
    plt.errorbar(np.array(val_count.index), np.zeros(val_count.shape), 
                 yerr=[np.zeros(val_count.shape), val_count.values],
                 fmt='k|', capsize=0, linewidth = 2)
#    plt.plot(data, np.zeros(data.shape), 'k+', ms=10, mew=1.5)
        # individual data
    plt.xlabel(var_name)
    plt.ylabel(_('Normalized relative frequency'))

    # percent on y axes http://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    def to_percent(y, position):
        s = str(100 * y)
        return s + r'$\%$' if matplotlib.rcParams['text.usetex'] is True else s + '%'
    from matplotlib.ticker import FuncFormatter
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    
    graph = plt.gcf()
    
    # QQ plot
    if csc.versions['statsmodels']:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sm.graphics.qqplot(data, line='s', ax=ax) # TODO set the color
        plt.title(_plt('Quantile-quantile plot'))
        graph2 = plt.gcf()
    else:
        text_result += '\n'+_('Sorry, QQ plot is only displayed if the statsmodels module is installed.')
        graph2 = None
    
    # Decide about normality
    norm = False if p < 0.05 else True
    
    return norm, text_result, graph, graph2


def one_t_test(pdf, data_measlevs, var_name, test_value=0):
    """One sample t-test
    
    arguments:
    var_name (str):
        Name of the variable to test.
    test_value (numeric):
        Test against this value.
        
    return:
    text_result (html str):
        Result in APA format.
    image (matplotlib):
        Bar chart with mean and confidence interval.
    """
    text_result = ''
    data = pdf[var_name].dropna()
    if data_measlevs[var_name] in ['int', 'unk']:
        if data_measlevs[var_name] == 'unk':
            text_result += warn_unknown_variable
        if len(set(data))==1:
            return _('One sample t-test cannot be run for constant variable.\n'), None
                    
        data = pdf[var_name].dropna()
        descr = DescrStatsW(data)
        t, p, df = descr.ttest_mean(float(test_value))
        if LooseVersion(csc.versions['statsmodels']) >= LooseVersion('0.5'):
            # Or we could use confidence_interval_t
            cil, cih = descr.tconfint_mean()
            ci = (cih-cil)/2
            prec = cs_util.precision(data)+1
            ci_text = '[%0.*f, %0.*f]' %(prec, cil, prec, cih)
        else:
            ci = 0  # only with statsmodels
            ci_text=_('Sorry, newer statsmodels module is required for confidence interval.\n')
        text_result += _('One sample t-test against %g')%float(test_value)+': <i>t</i>(%d) = %0.3g, %s\n' %(df, t, cs_util.print_p(p))
        
        # Graph
        plt.figure(figsize=(csc.fig_size_x, csc.fig_size_y*0.35))
        plt.barh([1], [data.mean()], xerr=[ci], color=theme_colors[0], ecolor='black')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlabel(var_name)  # TODO not visible yet, maybe matplotlib bug, cannot handle figsize consistently
        plt.title(_plt('Mean value with 95% confidence interval'))
        image = plt.gcf()
    else:
        text_result += _('One sample t-test is computed only for interval variables.')
        image = None
    return ci_text, text_result, image


def wilcox_sign_test(pdf, data_measlevs, var_name, value=0):
    """Wilcoxon signed-rank test
    
    arguments:
    var_name (str):
    value (numeric):
    """

    text_result = ''
    data = pdf[var_name].dropna()
    if data_measlevs[var_name] in ['int', 'ord', 'unk']:
        if data_measlevs[var_name] == 'unk':
            text_result += warn_unknown_variable
        '''if csc.versions['r']:
            # R version
            # http://ww2.coastal.edu/kingw/statistics/R-tutorials/singlesample-t.html
            r_data = robjects.FloatVector(pdf[var_name])
            r_test = robjects.r('wilcox.test')
            r_result = r_test(r_data, mu=float(value))
            v, p = r_result[0][0], r_result[2][0]
            text_result += _('Result of Wilcoxon signed-rank test')+': <i>W</i> = %0.3g, %s\n' % (v, cs_util.print_p(p))
        '''
        T, p = stats.wilcoxon(np.array(pdf[var_name] - float(value)), correction=True)
            # we need to convert the pandas dataframe to numpy arraym because pdf cannot be always handled
            # correction=True in order to work like the R wilcox.test
        text_result += _('Result of Wilcoxon signed-rank test')+': <i>T</i> = %0.3g, %s\n' % (T, cs_util.print_p(p))

        # Graph
        plt.figure(figsize=(csc.fig_size_x, csc.fig_size_y*0.35))
        plt.barh([1], [np.median(data)], color=theme_colors[0], ecolor='black')  # TODO error bar
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlabel(var_name)  # TODO not visible yet, maybe matplotlib bug, cannot handle figsize consistently
        plt.title(_plt('Median value'))
        image = plt.gcf()
    else:
        text_result += _('Wilcoxon signed-rank test is computed only for interval or ordinal variables.')
        image = None
    return text_result, image


def print_var_stats(pdf, var_names, groups=None, statistics=[]):
    """
    Computes descriptive stats for variables and/or groups.

    arguments:
    var_names: list of variable names to use
    groups: list of grouping variable names
    statistics: list of strings, they can be numpy functions, such as 'mean, 'median', and they should be included in the
            stat_names list

    Now it only handles a single dependent variable and a single grouping variable.
    """
    stat_names = {'mean': _('Mean'), 'median': _('Median'), 'std': _('Standard deviation'), 'amin': _('Minimum'),
                  'amax': _('Maximum'), 'lower_quartile': 'Lower quartile', 'upper_quartile': _('Upper quartile'),
                  'skew': _('Skewness'), 'kurtosis': _('Kurtosis'), 'ptp':_('Range')}
    # Create these functions in numpy namespace to enable simple getattr call of them below
    np.lower_quartile = lambda x: np.percentile(x, 25)
    np.upper_quartile = lambda x: np.percentile(x, 75)
    # with the bias=False it gives the same value as SPSS
    np.skew = lambda x: stats.skew(x, bias=False)
    # with the bias=False it gives the same value as SPSS
    np.kurtosis = lambda x: stats.kurtosis(x, bias=False)

    text_result = ''
    if sum([pdf[var_name].dtype == 'object' for var_name in var_names]):
         raise RuntimeError('only numerical variables can be used in print_var_stats')
    # Compute only variable statistics
    if not groups:
        # drop all data with NaN pair
        data = pdf[var_names].dropna()
        pdf_result = pd.DataFrame(columns=var_names)
        text_result += _('Descriptives for the variables') if len(var_names) > 1 else _('Descriptives for the variable')
        for var_name in var_names:
            prec = cs_util.precision(data[var_name])+1
            for stat in statistics:
                pdf_result.loc[stat_names[stat], var_name] = '%0.*f' % \
                                                             (prec, getattr(np, stat)(data[var_name].dropna()))
    # There is at least one grouping variable
    else:
        # missing groups and values will be dropped

        groups, grouped_data = _split_into_groups(pdf, var_names[0], groups)
        groups = [' : '.join(map(str, group)) for group in groups]
        pdf_result = pd.DataFrame(columns=groups)

        text_result += _('Descriptives for the groups')
        # Not sure if the precision can be controlled per cell with this method;
        # Instead we make a pandas frame with str cells
#        pdf_result = pd.DataFrame([np.mean(group_data.dropna()) for group_data in grouped_data], columns=[_('Mean')], index=groups)
#        text_result += pdf_result.T.to_html()
        for group_label, group_data in zip(groups, grouped_data):
            if len(group_data):
                prec = cs_util.precision(group_data) + 1
                for stat in statistics:
                    pdf_result.loc[stat_names[stat], group_label] = '%0.*f' % \
                                                                    (prec, getattr(np, stat)(group_data.dropna()))
            else:  # TODO can we remove this part?
                text_result += _('No data')
                for stat in statistics:
                    pdf_result.loc[stat_names[stat], group_label] = _('No data')
    text_result += _format_html_table(pdf_result.to_html(bold_rows=False))
    return text_result


def confidence_interval_t(data, ci_only=True):
    """95%, two-sided CI based on t-distribution
    http://statsmodels.sourceforge.net/stable/_modules/statsmodels/stats/weightstats.html#DescrStatsW.tconfint_mean
    """
    # FIXME is this solution slow? Should we write our own CI function?
    if LooseVersion(csc.versions['statsmodels']) >= LooseVersion('0.5'):
        descr = DescrStatsW(data)
        cil, cih = descr.tconfint_mean()
        ci = (cih-cil)/2
    else:
        cil = cih = ci = [None for i in data]  # FIXME maybe this one is not correct
    if ci_only:
        if isinstance(data, pd.Series):
            return ci  # FIXME this one is for series? The other is for dataframes?
        elif isinstance(data, pd.DataFrame):
            return pd.Series(ci, index=data.columns)
            # without var names the call from comp_group_graph_cum fails
    else:
        return ci, cil, cih

### Variable pairs ###


def var_pair_graph(data, meas_lev, slope, intercept, x, y, data_frame, raw_data=False):
    if meas_lev in ['int', 'ord']:
        text_result = ''
        suptitle_text = None

        # Prepare the frequencies for the plot
        xy = [(i, j) for i, j in zip(data.iloc[:, 0], data.iloc[:, 1])]
        xy_set_freq = [[element[0], element[1], xy.count(element)] for element in set(xy)]
        [xvalues, yvalues, xy_freq] = list(zip(*xy_set_freq))
        xy_freq = np.array(xy_freq, dtype=float)
        max_freq = max(xy_freq)
        if max_freq>10:
            xy_freq = (xy_freq-1)/((max_freq-1)/9.0)+1
            # largest dot shouldn't be larger than 10 × of the default size
            # smallest dot is 1 unit size
            suptitle_text = _plt('Largest sign on the graph displays %d cases.') % max_freq
        xy_freq *= 20.0

        # Draw figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if meas_lev == 'int':
            # Display the data
            ax.scatter(xvalues, yvalues, xy_freq, color=theme_colors[0], marker='o')
            # Display the linear fit for the plot
            if not raw_data:
                fit_x = [min(data.iloc[:, 0]), max(data.iloc[:, 0])]
                fit_y = [slope*i+intercept for i in fit_x]
                ax.plot(fit_x, fit_y, color=theme_colors[0])
            # Set the labels
            plt.title(_plt('Scatterplot of the variables'))
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        elif meas_lev == 'ord':
            # Display the data
            ax.scatter(stats.rankdata(xvalues), stats.rankdata(yvalues),
                       xy_freq, color=theme_colors[0], marker='o')
            ax.set_xlim(0, len(xvalues)+1)
            ax.set_ylim(0, len(yvalues)+1)
            ax.tick_params(top=False, right=False)
            # Create new tick labels, with the rank and the value of the corresponding rank
            ax.set_xticklabels(['%i\n(%s)' % (i, sorted(xvalues)[int(i-1)])
                                if i-1 in range(len(xvalues)) else '%i' % i for i in ax.get_xticks()])
            try:
                ax.set_yticklabels(['%i\n(%s)' % (i, sorted(yvalues)[int(i-1)])
                                if i-1 in range(len(yvalues)) else '%i' % i for i in ax.get_yticks()],
                               wrap=True)
            except:  # for matplotlib before 1.5
                ax.set_yticklabels(['%i\n(%s)' % (i, sorted(yvalues)[int(i-1)])
                                if i-1 in range(len(yvalues)) else '%i' % i for i in ax.get_yticks()])
            _set_axis_measurement_level(ax, 'ord', 'ord')
            # Display the labels
            plt.title(_plt('Scatterplot of the rank of the variables'))
            ax.set_xlabel(_plt('Rank of %s') % x)
            ax.set_ylabel(_plt('Rank of %s') % y)
        if suptitle_text:
            plt.suptitle(suptitle_text, x=0.9, y=0.025, horizontalalignment='right', fontsize=10)
        graph = plt.gcf()
    elif meas_lev in ['nom']:
        cont_table_data = pd.crosstab(data_frame[y], data_frame[x])#, rownames = [x], colnames = [y]) # TODO use data instead?
        text_result = '\n%s\n%s\n' % (_('Contingency table'),
                                      _format_html_table(cont_table_data.to_html(bold_rows=False)))
        if LooseVersion(csc.versions['statsmodels']) >= LooseVersion('0.5'):
            #mosaic(data_frame, [x, y])  # Previous version
            if 0 in cont_table_data.values:
                fig, rects = mosaic(cont_table_data.unstack()+1e-9, label_rotation=[0.0, 90.0])
                # this is a workaround for mosaic limitation, which cannot draw cells with 0 frequency
                # see https://github.com/cogstat/cogstat/issues/1
            else:
                fig, rects = mosaic(cont_table_data.unstack(), label_rotation=[0.0, 90.0])
            fig.set_facecolor(csc.bg_col)
            ax = plt.subplot(111)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            plt.title(_plt('Mosaic plot of the variables'))
            _set_axis_measurement_level(ax, 'nom', 'nom')
            try:
                graph = plt.gcf()
            except:  # in some cases mosaic cannot be drawn  # TODO how to solve this?
                text_result += '\n'+_('Sorry, the mosaic plot can not be drawn with those data.')
                graph = None
        else:
            text_result += '\n'+_('Sorry, mosaic plot can be drawn only if statsmodels 0.5 or later module is installed.')
            graph = None
    return text_result, graph

### Compare variables ###


def comp_var_graph(data, var_names, meas_level, data_frame, raw_data=False):
    intro_result = ''
    graph = None
    if meas_level in ['int', 'ord', 'unk']:
    # TODO is it OK for ordinals?
        variables = np.array(data)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if raw_data:
            plt.title(_plt('Individual data of the variables'))
        else:
            plt.title(_plt('Boxplots and individual data of the variables'))
        # Display individual data
        for i in range(len(variables.transpose())-1):  # for all pairs
            # Prepare the frequencies for the plot
            xy = [(x, y) for x, y in zip(variables.transpose()[i], variables.transpose()[i+1])]
            xy_set_freq = [[element[0], element[1], xy.count(element)] for element in set(xy)]
            [xvalues, yvalues, xy_freq] = list(zip(*xy_set_freq))
            xy_freq = np.array(xy_freq, dtype=float)
            max_freq = max(xy_freq)
            if max_freq > 10:
                xy_freq = (xy_freq-1)/((max_freq-1)/9.0)+1
                # largest dot shouldn't be larger than 10 × of the default size
                # smallest dot is 1 unit size
                intro_result += '\n'+_('Thickest line displays %d cases.') % max_freq + '\n'
            for data1, data2, data_freq in zip(xvalues, yvalues, xy_freq):
                plt.plot([i+1, i+2], [data1, data2], '-', color = csc.ind_line_col, lw=data_freq)
            
        # Display boxplots
        if not raw_data:
            box1 = ax.boxplot(variables, whis='range')
            # ['medians', 'fliers', 'whiskers', 'boxes', 'caps']
            plt.setp(box1['boxes'], color=theme_colors[0])
            plt.setp(box1['whiskers'], color=theme_colors[0])
            plt.setp(box1['caps'], color=theme_colors[0])
            plt.setp(box1['medians'], color=theme_colors[0])
            plt.setp(box1['fliers'], color=theme_colors[0])
        else:
            ax.set_xlim(0.5, len(var_names)+0.5)
        plt.xticks(list(range(1, len(var_names)+1)), _wrap_labels(var_names))
        plt.ylabel(_('Value'))
        graph = plt.gcf()
    elif meas_level == 'nom':
        import itertools
        graph = []
        if LooseVersion(csc.versions['statsmodels']) >= LooseVersion('0.5'):
            for var_pair in itertools.combinations(var_names, 2):
                # workaround to draw mosaic plots with zero cell, see #1
                #fig, rects = mosaic(data_frame, [var_pair[1], var_pair[0]]) # previous version
                ct = pd.crosstab(data_frame[var_pair[0]], data_frame[var_pair[1]]).sort_index(axis='index', ascending=False)\
                    .unstack()
                if 0 in ct.values:
                    fig, rects = mosaic(ct+1e-9, label_rotation=[0.0, 90.0])
                else:
                    fig, rects = mosaic(ct, label_rotation=[0.0, 90.0])
                fig.set_facecolor(csc.bg_col)
                ax = plt.subplot(111)
                ax.set_xlabel(var_pair[1])
                ax.set_ylabel(var_pair[0])
                plt.title(_plt('Mosaic plot of the variables'))
                _set_axis_measurement_level(ax, 'nom', 'nom')
                try:
                    graph.append(plt.gcf())
                except:  # in some cases mosaic cannot be drawn  # TODO how to solve this?
                    intro_result = '\n'+_('Sorry, the mosaic plot can not be drawn with those data.')
        else:
            intro_result = '\n'+_('Sorry, mosaic plot can be drawn only if statsmodels 0.5 or later module is installed.')
    return intro_result, graph


def comp_var_graph_cum(data, var_names, meas_level, data_frame):
    """Draw means with CI for int vars, and medians for ord vars.
    """
    graph = None
    condition_means_pdf = pd.DataFrame()
    if meas_level in ['int', 'unk']:
        # ord is excluded at the moment
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if meas_level in ['int', 'unk']:
            plt.title(_plt('Means and 95% confidence intervals for the variables'))
            means = np.mean(data)
            cis, cils, cihs = confidence_interval_t(data, ci_only=False)
            ax.bar(list(range(len(data.columns))), means, 0.5, yerr=cis, align='center', 
                   color=theme_colors[0], ecolor='0')
            condition_means_pdf[_('Point estimation')] = means
            # APA format, but cannot be used the numbers if copied to spreadsheet
            #group_means_pdf[_('95% confidence interval')] = '['+ cils.map(str) + ', ' + cihs.map(str) + ']'
            condition_means_pdf[_('95% CI (low)')] = cils
            condition_means_pdf[_('95% CI (high)')] = cihs

        elif meas_level in ['ord']:
            plt.title(_plt('Medians for the variables'))
            medians = np.median(data)
            ax.bar(list(range(len(data.columns))), medians, 0.5, align='center', 
                   color=theme_colors[0], ecolor='0')
        plt.xticks(list(range(len(var_names))), _wrap_labels(var_names))
        plt.ylabel(_plt('Value'))
        graph = plt.gcf()
    return condition_means_pdf, graph


def paired_t_test(pdf, var_names):
    """Paired sample t-test
    
    arguments:
    pdf (pandas dataframe)
    var_names (list of str): two variable names to compare
    
    return:
    text_result (string)
    """
    # Not available in statsmodels
    if len(var_names) != 2:
        return _('Paired t-test requires two variables.')
    
    variables = pdf[var_names].dropna()
    df = len(variables)-1
    t, p = stats.ttest_rel(variables.iloc[:, 0], variables.iloc[:, 1])
    text_result = _('Result of paired samples t-test')+': <i>t</i>(%d) = %0.3g, %s\n' %(df, t, cs_util.print_p(p))

    return text_result

def paired_wilcox_test(pdf, var_names):
    """Paired Wilcoxon Signed Rank test
    http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    
    arguments:
    pdf
    var_names (list of str): two variable names to compare
    
    return:
    """
    # Not available in statsmodels
    text_result = ''
    if len(var_names) != 2:
        return _('Paired Wilcoxon test requires two variables.')
    
    variables = pdf[var_names].dropna()
    T, p = stats.wilcoxon(variables.iloc[:, 0], variables.iloc[:, 1])
    text_result += _('Result of Wilcoxon signed-rank test') + ': <i>T</i> = %0.3g, %s\n' % (T, cs_util.print_p(p))
    # The test does not use df, despite some of the descriptions on the net.
    # So there's no need to display df.
    
    return text_result


def mcnemar_test(pdf, var_names):
    chi2, p = mcnemar(pdf[var_names[0]], pdf[var_names[1]], exact=False)
    return _('Result of the McNemar test') + ': &chi;<sup>2</sup>(1, <i>N</i> = %d) = %0.3g, %s\n' % \
                                              (len(pdf[var_names[0]]), chi2, cs_util.print_p(p))


def cochran_q_test(pdf, var_names):
    q, p = cochrans_q(pdf[var_names])
    return _("Result of Cochran's Q test") + ': <i>Q</i>(%d, <i>N</i> = %d) = %0.3g, %s\n' % \
                                              (len(var_names)-1, len(pdf[var_names[0]]), q, cs_util.print_p(p))


def repeated_measures_anova(pdf, var_names):
    [dfn, dfd, f, pf, w, pw], corr_table = cs_stat_num.repeated_measures_anova(pdf[var_names].dropna(), var_names)
    # Choose df correction depending on sphericity violation
    text_result = _("Result of Mauchly's test to check sphericity") + \
                   ': <i>W</i> = %0.3g, %s. ' % (w, cs_util.print_p(pw))
    if pw < 0.05:  # sphericity is violated
        p = corr_table[0, 1]
        text_result += '\n<decision>'+_('Sphericity is violated.') + ' >> ' \
                       +_('Using Greenhouse-Geisser correction.') + '\n<default>' + \
                       _('Result of repeated measures ANOVA') + ': <i>F</i>(%0.3g, %0.3g) = %0.3g, %s\n' \
                        % (dfn * corr_table[0, 0], dfd * corr_table[0, 0], f, cs_util.print_p(p))
    else:  # sphericity is not violated
        p = pf
        text_result += '\n<decision>'+_('Sphericity is not violated. ') + '\n<default>' + \
                       _('Result of repeated measures ANOVA') + ': <i>F</i>(%d, %d) = %0.3g, %s\n' \
                                                                % (dfn, dfd, f, cs_util.print_p(p))

    # Post-hoc tests
    if p < 0.05:
        pht = cs_stat_num.pairwise_ttest(pdf[var_names].dropna(), var_names).sort_index()
        text_result += '\n' + _('Comparing variables pairwise with the Holm-Bonferroni correction:')
        #print pht
        pht['text'] = pht.apply(lambda x: '<i>t</i> = %0.3g, %s' % (x['t'], cs_util.print_p(x['p (Holm)'])), axis=1)

        pht_text = pht[['text']]
        text_result += _format_html_table(pht_text.to_html(bold_rows=True, escape=False, header=False))

        # Or we can print them in a matrix
        #pht_text = pht[['text']].unstack()
        #np.fill_diagonal(pht_text.values, '')
        #text_result += pht_text.to_html(bold_rows=True, escape=False))

        #print text_result

    return text_result

def friedman_test(pdf, var_names):
    """Friedman t-test
    
    arguments:
    var_names (list of str):
    """
    # Not available in statsmodels
    text_result = ''
    if len(var_names) < 2:
        return _('Friedman test requires at least two variables.')
    
    variables = pdf[var_names].dropna()
    chi2, p = stats.friedmanchisquare(*[np.array(var) for var in variables.T.values])
    df = len(var_names)-1
    n = len(variables)
    text_result += _('Result of the Friedman test: ')+'&chi;<sup>2</sup>(%d, <i>N</i> = %d) = %0.3g, %s\n' % \
                                                      (df, n, chi2, cs_util.print_p(p))  #χ2(1, N=90)=0.89, p=.35

    return text_result

### Compare groups ###


def comp_group_graph(data_frame, meas_level, var_names, groups, group_levels, raw_data_only=False):
    """Display the boxplot of the groups with individual data or the mosaic plot

    :param data_frame: The data frame
    :param meas_level:
    :param var_names:
    :param groups: List of names of the grouping variables
    :param group_levels: List of lists or tuples with group levels (1 grouping variable) or group level combinations
    (more than 1 grouping variables)
    :param raw_data_only: Only the raw data are displayed
    :return:
    """
    intro_result = ''
    if meas_level in ['int', 'ord']:  # TODO 'unk'?
        # TODO is this OK for ordinal?
        # Get the data to display
        # group the raw the data according to the level combinations
        if len(groups) == 1:
            group_levels = [[group_level] for group_level in group_levels]
        variables = [data_frame[var_names[0]][(data_frame[groups] == pd.Series({group: level for group, level in zip(groups, group_level)})).all(axis=1)].dropna() for group_level in group_levels]
        if meas_level == 'ord':  # Calculate the rank information # FIXME is there a more efficient way to do this?
            index_ranks = dict(list(zip(pd.concat(variables).index, stats.rankdata(pd.concat(variables)))))
            variables_value = pd.concat(variables).values  # original values
            for var_i in range(len(variables)):  # For all groups
                for i in variables[var_i].index:  # For all values in that group
                    variables[var_i][i] = index_ranks[i]
                    #print i, variables[var_i][i], index_ranks[i]
        # TODO graph: mean, etc.
        #means = [np.mean(self.data_values[self.data_names.index(var_name)]) for var_name in var_names]
        #stds = [np.std(self.data_values[self.data_names.index(var_name)]) for var_name in var_names]
        #rects1 = ax.bar(range(1,len(variables)+1), means, color=theme_colors[0], yerr=stds)
        # Create graph
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Add boxplot
        if not raw_data_only:
            box1 = ax.boxplot(variables, whis='range')
            plt.setp(box1['boxes'], color=theme_colors[0])
            plt.setp(box1['whiskers'], color=theme_colors[0])
            plt.setp(box1['caps'], color=theme_colors[0])
            plt.setp(box1['medians'], color=theme_colors[0])
            plt.setp(box1['fliers'], color=theme_colors[0])
        # Display individual data
        for var_i in range(len(variables)):
            val_count = variables[var_i].value_counts()
            max_freq = max(val_count)
            if max_freq>10:
                val_count = (val_count-1)/((max_freq-1)/9.0)+1
                # largest dot shouldn't be larger than 10 × of the default size
                # smallest dot is 1 unit size
                plt.suptitle(_plt('Largest individual sign displays %d cases.') % max_freq, x=0.9, y=0.025,
                             horizontalalignment='right', fontsize=10)
            ax.scatter(np.ones(len(val_count))+var_i, val_count.index, val_count.values*5, color='#808080', marker='o')
            #plt.plot(np.ones(len(variables[i]))+i, variables[i], '.', color = '#808080', ms=3) # TODO color should be used from ini file
        # Add labels
        plt.xticks(list(range(1, len(group_levels)+1)), _wrap_labels([' : '.join(map(str, group_level)) for group_level in group_levels]))
        plt.xlabel(' : '.join(groups))
        if meas_level == 'ord':
            plt.ylabel(_('Rank of %s') % var_names[0])
            if raw_data_only:
                plt.title(_plt('Individual data of the rank data of the groups'))
            else:
                plt.title(_plt('Boxplots and individual data of the rank data of the groups'))
            ax.tick_params(top=False, right=False)
            # Create new tick labels, with the rank and the value of the corresponding rank
            try:
                ax.set_yticklabels(['%i\n(%s)' % (i, sorted(variables_value)[int(i)-1])
                                    if i-1 in range(len(variables_value)) else '%i' % i for i in ax.get_yticks()],
                                   wrap=True)
            except:  # for matplotlib before 1.5
                ax.set_yticklabels(['%i\n(%s)' % (i, sorted(variables_value)[int(i)-1])
                                    if i-1 in range(len(variables_value)) else '%i' % i for i in ax.get_yticks()])
            _set_axis_measurement_level(ax, 'nom', 'ord')
        else:
            plt.ylabel(var_names[0])
            if raw_data_only:
                plt.title(_plt('Individual data of the groups'))
            else:
                plt.title(_plt('Boxplots and individual data of the groups'))
            _set_axis_measurement_level(ax, 'nom', 'int')
        graph = fig
    elif meas_level in ['nom']:
        if LooseVersion(csc.versions['statsmodels']) >= LooseVersion('0.5'):
            # workaround to draw mosaic plots with zero cell, see #1
            #fig, rects = mosaic(data_frame, [groups[0], var_names[0]])  # previous version
            ct = pd.crosstab(data_frame[var_names[0]], [data_frame[groups[i]] for i in range(len(groups))]).sort_index(axis='index', ascending=False).unstack()
            #print ct
            if 0 in ct.values:
                fig, rects = mosaic(ct+1e-9, label_rotation=[0.0, 90.0])
            else:
                fig, rects = mosaic(ct, label_rotation=[0.0, 90.0])
            fig.set_facecolor(csc.bg_col)
            ax = plt.subplot(111)
            ax.set_xlabel(' : '.join(groups))
            ax.set_ylabel(var_names[0])
            plt.title(_plt('Mosaic plot of the groups'))
            _set_axis_measurement_level(ax, 'nom', 'nom')
            try:
                graph = fig
            except:  # in some cases mosaic cannot be drawn  # TODO how to solve this?
                intro_result = '\n'+_('Sorry, the mosaic plot can not be drawn with those data.')
                graph = None
        else:
            intro_result += '\n'+_('Sorry, mosaic plot can be drawn only if statsmodels 0.5 or later module is installed.')
            graph = None
    else:
        graph = None
    return intro_result, graph


def comp_group_graph_cum(data_frame, meas_level, var_names, groups, group_levels):
    """Draw means with CI for int vars, and medians for ord vars.
    """
    graph = None
    group_means_pdf = pd.DataFrame()
#    if len(groups) == 1:
#        group_levels = [[group_level] for group_level in group_levels]
    if meas_level in ['int', 'unk']:
        # ord is excluded at the moment
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        pdf = data_frame.dropna(subset=[var_names[0]])[[var_names[0]] + groups]
        if meas_level in ['int', 'unk']:
            plt.title(_plt('Means and 95% confidence intervals for the groups'))
            means = pdf.groupby(groups, sort=False).aggregate(np.mean)[var_names[0]]
            cis = pdf.groupby(groups, sort=False).aggregate(confidence_interval_t)[var_names[0]]
            ax.bar(list(range(len(means.values))), means.reindex(group_levels), 0.5, yerr=np.array(cis.reindex(group_levels)),
                   align='center', color=theme_colors[0], ecolor='0')
                   # pandas series is converted to np.array to be able to handle numeric indexes (group levels)
            _set_axis_measurement_level(ax, 'nom', 'int')
            group_means_pdf[_('Point estimation')] = means
            # APA format, but cannot be used the numbers if copied to spreadsheet
            #group_means_pdf[_('95% confidence interval')] = '['+ (means-cis).map(str) + ', ' + (means+cis).map(str) + ']'
            group_means_pdf[_('95% CI (low)')] = means - cis
            group_means_pdf[_('95% CI (high)')] = means + cis
        elif meas_level in ['ord']:
            plt.title(_plt('Medians for the groups'))
            medians = pdf.groupby(groups[0], sort=False).aggregate(np.median)[var_names[0]]
            ax.bar(list(range(len(medians.values))), medians.reindex(group_levels), 0.5, align='center', 
                   color=theme_colors[0], ecolor='0')
        if len(groups) == 1:
            group_levels = [[group_level] for group_level in group_levels]
        plt.xticks(list(range(len(group_levels))), _wrap_labels([' : '.join(map(str, group_level)) for group_level in group_levels]))
        plt.xlabel(' : '.join(groups))
        plt.ylabel(var_names[0])
        graph = fig
    return group_means_pdf, graph


def levene_test(pdf, var_name, group_name):
    """
    
    arguments:
    var_name (str):
    group_name (str):
    
    return
    p: p
    text_result: APA format
    """
    # Not available in statsmodels
    text_result = ''

    dummy_groups, var_s = _split_into_groups(pdf, var_name, group_name)
    for i, var in enumerate(var_s):
        var_s[i] = var_s[i].dropna()
    w, p = stats.levene(*var_s)
    text_result += _('Levene test')+': <i>W</i> = %0.3g, %s\n' %(w, cs_util.print_p(p))
            
    return p, text_result


def independent_t_test(pdf, var_name, grouping_name):
    """Independent samples t-test
    
    arguments:
    var_name (str):
    grouping_name (str):
    """
    from statsmodels.stats.weightstats import ttest_ind
    text_result = ''
    
    dummy_groups, [var1, var2] = _split_into_groups(pdf, var_name, grouping_name)
    var1 = var1.dropna()
    var2 = var2.dropna()
    t, p, df = ttest_ind(var1, var2)
    # CI http://onlinestatbook.com/2/estimation/difference_means.html
    # However, there are other computtional methods:
    # http://dept.stat.lsa.umich.edu/~kshedden/Python-Workshop/stats_calculations.html
    # http://www.statisticslectures.com/topics/ciindependentsamplest/
    mean_diff = np.mean(var1)-np.mean(var2)
    sse = np.sum((np.mean(var1)-var1)**2) + np.sum((np.mean(var2)-var2)**2)
    mse = sse / (df)
    nh = 2.0/(1.0/len(var1)+1.0/len(var2))
    s_m1m2 = np.sqrt(2*mse / (nh))
    t_cl = stats.t.ppf(1-(0.05/2), df) # two-tailed
    lci = mean_diff - t_cl*s_m1m2
    hci = mean_diff + t_cl*s_m1m2
    prec = cs_util.precision(var1.append(var2))+1
    text_result += _('Difference between the two groups:') +' %0.*f, ' % (prec, mean_diff) + \
                   _('95%% confidence interval [%0.*f, %0.*f]') % (prec, lci, prec, hci)+'\n'
    text_result += _('Result of independent samples t-test:')+' <i>t</i>(%0.3g) = %0.3g, %s\n' % \
                                                              (df, t, cs_util.print_p(p))
    return text_result


def single_case_task_extremity(pdf, var_name, grouping_name, se_name = None, n_trials=None):
    """Modified t-test for comparing a single case with a group.
    Used typically in case studies.

    arguments:
    pdf (pandas dataframe) including the data
    var_name (str): name of the dependent variable
    grouping_name (str): name of the grouping variable
    se_name (str): optional, name of the slope SE variable - use only for slope based calculation
    n_trials (int): optional, number of trials the slopes were calculated of - use only for slope based calculation
    """
    text_result = ''
    group_levels, [var1, var2] = _split_into_groups(pdf, var_name, grouping_name)
    if not se_name:  # Simple performance score
        try:
            if len(var1) == 1:
                ind_data = var1
                group_data = var2.dropna()
            else:
                ind_data = var2
                group_data = var1.dropna()
            t, p, df = cs_stat_num.modified_t_test(ind_data, group_data)
            text_result += _('Result of the modified independent samples t-test:') + \
                           ' <i>t</i>(%0.3g) = %0.3g, %s\n' % (df, t, cs_util.print_p(p))
        except ValueError:
            text_result += _('One of the groups should include only a single data.')
    else:  # slope performance
        group_levels, [se1, se2] = _split_into_groups(pdf, se_name, grouping_name)
        if len(var1)==1:
            case_var = var1[0]
            control_var = var2
            case_se = se1[0]
            control_se = se2
        else:
            case_var = var2[0]
            control_var = var1
            case_se = se2[0]
            control_se = se1
        t, df, p, test = cs_stat_num.slope_extremity_test(n_trials, case_var, case_se, control_var, control_se)
        text_result += _('Result of slope test with %s:')%(test) + \
                       ' <i>t</i>(%0.3g) = %0.3g, %s\n' % (df, t, cs_util.print_p(p))
    return text_result


def welch_t_test(pdf, var_name, grouping_name):
    """ Welch's t-test

    :param pdf: pandas data frame
    :param var_name: name of the dependent variable
    :param grouping_name: name of the grouping variable
    :return: html text with APA format result
    """
    dummy_groups, [var1, var2] = _split_into_groups(pdf, var_name, grouping_name)
    t, p = stats.ttest_ind(var1.dropna(), var2.dropna(), equal_var=False)
    # http://msemac.redwoods.edu/~darnold/math15/spring2013/R/Activities/WelchTTest.html
    n1 = len(var1)
    n2 = len(var2)
    A = np.std(var1)/n1
    B = np.std(var2)/n2
    df = (A+B)**2/(A**2/(n1-1)+B**2/(n2-1))
    return _("Result of Welch's unequal variances t-test:") + \
           ' <i>t</i>(%0.3g) = %0.3g, %s\n' % (df, t, cs_util.print_p(p))

def mann_whitney_test(pdf, var_name, grouping_name):
    """Mann-Whitney test
    
    arguments:
    var_name (str):
    grouping_name (str):
    """
    # Not available in statsmodels
    text_result = ''
    
    dummy_groups, [var1, var2] = _split_into_groups(pdf, var_name, grouping_name)
    try:
        u, p = stats.mannwhitneyu(var1.dropna(), var2.dropna(), alternative='two-sided')
        text_result += _('Result of independent samples Mann-Whitney rank test: ')+'<i>U</i> = %0.3g, %s\n' % \
                                                                                   (u, cs_util.print_p(p))
    except:
        try:  # older versions of mannwhitneyu do not include the alternative parameter
            u, p = stats.mannwhitneyu(var1.dropna(), var2.dropna())
            text_result += _('Result of independent samples Mann-Whitney rank test: ')+'<i>U</i> = %0.3g, %s\n' % \
                                                                                       (u, cs_util.print_p(p * 2))
        except Exception as e:
            text_result += _('Result of independent samples Mann-Whitney rank test: ')+str(e)

    return text_result


def one_way_anova(pdf, var_name, grouping_name):
    """One-way ANOVA

    Arguments:
    var_name (str):
    grouping_name (str):
    """
    text_result = ''
    
    # http://statsmodels.sourceforge.net/stable/examples/generated/example_interactions.html#one-way-anova
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    data = pdf.dropna(subset=[var_name, grouping_name])
    # from IPython import embed; embed()
    # FIXME If there is a variable called 'C', then patsy is confused whether C is the variable or the categorical variable
    # http://gotoanswer.stanford.edu/?q=Statsmodels+Categorical+Data+from+Formula+%28using+pandas%
    # http://stackoverflow.com/questions/22545242/statsmodels-categorical-data-from-formula-using-pandas
    # http://stackoverflow.com/questions/26214409/ipython-notebook-and-patsy-categorical-variable-formula
    anova_model = ols(str(var_name+' ~ C('+grouping_name+')'), data=data).fit()
    # Type I is run, and we want to run type III, but for a one-way ANOVA different types give the same results
    anova_result = anova_lm(anova_model)
    text_result += _('Result of one-way ANOVA: ') + '<i>F</i>(%d, %d) = %0.3g, %s\n' % \
                                                    (anova_result['df'][0], anova_result['df'][1], anova_result['F'][0],
                                                     cs_util.print_p(anova_result['PR(>F)'][0]))
    # http://en.wikipedia.org/wiki/Effect_size#Omega-squared.2C_.CF.892
    omega2 = (anova_result['sum_sq'][0] - (anova_result['df'][0] * anova_result['mean_sq'][1]))/\
             ((anova_result['sum_sq'][0]+anova_result['sum_sq'][1]) +anova_result['mean_sq'][1])
    effect_size_result = _('Effect size: ') + '&omega;<sup>2</sup> = %0.3g\n' % omega2
    # http://statsmodels.sourceforge.net/stable/stats.html#multiple-tests-and-multiple-comparison-procedures
    if anova_result['PR(>F)'][0] < 0.05:  # post-hoc
        post_hoc_res = sm.stats.multicomp.pairwise_tukeyhsd(np.array(data[var_name]), np.array(data[grouping_name]),
                                                            alpha=0.05)
        text_result += '\n'+_('Groups differ. Post-hoc test of the means.')+'\n'
        text_result += ('<fix_width_font>%s\n<default>' % post_hoc_res).replace(' ', '\u00a0')
        ''' # TODO create our own output
        http://statsmodels.sourceforge.net/devel/generated/statsmodels.sandbox.stats.multicomp.TukeyHSDResults.html#statsmodels.sandbox.stats.multicomp.TukeyHSDResults
        These are the original data:
        post_hoc_res.data
        post_hoc_res.groups

        These are used for the current output:
        post_hoc_res.groupsunique
        post_hoc_res.meandiffs
        post_hoc_res.confint
        post_hoc_res.reject
        '''
    return text_result, effect_size_result

def two_way_anova(pdf, var_name, grouping_names):
    """Two-way ANOVA

    Arguments:
    pdf (pd dataframe)
    var_name (str):
    grouping_names (list of str):
    """
    # TODO extend it to multi-way ANOVA
    text_result = ''

    # http://statsmodels.sourceforge.net/stable/examples/generated/example_interactions.html#one-way-anova
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    data = pdf.dropna(subset=[var_name] + grouping_names)
    # from IPython import embed; embed()
    # FIXME If there is a variable called 'C', then patsy is confused whether C is the variable or the categorical variable
    # http://gotoanswer.stanford.edu/?q=Statsmodels+Categorical+Data+from+Formula+%28using+pandas%
    # http://stackoverflow.com/questions/22545242/statsmodels-categorical-data-from-formula-using-pandas
    # http://stackoverflow.com/questions/26214409/ipython-notebook-and-patsy-categorical-variable-formula
    anova_model = ols(str('%s ~ C(%s) + C(%s) + C(%s):C(%s)' % (var_name, grouping_names[0], grouping_names[1], grouping_names[0], grouping_names[1])), data=data).fit()
    anova_result = anova_lm(anova_model, typ=3)
    text_result += _('Result of two-way ANOVA:' + '\n')
    # Main effects
    for group_i, group in enumerate(grouping_names):
        text_result += _('Main effect of %s: ' % group) + '<i>F</i>(%d, %d) = %0.3g, %s\n' % \
                       (anova_result['df'][group_i+1], anova_result['df'][4], anova_result['F'][group_i+1],
                        cs_util.print_p(anova_result['PR(>F)'][group_i+1]))
    # Interaction effects
    text_result += _('Interaction of %s and %s: ') % (grouping_names[0], grouping_names[1]) + '<i>F</i>(%d, %d) = %0.3g, %s\n' % \
                   (anova_result['df'][3], anova_result['df'][4], anova_result['F'][3], cs_util.print_p(anova_result['PR(>F)'][3]))

    """ # TODO
    # http://en.wikipedia.org/wiki/Effect_size#Omega-squared.2C_.CF.892
    omega2 = (anova_result['sum_sq'][0] - (anova_result['df'][0] * anova_result['mean_sq'][1])) / (
                (anova_result['sum_sq'][0] + anova_result['sum_sq'][1]) + anova_result['mean_sq'][1])
    text_result += _('Effect size: ') + '&omega;<sup>2</sup> = %0.3g\n' % omega2
    """

    """ # TODO
    # http://statsmodels.sourceforge.net/stable/stats.html#multiple-tests-and-multiple-comparison-procedures
    if anova_result['PR(>F)'][0] < 0.05:  # post-hoc
        post_hoc_res = sm.stats.multicomp.pairwise_tukeyhsd(np.array(data[var_name]), np.array(data[grouping_name]),
                                                            alpha=0.05)
        text_result += '\n' + _(u'Groups differ. Post-hoc test of the means.') + '\n'
        text_result += ('<fix_width_font>%s\n<default>' % post_hoc_res).replace(' ', u'\u00a0')
        ''' # TODO create our own output
        http://statsmodels.sourceforge.net/devel/generated/statsmodels.sandbox.stats.multicomp.TukeyHSDResults.html#statsmodels.sandbox.stats.multicomp.TukeyHSDResults
        These are the original data:
        post_hoc_res.data
        post_hoc_res.groups

        These are used for the current output:
        post_hoc_res.groupsunique
        post_hoc_res.meandiffs
        post_hoc_res.confint
        post_hoc_res.reject
        '''
    """
    return text_result

def kruskal_wallis_test(pdf, var_name, grouping_name):
    """Kruskal-Wallis test

    Arguments:
    var_name (str):
    grouping_name (str):
    """
    # Not available in statsmodels
    text_result = ''

    dummy_groups, variables = _split_into_groups(pdf, var_name, grouping_name)
    variables = [variable.dropna() for variable in variables]
    try:
        H, p = stats.kruskal(*variables)
        df = len(dummy_groups)-1
        n = len(pdf[var_name].dropna())  # TODO Is this OK here?
        text_result += _('Result of the Kruskal-Wallis test: ')+'&chi;<sup>2</sup>(%d, <i>N</i> = %d) = %0.3g, %s\n' % \
                                                                (df, n, H, cs_util.print_p(p))  # χ2(1, N=90)=0.89, p=.35
    except Exception as e:
        text_result += _('Result of the Kruskal-Wallis test: ')+str(e)
    # TODO post-hoc not available yet in statsmodels http://statsmodels.sourceforge.net/stable/generated/statsmodels.sandbox.stats.multicomp.MultiComparison.kruskal.html#statsmodels.sandbox.stats.multicomp.MultiComparison.kruskal

    return text_result


def chi_square_test(pdf, var_name, grouping_name):
    """Chi-Square test
    Cramer's V: http://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    
    Arguments:
    var_name (str):
    grouping_name (str):
    """
    text_result = ''
    cont_table_data = pd.crosstab(pdf[grouping_name], pdf[var_name])#, rownames = [x], colnames = [y])
    if LooseVersion(csc.versions['scipy'])>=LooseVersion('0.10'):
        chi2, p, dof, expected = stats.chi2_contingency(cont_table_data.values)
        try:
            cramersv = (chi2 / (cont_table_data.values.sum()*(min(cont_table_data.shape)-1)))**0.5
            cramer_result = _('Cramér\'s V measure of association: ')+'&phi;<i><sub>c</sub></i> = %.3f\n' % cramersv
        except ZeroDivisionError:  # TODO could this be avoided?
            cramer_result = _('Cramér\'s V measure of association cannot be computed (division by zero).')
        chi_result = _("Result of the Pearson's Chi-square test: ")+'</i>&chi;<sup>2</sup></i>(%g, <i>N</i> = %d) = %.3f, %s' % \
                                                                      (dof, cont_table_data.values.sum(), chi2, cs_util.print_p(p))
    else:
        return _("Sorry, at least SciPy 0.10 is required to calculate Cramér\'s V or Chi-Square test.", None)
    return cramer_result, chi_result

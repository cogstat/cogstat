# -*- coding: utf-8 -*-
"""
This module contains functions for creating charts.

Functions get the raw data (as pandas dataframe or as pandas series), and the variable name(s). Optionally, use further
necessary parameters, but try to solve everything inside the chart to minimize the number of needed additional
parameters. The functions return one or several graphs.
"""

import gettext
import os
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
import statsmodels.api as sm

from . import cogstat_config as csc
from . import cogstat_stat as cs_stat

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

# store the first matplotlib theme color in cogstat config for the GUI-specific html heading styles
csc.mpl_theme_color = matplotlib.colors.to_hex(theme_colors[0])

# Overwrite style parameters when needed
# https://matplotlib.org/tutorials/introductory/customizing.html
# Some dashed and dotted axes styles (which are simply line styles) are hard to differentiate, so we overwrite the style
#print matplotlib.rcParams['lines.dashed_pattern'], matplotlib.rcParams['lines.dotted_pattern']
matplotlib.rcParams['lines.dashed_pattern'] = [6.0, 6.0]
matplotlib.rcParams['lines.dotted_pattern'] = [1.0, 3.0]
#print matplotlib.rcParams['axes.spines.left']
#print(matplotlib.rcParams['font.size'], matplotlib.rcParams['font.serif'], matplotlib.rcParams['font.sans-serif'])
if csc.language == 'th':
    matplotlib.rcParams['font.sans-serif'][0:0] = ['Umpush', 'Loma', 'Laksaman', 'KoHo', 'Garuda']
if csc.language == 'ko':
    matplotlib.rcParams['font.sans-serif'][0:0] = ['NanumGothic', 'NanumMyeongjo']
#print matplotlib.rcParams['axes.titlesize'], matplotlib.rcParams['axes.labelsize']
matplotlib.rcParams['axes.titlesize'] = csc.graph_title_size  # title of the charts
matplotlib.rcParams['axes.labelsize'] = csc.graph_font_size  # labels of the axis
#print matplotlib.rcParams['xtick.labelsize'], matplotlib.rcParams['ytick.labelsize']
#print matplotlib.rcParams['figure.facecolor']
# Make sure that the axes are visible
#print matplotlib.rcParams['axes.facecolor'], matplotlib.rcParams['axes.edgecolor']
if matplotlib.colors.to_rgba(matplotlib.rcParams['figure.facecolor']) == \
        matplotlib.colors.to_rgba(matplotlib.rcParams['axes.edgecolor']):
    #print matplotlib.colors.to_rgba(matplotlib.rcParams['axes.edgecolor'])
    matplotlib.rcParams['axes.edgecolor'] = \
        'w' if matplotlib.colors.to_rgba(matplotlib.rcParams['axes.edgecolor']) == (0, 0, 0, 0) else 'k'

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext

# matplotlib does not support rtl Unicode yet (http://matplotlib.org/devel/MEP/MEP14.html),
# so we have to handle rtl text on matplotlib plots
rtl_lang = True if csc.language in ['he', 'fa', 'ar'] else False
if rtl_lang:
    from bidi.algorithm import get_display
    _plt = lambda text: get_display(t.gettext(text))
else:
    _plt = t.gettext


def _wrap_labels(labels):
    """
    labels: list of strings
            or list of lists of single strings
    """
    label_n = len(labels)
    max_chars_in_row = 55
        # TODO need a more precise method; should depend on font size and graph size;
        # but it cannot be a very precise method unless the font has fixed width
    if isinstance(labels[0], (list, tuple)):
        wrapped_labels = [textwrap.fill(' : '.join(map(str, label)), max(5, int(max_chars_in_row/label_n))) for label in
                          labels]
    else:
        wrapped_labels = [textwrap.fill(str(label), max(5, int(max_chars_in_row / label_n))) for label in
                          labels]
        # the width should not be smaller than a min value, here 5
        # use the unicode() to convert potentially numerical labels
        # TODO maybe for many lables use rotation, e.g.,
        # http://stackoverflow.com/questions/3464359/is-it-possible-to-wrap-the-text-of-xticks-in-matplotlib-in-python
    return wrapped_labels


def _set_axis_measurement_level(ax, x_measurement_level, y_measurement_level):
    """
    Set the axes types of the graph according to the measurement levels of the variables.
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


def _create_default_mosaic_properties(data):
    """"    Code from
    https://www.statsmodels.org/stable/_modules/statsmodels/graphics/mosaicplot.html
    and adapted to CogStat to use the matplotlib style colors

    Create the default properties of the mosaic given the data
    first it will varies the color hue (first category) then the color
    saturation (second category) and then the color value
    (third category).  If a fourth category is found, it will put
    decoration on the rectangle.  Does not manage more than four
    level of categories
    """
    from statsmodels.compat.python import lzip
    from collections import OrderedDict
    from itertools import product
    from numpy import array
    from matplotlib.colors import rgb_to_hsv

    def _single_hsv_to_rgb(hsv):
        """Transform a color from the hsv space to the rgb."""
        from matplotlib.colors import hsv_to_rgb
        return hsv_to_rgb(array(hsv).reshape(1, 1, 3)).reshape(3)

    def _tuplify(obj):
        """convert an object in a tuple of strings (even if it is not iterable,
        like a single integer number, but keep the string healthy)
        """
        if np.iterable(obj) and not isinstance(obj, str):
            res = tuple(str(o) for o in obj)
        else:
            res = (str(obj),)
        return res

    def _categories_level(keys):
        """use the Ordered dict to implement a simple ordered set
        return each level of each category
        [[key_1_level_1,key_2_level_1],[key_1_level_2,key_2_level_2]]
        """
        res = []
        for i in zip(*(keys)):
            tuplefied = _tuplify(i)
            res.append(list(OrderedDict([(j, None) for j in tuplefied])))
        return res

    categories_levels = _categories_level(list(iter(data.keys())))
    Nlevels = len(categories_levels)
    # first level, the hue
    L = len(categories_levels[0])
    # hue = np.linspace(1.0, 0.0, L+1)[:-1]
    #hue = np.linspace(0.0, 1.0, L + 2)[:-2]
    # CogStat specific code: Apply the hues of the matplotlib style color hues
    theme_colors_long = theme_colors * int(np.ceil(L / len(theme_colors)))
        # if we have less colors than categories then cycle through the colors
    hue = np.array([rgb_to_hsv(matplotlib.colors.to_rgb(theme_colors_long[i]))[0] for i in range(L)])
    # second level, the saturation
    L = len(categories_levels[1]) if Nlevels > 1 else 1
    saturation = np.linspace(0.5, 1.0, L + 1)[:-1]
    # third level, the value
    L = len(categories_levels[2]) if Nlevels > 2 else 1
    value = np.linspace(0.5, 1.0, L + 1)[:-1]
    # fourth level, the hatch
    L = len(categories_levels[3]) if Nlevels > 3 else 1
    hatch = ['', '/', '-', '|', '+'][:L + 1]
    # convert in list and merge with the levels
    hue = lzip(list(hue), categories_levels[0])
    saturation = lzip(list(saturation),
                      categories_levels[1] if Nlevels > 1 else [''])
    value = lzip(list(value),
                 categories_levels[2] if Nlevels > 2 else [''])
    hatch = lzip(list(hatch),
                 categories_levels[3] if Nlevels > 3 else [''])
    # create the properties dictionary
    properties = {}
    for h, s, v, t in product(hue, saturation, value, hatch):
        hv, hn = h
        sv, sn = s
        vv, vn = v
        tv, tn = t
        level = (hn,) + ((sn,) if sn else tuple())
        level = level + ((vn,) if vn else tuple())
        level = level + ((tn,) if tn else tuple())
        hsv = array([hv, sv, vv])
        prop = {'color': _single_hsv_to_rgb(hsv), 'hatch': tv, 'lw': 0}
        properties[level] = prop
    return properties


def _mosaic_labelizer(crosstab_data, l, separator='\n'):
    """Custom labelizer function for statsmodel mosaic function
    """
    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    try:
        return separator.join(l) if crosstab_data[l] != 0 else ""
    except KeyError:
        # nominal variable might be coded as integers or interval/ordinal numerical variable can be used with nominal
        # variable in a pair, which values cannot be handled by statsmodels mosaic
        ll = tuple([int(float(l_x)) if isfloat(l_x) else l_x for l_x in l])
        return separator.join(l) if crosstab_data[ll] != 0 else ""


############################
### Charts for filtering ###
############################

def create_filtered_cases_chart(included_cases, excluded_cases, var_name, lower_limit, upper_limit):
    """Displays the filtered and kept cases for a variable.

    Parameters
    ----------
    included_cases
    excluded_cases
    var_name : str
    lower_limit : float
    upper_limit : float

    Returns
    -------
    matplotlib chart
    """

    # Follow the structure of create_variable_raw_chart structure for interval variables.

    fig = plt.figure(figsize=(csc.fig_size_x, csc.fig_size_y * 0.25))
    ax = plt.gca()

    # Add individual data
    # Excluded cases and limit lines are denoted with the second color in the theme.
    plt.scatter(included_cases, np.random.random(size=len(included_cases)), color=theme_colors[0], marker='o')
    plt.scatter(excluded_cases, np.random.random(size=len(excluded_cases)), color=theme_colors[1], marker='o')
    if (lower_limit != None) and (upper_limit != None):
        plt.vlines([lower_limit, upper_limit], ymin=-1, ymax=2, colors=theme_colors[1])
    ax.axes.set_ylim([-1.5, 2.5])
    fig.subplots_adjust(top=0.85, bottom=0.4)

    # Add labels
    if (lower_limit == None) and (upper_limit == None):
        plt.title(_plt('Included and excluded cases'))
    else:
        plt.title(_plt('Included and excluded cases with exclusion criteria'))
    plt.xlabel(var_name)
    ax.axes.get_yaxis().set_visible(False)
    _set_axis_measurement_level(ax, 'int', 'nom')

    return plt.gcf()

####################################
### Charts for Explore variables ###
####################################


def create_variable_raw_chart(pdf, data_measlevs, var_name):
    """

    Parameters
    ----------
    pdf : pandas dataframe
        It is sufficient to include only the relevant variable. It is assumed that nans are dropped.
    data_measlevs :

    var_name : str
        Name of the variable to display

    Returns
    -------
    matplotlib chart
    """
    if data_measlevs[var_name] == 'ord':
        data_orig_value = pdf[var_name].dropna()
        data = pd.Series(stats.rankdata(data_orig_value))
    else:
        data = pdf[var_name].dropna()

    if data_measlevs[var_name] in ['int', 'ord', 'unk']:
        fig = plt.figure(figsize=(csc.fig_size_x, csc.fig_size_y * 0.25))
        ax = plt.gca()
        # Add individual data
        plt.scatter(data, np.random.random(size=len(data)), color=theme_colors[0], marker='o')
        ax.axes.set_ylim([-1.5, 2.5])
        fig.subplots_adjust(top=0.85, bottom=0.4)
        # Add labels
        if data_measlevs[var_name] == 'ord':
            plt.title(_plt('Rank of the raw data'))
            plt.xlabel(_('Rank of %s') % var_name)
        else:
            plt.title(_plt('Raw data'))
            plt.xlabel(var_name)
        ax.axes.get_yaxis().set_visible(False)
        if data_measlevs[var_name] in ['int', 'unk']:
            _set_axis_measurement_level(ax, 'int', 'nom')
        elif data_measlevs[var_name] == 'ord':
            ax.tick_params(top=False, right=False)
            # Create new tick labels, with the rank and the value of the corresponding rank
            ax.set_xticklabels(['%i\n(%s)' % (i, sorted(data_orig_value)[int(i)-1])
                                if i-1 in range(len(data_orig_value)) else '%i' % i for i in ax.get_xticks()])
            _set_axis_measurement_level(ax, 'ord', 'nom')
    elif data_measlevs[var_name] in ['nom']:
        # For nominal variables the histogram is a frequency graph
        plt.figure()
        values = list(set(data))
        freqs = [list(data).count(i) for i in values]
        locs = np.arange(len(values))
        plt.title(_plt('Histogram'))
        plt.bar(locs, freqs, 0.9, color=theme_colors[0])
        plt.xticks(locs+0.9/2., _wrap_labels(values))
        plt.ylabel(_plt('Frequency'))
        ax = plt.gca()
        _set_axis_measurement_level(ax, 'nom', 'int')

    return plt.gcf()


def create_histogram_chart(pdf, data_measlevs, var_name):
    """Histogram with individual data and boxplot

    Parameters
    ----------
    pdf : pandas dataframe
        It is sufficient to include only the relevant variable. It is assumed that nans are dropped.
    data_measlevs :

    var_name : str
        name of the variable

    Returns
    -------

    """
    chart_result = ''
    max_length = 10  # maximum printing length of an item # TODO print ... if it's exceeded
    data = pdf[var_name]
    if data_measlevs[var_name] == 'ord':
        data_value = data.copy(deep=True)  # The original values of the data
        data = pd.Series(stats.rankdata(data_value))  # The ranks of the data
    if data_measlevs[var_name] in ['int', 'ord', 'unk']:
        categories_n = len(set(data))
        if categories_n < 10:
            freq, edge = np.histogram(data, bins=categories_n)
        else:
            freq, edge = np.histogram(data)
        #        text_result = _(u'Edge\tFreq\n')
        #        text_result += u''.join([u'%.2f\t%s\n'%(edge[j], freq[j]) for j in range(len(freq))])

        plt.figure()

        # Prepare the frequencies for the plot
        val_count = data.value_counts()
        if max(val_count) > 1:
            plt.suptitle(_plt('Largest tick on the x axes displays %d cases.') % max(val_count),
                         x=0.9, y=0.025, horizontalalignment='right', fontsize=10)
        val_count = (val_count * (max(freq) / max(val_count))) / 20.0

        # Upper part with histogram and individual data
        ax_up = plt.axes([0.1, 0.3, 0.8, 0.6])
        plt.hist(data.values, bins=len(edge) - 1, color=theme_colors[0])
        # .values needed, otherwise it gives error if the first case is missing data
        # Add individual data
        plt.errorbar(np.array(val_count.index), np.zeros(val_count.shape),
                     yerr=[np.zeros(val_count.shape), val_count.values],
                     fmt='k|', capsize=0, linewidth=2)
        # plt.plot(np.array(val_count.index), np.zeros(val_count.shape), 'k|', markersize=10, markeredgewidth=1.5)
        # Add labels
        if data_measlevs[var_name] == 'ord':
            plt.title(_plt('Histogram of rank data with individual data and boxplot'))
        else:
            plt.title(_plt('Histogram with individual data and boxplot'))
        plt.gca().axes.get_xaxis().set_visible(False)
        if data_measlevs[var_name] in ['int', 'unk']:
            _set_axis_measurement_level(ax_up, 'int', 'int')
        elif data_measlevs[var_name] == 'ord':
            _set_axis_measurement_level(ax_up, 'ord', 'int')
        plt.ylabel(_plt('Frequency'))
        # Lower part showing the boxplot
        ax_low = plt.axes([0.1, 0.13, 0.8, 0.17], sharex=ax_up)
        box1 = plt.boxplot(data.values, vert=0,
                           whis=[0, 100])  # .values needed, otherwise error when the first case is missing data
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
        if data_measlevs[var_name] in ['int', 'unk']:
            _set_axis_measurement_level(ax_low, 'int', 'nom')
            ax_low.spines['top'].set_visible(True)
        if data_measlevs[var_name] == 'ord':
            ax_low.tick_params(top=False, right=False)
            # Create new tick labels, with the rank and the value of the corresponding rank
            ax_low.set_xticklabels(['%i\n(%s)' % (i, sorted(data_value)[int(i - 1)])
                                    if i - 1 in range(len(data_value)) else '%i' % i for i in ax_low.get_xticks()])
            _set_axis_measurement_level(ax_low, 'ord', 'nom')
            ax_low.spines['top'].set_visible(True)
            ax_low.spines['top'].set_linestyle('dashed')
    # For nominal variables the histogram is a frequency graph, which has already been displayed in the Raw data, so it
    # is not repeated here
    return plt.gcf()


def create_normality_chart(pdf, var_name):
    """

    Parameters
    ----------
    pdf : pandas dataframe

    var_name : str

    Returns
    -------

    """

    data = pdf[var_name]
    # Prepare the frequencies for the plot
    val_count = data.value_counts()
    plt.figure()  # Otherwise the next plt.hist will modify the actual (previously created) graph
    n, bins, patches = plt.hist(data.values, density=True, color=theme_colors[0])
    if max(val_count) > 1:
        plt.suptitle(_plt('Largest tick on the x axes displays %d cases.') % max(val_count),
                     x=0.9, y=0.025, horizontalalignment='right', fontsize=10)
    val_count = (val_count * (max(n) / max(val_count))) / 20.0

    # Graphs
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # 1. Histogram
    n, bins, patches = ax1.hist(data.values, density=True, color=theme_colors[0])
    ax1.plot(bins, stats.norm.pdf(bins, np.mean(data), np.std(data)), color=theme_colors[1], linestyle='--',
             linewidth=3)
    ax1.set_title(_plt('Histogram with individual data and normal distribution'))
    ax1.errorbar(np.array(val_count.index), np.zeros(val_count.shape),
                 yerr=[np.zeros(val_count.shape), val_count.values],
                 fmt='k|', capsize=0, linewidth=2)
    #    plt.plot(data, np.zeros(data.shape), 'k+', ms=10, mew=1.5)
    # individual data
    ax1.set_xlabel(var_name)
    ax1.set_ylabel(_('Normalized relative frequency'))

    # percent on y axes http://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    def to_percent(y, position):
        s = str(100 * y)
        return s + r'$\%$' if matplotlib.rcParams['text.usetex'] is True else s + '%'

    from matplotlib.ticker import FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(to_percent))
    _set_axis_measurement_level(ax1, 'int', 'int')

    # 2. QQ plot
    sm.graphics.qqplot(data, line='s', ax=ax2, color=theme_colors[0])
    # Change the red line color (otherwise we should separately call the sm.qqline() function)
    lines = fig.findobj(lambda x: hasattr(x, 'get_color') and x.get_color() == 'r')
    [d.set_color(theme_colors[1]) for d in lines]
    ax2.set_title(_plt('Quantile-quantile plot'))
    ax2.set_xlabel(_plt('Normal theoretical quantiles'))
    ax2.set_ylabel(_plt('Sample quantiles'))
    _set_axis_measurement_level(ax2, 'int', 'int')
    normality_qq = plt.gcf()

    return normality_qq


def create_variable_population_chart(data, var_name, stat, ci=None):
    """

    Parameters
    ----------
    data :  pandas series
        It is assumed that nans are dropped.
    var_name : str

    stat : {'mean', 'median'}

    ci :

    Returns
    -------
    matplotlib chart
    """
    plt.figure(figsize=(csc.fig_size_x, csc.fig_size_y * 0.35))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.xlabel(var_name)  # TODO not visible yet, maybe matplotlib bug, cannot handle figsize consistently
    if stat == 'mean':
        plt.barh([1], [data.mean()], xerr=[ci], color=theme_colors[0], ecolor='black')
        plt.title(_plt('Mean value with 95% confidence interval'))
        _set_axis_measurement_level(plt.gca(), 'int', 'int')
    elif stat == 'median':
        plt.barh([1], [np.median(data)], color=theme_colors[0], ecolor='black')  # TODO error bar
        plt.title(_plt('Median value'))
        _set_axis_measurement_level(plt.gca(), 'ord', 'nom')
    return plt.gcf()


#########################################
### Charts for Explore variable pairs ###
#########################################

def create_residual_chart(data, meas_lev, x, y):
    """Draw a chart with residual plot and histogram of residuals

    Parameters
    ----------
    data : pandas dataframe
    meas_lev : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    x : str
        Name of the x variable.
    y : str
        Name of the x variable.

    Returns
    -------
    matplotlib chart
        A residual plot and a histogram of residuals.
    """

    if meas_lev == 'int':
        val_count = data.value_counts()
        if max(val_count) > 1:
            plt.suptitle(_plt('Largest tick on the x axes displays %d cases.') % max(val_count),
                         x=0.9, y=0.025, horizontalalignment='right', fontsize=10)

        import statsmodels.regression
        import statsmodels.tools
        residuals = statsmodels.regression.linear_model.OLS(data[y],statsmodels.tools.add_constant(data[x]))\
            .fit().resid

        # Two third on left for residual plot, one third on right for histogram of residuals
        fig = plt.figure()
        gs = plt.GridSpec(1, 3, figure=fig)
        ax_res_plot = fig.add_subplot(gs[0, :2])
        ax_hist = fig.add_subplot(gs[0, 2], sharey=ax_res_plot)

        # Residual plot (scatter of x vs. residuals)
        ax_res_plot.plot(data[x], residuals, '.')
        ax_res_plot.axhline(y=0)
        ax_res_plot.set_title(_plt("Residual plot"))
        ax_res_plot.set_xlabel(x)
        ax_res_plot.set_ylabel(_plt("Residuals"))

        # Histogram of residuals
        n, bins, patches = ax_hist.hist(residuals, density=True, orientation='horizontal')
        normal_distribution = stats.norm.pdf(bins, np.mean(residuals), np.std(residuals))
        ax_hist.plot(normal_distribution, bins, "--")
        ax_hist.set_title(_plt("Histogram of residuals"))
        # ax_hist.set_xlabel("Frequency")

        plt.setp(ax_hist.get_yticklabels(), visible=False)
        plt.setp(ax_hist.get_yticklabels(minor=True), visible=False)
        plt.setp(ax_hist.get_xticklabels(), visible=False)
        plt.setp(ax_hist.get_xticklabels(minor=True), visible=False)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05)
        graph = plt.gcf()
    else:
        graph = None

    return graph


def create_variable_pair_chart(data, meas_lev, x, y, result=None, raw_data=False, regression=True, CI=False,
                               xlims=[None, None], ylims=[None, None]):

    """Draw a chart relating two variables with optional model fit and raw data

    Parameters
    ----------
    data : pandas dataframe
    meas_lev : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    x : str
        Name of the x variable.
    y : str
        Name of the y variable.
    result : statsmodels regression result object
        Result of the regression analysis.
    raw_data : bool
        Displays raw data when True.
    regression : bool
        Displays the regression line when True.
    CI : bool
        Displays the CI band of the regression line if True. This has an effect only if the regression parameter is set
        to True.
    xlims : list of two floats
        List of values that may overwrite the automatic ylim values for interval and ordinal variables
    ylims : list of two floats
        List of values that may overwrite the automatic ylim values for interval and ordinal variables

    Returns
    -------
    matplotlib chart
        A plot of the two variables optionally containing the raw data, regression line and its CI band.
    """

    if meas_lev in ['int', 'ord']:

        # Prepare the frequencies for the plot
        xy_set_freq = data.iloc[:, 0:2].value_counts().reset_index()
        [xvalues, yvalues, xy_freq] = xy_set_freq.values.T.tolist()
        xy_freq = np.array(xy_freq, dtype=float)
        max_freq = max(xy_freq)
        if max_freq > 10:
            xy_freq = (xy_freq-1)/((max_freq-1)/9.0)+1
            # largest dot shouldn't be larger than 10 × of the default size
            # smallest dot is 1 unit size

        xy_freq *= 20.0

        # Draw figure
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if max_freq > 1:
            plt.suptitle(_plt('Largest sign on the graph displays %d cases.') % max_freq,
                         x=0.9, y=0.025, horizontalalignment='right', fontsize=10)

        if meas_lev == 'int':
            # Display the data
            if raw_data:
                ax.scatter(xvalues, yvalues, xy_freq, color=theme_colors[0], marker='o')
                # this version in the comment would not make number pairs with multiple cases larger
                # ax.scatter(data[x], data[y], color=theme_colors[0], marker='o')
                plt.title(_plt('Scatterplot of the variables'))
            # Display the linear fit for the plot
            if regression:
                from statsmodels.stats.outliers_influence import summary_table
                data_sorted = data.sort_values(by=x)
                st, summary, ss2 = summary_table(result, alpha=0.05)

                # Plotting regression line from statsmodels fitted values
                fittedvalues = summary[:, 2]
                ax.plot(data_sorted[x], fittedvalues, color=theme_colors[0])

                if CI:
                    # this will overwrite plot title that was set when raw data are displayed
                    # It assumes that regression line and CI are displayed
                    plt.title(_plt('Linear regression line with 95% CI'))
                    # Calculate CIs
                    predict_mean_ci_low, predict_mean_ci_upp = summary[:, 4:6].T

                    # Plot CI band
                    ax.plot(data_sorted[x], data_sorted[y], 'o', alpha=0)
                    ax.plot(data_sorted[x], predict_mean_ci_low, '--', color=theme_colors[0])
                    ax.plot(data_sorted[x], predict_mean_ci_upp, '--', color=theme_colors[0])
                    ax.fill_between(data_sorted[x], predict_mean_ci_low, predict_mean_ci_upp, color=theme_colors[0],
                                    alpha=0.2)
                else:
                    # This will overwrite plot title that was set when raw data are displayed
                    # It assumes that regression and raw data are displayed
                    plt.title(_plt('Scatter plot with linear regression line'))

            # Set the labels
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            _set_axis_measurement_level(ax, 'int', 'int')

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
            except TypeError:  # for matplotlib before 1.5
                ax.set_yticklabels(['%i\n(%s)' % (i, sorted(yvalues)[int(i-1)])
                                    if i-1 in range(len(yvalues)) else '%i' % i for i in ax.get_yticks()])
            _set_axis_measurement_level(ax, 'ord', 'ord')
            # Display the labels
            plt.title(_plt('Scatterplot of the rank of the variables'))
            ax.set_xlabel(_plt('Rank of %s') % x)
            ax.set_ylabel(_plt('Rank of %s') % y)
        # Set manual xlim values
        ax.set_xlim(xlims)  # Default None values do not change the limit
        # Set manual ylim values
        ax.set_ylim(ylims)  # Default None values do not change the limit
        graph = plt.gcf()
    elif meas_lev in ['nom']:
        cont_table_data = pd.crosstab(data[y], data[x])
        mosaic_data = cont_table_data.sort_index(ascending=False, level=1).unstack()
            # sort the index to have the same order on the chart as in the table
        fig, rects = mosaic(mosaic_data, label_rotation=[0.0, 90.0],
                            properties=_create_default_mosaic_properties(mosaic_data),
                            labelizer=lambda x: _mosaic_labelizer(mosaic_data, x, '\n'))
        ax = fig.get_axes()[0]
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.title(_plt('Mosaic plot of the variables'))
        _set_axis_measurement_level(ax, 'nom', 'nom')
        graph = plt.gcf()
    return graph

def create_multiple_variable_chart(data, meas_lev):
    """Draw a chart relating more than two variables displaying raw data

    Parameters
    ----------
    data : pandas dataframe
    meas_lev : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    var_names : list of str
        Names of the explanatory variables.

    Returns
    -------
    matplotlib chart
        A matrix plot of the variables optionally containing the raw data.
    """

    if meas_lev == "int":
        fig, ax = plt.subplots(len(data.columns), len(data.columns), tight_layout=True)
        fig.suptitle(_plt("Scatterplot matrix of variables"))
        for i in range(0, len(data.columns)):
            ax[i, 0].set_ylabel(data.columns[i])
            ax[len(data.columns) - 1, i].set_xlabel(data.columns[i])
            for j in range(0, len(data.columns)):
                if i == j:
                    ax[i, j].hist(data.iloc[:, i])
                else:
                    ax[i, j].scatter(data.iloc[:, i], data.iloc[:, j])

        graph = plt.gcf()
        return graph
    else:
        return None

def create_multicollinearity_chart(data, meas_lev, var_names):
    """Draw a chart relating the explanatory variables in a multiple regression displaying raw data

    Parameters
    ----------
    data : pandas dataframe
    meas_lev : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    var_names : list of str
        Names of the explanatory variables.

    Returns
    -------
    matplotlib chart
        A matrix plot of the variables optionally containing the raw data.
    """
    if meas_lev == "int":
        data = data[var_names]
        if len(var_names) == 2:
            ncols = 1
        elif len(x) >= 2:
            ncols = 2
        import math
        nrows = math.ceil(len(x)/2)
        fig = plt.figure(tight_layout=True)
        fig.suptitle(_plt("Scatterplot matrix of explanatory variables"))
        x_done = []
        ind = 1
        for index_1, x_i in enumerate(x):
            for index_2, x_j in enumerate(x):
                if x_i != x_j and [x_i, x_j] not in x_done and [x_j, x_i] not in x_done:
                    ax = plt.subplot(nrows, ncols, ind)
                    ax.scatter(data[x_i], data[x_j])
                    ax.set_xlabel(x_i)
                    ax.set_ylabel(x_j)
                    x_done.append([x_i, x_j])
                    ind += 1

        graph = plt.gcf()
        return graph
    else:
        return None

def part_regress_plots(data, dependent, var_names):
    """Draw a matrix of partial regression plots.

    Parameters
    ----------
    data : pandas dataframe
        The dataframe analysed.
    var_names : list of str
        Names of the explanatory variables.
    dependent : str
        Name of the dependent variable.

    Returns
    -------
    matplotlib chart
        For all explanatory variables, the function plots the residuals from the regression of the dependent variable
        and the other explanatory variables against the residuals from the regression of the chosen explanatory variable
        and all other explanatory variables. Plots for all explanatory variables shown in a matrix.
        This allows the visualization of the bivariate relationship while factoring out all other explanatory variables.
    """

    if len(var_names) == 2:
        ncols = 1
    elif len(var_names) >= 3:
        ncols = 2
    import math
    nrows = math.ceil(len(x) / 2)

    fig = plt.figure(tight_layout=True)
    fig.suptitle("Partial regression plots")
    for index, x_i in enumerate(var_names):
        x_other = var_names.copy()
        x_other.remove(x_i) # Remove the chosen explanatory variable from the list of explanatory variables

        # Calculating residuals from regressing the dependent variable on the remaining explanatory variables
        resid_dependent = sm.OLS(data[dependent], sm.add_constant(data[x_other])).fit().resid
        # Calculating the residuals from regressing the chosen explanatory variable on the remaining
        # explanatory variables
        resid_x_i = sm.OLS(data[x_i], sm.add_constant(data[x_other])).fit().resid

        ax = plt.subplot(nrows, ncols, index+1)
        ax.scatter(resid_x_i, resid_dependent)
        ax.set_xlabel(x_i + " | other X")
        ax.set_ylabel(dependent + " | other X")

    graph = plt.gcf()

    return graph

#########################################
### Charts for Repeated measures vars ###
#########################################


def create_repeated_measures_sample_chart(data, var_names, meas_level, raw_data_only=False, ylims=[None, None]):
    """

    Parameters
    ----------
    data : pandas dataframe
        It is assumed that the missing cases are dropped.
    var_names : list of str

    meas_level : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    raw_data_only : bool
        Only the raw data should be displayed? Or the box plots too?
    ylims : list of two floats
        List of values that may overwrite the automatic ylim values for interval and ordinal variables

    Returns
    -------
    matplotlib chart
    """
    graph = None
    if meas_level in ['int', 'ord', 'unk']:
        # TODO is it OK for ordinals?
        variables = np.array(data)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if raw_data_only:
            plt.title(_plt('Individual data of the variables'))
        else:
            plt.title(_plt('Boxplots and individual data of the variables'))
        # Display individual data
        # Find the value among all variables with the largest frequency
        max_freq = max([max(data.iloc[:, [i, i+1]].groupby([data.columns[i], data.columns[i+1]]).size()) for i in
                        range(len(data.columns) - 1)])
        for i in range(len(data.columns) - 1):  # for all pairs
            # Prepare the frequencies for the plot
            # Create dataframe with value pairs and their frequencies
            xy_set_freq = data.iloc[:, [i, i+1]].groupby([data.columns[i], data.columns[i+1]]).size().reset_index()
            if max_freq > 10:
                xy_set_freq.iloc[:, 2] = (xy_set_freq.iloc[:, 2] - 1) / ((max_freq - 1) / 9.0) + 1
                # largest dot shouldn't be larger than 10 × of the default size
                # smallest dot is 1 unit size
            for j, row in xy_set_freq.iterrows():
                plt.plot([i + 1, i + 2], [row.values[0], row.values[1]], '-', color=csc.ind_line_col, lw=row.values[2],
                         solid_capstyle='round')
        if max_freq > 1:
            plt.suptitle(_plt('Thickest line displays %d cases.') % max_freq, x=0.9, y=0.025,
                         horizontalalignment='right', fontsize=10)
        # Display boxplots
        if not raw_data_only:
            box1 = ax.boxplot(variables, whis=[0, 100])
            # ['medians', 'fliers', 'whiskers', 'boxes', 'caps']
            plt.setp(box1['boxes'], color=theme_colors[0])
            plt.setp(box1['whiskers'], color=theme_colors[0])
            plt.setp(box1['caps'], color=theme_colors[0])
            plt.setp(box1['medians'], color=theme_colors[0])
            plt.setp(box1['fliers'], color=theme_colors[0])
        else:
            ax.set_xlim(0.5, len(var_names) + 0.5)
        plt.xticks(list(range(1, len(var_names) + 1)), _wrap_labels(var_names))
        plt.ylabel(_('Value'))
        # Set manual ylim values
        ax.set_ylim(ylims)  # Default None values do not change the limit

        if meas_level in ['int', 'unk']:
            _set_axis_measurement_level(ax, 'nom', 'int')
        graph = plt.gcf()
    elif meas_level == 'nom':
        import itertools
        graph = []
        for var_pair in itertools.combinations(var_names, 2):
            ct = pd.crosstab(data[var_pair[0]], data[var_pair[1]]).sort_index(axis='index',
                                                                                          ascending=False) \
                .unstack()  # sort the index to have the same order on the chart as in the table
            fig, rects = mosaic(ct, label_rotation=[0.0, 90.0],
                                properties=_create_default_mosaic_properties(ct),
                                labelizer=lambda x: _mosaic_labelizer(ct, x, '\n'))
            ax = fig.get_axes()[0]
            ax.set_xlabel(var_pair[1])
            ax.set_ylabel(var_pair[0])
            plt.title(_plt('Mosaic plot of the variables'))
            _set_axis_measurement_level(ax, 'nom', 'nom')
            graph.append(plt.gcf())
    return graph


def create_repeated_measures_population_chart(data, var_names, meas_level, ylims=[None, None]):
    """Draw means with CI for int vars, and medians for ord vars.

    Parameters
    ----------
    data : pandas dataframe
    var_names : list of str
    meas_level : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    ylims : list of two floats
        List of values that may overwrite the automatic ylim values for interval and ordinal variables

    Returns
    -------

    """
    graph = None
    if meas_level in ['int', 'unk']:
        # ord is excluded at the moment
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if meas_level in ['int', 'unk']:
            plt.title(_plt('Means and 95% confidence intervals for the variables'))
            means = np.mean(data)
            cis = cs_stat.confidence_interval_t(data)
            ax.bar(list(range(len(data.columns))), means, 0.5, yerr=cis, align='center',
                   color=theme_colors[0], ecolor='0')

        elif meas_level in ['ord']:
            plt.title(_plt('Medians for the variables'))
            medians = np.median(data)
            ax.bar(list(range(len(data.columns))), medians, 0.5, align='center',
                   color=theme_colors[0], ecolor='0')
        plt.xticks(list(range(len(var_names))), _wrap_labels(var_names))
        plt.ylabel(_plt('Value'))
        # Set manual ylim values
        ax.set_ylim(ylims)  # Default None values do not change the limit

        _set_axis_measurement_level(ax, 'nom', 'int')
        graph = plt.gcf()
    return graph


#################################
### Charts for Compare groups ###
#################################


def create_compare_groups_sample_chart(data_frame, meas_level, var_names, groups, group_levels, raw_data_only=False,
                                       ylims=[None, None]):
    """Display the boxplot of the groups with individual data or the mosaic plot

    Parameters
    ----------
    data_frame: pandas data frame
        It is assumed that the missing cases are dropped.
    meas_level : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    var_names : list of str
    groups : list of str
        Grouping variables
    group_levels
        List of lists or tuples with group levels (1 grouping variable) or group level combinations
        (more than 1 grouping variables)
    raw_data_only : bool
        Only the raw data are displayed
    ylims : list of two floats
        List of values that may overwrite the automatic ylim values for interval and ordinal variables

    Returns
    -------
    matplotlib chart
    """
    if meas_level in ['int', 'ord']:  # TODO 'unk'?
        # TODO is this OK for ordinal?
        # Get the data to display
        # group the raw the data according to the level combinations
        variables = [data_frame[var_names[0]][(data_frame[groups] ==
                                               pd.Series({group: level for group, level in zip(groups, group_level)})).
            all(axis=1)].dropna() for group_level in group_levels]
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
            box1 = ax.boxplot(variables, whis=[0, 100])
            plt.setp(box1['boxes'], color=theme_colors[0])
            plt.setp(box1['whiskers'], color=theme_colors[0])
            plt.setp(box1['caps'], color=theme_colors[0])
            plt.setp(box1['medians'], color=theme_colors[0])
            plt.setp(box1['fliers'], color=theme_colors[0])
        else:
            ax.set_xlim(0.5, len(group_levels) + 0.5)
        # Display individual data
        # Find the value among all groups with the largest frequency
        max_freq = max([max(variables[var_i].value_counts(), default=0) for var_i in range(len(variables))])
            # default=0 parameter is needed when a group level combination does not include any cases
        for var_i in range(len(variables)):
            val_count = variables[var_i].value_counts()
            # If max_freq is larger than 10,then make the largest item size 10
            if max_freq > 10:
                val_count = (val_count-1)/((max_freq-1)/9.0)+1
                # largest dot shouldn't be larger than 10 × of the default size
                # smallest dot is 1 unit size
            ax.scatter(np.ones(len(val_count))+var_i, val_count.index, val_count.values*5, color='#808080', marker='o')
            # TODO color should be used from ini file or from style
            #plt.plot(np.ones(len(variables[i]))+i, variables[i], '.', color = '#808080', ms=3)
        if max_freq > 1:
            plt.suptitle(_plt('Largest individual sign displays %d cases.') % max_freq, x=0.9, y=0.025,
                         horizontalalignment='right', fontsize=10)
        # Set manual ylim values
        ax.set_ylim(ylims)  # Default None values do not change the limit
        # Add labels
        plt.xticks(list(range(1, len(group_levels)+1)), _wrap_labels([' : '.join(map(str, group_level)) for
                                                                      group_level in group_levels]))
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
            except TypeError:  # for matplotlib before 1.5
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
        graph = []
        """
        # TODO keep implementing several grouping variables mosaic plot
        # Current issues:
        # - if there are cells with zero value, three variables mosaic plot will not run - probably statsmodels issue
        # - ax = plt.subplot(111) removes the labels of the third variable at the top
        # - dependent variable should go to the y axis when there are three variables
        ct = pd.crosstab(data_frame[var_names[0]], [data_frame[ddd] for ddd in data_frame[groups]]).
                sort_index(axis='index', ascending=False).unstack()  
                # sort the index to have the same order on the chart as in the table
        print(ct)
        fig, rects = mosaic(ct, label_rotation=[0.0, 90.0] if len(groups) == 1 else [0.0, 90.0, 0.0],
                            labelizer=lambda x: _mosaic_labelizer(ct, x, ' : '),
                            properties=_create_default_mosaic_properties(ct))
        ax = plt.subplot(111)
        # ax.set_xlabel(' : '.join(groups))
        ax.set_xlabel(groups[0])
        ax.set_ylabel(groups[1] if len(groups) > 1 else var_names[0])
        plt.title(_plt('Mosaic plot of the groups'))
        _set_axis_measurement_level(ax, 'nom', 'nom')
        graph.append(fig)
        #"""
        for group in groups:
            ct = pd.crosstab(data_frame[var_names[0]], data_frame[group]).sort_index(axis='index', ascending=False).\
                unstack()  # sort the index to have the same order on the chart as in the table
            #print(ct)
            fig, rects = mosaic(ct, label_rotation=[0.0, 90.0],
                                properties=_create_default_mosaic_properties(ct),
                                labelizer=lambda x: _mosaic_labelizer(ct, x, ' : '))
            ax = fig.get_axes()[0]
            ax.set_xlabel(group)
            ax.set_ylabel(var_names[0])
            plt.title(_plt('Mosaic plot of the groups'))
            _set_axis_measurement_level(ax, 'nom', 'nom')
            graph.append(fig)
    else:
        graph = None
    return graph


def create_compare_groups_population_chart(pdf, meas_level, var_names, groups, group_levels, ylims=[None, None]):
    """Draw means with CI for int vars, and medians for ord vars.

    Parameters
    ----------
    pdf : pandas dataframe
        It is asssumed that missing cases are removed.
    meas_level : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    var_names : list of str
    groups : list of str
    group_levels
    ylims : list of two floats
        List of values that may overwrite the automatic ylim values for interval and ordinal variables

    Returns
    -------

    """
    graph = None
    group_levels = [level[0] for level in group_levels] if len(group_levels[0]) == 1 else group_levels
    if meas_level in ['int', 'unk']:
        # ord is excluded at the moment
        fig = plt.figure()
        ax = fig.add_subplot(111)

        pdf = pdf[[var_names[0]] + groups]
        if meas_level in ['int', 'unk']:
            plt.title(_plt('Means and 95% confidence intervals for the groups'))
            means = pdf.groupby(groups, sort=False).aggregate(np.mean)[var_names[0]]
            cis = pdf.groupby(groups, sort=False).aggregate(cs_stat.confidence_interval_t)[var_names[0]]
            ax.bar(list(range(len(group_levels))), means.reindex(group_levels), 0.5,
                   yerr=np.array(cis.reindex(group_levels)),
                   align='center', color=theme_colors[0], ecolor='0')
            # pandas series is converted to np.array to be able to handle numeric indexes (group levels)
            _set_axis_measurement_level(ax, 'nom', 'int')
        elif meas_level in ['ord']:
            plt.title(_plt('Medians for the groups'))
            medians = pdf.groupby(groups[0], sort=False).aggregate(np.median)[var_names[0]]
            ax.bar(list(range(len(group_levels))), medians.reindex(group_levels), 0.5, align='center',
                   color=theme_colors[0], ecolor='0')
        if len(groups) == 1:
            group_levels = [[group_level] for group_level in group_levels]
        plt.xticks(list(range(len(group_levels))),
                   _wrap_labels([' : '.join(map(str, group_level)) for group_level in group_levels]))
        plt.xlabel(' : '.join(groups))
        plt.ylabel(var_names[0])

        # Set manual ylim values
        ax.set_ylim(ylims)  # Default None values do not change the limit

        graph = fig
    return graph

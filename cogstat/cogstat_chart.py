# -*- coding: utf-8 -*-
"""
This module contains functions for creating charts.

Functions get the raw data (as pandas dataframe or as pandas series), and the variable name(s). Optionally, use further
necessary parameters, but try to solve everything inside the chart to minimize the number of needed additional
parameters. The functions return one or several graphs.

One exception is the create_repeated_measures_groups_chart function that returns pandas DataFrame too.
"""

import gettext
import os
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab
import numpy as np
import pandas as pd
import statsmodels.api
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
import statsmodels.api as sm

from . import cogstat_config as csc
from . import cogstat_stat as cs_stat
from . import cogstat_util as cs_util
from . import cogstat_stat_num as cs_stat_num

matplotlib.pylab.rcParams['figure.figsize'] = csc.fig_size_x, csc.fig_size_y

def set_matplotlib_theme():
    """Function to set the matplotlib theme. This is in a function so that the theme can be changed right after it is
    set in Preferences.
    """
    global theme_colors

    # Set the styles
    # User ini file includes a single theme name (unless it is freshly created based on the default ini file.
    # Default ini files includes several theme names so that preferred themes with different names in various matplotlib
    #  versions can be used.
    theme_is_set = False  # Can we set the theme?
    for theme in csc.theme if type(csc.theme) is list else [csc.theme]:
        try:
            plt.style.use(theme)
            if ('seaborn' in theme) and (theme not in plt.style.available):
                theme = theme.replace('seaborn', 'seaborn-v0_8')
                # this is needed if the ini file includes the old theme name, but the matplotlib version changed
                # from <3.6 to >=3.6
                plt.style.use(theme)
            theme_is_set = True
            # if csc.theme is a list, then overwrite the theme in csc.theme and in cogstat.ini with the first available theme
            if type(csc.theme) is list:  # list is used only in the default file, and it means that the ini file has just
                                         # been created
                csc.theme = theme
                csc.save('theme', theme)
            break
        except IOError:  # if the given theme is not available, try the next one
            continue
    if not theme_is_set:  # If the theme couldn't be set based on preferences/ini, set the first available theme
        csc.theme = sorted(plt.style.available)[0]
        csc.save('theme', csc.theme)

    # Set the style to default first, so that if a property is not set in a style, then not the arbitrary property of
    #  the previous style is used
    plt.style.use('default')
    plt.style.use(csc.theme)
    #print(plt.style.available)
    #style_num = 15
    #print(plt.style.available[style_num])
    #plt.style.use(plt.style.available[style_num])
    theme_colors = [col['color'] for col in list(plt.rcParams['axes.prop_cycle'])]  # set module variable
    #print(theme_colors)
    # this is a workaround, as 'C0' notation does not seem to work

    # store the first matplotlib theme color in cogstat config for the GUI-specific html heading styles
    csc.mpl_theme_color = theme_colors[0]

    # Overwrite style parameters when needed
    # https://matplotlib.org/tutorials/introductory/customizing.html
    # Some dashed and dotted axes styles (which are simply line styles) are hard to differentiate, so we overwrite the style
    #print(matplotlib.rcParams['lines.dashed_pattern'], matplotlib.rcParams['lines.dotted_pattern'])
    matplotlib.rcParams['lines.dashed_pattern'] = [6.0, 6.0]
    matplotlib.rcParams['lines.dotted_pattern'] = [1.0, 3.0]
    #print(matplotlib.rcParams['axes.spines.left'])
    #print(matplotlib.rcParams['font.size'], matplotlib.rcParams['font.serif'], matplotlib.rcParams['font.sans-serif'])
    if csc.language == 'th':
        matplotlib.rcParams['font.sans-serif'][0:0] = ['Umpush', 'Loma', 'Laksaman', 'KoHo', 'Garuda']
    if csc.language == 'ko':
        matplotlib.rcParams['font.sans-serif'][0:0] = ['NanumGothic', 'NanumMyeongjo']
    if csc.language == 'zh':
        matplotlib.rcParams['font.sans-serif'][0:0] = ['SimHei', 'Heiti TC', 'WenQuanYi Zen Hei', 'SimSun']
    #print(matplotlib.rcParams['axes.titlesize'], matplotlib.rcParams['axes.labelsize'])
    matplotlib.rcParams['axes.titlesize'] = csc.graph_title_size  # title of the charts
    matplotlib.rcParams['axes.labelsize'] = csc.graph_font_size  # labels of the axis
    #print(matplotlib.rcParams['xtick.labelsize'], matplotlib.rcParams['ytick.labelsize'])
    #print(matplotlib.rcParams['figure.facecolor'])
    # Make sure that the axes are visible
    #print(matplotlib.rcParams['axes.facecolor'], matplotlib.rcParams['axes.edgecolor'])
    if matplotlib.colors.to_rgba(matplotlib.rcParams['figure.facecolor']) == \
            matplotlib.colors.to_rgba(matplotlib.rcParams['axes.edgecolor']):
        #print(matplotlib.colors.to_rgba(matplotlib.rcParams['axes.edgecolor']))
        matplotlib.rcParams['axes.edgecolor'] = \
            'w' if matplotlib.colors.to_rgba(matplotlib.rcParams['axes.edgecolor']) == (0, 0, 0, 0) else 'k'

theme_colors = None
set_matplotlib_theme()

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


def _value_count(data, max_freq):
    """
    Count the values in a data series or the value pairs in a series of data pairs.
    Value(pair) counts will be decreased based on max_freq if needed, so that the signs reflecting the count will not
     be too large.
    This function should be used for all relevant chart creation, so that max_freq-dependent scaling works the same way
     in all charts.

    Parameters
    ----------
    data : pandas series or dataframe
    max_freq : int
        The maximum frequency in a set of data series (panels, etc.) that the potential decrease of count relies on.

    Returns
    -------
    pandas series
        Count of the value(pairs), potentially decreased if the max_freq is too high.
        (Multi)indexes are the value(pairs).
    """
    val_count = data.value_counts()
    # If max_freq is larger than 10,then make the largest item size 10
    if max_freq > 10:
        val_count = (val_count - 1) / ((max_freq - 1) / 9.0) + 1
        # largest dot shouldn't be larger than 10 × of the default size
        # smallest dot is 1 unit size
    return val_count


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
    # If we have less colors than categories then cycle through the colors
    hue = np.array([rgb_to_hsv(matplotlib.colors.to_rgb(theme_colors[i % len(theme_colors)]))[0] for i in range(L)])
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

    Nominal variable might be coded as integers, floats or interval/ordinal numerical variable can be used with
    nominal variable in a pair, which values cannot be handled by statsmodels mosaic
    """
    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # l is a tuple of string, but crosstab_data potentially has numerical index, so first we convert strings to
    # numerical if needed
    l_new = []
    for lx in l:
        if isfloat(lx):
            l_new.append(float(lx))
        elif lx.isdigit():
            l_new.append(int(lx))
        else:
            l_new.append(lx)
    return separator.join(l) if crosstab_data[tuple(l_new)] != 0 else ""


############################
### Charts for filtering ###
############################

def create_filtered_cases_chart(included_cases, excluded_cases, var_name, lower_limit=None, upper_limit=None):
    """Displays the filtered and kept cases for a variable.

    Parameters
    ----------
    included_cases
    excluded_cases
    var_name : str
    lower_limit : float or None
    upper_limit : float or None

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
    if (lower_limit is not None) and (upper_limit is not None):
        plt.vlines([lower_limit, upper_limit], ymin=-1, ymax=2, colors=theme_colors[1])
    ax.axes.set_ylim([-1.5, 2.5])
    fig.subplots_adjust(top=0.85, bottom=0.4)

    # Add labels
    if (lower_limit is None) and (upper_limit is None):
        plt.title(_plt('Included and excluded cases'))
    else:
        plt.title(_plt('Included and excluded cases with exclusion criteria'))
    plt.xlabel(var_name)
    ax.axes.get_yaxis().set_visible(False)
    _set_axis_measurement_level(ax, 'int', 'nom')

    return plt.gcf()


#######################################
### Charts for Reliability Analyses ###
#######################################


def create_item_total_matrix(data, regression=True):
    """Draw a grid of charts relating item scores and the total score with item-removal displaying raw data and
    optionally the regression line.
    The function assumes that reversed items have been reversed already.

    # TODO should we use the pandas or seaborn solution?

    Parameters
    ----------
    data : pandas dataframe
    regression : bool
        Display the regression line along with the scatterplot of raw data if True,
        display only scatterplot of raw data if False.

    Returns
    -------
    matplotlib chart
        A matrix plot of the variables optionally containing the regression line.
    """

    if regression:
        from statsmodels.stats.outliers_influence import summary_table
    fig = plt.figure(tight_layout=True)
    fig.suptitle(_plt('Scatterplots of item scores and total scores with item-removal'))
    ncols = len(data.columns.tolist()) if len(data.columns.tolist()) < 4 else 3
    import math
    nrows = math.ceil(len(data.columns.tolist())/3)

    items_list = data.columns.tolist()
    total_scores = [data[data.columns.difference([var])].sum(axis=1) for var in items_list]  # Total scores with item-removal
    total_scores_df = pd.DataFrame(total_scores).T
    total_scores_df.columns = ["%s_total" % var for var in items_list]

    data_temp_all_vars = pd.concat([data, total_scores_df], axis=1)
    global_max_freq = max([max(data_temp_all_vars[[var, var+'_total']].value_counts()) for var in items_list])

    for index, item in enumerate(items_list):
        ax = plt.subplot(nrows, ncols, index + 1)

        # Prepare the frequencies for the plot
        data_temp = data_temp_all_vars[[item, item+'_total']]
        val_count = _value_count(data_temp, global_max_freq)

        ax.scatter(*zip(*val_count.index), val_count.values*20, color=theme_colors[0], marker='o')
        if regression:
            regress_result = statsmodels.api.OLS(total_scores_df[item+"_total"],
                                                 statsmodels.api.add_constant(data[item])).fit()
            ax.plot(data[item], regress_result.predict(statsmodels.api.add_constant(data[item])),
                    color=theme_colors[1])

        ax.set_ylabel(_plt('Total (rest)'))
        ax.set_xlabel(item)
    if global_max_freq > 1:
        fig.text(x=0.9, y=0.005, s=_plt('Largest sign on the graph displays %d cases.') % global_max_freq,
                 horizontalalignment='right', fontsize=10)

    graph = plt.gcf()
    return graph


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
    matplotlib chart
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
            plt.suptitle(_plt('Largest tick on the x-axis displays %d cases') % max(val_count) + '.',
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
    matplotlib chart
    """

    data = pdf[var_name]
    # Prepare the frequencies for the plot
    val_count = data.value_counts()
    plt.figure()  # Otherwise the next plt.hist will modify the actual (previously created) graph
    n, bins, patches = plt.hist(data.values, density=True, color=theme_colors[0])
    if max(val_count) > 1:
        plt.suptitle(_plt('Largest tick on the x-axis displays %d cases') % max(val_count) + '.',
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

    # percent on y-axes http://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
    def to_percent(y, position):
        s = str(100 * y)
        return s + r'$\%$' if matplotlib.rcParams['text.usetex'] is True else s + '%'

    from matplotlib.ticker import FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(to_percent))
    _set_axis_measurement_level(ax1, 'int', 'int')

    # 2. QQ plot
    sm.graphics.qqplot(data, line='s', ax=ax2, color=theme_colors[0])
    # Change the red line color (otherwise we should separately call the sm.qqline() function)
    def to_python_bool(value):
        """Return the value itself, if it is a Python boolean,
        otherwise, it is a numpy boolean array, and any() is returned"""
        return value if isinstance(value, bool) else value.any()
    lines = fig.findobj(lambda x: hasattr(x, 'get_color') and to_python_bool(x.get_color() == 'r'))
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

def create_residual_chart(data, meas_lev, predictors, y):
    """Draw a chart with residuals vs. explanatory variable, and histogram of residuals.
    Draw a matrix of plots in case of multiple explanatory variables.

    Parameters
    ----------
    data : pandas dataframe
    meas_lev : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    predictors : list of str
        Names of the predictor variables.
    y : str
        Name of the x variable.

    Returns
    -------
    matplotlib chart
        A residual plot and a histogram of residuals.
    """

    if meas_lev == 'int':
        import math
        import statsmodels.regression
        import statsmodels.tools

        nrows = math.ceil(len(predictors) / 2)
        ncols = 3 if len(predictors) == 1 else 7
        plot_cols = [[0, 3]] if len(predictors) == 1 else [[0, 3], [4, 7]]

        residuals = statsmodels.regression.linear_model.OLS(data[y],statsmodels.tools.add_constant(data[predictors]))\
            .fit().resid

        # Calculate maximum frequency
        global_max_freq = 1
        for predictor in predictors:
            res_df = pd.DataFrame(np.array([data[predictor], residuals]).T)
            local_max = np.max(res_df.value_counts())
            global_max_freq = np.max([global_max_freq, local_max])

        # Two third on left for residual plot, one third on right for histogram of residuals
        fig = plt.figure()
        gs = plt.GridSpec(nrows, ncols, figure=fig)
        i = 0
        for row in range(nrows):
            for col_1, col_2 in plot_cols:
                ax_res_plot = fig.add_subplot(gs[row, col_1:col_2-1])
                ax_hist = fig.add_subplot(gs[row, col_2-1], sharey=ax_res_plot)

                # Preparing frequencies
                res_df = pd.DataFrame(np.array([data[predictors[i]], residuals]).T)
                val_count = _value_count(res_df, global_max_freq)
                if global_max_freq > 1:
                    plt.suptitle(_plt('Largest tick on the x-axis displays %d cases') % max(val_count) + '.',
                                 x=0.9, y=0.025, horizontalalignment='right', fontsize=10)

                # Residual plot (scatter of x vs. residuals)
                ax_res_plot.scatter(*zip(*val_count.index), val_count.values * 20, color=theme_colors[0], marker='o')
                ax_res_plot.axhline(y=0)
                ax_res_plot.set_xlabel(predictors[i])
                ax_res_plot.set_ylabel(_plt("Residuals"))

                # Histogram of residuals
                n, bins, patches = ax_hist.hist(residuals, density=True, orientation='horizontal')
                normal_distribution = stats.norm.pdf(bins, np.mean(residuals), np.std(residuals))
                # TODO histograms are the same for every variable, should we only display them once?
                ax_hist.plot(normal_distribution, bins, "--")
                # ax_hist.set_title(_plt("Histogram of residuals"))
                ax_hist.set_xlabel("Freq")

                # Set histogram axis ticks invisible
                plt.setp(ax_hist.get_yticklabels(), visible=False)
                plt.setp(ax_hist.get_yticklabels(minor=True), visible=False)
                plt.setp(ax_hist.get_xticklabels(), visible=False)
                plt.setp(ax_hist.get_xticklabels(minor=True), visible=False)

                i += 1
                if i+1 > len(predictors):
                    break

        fig.suptitle("Residual plot and histogram of residuals")
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

def create_scatter_matrix(data, meas_lev):
    """Draw a chart relating more than two variables displaying raw data

    # TODO should we use the pandas or seaborn solution?

    Parameters
    ----------
    data : pandas dataframe
        Include only the variables that are involved in the analyses, not the whole dataset
        # TODO this is inconsistent with the rest of the interfaces
    meas_lev : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables

    Returns
    -------
    matplotlib chart
        A matrix plot of the variables optionally containing the raw data.
    """
    if meas_lev == 'int':
        fig, ax = plt.subplots(len(data.columns), len(data.columns), tight_layout=True)
        fig.suptitle(_plt('Scatterplot matrix of variables'))
        # Preparing frequencies
        global_max_freq = 1
        for i in range(len(data.columns)):
            for j in range(len(data.columns)):
                if i != j:
                    local_max = np.max(data.iloc[:, [i, j]].value_counts())
                    global_max_freq = np.max([global_max_freq, local_max])
        # Draw plots
        for i in range(0, len(data.columns)):
            ax[i, 0].set_ylabel(data.columns[i])
            ax[len(data.columns) - 1, i].set_xlabel(data.columns[i])
            for j in range(0, len(data.columns)):
                if i == j:
                    ax[i, j].hist(data.iloc[:, i])
                else:
                    val_count = _value_count(data.iloc[:, [i, j]], global_max_freq)
                    ax[i, j].scatter(*zip(*val_count.index), val_count.values * 20, color=theme_colors[0], marker='o')
        if global_max_freq > 1:
            fig.text(x=0.9, y=0.005, s=_plt('Largest sign on the graph displays %d cases.') % global_max_freq,
                     horizontalalignment='right', fontsize=10)
        graph = plt.gcf()
        return graph
    else:
        return None


def multi_regress_plots(data, predicted, predictors, partial=True, params=None):
    """Draw a matrix of scatterplots with regression lines or partial regression plots to visualize multiple regression.
    Scatterplots:
    Scatterplots of all explanatory variables vs. the dependent variable are shown. Regression lines are derived from
    the multiple regression equation by using the intercept and the appropriate slope for each explanatory variable.

    Partial regression plots:
    For all explanatory variables, the function plots the residuals from the regression of the dependent variable
    and the other explanatory variables against the residuals from the regression of the chosen explanatory variable
    and all other explanatory variables. Plots for all explanatory variables shown in a matrix.
    This allows the visualization of the bivariate relationships while factoring out all other explanatory variables.
    Regression lines are derived from the regression of the residuals plotted.

    Parameters
    ----------
    data : pandas dataframe
        The dataframe analysed.
    predicted : str
        Name of the dependent variable.
    predictors : list of str
        Names of the explanatory variables.
    partial : bool
        Display partial regression plots if True, scatterplots with regression lines if False. Default is True.
    params : pandas series
        Model parameters from statsmodels. Series index has to contain the variable names. Default None.

    Returns
    -------
    matplotlib chart

    """

    import math
    ncols = 2 if len(predictors) < 5 else 3
    # Using a sigmoid function to determine number of rows: 1 row when predictors < 3, 2 rows when predictors < 7,
    # after that nrows increases every 3 additional predictors
    nrows = round(1 / (1 + np.exp(-len(predictors) + 2.5))) + 1 if len(predictors) < 7 else math.ceil(len(predictors) / 3)

    fig = plt.figure(tight_layout=True)
    if partial:
        fig.suptitle(_plt('Partial regression plots with regression lines'))
    else:
        fig.suptitle(_plt('Sample scatterplots with model fitted lines'))

    # Calculate residuals
    global_max_freq = 1
    residuals = []
    for index, predictor in enumerate(predictors):
        if partial:
            predictors_other = predictors.copy()
            predictors_other.remove(predictor)
            # Calculating residuals from regressing the dependent variable on the remaining explanatory variables
            dependent = sm.OLS(data[predicted], sm.add_constant(data[predictors_other])).fit().resid
            # Calculating the residuals and fitted values from regressing the chosen explanatory variable on the
            # remaining explanatory variables
            x_i = sm.OLS(data[predictor], sm.add_constant(data[predictors_other])).fit().resid
            fitted_x_i = sm.OLS(dependent, sm.add_constant(x_i)).fit().predict()
            residuals += [[dependent, x_i, fitted_x_i]]
        else:
            dependent, x_i = predicted, predictor

        # Preparing frequencies
        local_max = np.max(pd.DataFrame([dependent, x_i]).value_counts())
        global_max_freq = np.max([global_max_freq, local_max])

    # Make plots
    for index, predictor in enumerate(predictors):
        ax = plt.subplot(nrows, ncols, index+1)

        if partial:
            dependent, x_i, fitted_x_i = residuals[index][0], residuals[index][1], residuals[index][2]
        else:
            dependent, x_i = data[predicted], data[predictor]

        val_count = _value_count(pd.concat([x_i, dependent], axis=1), global_max_freq)
        ax.scatter(*zip(*val_count.index), val_count.values*20, color=theme_colors[0], marker='o')

        if partial:
            ax.plot(x_i, fitted_x_i, color=theme_colors[0])  # Partial regression line
            ax.set_xlabel(predictor + ' ' + _plt('residuals'))
            ax.set_ylabel(predicted + ' ' + _plt('residuals'))
        else:
            x_vals = np.array(ax.get_xlim())
            y_vals = params[0] + params[predictor] * x_vals
            ax.plot(x_vals, y_vals, color=theme_colors[0])
            ax.set_xlabel(predictor)
            ax.set_ylabel(predicted)



    if global_max_freq > 1:
        fig.text(x=0.9, y=0.005, s=_plt('Largest sign on the graph displays %d cases.') % global_max_freq,
                 horizontalalignment='right', fontsize=10)

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
        individual_line_color = cs_util.change_color(theme_colors[0], saturation=0.4, brightness=1.3)
        for i in range(len(data.columns) - 1):  # for all pairs
            # Prepare the frequencies for the plot
            # Create dataframe with value pairs and their frequencies
            xy_set_freq = data.iloc[:, [i, i+1]].groupby([data.columns[i], data.columns[i+1]]).size().reset_index()
            if max_freq > 10:
                xy_set_freq.iloc[:, 2] = (xy_set_freq.iloc[:, 2] - 1) / ((max_freq - 1) / 9.0) + 1
                # largest dot shouldn't be larger than 10 × of the default size
                # smallest dot is 1 unit size
            for j, row in xy_set_freq.iterrows():
                plt.plot([i + 1, i + 2], [row.values[0], row.values[1]], '-', color=individual_line_color,
                         lw=row.values[2], solid_capstyle='round')
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
            ct = pd.crosstab(data[var_pair[0]], data[var_pair[1]]).sort_index(axis='index', ascending=False) \
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
    matplotlib chart
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


def create_compare_groups_sample_chart(data_frame, meas_level, var_names, grouping_variables, group_levels, raw_data_only=False,
                                       ylims=[None, None]):
    """Display the boxplot of the groups with individual data or the mosaic plot

    Parameters
    ----------
    data_frame: pandas data frame
        It is assumed that the missing cases are dropped.
    meas_level : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the variables
    var_names : list of a single str
        Dependent variable
    grouping_variables : list of str
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
    matplotlib chart or list of matplotlib charts
    """
    if meas_level in ['int', 'ord']:  # TODO 'unk'?
        # TODO is this OK for ordinal?
        # Get the data to display
        # group the raw the data according to the level combinations
        variables = [data_frame[var_names[0]][(data_frame[grouping_variables] ==
                                               pd.Series({group: level for group, level in zip(grouping_variables, group_level)})).
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
            val_count = _value_count(variables[var_i], max_freq)
            # size parameter must be float, not int
            ax.scatter(np.ones(len(val_count)) + var_i, val_count.index, val_count.values.astype(float) * 5,
                       color=theme_colors[0], marker='o')
            #plt.plot(np.ones(len(variables[i]))+i, variables[i], '.', color = '#808080', ms=3)
        if max_freq > 1:
            plt.suptitle(_plt('Largest individual sign displays %d cases.') % max_freq, x=0.9, y=0.025,
                         horizontalalignment='right', fontsize=10)
        # Set manual ylim values
        ax.set_ylim(ylims)  # Default None values do not change the limit
        # Add labels
        plt.xticks(list(range(1, len(group_levels)+1)), _wrap_labels([' : '.join(map(str, group_level)) for
                                                                      group_level in group_levels]))
        plt.xlabel(' : '.join(grouping_variables))
        if meas_level == 'ord':
            plt.ylabel(_('Rank of %s') % var_names[0])
            if raw_data_only:
                plt.title(_plt('Individual data of the rank data of the groups'))
            else:
                plt.title(_plt('Boxplots and individual data of the rank data of the groups'))
            # Create new tick labels, with the rank and the value of the corresponding rank
            ax.set_yticklabels(['%i\n(%s)' % (i, sorted(variables_value)[int(i)-1])
                                if i-1 in range(len(variables_value)) else '%i' % i for i in ax.get_yticks()],
                               wrap=True)
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
        # - dependent variable should go to the y-axis when there are three variables
        ct = pd.crosstab(data_frame[var_names[0]], [data_frame[ddd] for ddd in data_frame[grouping_variables]]).
                sort_index(axis='index', ascending=False).unstack()  
                # sort the index to have the same order on the chart as in the table
        print(ct)
        fig, rects = mosaic(ct, label_rotation=[0.0, 90.0] if len(grouping_variables) == 1 else [0.0, 90.0, 0.0],
                            labelizer=lambda x: _mosaic_labelizer(ct, x, ' : '),
                            properties=_create_default_mosaic_properties(ct))
        ax = plt.subplot(111)
        # ax.set_xlabel(' : '.join(grouping_variables))
        ax.set_xlabel(grouping_variables[0])
        ax.set_ylabel(grouping_variables[1] if len(grouping_variables) > 1 else var_names[0])
        plt.title(_plt('Mosaic plot of the groups'))
        _set_axis_measurement_level(ax, 'nom', 'nom')
        graph.append(fig)
        #"""
        for grouping_variable in grouping_variables:
            ct = pd.crosstab(data_frame[var_names[0]], data_frame[grouping_variable]).sort_index(axis='index', ascending=False).\
                unstack()  # sort the index to have the same order on the chart as in the table
            #print(ct)
            fig, rects = mosaic(ct, label_rotation=[0.0, 90.0],
                                properties=_create_default_mosaic_properties(ct),
                                labelizer=lambda x: _mosaic_labelizer(ct, x, ' : '))
            ax = fig.get_axes()[0]
            ax.set_xlabel(grouping_variable)
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
    matplotlib chart
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


def create_repeated_measures_groups_chart(data, dep_meas_level, dep_names=None, factor_info=None,
                                          indep_x=None, indep_color=None, indep_panel=None,
                                          ylims=[None, None], show_factor_names_on_x_axis=True,
                                          raw_data=False, box_plots=False, descriptives=False, estimations=False,
                                          descriptives_table=False, estimation_table=False, statistics=None):
    """Create repeated measures and group data charts. Return related results in numerical format, too.

    Overall, when calling the function, provide
    - the dataframe (data),
    - the dependent variables (in dep_names; and optionally in factor_info, if there are repeated measures factors),
    - the (optional) independent variables that are specific to the display methods (either grouping names or factors
    in factor_info or both or none) (all independent variables should be given in indep_x, indep_color, or indep_panel),
    - and the information that should be displayed (raw_data, box_plots, etc.)

    Parameters
    ----------
    data : pandas DataFrame that include the table with all the relevant data
    dep_meas_level : {'int', 'ord', 'nom', 'unk'}
        Measurement level of the dependent variable
    dep_names : list of str
        Name(s) of the dependent variable(s). If there are more than one variable, set factor_info too.
    factor_info : multiindex pandas DataFrame
        Repeated measures design info
        Use this if there are multiple variables in dep_names. factor_info should information about all dep_names.
        Indexes are the names of the levels, and values are the names of the variables (i.e., indexes include the
        repeated measures independent factors (name of the indexes) and their level names (index levels)).
        Add all factors to either indep_x, indep_color, or indep_panel.
    indep_x : list of str
        Independent variables to be displayed on the x-axes
    indep_color : list of str
        Independent variables to be displayed as different colors
    indep_panel : list of str
        Independent variables to be displayed on different panels. Only grouping variables can be used here.
    ylims : list of two floats
        Minimum and maximum values of the y-axes
    show_factor_names_on_x_axis : bool
        Should the factor names and original variable names shown on x-axis, or only the original names
    raw_data : bool
        Should the raw data chart displayed?
    box_plots : bool
        Should box plots chart displayed?
    descriptives : bool
        Should the descriptives chart displayed?
    estimations : bool
        Should the parameter estimations chart displayed?
    descriptives_table : bool
        Should we add a table of the descriptives?
    estimation_table : bool
        Should we add a table of the estimations?
    statistics : list of str
        Statistics to be shown in descriptives_table
        They can be names that are included in the stat_names and stat_functions list in this function.

    Returns
    -------
    list of pandas dataframe(s) and matplotlib chart
        If descriptives_table or estimation_table is True, then the first item(s) of the list is/are pandas
          dataframe(s).
        If any chart request is True, then the last item(s) is/are the chart(s).
    """

    """
    Technically, instead of using the variables of the original dataframe, we modify the data so that both within-
    and between-subject independent variables are equally columns of the dataframe. For this, between-subject grouping 
    variables shouldn't be changed, but only the repeated measures within-subject variables.

    The function should handle all of these cases technically: TODO check these ones more systematically
    - no independent variables (TODO not implemented yet), between-subject, within-subject, and mixed designs
    - any of the independent variables have one or more levels
    - missing cells for multiple factors/groups for some factor level combinations
    - in any independent variables display dimensions (panel, color, x) there could be 0, 1, or multiple factors/groups
    
    In the descriptive or estimation tables, the arrangement follows the display dimensions in the sense that 
    multiindex follows the panel, x, color order (panel is the top, color is the bottom in columns/heading). 
    On the other hand, we don't follow strictly the charts in the sense that separate tables are not used for separate 
    panels, because that leads to hard to read/review tables.
    
    Handling tables in this module is not an entirely coherent solution, however, it makes maintaining the code more
    reasonable.
    
    Sorting. For between-subjects design sort alphabetically, for repeated measures keep the originally
    specified orders.

    In panels, only between-subject variables can be used; therefore, simple sorting is sufficient.
    Since now the repeated measures factor level labels are only the factor name with a number
    (e.g., factor 1, factor 2, factor 3), the simple sorting does the job. If custom factor level names can be
    specified, then the ordering parts should be adjusted.
    
    TODO For the testing period (until the beta/RC), both this function's and the older parallel functions' charts and 
    tables can be displayed if cs_config.test_functions is set to True.
    """

    # TODO nominal variable
    # TODO what if only the tables are needed?
    # TODO should we drop missing data? Or is it the job of the caller? Either the missing data should be dropped or
    #  it should be checked if there are missing data

    # 0. Check parameter constraints and find dependent and independent variables
    # Check if at least one chart or table is asked for
    if (raw_data + box_plots + descriptives + estimations + descriptives_table + estimation_table) == 0:
        return None
    # Check if dep_names and factor_info are coherent. It is assumed that factor_info includes information about all
    # dep_names if dep_names includes more than 1 variable
    if len(dep_names) > 1:
        if sorted(dep_names) != sorted(factor_info.values[0]):
            raise RuntimeError('The variables in dep_names and factor_info do not match')
    # Independent variable(s) for displaying/calculating the results
    if indep_x is None:
        indep_x = []
    if indep_color is None:
        indep_color = []
    if indep_panel is None:
        indep_panel = []
    # All independent variable names for displaying/calculating the results
    # The order is relevant in descriptive tables: panel, x, then color is the hierarchy that matches the charts
    indep_names = indep_panel + indep_x + indep_color
    # All independent variables can be used only in a single display option
    if len(set(indep_names)) != len(indep_names):
        raise RuntimeError('Some of the independent variables are used in several dimensions')
    # Check if all repeated measures factors are included in either indep_x, indep_color, or indep_panel
    if factor_info is not None:
        if not(all(var in indep_names for var in factor_info.columns.names)):
            raise RuntimeError('Some repeated measures factors were not specified as independent variable')
    # Within-subject (repeated measures) independent variables (they are the same as the factor names)
    within_indep_names = factor_info.columns.names if (factor_info is not None) else []
    # All independent variables that are not within-subject variables are between-subject variables (grouping variables)
    # Between-subject independent variables
    between_indep_names = list(set(indep_names) - set(within_indep_names))
    # Only grouping variables can be user in panels
    if len(set(indep_panel) - set(between_indep_names)) != 0:
        raise RuntimeError('Only grouping variables can be used in panels')
    # Currently, at least one independent variable should be given
    if len(within_indep_names + between_indep_names) == 0:
        raise RuntimeError('At least one independent variable should be used')
    # Currently, nominal variables are not handled yet
    if dep_meas_level == 'nom':
        raise RuntimeError('Nominal variable handling is not implemented yet')

    # Available statistics
    # Used for descriptives tables
    if statistics is None:
        statistics = []
    stat_names = {'mean': _('Mean'),
                  'median': _('Median'),
                  'std': _('Standard deviation'),
                  'min': _('Minimum'),
                  'max': _('Maximum'),
                  'range': _('Range'),
                  'lower quartile': _('Lower quartile'),
                  'upper quartile': _('Upper quartile'),
                  'skewness': _('Skewness'),
                  'kurtosis': _('Kurtosis'),
                  'variation ratio': _('Variation ratio')
                  }
    stat_functions = {'mean': np.mean,
                      'median': np.median,
                      'std': np.std,
                      'min': np.amin,
                      'max': np.amax,
                      'range': np.ptp,
                      'lower quartile': lambda x: np.percentile(x, 25),
                      'upper quartile': lambda x: np.percentile(x, 75),
                      # with the bias=False it gives the same value as SPSS
                      'skewness': lambda x: stats.skew(x, bias=False),
                      # with the bias=False it gives the same value as SPSS
                      'kurtosis': lambda x: stats.kurtosis(x, bias=False),
                      'variation ratio': lambda x: 1 - (sum(x == stats.mode(x)[0][0]) / len(x))
                      }

    # 1a. Prepare raw data: Create long format raw data
    # For a unified handling of both within-subject and between-subject variables, we transform the original data into
    #  a long format table, so that all independent variables will be separate columns, and the dependent variable will
    #  be a single column.
    long_raw_data = data[dep_names]
    # Repeated measures data requires some transformations. Melting and renaming is easier if only the relevant
    # variables included in the data. So we do this first separately.
    if factor_info is not None:  # if there are within-subject factors
        # Rename the dependent variables (dep_names) to the factor levels (factor_info.columns)
        # TODO what should be the policy: when to use the original name of the variable and when to use the factor levels?
        long_raw_data.columns = factor_info.columns
        # Change the data into long format so that all independent variables will be a separate column.
        # The ignore_index keeps the original indexes when several new rows are created for a previously single row in
        #  the new long format, so that grouping variable information can be added to all relevant rows (data are
        #  joined on indexes).
        long_raw_data = long_raw_data.melt(ignore_index=False, value_name='repeated_measures_dependent')
        # This will be the name of the dependent variable in the dataframe. Therefore, dep_name variable can be used
        #  not only in between-subject design, but in design including within-subject variables (including mixed design)
        dep_name = 'repeated_measures_dependent'
    else:
        dep_name = dep_names[0]
    long_raw_data = long_raw_data.join(data[between_indep_names])
    # This is used in the code where the code is easier to handle with a grouping column that includes all cases.
    long_raw_data['all_raw_rows'] = 1


    # 1b. Calculate descriptives and estimations for tables and charts in long format

    # Make a large pivot table where multi-index levels are the factors and groups
    #   One level is an extra for the statistics
    #   Another level is an extra constant to be used for general algorithms

    # These tables are not needed when only raw data or boxplots charts are needed, however, it is easier to set up the
    #  chart loops with these.

    # TODO add a solution when this is not calculated when not needed

    if dep_meas_level in ['int', 'unk']:
        means = long_raw_data.pivot_table(values=dep_name,
                                          index=(indep_names if indep_names else 'all_raw_rows'),
                                          aggfunc=np.mean)
        # TODO when there is only a single case, and CI is missing, no error bar is given; this looks like an exact
        #  estimation which can be misleading
        cis = long_raw_data.pivot_table(values=dep_name,
                                        index=(indep_names if indep_names else 'all_raw_rows'),
                                        aggfunc=cs_stat.confidence_interval_t, dropna=False)  # TODO do we need dropna?
        long_stat_data = pd.concat([means, cis], axis=1, keys=['means', 'cis'], names=['cogstat statistics'])
    elif dep_meas_level == 'ord':
        medians = long_raw_data.pivot_table(values=dep_name,
                                            index=(indep_names if indep_names else 'all_raw_rows'),
                                            aggfunc=np.median)  # sort=False - in pandas 1.3
        cis_low = long_raw_data.pivot_table(values=dep_name,
                                            index=(indep_names if indep_names else 'all_raw_rows'),
                                            aggfunc=lambda x: cs_stat_num.quantile_ci(x)[0][0])
        cis_high = long_raw_data.pivot_table(values=dep_name,
                                             index=(indep_names if indep_names else 'all_raw_rows'),
                                             aggfunc=lambda x: cs_stat_num.quantile_ci(x)[1][0])
        long_stat_data = pd.concat([medians, cis_low, cis_high], axis=1, keys=['medians', 'cis_low', 'cis_high'],
                                   names=['cogstat statistics'])
    elif dep_meas_level == 'nom':
        pass  # TODO
        return ([pd.DataFrame()] if estimation_table else []) + [None]

    long_stat_data = long_stat_data.stack('cogstat statistics', dropna=False)
    # long_stat_data is expected to be Series in the following parts
    long_stat_data = long_stat_data.squeeze()
    # add new index level
    long_stat_data = pd.concat([long_stat_data], keys=[1], names=['all_stat_rows'])
    # The order of the independent variables is relevant in tables: panel, x, then color is the hierarchy that matches
    #  the charts
    long_stat_data = long_stat_data.reorder_levels(['cogstat statistics'] +
                                                   indep_panel + indep_x + indep_color +
                                                   ([] if indep_names else ['all_raw_rows']) +
                                                   ['all_stat_rows'])
    long_stat_data.sort_index(inplace=True)


    # 2. Create descriptive and estimation tables
    # Independent variable levels follow panel, color, x order
    if descriptives_table:
        descriptives_table_df = long_raw_data.pivot_table(values=dep_name,
                                        index=(indep_names if indep_names else 'all_raw_rows'),
                                        aggfunc=[stat_functions[statistic] for statistic in statistics], dropna=False)
        # dropna=False is needed (keeping columns with nans), otherwise, the column names and the calculated columns
        # may not match
        descriptives_table_df.columns = [stat_names[statistic] for statistic in statistics]
        # If there is/are repeated measures variables, add the variable name to the table (not only the factor names
        # with the levels)
        if factor_info is not None:
            # Select the index axes that include within-subject variables
            factor_level_combinations = descriptives_table_df.index.to_frame()[factor_info.columns.names]
            # Find the appropriate names for the factor level combinations
            var_names = [factor_info.loc[0, tuple(row)] for index, row in factor_level_combinations.iterrows()]
            # Add the original variable names (var_names) to the multiindex
            descriptives_table_df['(' + _('Original variable name') + ')'] = var_names
            descriptives_table_df.set_index('(' + _('Original variable name') + ')', append=True, inplace=True)
        prec = cs_util.precision(long_raw_data[dep_name]) + 1
        # TODO use different precision for variation ratio; this should be done row-wise
        #formatters = ['%0.{}f'.format(2 if stat_names[statistic] == 'variation ratio' else prec) for statistic in statistics]
        descriptives_table_styler = descriptives_table_df.T.style.format('{:.%sf}' % prec)

    # Create estimations table
    # For interval variables: mean, and 95% CI ranges
    # For ordinal variables: median and 95% CI ranges
    if estimation_table:
        if dep_meas_level in ['int', 'unk', 'ord']:
            if dep_meas_level in ['int', 'unk']:
                estimation_table_df = pd.concat([long_stat_data['means'],
                                                 long_stat_data['means'] - long_stat_data['cis'],
                                                 long_stat_data['means'] + long_stat_data['cis']],
                                                axis=1)
            else:  # ordinal
                estimation_table_df = pd.concat([long_stat_data['medians'], long_stat_data['cis_low'],
                                                 long_stat_data['cis_high']], axis=1)

            estimation_table_df.columns = [_('Point estimation'), _('95% CI (low)'), _('95% CI (high)')]
            estimation_table_df.index = estimation_table_df.index.droplevel('all_stat_rows')
            prec = cs_util.precision(long_raw_data[dep_name]) + 1
            # If there is/are repeated measures variables, add the variable name to the table (not only the factor names
            # with the levels)
            if factor_info is not None:
                # Select the index axes that include within-subject variables
                factor_level_combinations = estimation_table_df.index.to_frame()[factor_info.columns.names]
                # Find the appropriate names for the factor level combinations
                var_names = [factor_info.loc[0, tuple(row)] for index, row in factor_level_combinations.iterrows()]
                # Add the original variable names (var_names) to the multiindex
                estimation_table_df['(' + _('Original variable name') + ')'] = var_names
                estimation_table_df.set_index('(' + _('Original variable name') + ')', append=True, inplace=True)
            estimation_table_styler = estimation_table_df.style.format('{:.%sf}' % prec)
            if dep_meas_level == 'ord':
                estimation_table_styler.pipe_func = lambda x: x.data.replace(np.nan, _('Out of the data range'))

    # 3. Create charts
    graphs = []

    # For all labels, handle the following scenarios: (a) the actual dimension is not used, (b) there is only one value,
    # (c) there are several values.
    # For all labels, display both the dimension name and the value.

    # If the dependent variable(s) is/are ordinal, then use the order information to display data.
    #  Here, we modify the long_raw_data. The rest of the function deals only with charts, and this will be used only
    #  in raw data and box plots.
    if dep_meas_level == 'ord':
        original_values = long_raw_data[dep_name].values
        long_raw_data[dep_name] = stats.rankdata(long_raw_data[dep_name])

    if raw_data:
        # Find most frequent value when data are split by all independent variable levels.
        # TODO this is needed only for interval and ordinal (but not nominal) variables
        # The max_freq_global stores the maximum frequency of a value in the whole analysis (note that maximum
        #   frequency could be smaller in some panels or other subgroups).
        # Global is used to set the size of the signs, so that they are comparable across panels. Panel version (see
        # below) is used for the notes to add to charts.
        # This is relevant only when there are multiple panels.
        if indep_names:  # there are independent variables
            indep_names_for_groupby = indep_names if len(indep_names) >1 else indep_names[0]
            max_freq_global = max([max(long_raw_data_subset[1][dep_name].value_counts(), default=0) for
                                   long_raw_data_subset in long_raw_data.groupby(by=indep_names_for_groupby)])
            # default=0 parameter is needed when a group level combination does not include any cases
        else:  # single variable
            max_freq_global = max(long_raw_data[dep_name].value_counts())
        if raw_data and (set(indep_x) - set(between_indep_names)):  # at least one x-axes independent variable
                                                                    #  is repeated measures
            # This is used for a mixed design, when different panels are groups.
            # Repeated measures factors in indep_x
            within_indep_x = list(set(indep_x) - set(between_indep_names))
            between_indep_x = list(set(indep_x) - set(within_indep_x))
            # TODO this could be faster with loops (saving time for repeated pivot())?
            max_freq_global_connec = max([max(long_raw_data_subset[1].
                                              pivot(columns=within_indep_x, values=dep_name).
                                              iloc[:, [column_i, column_i + 1]].value_counts(), default=0)
                                          for long_raw_data_subset
                                          in long_raw_data.groupby(by=indep_panel + indep_color + between_indep_x if
                                                                   indep_panel + indep_color + between_indep_x
                                                                   else 'all_raw_rows')
                                          for column_i
                                          in range(len(long_raw_data_subset[1].
                                                       pivot(columns=within_indep_x, values=dep_name).columns) - 1)])

    # A. Panels level
    # (1) Create new dataframe for all separate panels (technically, panels are charts) and (2) add title
    for (panel_stat_name, panel_stat_group), (panel_raw_name, panel_raw_group) \
            in zip(long_stat_data.groupby(level=(indep_panel if indep_panel else 'all_stat_rows')),
                   long_raw_data.groupby(by=(indep_panel if indep_panel else 'all_raw_rows'))):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        suptitle_text_line = ''
        suptitle_text_sign = ''

        # Calculate the max value specifically for this panel
        if indep_names:  # there are independent variables
            indep_names_for_groupby = indep_names if len(indep_names) > 1 else indep_names[0]
            max_freq_panel = max([max(long_raw_data_subset[1][dep_name].value_counts(), default=0) for
                                  long_raw_data_subset in panel_raw_group.groupby(by=indep_names_for_groupby)])
            # default=0 parameter is needed when a group level combination does not include any cases
        else:  # single variable
            max_freq_panel = max(panel_raw_group[dep_name].value_counts())

        # B. Colors level
        # (1) Create new dataframe for all separate colors and (2) add legend
        color_n = len(panel_stat_group.groupby(indep_color if indep_color else 'all_stat_rows'))
            # used for the widths of the columns
        for i, ((color_stat_name, color_stat_group), (color_raw_name, color_raw_group)) in \
                enumerate(zip(panel_stat_group.groupby(indep_color if indep_color else 'all_stat_rows'),
                              panel_raw_group.groupby(by=(indep_color if indep_color else 'all_raw_rows')))):
            color_label_set = False  # check that the label was set only once when drawing various components
            color_label = (color_stat_name if len(indep_color) == 1 else ' : '.join(map(str, color_stat_name))) \
                if indep_color else ''

            # C. X level
            if raw_data or box_plots:
                # raw data and boxplots rely on long_raw_data
                # TODO can we have a better solution omitting the loops on the x-axes level while drawing the chart?

                # For repeated measures data, display the connections
                # This will handle mixed designs too: if repeated measures factors are the lowest level, then they'll
                #  be connected. When the between-subject factor level changes, the pivot row has missing data. This
                #  also means that if repeated maeasures variables are not the lowest levels, then the data will not be
                #  connected.
                # TODO connect color conditions too; within a single x value, the neighboring colors could be connected;
                #  this would change max_freq_global_connec;
                if raw_data and (set(indep_x) - set(between_indep_names)):  # at least one x-axes independent variable
                                                                            #  is repeated measures
                    # Find the value among all variables with the largest frequency
                    data_con = color_raw_group.pivot(columns=indep_x, values=dep_name).sort_index(axis=1)
                    # max_freq_panel_connec is the specific maximum frequency for the connected items per panel
                    max_freq_panel_connec = 1
                    individual_line_color = cs_util.change_color(theme_colors[i], saturation=0.4, brightness=1.3)
                    for column_i in range(len(data_con.columns) - 1):  # for all x level pairs
                        xy_set_freq = _value_count(data_con.iloc[:, [column_i, column_i + 1]],
                                                   max_freq=max_freq_global_connec)
                        for index, value in xy_set_freq.items():
                            plt.plot([column_i + 1 + i/(color_n+1), column_i + 2 + i/(color_n+1)], [index[0], index[1]],
                                     '-', color=individual_line_color, lw=value, solid_capstyle='round', zorder=0)
                        max_freq_panel_connec = max(max_freq_panel_connec,
                                                    max(data_con.iloc[:, [column_i, column_i + 1]].value_counts().values,
                                                        default=0))
                    if max_freq_panel_connec > 1:
                        suptitle_text_line = _plt('Thickest line displays %d cases.') % max_freq_panel_connec + ' '
                if indep_x:
                    if len(indep_x) > 1:
                        indep_x_for_groupby = indep_x
                    else:
                        indep_x_for_groupby = indep_x[0]
                else:
                    indep_x_for_groupby = 'all_raw_rows'
                for j, (x_raw_name, x_raw_group) in \
                        enumerate(color_raw_group.groupby(by=indep_x_for_groupby)):
                    if raw_data:
                        val_count = _value_count(x_raw_group[dep_name], max_freq_global)
                        # size parameter must be a float, not an int
                        ax.scatter(np.ones(len(val_count)) + j + i/(color_n+1),
                                   val_count.index, val_count.values.astype(float) * 5,
                                   color=theme_colors[i % len(theme_colors)], marker='o',
                                   label=color_label if not color_label_set else '')
                        color_label_set = True
                        if max_freq_global > 1:
                            suptitle_text_sign = _plt('Largest individual sign displays %d cases.') % max_freq_panel
                    if box_plots:
                        box1 = ax.boxplot(x_raw_group[dep_name],
                                          positions=[1 + j + i / (color_n + 1)], widths=0.5 / color_n, whis=[0, 100])

                        # TODO set color label: https://stackoverflow.com/questions/32172164/what-is-the-use-of-the-label-property-in-matplotlib-box-plots
                        # the label is: color_label if not color_label_set else ''
                        # color_label_set = True

                        for prop in ['boxes', 'whiskers', 'caps', 'medians', 'fliers']:
                            plt.setp(box1[prop], color=theme_colors[i % len(theme_colors)])
                # Refine "margins" if only raw data are drawn so that the dots will not be too close to the edges
                if [raw_data, box_plots, descriptives, estimations] == [1, 0, 0, 0] and not(indep_color):
                    ax.set_xlim(0.5,
                                len(color_raw_group.groupby(by=(indep_x if indep_x else 'all_raw_rows')).groups.keys())
                                    + 0.5)

            if descriptives or estimations:
                # descriptives and estimations rely on long_stat_data
                color_group_sorted = color_stat_group.copy(deep=True)
                color_group_sorted.index = color_group_sorted.index.droplevel(level='all_stat_rows')

                # TODO descriptives
                if descriptives:
                    pass
                if estimations:
                    def _my_len(object):
                        """Return 1 if float, otherwise return the length of the object"""
                        return 1 if isinstance(object, float) else len(object)

                    if dep_meas_level in ['int', 'unk']:
                        ax.bar(x=np.arange(_my_len(color_group_sorted['means'])) + 1 + i/(color_n+1),
                               height=color_group_sorted['means'], width=1/(color_n+1),
                               yerr=np.array(color_group_sorted['cis']),
                               label=color_label if not color_label_set else '',
                               ecolor='0')
                    elif dep_meas_level in ['ord']:
                        ax.bar(x=np.arange(_my_len(color_group_sorted['medians'])) + 1 + i/(color_n+1),
                               height=color_group_sorted['medians'], width=1/(color_n+1),
                               label=color_label if not color_label_set else '',
                               ecolor='0')

            if indep_color:
                ax.legend(title=indep_color[0] if len(indep_color) == 1 else ' : '.join(indep_color))

        # panel (chart) labels
        plt_title = ''
        # Currently, it handles only the cases that are needed in the main module
        if [raw_data, box_plots, descriptives, estimations] == [1, 0, 0, 0]:
            if dep_meas_level in ['int', 'unk']:
                plt_title = _plt('Individual data')
            elif dep_meas_level in ['ord']:
                plt_title = _plt('Individual rank data')
        elif [raw_data, box_plots, descriptives, estimations] == [1, 1, 0, 0]:
            if dep_meas_level in ['int', 'unk']:
                plt_title = _plt('Boxplots and individual data')
            elif dep_meas_level in ['ord']:
                plt_title = _plt('Boxplots and individual rank data')
        elif [raw_data, box_plots, descriptives, estimations] == [0, 0, 0, 1]:
            if dep_meas_level in ['int', 'unk']:
                plt_title = _plt('Means and 95% confidence intervals')
            elif dep_meas_level in ['ord']:
                plt_title = _plt('Medians')
        if indep_panel:  # only if there are panel independent variables - otherwise, no variable info is needed
            if len(indep_panel) == 1:
                plt.title(plt_title + '\n%s (%s)' % (panel_stat_name, indep_panel[0]))
            else:
                plt.title(plt_title + '\n%s (%s)' % (' : '.join(map(str, panel_stat_name)),
                                                     ' : '.join(map(str, indep_panel))))
        else:
            plt.title(plt_title)

        # set x ticks and x label
        if indep_x:
            xtick_labels = color_raw_group.groupby(by=(indep_x if indep_x else 'all_raw_rows')).groups.keys()
            # If all repeated measures factors are included, then display the variable names too, and not only the
            # factor levels
            if all(within_indep_name in indep_x for within_indep_name in within_indep_names) and within_indep_names:
                # Select the repeated measures independent factors in the order specified in indep_x
                within_indep_x = [indep_x_item for indep_x_item in indep_x if indep_x_item in within_indep_names]
                # Select the factor level combinations that include within-subject variables
                factor_level_combinations = color_raw_group.groupby(by=(indep_x if indep_x else 'all_raw_rows')).dtypes.index.to_frame()[within_indep_x]
                factor_level_combinations.sort_index(axis='columns', level=within_indep_names, inplace=True)
                # Find the appropriate names for the factor level combinations
                var_names = [factor_info.loc[0, tuple(row)] for index, row in factor_level_combinations.iterrows()]
                if show_factor_names_on_x_axis:
                    # Add the original variable names (var_names) to the xtick_labels
                    xtick_labels = [(xtick_label + ('(' + var_name + ')', )) if isinstance(xtick_label, tuple)  # else str
                                    else (xtick_label + ' (' + var_name + ')')
                                    for xtick_label, var_name in zip(xtick_labels, var_names)]
                else:
                    # Show only the original variable names on xtick_labels
                    xtick_labels = [var_name for xtick_label, var_name in zip(xtick_labels, var_names)]
            xtick_labels_formatted = [(' : '.join(map(str, group_level)) if isinstance(group_level, tuple)
                                       else group_level) for group_level in xtick_labels]
            plt.xticks(np.arange(len(xtick_labels)) + 1 + ((color_n - 1) / 2 / (color_n + 1)),
                       _wrap_labels(xtick_labels_formatted))
            if indep_x[0] != _('Unnamed factor'):
                plt.xlabel(' : '.join(indep_x))
        else:
            ax.tick_params(bottom=False, labelbottom=False)

        # set y label
        if dep_meas_level in ['int', 'unk']:
            plt.ylabel(_('Value') if factor_info is not None else dep_name)
        elif dep_meas_level == 'ord':
            plt.ylabel(_('Rank value') if factor_info is not None else _('Rank of %s') % dep_name)

        # set y ticks
        if dep_meas_level == 'ord':
            # Create new tick labels, with the rank and the value of the corresponding rank
            ax.set_yticklabels(['%i\n(%s)' % (i, sorted(original_values)[int(i)-1])
                                if i-1 in range(len(original_values)) else '%i' % i for i in ax.get_yticks()],
                               wrap=True)

        # set manual ylim values
        ax.set_ylim(ylims)  # Default None values do not change the limit

        # set axes styles
        # TODO consider different axes style when no independent variable is used here
        if dep_meas_level in ['int', 'unk']:
            _set_axis_measurement_level(ax, 'nom', 'int')
        elif dep_meas_level in ['ord']:
            _set_axis_measurement_level(ax, 'nom', 'ord')

        if suptitle_text_line or suptitle_text_sign:
            plt.suptitle(suptitle_text_line + ' ' + suptitle_text_sign, x=0.9, y=0.025, horizontalalignment='right',
                         fontsize=10)

        graphs.append(fig)

    results_list = []
    if descriptives_table:
        results_list.append([descriptives_table_styler])
    if estimation_table:
        results_list.append([estimation_table_styler])
    results_list.append(graphs)

    return results_list

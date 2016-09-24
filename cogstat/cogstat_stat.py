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
from cStringIO import StringIO
from distutils.version import LooseVersion
from scipy import stats

import cogstat_config as csc
import cogstat_util as cs_util
import cogstat_stat_num as cs_stat_num

try:
    from statsmodels.graphics.mosaicplot import mosaic
except:
    pass
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.sandbox.stats.runs import mcnemar
from statsmodels.sandbox.stats.runs import cochrans_q
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab
from bidi.algorithm import get_display
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr  # requires at least rpy 2.1.0; use instead:
    #from rpy2.robjects import r # http://stackoverflow.com/questions/2128806/python-rpy2-cant-import-a-bunch-of-packages
except:
    pass

matplotlib.pylab.rcParams['figure.figsize'] = csc.fig_size_x, csc.fig_size_y

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.ugettext

# matplotlib does not support rtl Unicode yet (http://matplotlib.org/devel/MEP/MEP14.html),
# so we have to handle rtl text on matplotlib plots
rtl_lang = True if csc.language in ['he', 'fa', 'ar'] else False
if rtl_lang:
    _plt = lambda text: get_display(t.ugettext(text))
else:
    _plt = t.ugettext

warn_unknown_variable = '<warning>'+_('The properties of the variables are not set. Set them in your data source.')+'\n<default>' # XXX ezt talán elég az importnál nézni, az elemzéseknél lehet már másként.

### Various things ###

def _get_R_output(obj):
    """
    Returns the output of R, printing obj.
    """
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    robjects.r('options')(width=200)  # Controls wrapping the output
    print obj
    sys.stdout = old_stdout
    return mystdout.getvalue()


def _split_into_groups(pdf, var_name, grouping_name):
    """
    arguments:
    var_name (str): name of the dependent var
    grouping_name (list of str): name of the grouping var
    
    return:
    groups (list of str): list of group levels
    grouped data: list of pandas series
    """
    groups = list(set(pdf[grouping_name].dropna()))
    grouped_data = [pdf.groupby(grouping_name).get_group(group)[var_name].dropna() for group in groups]
    return groups, grouped_data


def _wrap_labels(labels):
    """
    """
    label_n = len(labels)
    max_chars_in_row = 55
        # TODO need a more precise method; should depend on font size and graph size;
        # but cannot be a very precise method unless the font is fixed width
    wrapped_labels = [textwrap.fill(unicode(label), max(5, max_chars_in_row/label_n)) for label in labels]
        # the width should not be smaller than a min value, here 5
        # use the unicode() to convert potentially numerical labels
        # TODO maybe for many lables use rotation, e.g., http://stackoverflow.com/questions/3464359/is-it-possible-to-wrap-the-text-of-xticks-in-matplotlib-in-python
    return wrapped_labels


def pivot(pdf, row_names, col_names, page_names, depend_name, function):
    """
    Build pivot table
    all parameters are lists # TODO doc
    """
            
    if len(depend_name) != 1:
        return _(u'Sorry, only one dependent variable can be used.')
    if pdf[depend_name[0]].dtype == 'object':
        return _(u'Sorry, string variables cannot be used in Pivot table.')
    function_code = {'N': 'len', 'Sum': 'np.sum', 'Mean': 'np.mean', 'Median': 'median', 'Lower quartile': 'perc25',
                     'Upper quartile': 'perc75', 'Standard deviation': 'np.std', 'Variance': 'np.var'}
    result = u''
    if page_names:
        result += _(u'Independent variable(s) - Pages: ') + u', '.join(x for x in page_names) + '\n'
    if col_names:
        result += _(u'Independent variable(s) - Columns: ') + u', '.join(x for x in col_names) + '\n'
    if row_names:
        result += _(u'Independent variable(s) - Rows: ') + u', '.join(x for x in row_names) + '\n'
    result += u'Dependent variable: ' + depend_name[0] + '\n' + _('Function: ') + function + '\n'

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
                ptable_result = '%s\n%s' % (ptable_result, ptable.
                                            to_html(bold_rows=False, sparsify=False, float_format=format_output).\
                                            replace('\n', '').\
                                            replace('border="1"', 'style="border:1px solid black;"'))
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
        names[i] = str(names[i]).translate(string.maketrans(u' -', u'__'))  # use underscore instead of space or dash
        if names[i][0].isdigit():  # do not start with number
            names[i]=u'_'+names[i]
        name_changed = False
        for j in range(i):
            while names[i]==names[j]:
                if not name_changed:
                    names[i] = names[i]+u'_1'
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


def frequencies(pdf, var_name):
    """Frequencies
    
    arguments:
    var_name (str): name of the variable
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
        freq[i].extend([rel_freq, running_total, running_rel_total])
    text_result = pd.DataFrame(freq, columns=[_('Value'), _('Freq'), _('Rel freq'), _('Cum freq'), _('Cum rel freq')]).\
        to_html(formatters={_('Rel freq'): as_percent, _('Cum rel freq'): as_percent}, bold_rows=False, index=False).\
        replace('\n', '').\
        replace('border="1"', 'style="border:1px solid black;"')
    return text_result


def histogram(pdf, data_measlevs, var_name):
    """Histogram
    
    arguments:
    var_name (str): name of the variable
    """
    text_result=''
    suptitle_text = None
    max_length = 10  # maximum printing length of an item # TODO print ... if it's exceeded
    data = pdf[var_name].dropna()
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
            suptitle_text = _plt(u'Largest tick on the x axes displays %d cases.') % max(val_count)
        val_count = (val_count*(max(freq)/max(val_count)))/20.0

        plt.figure(facecolor=csc.bg_col)
        plt.axes([0.1, 0.3, 0.8, 0.6])
        plt.hist(data.values, bins=len(edge)-1, color=csc.fig_col)
            # .values needed, otherwise error if the first case is missing data
        plt.errorbar(np.array(val_count.index), np.zeros(val_count.shape), 
                     yerr=[np.zeros(val_count.shape), val_count.values],
                     fmt='k|', capsize=0, linewidth=2)
#        plt.plot(np.array(val_count.index), np.zeros(val_count.shape), 'k|', markersize=10, markeredgewidth=1.5)
            # individual data
        plt.title(_plt('Histogram with individual data and boxplot'), fontsize=csc.graph_font_size)
        if suptitle_text:
            plt.suptitle(suptitle_text, x=0.9, y=0.02, horizontalalignment='right', fontsize=10)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.ylabel(_plt('Frequency'))
        plt.axes([0.1, 0.1, 0.8, 0.2])
        box1 = plt.boxplot(data.values, vert=0)  # .values needed, otherwise error if the first case is missing data
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlabel(var_name)
        plt.setp(box1['boxes'], color=csc.fig_col_bold)
        plt.setp(box1['whiskers'], color=csc.fig_col_bold)
        plt.setp(box1['caps'], color=csc.fig_col_bold)
        plt.setp(box1['medians'], color=csc.fig_col_bold)
        plt.setp(box1['fliers'], color=csc.fig_col_bold)
    elif data_measlevs[var_name] in ['nom']:
        # For nominal variables the histogram is a frequency graph
        plt.figure(facecolor=csc.bg_col)
        values = list(set(pdf[var_name]))
        freqs = [list(pdf[var_name]).count(i) for i in values]
        locs = np.arange(len(values))
        plt.title(_plt('Histogram'), fontsize=csc.graph_font_size)
        plt.bar(locs, freqs, 0.9, color=csc.fig_col)
        plt.xticks(locs+0.9/2., _wrap_labels(values))
        plt.ylabel(_plt('Frequency'))
    return text_result, plt.gcf()


def descriptives(pdf, data_measlevs, var_name):
    """Descriptives
    
    Mode is left out intentionally: it is not too useful.
    """
    data = pdf[var_name].dropna()
    
    data_measlev = data_measlevs[var_name]
    text_result = u''
    text_result += _(u'N of valid cases: %g') % len(data) + '\n'
    invalid_cases = len(pdf[var_name])-len(data)
    text_result += _(u'N of invalid cases: %g') % invalid_cases
    text_result += '\n\n'
    if data_measlev in ['int', 'ord', 'unk']:
        prec = cs_util.precision(data)+1
    if data_measlev in ['int', 'unk']:
        text_result += _(u'Mean: %0.*f') % (prec, np.mean(data))+'\n'
        text_result += _(u'Standard deviation: %0.*f') % (prec, np.std(data, ddof=1))+'\n'
            # ddof=1 gives the sample stat as in SPSS
        text_result += _(u'Skewness: %0.*f') % (prec, stats.skew(data, bias=False))+'\n'
            # with the bias=False it gives the same value as SPSS
        text_result += _(u'Kurtosis: %0.*f') % (prec, stats.kurtosis(data, bias=False))+'\n'
            # with the bias=False it gives the same value as SPSS
        text_result += u'\n'
    if data_measlev in ['int', 'ord', 'unk']:
        text_result += _(u'Median: %0.*f') % (prec, np.median(data))+'\n'
        text_result += _(u'Range: %0.*f') % (prec-1, (np.max(data)-np.min(data)))+'\n'
        text_result += u'\n'
    return text_result[:-2]


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

    if data_measlevs[var_name] in [u'nom', u'ord']:
        return False, '<decision>'+_(u'Normality can be checked only for interval variables.')+'\n<default>', None, None
    if len(set(data)) == 1:
        return False, _(u'Normality cannot be checked for constant variable in %s%s.\n' % (var_name, ' (%s: %s)' % (group_name, group_value) if group_name else '')), None, None
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
        return False, _(u'Too small sample to test normality in variable %s%s.\n' % (var_name, ' (%s: %s)' % (group_name, group_value) if group_name else '')), None, None
    else:
        W, p = stats.shapiro(data)
        text_result += _('Shapiro-Wilk normality test in variable %s%s') % (var_name, ' (%s: %s)' % (group_name, group_value) if group_name else '') +': <i>W</i> = %0.3g, %s\n' %(W, cs_util.print_p(p))

    # Prepare the frequencies for the plot
    val_count = data.value_counts()
    plt.figure()  # Otherwise the next plt.hist will modify the actual (previously created) graph
    n, bins, patches = plt.hist(data.values, normed=True, color=csc.fig_col)
    if max(val_count) > 1:
        suptitle_text = _plt(u'Largest tick on the x axes displays %d cases.') % max(val_count)
    val_count = (val_count*(max(n)/max(val_count)))/20.0

    # Graph
    plt.figure(facecolor=csc.bg_col)
    n, bins, patches = plt.hist(data.values, normed=True, color=csc.fig_col)
    plt.plot(bins, matplotlib.pylab.normpdf(bins, np.mean(data), np.std(data)), 'g--', linewidth=3)
    plt.title(_plt('Histogram with individual data and normal distribution'), fontsize=csc.graph_font_size)
    if suptitle_text:
        plt.suptitle(suptitle_text, x=0.9, y=0.02, horizontalalignment='right', fontsize=10)
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
        fig = plt.figure(facecolor=csc.bg_col)
        ax = fig.add_subplot(111)
        sm.graphics.qqplot(data, line='s', ax=ax)
        plt.title(_plt('Quantile-quantile plot'), fontsize=csc.graph_font_size)
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
            return _(u'One sample t-test cannot be run for constant variable.\n'), None
                    
        data = pdf[var_name].dropna()
        descr = DescrStatsW(data)
        t, p, df = descr.ttest_mean(float(test_value))
        if LooseVersion(csc.versions['statsmodels']) >= LooseVersion('0.5'):
            # Or we could use confidence_interval_t
            cil, cih = descr.tconfint_mean()
            ci = (cih-cil)/2
            prec = cs_util.precision(data)+1
            ci_text = _(u'95%% Confidence interval [%0.*f, %0.*f]') %(prec, cil, prec, cih)+'\n'
        else:
            ci = 0  # only with statsmodels
            ci_text=_('Sorry, newer statsmodels module is required for confidence interval.\n')
        text_result += ci_text+_('One sample t-test against %g')%float(test_value)+': <i>t</i>(%d) = %0.3g, %s\n' %(df, t, cs_util.print_p(p))
        
        # Graph
        plt.figure(figsize=(csc.fig_size_x, csc.fig_size_y*0.35), facecolor=csc.bg_col)
        plt.barh([1], [data.mean()], xerr=[ci], color=csc.fig_col, ecolor='black')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlabel(var_name)  # TODO not visible yet, maybe matplotlib bug, cannot handle figsize consistently
        plt.title(_plt('Mean value with 95% confidence interval'), fontsize=csc.graph_font_size)
        image = plt.gcf()
    else:
        text_result += _('One sample t-test is computed only for interval variables.')
        image = None
    return text_result, image


def wilcox_sign_test(pdf, data_measlevs, var_name, value=0):
    """Wilcoxon signed-rank test
    
    arguments:
    var_name (str):
    value (numeric):
    """
    # Not available in Python
    # http://comments.gmane.org/gmane.comp.python.scientific.user/33447
    # https://gist.github.com/mblondel/1761714 # is this correct?

    text_result = ''
    data = pdf[var_name].dropna()
    if data_measlevs[var_name] in ['int', 'ord', 'unk']:
        if data_measlevs[var_name] == 'unk':
            text_result += warn_unknown_variable
        
        if csc.versions['r']:
            # R version
            # http://ww2.coastal.edu/kingw/statistics/R-tutorials/singlesample.html
            r_data = robjects.FloatVector(pdf[var_name])
            r_test = robjects.r('wilcox.test')
            r_result = r_test(r_data, mu=float(value))
            v, p = r_result[0][0], r_result[2][0]
            text_result += _('Result of Wilcoxon signed rank test')+': <i>W</i> = %0.3g, %s\n' % (v, cs_util.print_p(p))
        else:
            text_result += _('Sorry, this function is not available if R is not installed.')
                    
        # Graph
        plt.figure(figsize=(csc.fig_size_x, csc.fig_size_y*0.35), facecolor=csc.bg_col)
        plt.barh([1], [np.median(data)], color=csc.fig_col, ecolor='black')  # TODO error bar
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlabel(var_name)  # TODO not visible yet, maybe matplotlib bug, cannot handle figsize consistently
        plt.title(_plt('Median value'), fontsize=csc.graph_font_size)
        image = plt.gcf()
    else:
        text_result += _('Wilcoxon signed-rank test is computed only for interval or ordinal variables.')
        image = None
    return text_result, image


def print_var_stats(pdf, var_names, group_names=None, stat=None):
    """
    Computes descriptive stats for variables and/or groups.

    arguments:
    var_names: list of variable names to use
    group_names: list of grouping variable names
    stat: can be 'mean, 'median'
    
    Now it only handles a single dependent variable and a single grouping variable.
    """
    text_result = u''
    stat_name = {'mean': _('Mean'), 'median': _('Median')}
    if sum([pdf[var_name].dtype == 'object' for var_name in var_names]):
         raise RuntimeError('only numerical variables can be used in print_var_stats')
    if not group_names: # compute only variable statistics
        # drop all data with NaN pair
        data = pdf[var_names].dropna()
        pdf_result = pd.DataFrame(columns=var_names, index=[stat_name[stat]])
        if stat == 'mean':
            text_result += _(u'Means for the variables')
        elif stat == 'median':
            text_result += _(u'Medians for the variables')
        for var_name in var_names:
            prec = cs_util.precision(data[var_name])+1
            if stat == 'mean':
                pdf_result.loc[stat_name[stat], var_name] = (u'%0.*f') % (prec, np.mean(data[var_name].dropna()))
            elif stat == 'median':
                pdf_result.loc[stat_name[stat], var_name] = (u'%0.*f') % (prec, np.median(data[var_name].dropna()))
        text_result += pdf_result.to_html(bold_rows=False).replace('\n', '').replace('border="1"', 'style="border:1px solid black;"')
    else:  # there is grouping variable
        # TODO now it only handles a single dependent variable and a single grouping variable
        # missing groups and values will be dropped
        groups, grouped_data = _split_into_groups(pdf, var_names[0], group_names[0])
        pdf_result = pd.DataFrame(columns=groups, index=[stat_name[stat]])
        if stat == 'mean':
            text_result += _(u'Means for the groups')
            # Not sure if the precision can be controlled per cell with this method;
            # Instead we make a pandas frame with str cells
#            pdf_result = pd.DataFrame([np.mean(group_data.dropna()) for group_data in grouped_data], columns=[_('Mean')], index=groups)
#            text_result += pdf_result.T.to_html().replace('\n','')
        elif stat == 'median':
            text_result += _(u'Medians for the groups')
        for group_label, group_data in zip(groups, grouped_data):
            if len(group_data):
                prec = cs_util.precision(group_data)+1
                if stat == 'mean':
                    pdf_result.loc[stat_name[stat], group_label] = (u'%0.*f') %(prec, np.mean(group_data.dropna()))
                elif stat == 'median':
                    pdf_result.loc[stat_name[stat], group_label] = (u'%0.*f') %(prec, np.median(group_data.dropna()))
            else:
                text_result += _('No data')
                pdf_result.loc[stat_name[stat], group_label] = _('No data')
        text_result += pdf_result.to_html(bold_rows=False).\
            replace('\n', '').\
            replace('border="1"', 'style="border:1px solid black;"')
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


def var_pair_graph(data, meas_lev, slope, intercept, x, y, data_frame):
    if meas_lev in ['int', 'ord']:
        text_result = ''
        suptitle_text = None

        # Prepare the frequencies for the plot
        xy = [(i, j) for i, j in zip(data.iloc[:, 0], data.iloc[:, 1])]
        xy_set_freq = [[element[0], element[1], xy.count(element)] for element in set(xy)]
        [xvalues, yvalues, xy_freq] = zip(*xy_set_freq)
        xy_freq = np.array(xy_freq, dtype=float)
        max_freq = max(xy_freq)
        if max_freq>10:
            xy_freq = (xy_freq-1)/((max_freq-1)/9.0)+1
            # largest dot shouldn't be larger than 10 × of the default size
            # smallest dot is 1 unit size
            suptitle_text = _plt(u'Largest sign on the graph displays %d cases.') % max_freq
        xy_freq *= 20.0
        # Prepare the linear fit for the plot
        if meas_lev == 'int':
            fit_x = [min(data.iloc[:,0]), max(data.iloc[:,0])]
            fit_y = [slope*i+intercept for i in fit_x]

        # Draw figure
        fig = plt.figure(facecolor=csc.bg_col)
        ax = fig.add_subplot(111)
        ax.scatter(xvalues, yvalues, xy_freq, color=csc.fig_col_bold, marker='o')
        if meas_lev == 'int':
            ax.plot(fit_x, fit_y, color=csc.fig_col_bold)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.title(_plt('Scatterplot of the variables'), fontsize=csc.graph_font_size)
        if suptitle_text:
            plt.suptitle(suptitle_text, x=0.9, y=0.02, horizontalalignment='right', fontsize=10)
        graph = plt.gcf()
    elif meas_lev in ['nom']:
        cont_table_data = pd.crosstab(data_frame[y], data_frame[x])#, rownames = [x], colnames = [y]) # TODO use data instead?
        text_result = '\n%s\n' % cont_table_data.to_html(bold_rows=False).replace('\n', '').\
            replace('border="1"', 'style="border:1px solid black;"')
        if LooseVersion(csc.versions['statsmodels']) >= LooseVersion('0.5'):
            #mosaic(data_frame, [x, y])  # Previous version
            if 0 in cont_table_data.values:
                fig, rects = mosaic(cont_table_data.unstack()+1e-9)
                # this is a workaround for mosaic limitation, which cannot draw cells with 0 frequency
                # see https://github.com/cogstat/cogstat/issues/1
            else:
                fig, rects = mosaic(cont_table_data.unstack())
            fig.set_facecolor(csc.bg_col)
            ax = plt.subplot(111)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            plt.title(_plt('Mosaic plot of the variables'), fontsize=csc.graph_font_size)
            try:
                graph = plt.gcf()
            except:  # in some cases mosaic cannot be drawn  # TODO how to solve this?
                text_result += '\n'+_(u'Sorry, the mosaic plot can not be drawn with those data.')
                graph = None
        else:
            text_result += '\n'+_(u'Sorry, mosaic plot can be drawn only if statsmodels 0.5 or later module is installed.')
            graph = None
    return text_result, graph

### Compare variables ###


def comp_var_graph(data, var_names, meas_level, data_frame):
    intro_result = ''
    graph = None
    if meas_level in ['int', 'ord', 'unk']:
    # TODO is it OK for ordinals?
        variables = np.array(data)

        fig = plt.figure(facecolor=csc.bg_col)
        ax = fig.add_subplot(111)
        plt.title(_plt('Boxplot and individual data of the variables'), fontsize=csc.graph_font_size)
        # Display individual data
        for i in range(len(variables.transpose())-1):  # for all pairs
            # Prepare the frequencies for the plot
            xy = [(x,y) for x,y in zip(variables.transpose()[i], variables.transpose()[i+1])]
            xy_set_freq = [[element[0], element[1], xy.count(element)] for element in set(xy)]
            [xvalues, yvalues, xy_freq] = zip(*xy_set_freq)
            xy_freq = np.array(xy_freq, dtype=float)
            max_freq = max(xy_freq)
            if max_freq > 10:
                xy_freq = (xy_freq-1)/((max_freq-1)/9.0)+1
                # largest dot shouldn't be larger than 10 × of the default size
                # smallest dot is 1 unit size
                intro_result += '\n'+_(u'Thickest line displays %d cases.') % max_freq + '\n'
            for data1, data2, data_freq in zip(xvalues, yvalues, xy_freq):
                plt.plot([i+1, i+2], [data1, data2], '-', color = csc.ind_line_col, lw=data_freq)
            
        # Display boxplots
        box1 = ax.boxplot(variables)
        # ['medians', 'fliers', 'whiskers', 'boxes', 'caps']
        plt.setp(box1['boxes'], color=csc.fig_col_bold)
        plt.setp(box1['whiskers'], color=csc.fig_col_bold)
        plt.setp(box1['caps'], color=csc.fig_col_bold)
        plt.setp(box1['medians'], color=csc.fig_col_bold)
        plt.setp(box1['fliers'], color=csc.fig_col_bold)
        plt.xticks(range(1,len(var_names)+1), _wrap_labels(var_names))
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
                    fig, rects = mosaic(ct+1e-9)
                else:
                    fig, rects = mosaic(ct)
                fig.set_facecolor(csc.bg_col)
                ax = plt.subplot(111)
                ax.set_xlabel(var_pair[1])
                ax.set_ylabel(var_pair[0])
                plt.title(_plt('Mosaic plot of the variables'), fontsize=csc.graph_font_size)
                try:
                    graph.append(plt.gcf())
                except:  # in some cases mosaic cannot be drawn  # TODO how to solve this?
                    intro_result = '\n'+_(u'Sorry, the mosaic plot can not be drawn with those data.')
        else:
            intro_result = '\n'+_(u'Sorry, mosaic plot can be drawn only if statsmodels 0.5 or later module is installed.')
    return intro_result, graph


def comp_var_graph_cum(data, var_names, meas_level, data_frame):
    """Draw means with CI for int vars, and medians for ord vars.
    """
    graph = None
    if meas_level in ['int', 'unk']:
        # ord is excluded at the moment
        fig = plt.figure(facecolor=csc.bg_col)
        ax = fig.add_subplot(111)

        if meas_level in ['int', 'unk']:
            plt.title(_plt('Means and 95% confidence intervals for the variables'), fontsize=csc.graph_font_size)
            means = np.mean(data)
            cis, cils, cihs = confidence_interval_t(data, ci_only=False)
            ax.bar(range(len(data.columns)), means, 0.5, yerr=cis, align='center', 
                   color=csc.bg_col, ecolor=csc.fig_col_bold, edgecolor=csc.fig_col)
        elif meas_level in ['ord']:
            plt.title(_plt('Medians for the variables'), fontsize=csc.graph_font_size)
            medians = np.median(data)
            ax.bar(range(len(data.columns)), medians, 0.5, align='center', 
                   color=csc.bg_col, ecolor=csc.fig_col_bold, edgecolor=csc.fig_col)
        plt.xticks(range(len(var_names)), _wrap_labels(var_names))
        plt.ylabel(_plt('Value'))
        graph = plt.gcf()
    return graph


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
    
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#scipy.stats.wilcoxon
    # df = number of pairs of data which have different values i.e., the difference is non-zero
    # http://web.anglia.ac.uk/numbers/biostatistics/wilcoxon/local_folder/critical_values.html
    variables = pdf[var_names].dropna()
    z, p = stats.wilcoxon(variables.iloc[:,0], variables.iloc[:,1])
    text_result += _('Result of Wilcoxon signed rank test')+': <i>W</i> = %0.3g, %s\n' %(z,cs_util.print_p(p))
    # The test does not use df, despite some of the descriptions on the net.
    # So there's no need to display df.
    
    return text_result


def mcnemar_test(pdf, var_names):
    chi2, p = mcnemar(pdf[var_names[0]], pdf[var_names[1]], exact=False)
    return _('Result of the McNemar test') + ': <i>&chi;<sup>2</sup></i>(1, <i>N</i> = %d) = %0.3g, %s\n' % \
                                              (len(pdf[var_names[0]]), chi2, cs_util.print_p(p))


def cochran_q_test(pdf, var_names):
    q, p = cochrans_q(pdf[var_names])
    return _("Result of Cochran's Q test") + ': <i>Q</i>(%d, <i>N</i> = %d) = %0.3g, %s\n' % \
                                              (len(var_names)-1, len(pdf[var_names[0]]), q, cs_util.print_p(p))


def repeated_measures_anova(pdf, var_names):

    [dfn, dfd, f, pf, w, pw], corr_table = cs_stat_num.repeated_measures_anova(pdf, var_names)
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
        pht = cs_stat_num.pairwise_ttest(pdf, var_names).sort_index()
        text_result += '\n' + _('Comparing variables pairwise with the Holm-Bonferroni correction:')
        #print pht
        pht['text'] = pht.apply(lambda x: '<i>t</i> = %0.3g, %s' % (x['t'], cs_util.print_p(x['p (Holm)'])), axis=1)

        pht_text = pht[['text']]
        text_result += pht_text.to_html(bold_rows=True, escape=False, header=False).replace('\n', ''). \
            replace('border="1"', 'style="border:1px solid black;"')

        # Or we can print them in a matrix
        #pht_text = pht[['text']].unstack()
        #np.fill_diagonal(pht_text.values, '')
        #text_result += pht_text.to_html(bold_rows=True, escape=False).replace('\n', '').\
        #                        replace('border="1"', 'style="border:1px solid black;"')

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
    text_result += _('Result of the Friedman test: ')+'<i>&chi;<sup>2</sup></i>(%d, <i>N</i> = %d) = %0.3g, %s\n' % \
                                                      (df, n, chi2, cs_util.print_p(p))  #χ2(1, N=90)=0.89, p=.35

    return text_result

### Compare groups ###


def comp_group_graph(data_frame, meas_level, var_names, groups, group_levels):
    intro_result = ''
    if meas_level in ['int', 'ord']: # TODO 'unk'?
    # TODO is this OK for ordinal?
        fig = plt.figure(facecolor=csc.bg_col)
        ax = fig.add_subplot(111)
        variables = [data_frame[var_names[0]][data_frame[groups[0]]==group_value].dropna() for group_value in list(group_levels)]
        # TODO graph: mean, etc.
        #means = [np.mean(self.data_values[self.data_names.index(var_name)]) for var_name in var_names]
        #stds = [np.std(self.data_values[self.data_names.index(var_name)]) for var_name in var_names]
        #rects1 = ax.bar(range(1,len(variables)+1), means, color=csc.fig_col, yerr=stds)
        box1 = ax.boxplot(variables)
        plt.setp(box1['boxes'], color=csc.fig_col_bold)
        plt.setp(box1['whiskers'], color=csc.fig_col_bold)
        plt.setp(box1['caps'], color=csc.fig_col_bold)
        plt.setp(box1['medians'], color=csc.fig_col_bold)
        plt.setp(box1['fliers'], color=csc.fig_col_bold)
        plt.xticks(range(1, len(group_levels)+1), _wrap_labels(list(group_levels)))
        plt.xlabel(groups[0])
        plt.ylabel(var_names[0])
        # Display individual data
        for i in range(len(variables)):
            val_count = variables[i].value_counts()
            max_freq = max(val_count)
            if max_freq>10:
                val_count = (val_count-1)/((max_freq-1)/9.0)+1
                # largest dot shouldn't be larger than 10 × of the default size
                # smallest dot is 1 unit size
                plt.suptitle(_plt(u'Largest individual data display %d cases.') % max_freq, x=0.9, y=0.02,
                             horizontalalignment='right', fontsize=10)
            ax.scatter(np.ones(len(val_count))+i, val_count.index, val_count.values*5, color='#808080', marker='o')
            #plt.plot(np.ones(len(variables[i]))+i, variables[i], '.', color = '#808080', ms=3) # TODO color should be used from ini file
        plt.title(_plt('Boxplot and individual data of the groups'), fontsize=csc.graph_font_size)
        graph = fig
    elif meas_level in ['nom']:
        if LooseVersion(csc.versions['statsmodels']) >= LooseVersion('0.5'):
            # workaround to draw mosaic plots with zero cell, see #1
            #fig, rects = mosaic(data_frame, [groups[0], var_names[0]])  # previous version
            ct = pd.crosstab(data_frame[var_names[0]], data_frame[groups[0]]).sort_index(axis='index', ascending=False)\
                .unstack()
            if 0 in ct.values:
                fig, rects = mosaic(ct+1e-9)
            else:
                fig, rects = mosaic(ct)
            fig.set_facecolor(csc.bg_col)
            ax = plt.subplot(111)
            ax.set_xlabel(groups[0])
            ax.set_ylabel(var_names[0])
            plt.title(_plt('Mosaic plot of the groups'), fontsize=csc.graph_font_size)
            try:
                graph = fig
            except:  # in some cases mosaic cannot be drawn  # TODO how to solve this?
                intro_result = '\n'+_(u'Sorry, the mosaic plot can not be drawn with those data.')
                graph = None
        else:
            intro_result += '\n'+_(u'Sorry, mosaic plot can be drawn only if statsmodels 0.5 or later module is installed.')
            graph = None
    else:
        graph = None
    return intro_result, graph


def comp_group_graph_cum(data_frame, meas_level, var_names, groups, group_levels):
    """Draw means with CI for int vars, and medians for ord vars.
    """
    graph = None
    if meas_level in ['int', 'unk']:
        # ord is excluded at the moment
        fig = plt.figure(facecolor=csc.bg_col)
        ax = fig.add_subplot(111)
        
        pdf = data_frame.dropna(subset=[var_names[0]])[[var_names[0], groups[0]]]
        if meas_level in ['int', 'unk']:
            plt.title(_plt('Means and 95% confidence intervals for the groups'), fontsize=csc.graph_font_size)
            means = pdf.groupby(groups[0], sort=False).aggregate(np.mean)[var_names[0]]
            cis = pdf.groupby(groups[0], sort=False).aggregate(confidence_interval_t)[var_names[0]]
            ax.bar(range(len(means.values)), means.reindex(group_levels), 0.5, yerr=np.array(cis.reindex(group_levels)), align='center', 
                   color=csc.bg_col, ecolor=csc.fig_col_bold, edgecolor=csc.fig_col)
                   # pandas series is converted to np.array to be able to handle numeric indexes (group levels)
        elif meas_level in ['ord']:
            plt.title(_plt('Medians for the groups'), fontsize=csc.graph_font_size)
            medians = pdf.groupby(groups[0], sort=False).aggregate(np.median)[var_names[0]]
            ax.bar(range(len(medians.values)), medians.reindex(group_levels), 0.5, align='center', 
                   color=csc.bg_col, ecolor=csc.fig_col_bold, edgecolor=csc.fig_col)
        plt.xticks(range(len(group_levels)), _wrap_labels(group_levels))
        plt.xlabel(groups[0])
        plt.ylabel(var_names[0])
        graph = fig
    return graph


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


def modified_t_test(pdf, var_name, grouping_name):
    """Modified t-test for comparing a single case with a group.
    Used typically in case studies.

    arguments:
    var_name (str):
    grouping_name (str):
    """
    text_result = ''
    group_levels, [var1, var2] = _split_into_groups(pdf, var_name, grouping_name)
    try:
        t, p, df = cs_stat_num.modified_t_test(var1, var2)
        text_result += _('Result of the modified independent samples t-test:') + \
                       ' <i>t</i>(%0.3g) = %0.3g, %s\n' % (df, t, cs_util.print_p(p))
    except ValueError:
        text_result += _('One of the groups should include only a single data.')
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
        u, p = stats.mannwhitneyu(var1.dropna(), var2.dropna())
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu
        # The reported p-value is for a one-sided hypothesis, to get the two-sided p-value multiply the returned p-value by 2.
        text_result += _('Result of independent samples Mann-Whitney rank test: ')+'<i>U</i> = %0.3g, %s\n' % \
                                                                                   (u, cs_util.print_p(p*2))
    except Exception as e:
        text_result += _('Result of independent samples Mann-Whitney rank test: ')+unicode(e)

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
    rehab_lm = ols(str(var_name+' ~ C('+grouping_name+')'), data=data).fit()
    ant = anova_lm(rehab_lm)  # TODO Type III to have the same result as SPSS
    text_result += _('Result of one-way ANOVA: ') + '<i>F</i>(%d, %d) = %0.3g, %s\n' % \
                                                    (ant['df'][0], ant['df'][1], ant['F'][0], cs_util.print_p(ant['PR(>F)'][0]))
    # http://en.wikipedia.org/wiki/Effect_size#Omega-squared.2C_.CF.892
    omega2 = (ant['sum_sq'][0] - (ant['df'][0] * ant['mean_sq'][1]))/((ant['sum_sq'][0]+ant['sum_sq'][1]) +ant['mean_sq'][1])
    text_result += _('Effect size: ') + '<i>&omega;<sup>2</sup></i> = %0.3g\n' % omega2
    # http://statsmodels.sourceforge.net/stable/stats.html#multiple-tests-and-multiple-comparison-procedures
    if ant['PR(>F)'][0] < 0.05:  # post-hoc
        post_hoc_res = sm.stats.multicomp.pairwise_tukeyhsd(np.array(data[var_name]), np.array(data[grouping_name]),
                                                            alpha=0.05)
        text_result += '\n'+_(u'Groups differ. Post-hoc test of the means.')+'\n'
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
        text_result += _('Result of the Kruskal-Wallis test: ')+'<i>&chi;<sup>2</sup></i>(%d, <i>N</i> = %d) = %0.3g, %s\n' % \
                                                                (df, n, H, cs_util.print_p(p))  # χ2(1, N=90)=0.89, p=.35
    except Exception as e:
        text_result += _('Result of the Kruskal-Wallis test: ')+unicode(e)
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
    #text_result = '%s\n%s\n'%(text_result, cont_table_data)
    if LooseVersion(csc.versions['scipy'])>=LooseVersion('0.10'):
        chi2, p, dof, expected = stats.chi2_contingency(cont_table_data.values)
        try:
            cramersv = (chi2 / (cont_table_data.values.sum()*(min(cont_table_data.shape)-1)))**0.5
            text_result += _(u'Cramér\'s V measure of association: ')+'<i>&phi;<sub>c</sub></i> = %.3f\n' % cramersv
        except ZeroDivisionError:  # TODO could this be avoided?
            text_result += _(u'Cramér\'s V measure of association cannot be computed (division by zero).')
        text_result += _("Result of the Pearson's Chi-square test: ")+'</i>&chi;<sup>2</sup></i>(%g, <i>N</i> = %d) = %.3f, %s' % \
                                                                      (dof, cont_table_data.values.sum(), chi2, cs_util.print_p(p))
    else:
        text_result += _(u"Sorry, at least SciPy 0.10 is required to calculate Chi-Square test.")
    return text_result

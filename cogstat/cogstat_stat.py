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
from io import StringIO
from scipy import stats
import itertools

from . import cogstat_config as csc
from . import cogstat_util as cs_util
from . import cogstat_stat_num as cs_stat_num
from . import cogstat_chart as cs_chart

try:
    from statsmodels.graphics.mosaicplot import mosaic
except:
    pass
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.sandbox.stats.runs import mcnemar
from statsmodels.sandbox.stats.runs import cochrans_q
from statsmodels.stats.anova import AnovaRM
import pandas as pd
import scikit_posthocs

'''
# r is not needed for some time, but may be necessary at some later point again, so keep the code
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr  # requires at least rpy 2.1.0; use instead:
    #from rpy2.robjects import r # http://stackoverflow.com/questions/2128806/python-rpy2-cant-import-a-bunch-of-packages
except:
    pass
'''


t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext


warn_unknown_variable = '<warning>'+_('The properties of the variables are not set. Set them in your data source.') \
                        + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                        % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                        + '\n<default>' # XXX ezt talán elég az importnál nézni, az elemzéseknél lehet már másként.

### Various things ###

'''
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
'''


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
    levels = [list(set(pdf[group].dropna())) for group in grouping_name]
    for i in range(len(levels)):
        levels[i].sort()

    # create all level combinations for the grouping variables
    level_combinations = list(itertools.product(*levels))
    grouped_data = [pdf[var_name][(pdf[grouping_name] == pd.Series({group: level for group, level in zip(grouping_name, group_level)})).all(axis=1)].dropna() for group_level in
                 level_combinations]
    return level_combinations, grouped_data


def _format_html_table(html_table):
    """Format html table

    :return: str html
    """
    if csc.output_type == 'gui':
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
    function_code = {_('N'): 'len', _('Sum'): 'np.sum', _('Mean'): 'np.mean', _('Median'): 'median',
                     _('Lower quartile'): 'perc25', _('Upper quartile'): 'perc75',
                     _('Standard deviation'): 'np.std', _('Variance'): 'np.var'}
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
                ptable = pd.pivot_table(df, values=depend_name, index=row_names, columns=col_names,
                                        aggfunc=eval(function_code[function]))
                ptable_result = '%s\n%s' % (ptable_result, _format_html_table(ptable.
                                            to_html(bold_rows=False, sparsify=False, float_format=format_output)))
            else:
                temp_result = eval(function_code[function])(np.array(df[depend_name]))
                # TODO convert to html output; when ready stop using fix_width_font
                # np.array conversion needed, np.mean requires axis name for pandas dataframe
                ptable_result = '%s\n%s' % (ptable_result, temp_result)
        return ptable_result
    result += '<fix_width_font>%s\n<default>' % print_pivot_page(pdf, page_names)
    return result


def diffusion(df, error_name=[], RT_name=[], participant_name=[], condition_names=[]):
    """Behavioral diffusion analysis"""
    if not (error_name and RT_name and participant_name and condition_names):
        result = _('Specify all the required parameters (reaction time, error, particpant and condition variables).')
        return result
    result = ''
    result += _('Error: %s, Reaction time: %s, Participant: %s, Condition(s): %s') % \
              (error_name[0], RT_name[0], participant_name[0], ','.join(condition_names))

    # Calculate RT and error rate statistics
    mean_correct_RT_table = pd.pivot_table(df[df[error_name[0]] == 0], values=RT_name[0], index=participant_name,
                                           columns=condition_names, aggfunc=np.mean)
    var_correct_RT_table = pd.pivot_table(df[df[error_name[0]] == 0], values=RT_name[0], index=participant_name,
                                          columns=condition_names, aggfunc=np.var)
    # TODO for the var function do we need a ddof=1 parameter?
    mean_percent_correct_table = 1 - pd.pivot_table(df, values=error_name[0], index=participant_name,
                                                    columns=condition_names,
                                                    aggfunc=cs_stat_num.diffusion_edge_correction_mean)
    previous_precision = pd.get_option('precision')
    pd.set_option('precision', 3)  # thousandth in error, milliseconds in RT, thousandths in diffusion parameters
    result += '\n\n' + _('Mean percent correct with edge correction') + _format_html_table(mean_percent_correct_table.to_html(bold_rows=False))
    result += '\n\n' + _('Mean correct reaction time') + _format_html_table(mean_correct_RT_table.to_html(bold_rows=False))
    result += '\n\n' + _('Correct reaction time variance') + _format_html_table(var_correct_RT_table.to_html(bold_rows=False))

    # Recover diffusion parameters
    EZ_parameters = pd.concat([mean_percent_correct_table.stack(condition_names),
                               var_correct_RT_table.stack(condition_names),
                               mean_correct_RT_table.stack(condition_names)],
                              axis=1).apply(lambda x: cs_stat_num.diffusion_get_ez_params(*x), axis=1)
    EZ_parameters.columns = ['drift rate', 'threshold', 'nondecision time']
    drift_rate_table = EZ_parameters['drift rate'].unstack(condition_names)
    threshold_table = EZ_parameters['threshold'].unstack(condition_names)
    nondecision_time_table = EZ_parameters['nondecision time'].unstack(condition_names)
    result += '\n\n' + _('Drift rate') + _format_html_table(drift_rate_table.to_html(bold_rows=False))
    result += '\n\n' + _('Threshold') + _format_html_table(threshold_table.to_html(bold_rows=False))
    result += '\n\n' + _('Nondecision time') + _format_html_table(nondecision_time_table.to_html(bold_rows=False))
    pd.set_option('precision', previous_precision)

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

    chart = cs_chart.create_variable_raw_chart(pdf, data_measlevs, var_name, data)

    return text_result, chart

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

    normality_histogram, qq_plot = cs_chart.create_normality_chart(data, var_name)
    
    # Decide about normality
    norm = False if p < 0.05 else True
    
    return norm, text_result, normality_histogram, qq_plot


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
        # Or we could use confidence_interval_t
        cil, cih = descr.tconfint_mean()
        ci = (cih-cil)/2
        prec = cs_util.precision(data)+1
        ci_text = '[%0.*f, %0.*f]' %(prec, cil, prec, cih)
        text_result += _('One sample t-test against %g')%float(test_value)+': <i>t</i>(%d) = %0.3g, %s\n' %(df, t, cs_util.print_p(p))
        
        # Graph
        image = cs_chart.create_variable_population_chart(data, var_name, ci)
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

        image = cs_chart.create_variable_population_chart_2(data, var_name)
    else:
        text_result += _('Wilcoxon signed-rank test is computed only for interval or ordinal variables.')
        image = None
    return text_result, image


def print_var_stats(pdf, var_names, meas_levs, groups=None, statistics=[]):
    """
    Computes descriptive stats for variables and/or groups.

    arguments:
    var_names: list of variable names to use
    groups: list of grouping variable names
    meas_levs:
    statistics: list of strings, they can be numpy functions, such as 'mean, 'median', and they should be included in the
            stat_names list

    Now it only handles a single dependent variable and a single grouping variable.
    """
    stat_names = {'mean': _('Mean'), 'median': _('Median'), 'std': _('Standard deviation'), 'amin': _('Minimum'),
                  'amax': _('Maximum'), 'lower_quartile': 'Lower quartile', 'upper_quartile': _('Upper quartile'),
                  'skew': _('Skewness'), 'kurtosis': _('Kurtosis'), 'ptp': _('Range'),
                  'variation_ratio': _('Variation ratio')}
    # Create these functions in numpy namespace to enable simple getattr call of them below
    np.lower_quartile = lambda x: np.percentile(x, 25)
    np.upper_quartile = lambda x: np.percentile(x, 75)
    # with the bias=False it gives the same value as SPSS
    np.skew = lambda x: stats.skew(x, bias=False)
    # with the bias=False it gives the same value as SPSS
    np.kurtosis = lambda x: stats.kurtosis(x, bias=False)
    np.variation_ratio = lambda x: 1 - (sum(x == stats.mode(x)[0][0]) / len(x))

    text_result = ''
    # Compute only variable statistics
    if not groups:
        # drop all data with NaN pair
        data = pdf[var_names].dropna()
        pdf_result = pd.DataFrame(columns=var_names)
        text_result += _('Descriptives for the variables') if len(var_names) > 1 else _('Descriptives for the variable')
        for var_name in var_names:
            if meas_levs[var_name] != 'nom':
                prec = cs_util.precision(data[var_name])+1
            for stat in statistics:
                pdf_result.loc[stat_names[stat], var_name] = '%0.*f' % \
                                                             (2 if stat == 'variation_ratio' else prec,
                                                              getattr(np, stat)(data[var_name].dropna()))
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
                if meas_levs[var_names[0]] != 'nom':
                    prec = cs_util.precision(group_data) + 1
                for stat in statistics:
                    pdf_result.loc[stat_names[stat], group_label] = '%0.*f' % \
                                                                    (2 if stat == 'variation_ratio' else prec,
                                                                     getattr(np, stat)(group_data.dropna()))
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
    descr = DescrStatsW(data)
    cil, cih = descr.tconfint_mean()
    ci = (cih-cil)/2
    if ci_only:
        if isinstance(data, pd.Series):
            return ci  # FIXME this one is for series? The other is for dataframes?
        elif isinstance(data, pd.DataFrame):
            return pd.Series(ci, index=data.columns)
            # without var names the call from comp_group_graph_cum fails
    else:
        return ci, cil, cih

### Variable pairs ###


def var_pair_contingency_table(meas_lev, x, y, data_frame):
    if meas_lev in ['nom']:
        cont_table_data = pd.crosstab(data_frame[y],
                                      data_frame[x])  # , rownames = [x], colnames = [y]) # TODO use data instead?
        text_result = '\n%s\n%s\n' % (_('Contingency table'),
                                      _format_html_table(cont_table_data.to_html(bold_rows=False)))
    else:
        text_result = None
    return text_result


### Compare variables ###


def repeated_measures_estimations(data, meas_level):
    """Draw means with CI for int vars, and medians for ord vars.
    """
    # TODO the same things are calculated in cs_chart.create_repeated_measures_population_chart()
    condition_means_pdf = pd.DataFrame()
    if meas_level in ['int', 'unk']:
        means = np.mean(data)
        cis, cils, cihs = confidence_interval_t(data, ci_only=False)
        condition_means_pdf[_('Point estimation')] = means
        # APA format, but cannot be used the numbers if copied to spreadsheet
        #group_means_pdf[_('95% confidence interval')] = '['+ cils.map(str) + ', ' + cihs.map(str) + ']'
        condition_means_pdf[_('95% CI (low)')] = cils
        condition_means_pdf[_('95% CI (high)')] = cihs
    return condition_means_pdf


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


def repeated_measures_anova(pdf, var_names, factors=[]):
    """
    TODO
    :param pdf:
    :param var_names:
    :param factors:
    :return:
    """

    if not factors:  # one-way comparison
        # TODO use statsmodels functions
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
    else:  # multi-way comparison

        # Prepare the dataset for the ANOVA
        # new temporary names are needed to set the independent factors in the long format
        # (alternatively, one might set it later in the long format directly)
        temp_var_names = ['']
        for factor in factors:
            # TODO this will not work if the factor name includes the current separator (_)
            temp_var_names = [previous_var_name+'_'+factor[0]+str(i)
                              for previous_var_name in temp_var_names for i in range(factor[1])]
        temp_var_names = [temp_var_name[1:] for temp_var_name in temp_var_names]
        #print(temp_var_names)

        pdf_temp = pdf[var_names].dropna()
        pdf_temp.columns = temp_var_names
        pdf_temp = pdf_temp.assign(ID=pdf_temp.index)
        pdf_long = pd.melt(pdf_temp, id_vars='ID', value_vars=temp_var_names)
        pdf_long = pd.concat([pdf_long, pdf_long['variable'].str.split('_', expand=True).
                             rename(columns={i: factors[i][0] for i in range(len(factors))})], axis=1)

        # Run ANOVA
        anovarm = AnovaRM(pdf_long, 'value', 'ID', [factor[0] for factor in factors])
        anova_res = anovarm.fit()

        # Create the text output
        #text_result = str(anova_res)
        text_result = ''
        for index, row in anova_res.anova_table.iterrows():
            factor_names = index.split(':')
            if len(factor_names) == 1:
                text_result += _('Main effect of %s') % factor_names[0]
            else:
                text_result += _('Interaction of factors %s') % ', '.join(factor_names)
            text_result += (': <i>F</i>(%d, %d) = %0.3g, %s\n' %
                            (row['Num DF'], row['Den DF'], row['F Value'], cs_util.print_p(row['Pr > F'])))

        # TODO post hoc - procedure for any number of factors (i.e., not only for two factors)
    #print(text_result)

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


def comp_group_graph_cum(data_frame, meas_level, var_names, groups, group_levels):
    pass
def comp_group_estimations(data_frame, meas_level, var_names, groups):
    """Draw means with CI for int vars, and medians for ord vars.
    """
    group_means_pdf = pd.DataFrame()
    if meas_level in ['int', 'unk']:
        pdf = data_frame.dropna(subset=[var_names[0]])[[var_names[0]] + groups]
        means = pdf.groupby(groups, sort=True).aggregate(np.mean)[var_names[0]]
        cis = pdf.groupby(groups, sort=True).aggregate(confidence_interval_t)[var_names[0]]
        group_means_pdf[_('Point estimation')] = means
        # APA format, but cannot be used the numbers if copied to spreadsheet
        #group_means_pdf[_('95% confidence interval')] = '['+ (means-cis).map(str) + ', ' + (means+cis).map(str) + ']'
        group_means_pdf[_('95% CI (low)')] = means - cis
        group_means_pdf[_('95% CI (high)')] = means + cis
    return group_means_pdf


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
        if p < 0.05:
            # Run the post hoc tests
            text_result += '\n' + _('Groups differ. Post-hoc test of the means.') + '\n'
            text_result += _("Results of Dunn's test (p values).") + '\n'
            posthoc_result = scikit_posthocs.posthoc_dunn(pdf.dropna(subset=[grouping_name]),
                                                          val_col=var_name, group_col=grouping_name)
            text_result += _format_html_table(posthoc_result.to_html(float_format=lambda x : '%.3f'%x))

    except Exception as e:
        text_result += _('Result of the Kruskal-Wallis test: ')+str(e)

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
    chi2, p, dof, expected = stats.chi2_contingency(cont_table_data.values)
    try:
        cramersv = (chi2 / (cont_table_data.values.sum()*(min(cont_table_data.shape)-1)))**0.5
        cramer_result = _('Cramér\'s V measure of association: ')+'&phi;<i><sub>c</sub></i> = %.3f\n' % cramersv
    except ZeroDivisionError:  # TODO could this be avoided?
        cramer_result = _('Cramér\'s V measure of association cannot be computed (division by zero).')
    chi_result = _("Result of the Pearson's Chi-square test: ")+'</i>&chi;<sup>2</sup></i>(%g, <i>N</i> = %d) = %.3f, %s' % \
                                                                  (dof, cont_table_data.values.sum(), chi2, cs_util.print_p(p))
    return cramer_result, chi_result

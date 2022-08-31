# -*- coding: utf-8 -*-

"""
This module contains functions for statistical analyses. The functions
calculate the text output that are compiled in the cogstat module.

The arguments usually include the pandas data frame (pdf), the names of the
relevant variables, and properties of the calculations and the output.

The output is usually a string (html with some custom notations).

Mostly scipy.stats, statsmodels, and pingouin are used to generate the results.
"""

import gettext
import itertools
import os
import string

import numpy as np
import pandas as pd
import pingouin
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW

from . import cogstat_config as csc
from . import cogstat_util as cs_util
from . import cogstat_stat_num as cs_stat_num

'''
# r is not needed for some time, but may be necessary at some later point again, so keep the code
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr  # requires at least rpy 2.1.0; use instead:
    #from rpy2.robjects import r # http://stackoverflow.com/questions/2128806/python-rpy2-cant-import-a-bunch-of-packages
except:
    pass
'''

run_power_analysis = False  # should this analysis be included?

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext


### Various things ###

'''
def _get_R_output(obj):
    """
    Returns the output of R, printing obj.
    """
    from io import StringIO
    import sys

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
    grouped_data = [pdf[var_name][(pdf[grouping_name] == pd.Series({group: level for group, level in
                                                                    zip(grouping_name, group_level)})).all(axis=1)].
                        dropna() for group_level in level_combinations]
    return level_combinations, grouped_data


def _format_html_table(html_table):
    """Format html table

    :return: str html
    """
    if csc.output_type == 'gui':
        # Because qt does not support table borders, use padding to have a more reviewable table
        return '<style> th, td {padding-right: 5px; padding-left: 5px} </style>' + html_table.replace('\n', '').\
            replace('border="1"', 'style="border:1px solid black;"')
    else:
        return html_table.replace('\n', '').replace('border="1"', 'style="border:1px solid black;"')


def pivot(pdf, row_names, col_names, page_names, depend_name, function):
    """Build a pivot table.

    Parameters
    ----------
    pdf : pandas dataframe
    row_names : list of str
    col_names : list of str
    page_names : list of str
    depend_name : str
    function : str
        Localized version of N, Sum, Mean, Median, Lower quartile, Upper quartile, Standard deviation, Variance

    Returns
    -------

    """
            
    if pdf[depend_name].dtype == 'object':
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
    result += _('Dependent variable: ') + depend_name + '\n' + _('Function: ') + function + '\n'

    if function == 'N':
        prec = 0
    else:
        prec = cs_util.precision(pdf[depend_name])+1

    def format_output(x):
        return '%0.*f' % (prec, x)
    
    def print_pivot_page(df, page_vars, page_value_list='', ptable_result='', case_unsensitive_index_sort=True):
        """
        Specify the pages recursively.

        Parameters
        ----------

        case_unsensitive_index_sort : bool
            Pdf.pivot_table() sorts the index, but unlike spreadsheet software packages, it is case sensitive.
            If this parameter is True, the indexes will be reordered to be case-insensitive
        """

        # A. These should be used to handle missing data correctly
        # 1. pandas.pivot_table cannot drop na-s
        # 2. np.percentile cannot drop na-s
        # B. These are used to wrap parameters

        def perc25(data):
            return np.percentile(data.dropna(), 25)

        def median(data):
            # TODO ??? Neither np.median nor stats.nanmedian worked corrrectly ???
            #  or Calc computes percentile in a different way?
            return np.percentile(data.dropna(), 50)

        def perc75(data):
            return np.percentile(data.dropna(), 75)

        if page_vars:
            page_values = set(df[page_vars[0]])
            for page_value in page_values:
                if page_value == page_value:  # skip nan
                    ptable_result = print_pivot_page(df[df[page_vars[0]] == page_value], page_vars[1:], '%s%s = %s%s' %
                                                     (page_value_list, page_vars[0], page_value, ', ' if page_vars[1:]
                                                     else ''), ptable_result)
        else:  # base case
            if page_value_list:
                ptable_result = '%s\n\n%s' % (ptable_result, page_value_list)
            if row_names or col_names:
                ptable = pd.pivot_table(df, values=depend_name, index=row_names, columns=col_names,
                                        aggfunc=eval(function_code[function]))
                if case_unsensitive_index_sort:
                    # default pivot_table() sort returns case-sensitive ordered indexes
                    # we reorder the tables to be case-insensitive
                    from pandas.api.types import is_string_dtype
                    if is_string_dtype(ptable.index):
                        ptable.sort_index(inplace=True, key=lambda x: x.str.lower())
                ptable_result = '%s\n%s' % (ptable_result, _format_html_table(ptable.
                                            to_html(bold_rows=False, sparsify=False, float_format=format_output)))
            else:
                temp_result = eval(function_code[function])(np.array(df[depend_name]))
                # TODO convert to html output; when ready stop using fix_width_font
                # np.array conversion needed, np.mean requires axis name for pandas dataframe
                ptable_result = '%s\n%s' % (ptable_result, temp_result)
        return ptable_result
    result += '%s\n' % print_pivot_page(pdf, page_names)
    return result


def diffusion(df, error_name=[], RT_name=[], participant_name=[], condition_names=[], case_unsensitive_index_sort=True):
    """
    Behavioral diffusion analysis

    Parameters
    ----------

    case_unsensitive_index_sort : bool
        Pdf.pivot_table() sorts the index, but unlike spreadsheet software packages, it is case sensitive.
        If this parameter is True, the indexes will be reordered to be case-insensitive
    """
    if not (error_name and RT_name):
        result = _('Specify the minimum required parameters (reaction time and error).')
        return result
    result = ''
    result += _('Error: %s, Reaction time: %s, Participant: %s, Condition(s): %s') % \
              (error_name[0], RT_name[0], participant_name[0] if participant_name != [] else _('None'),
               ','.join(condition_names) if condition_names != [] else _('None'))
    df_diff = df.copy()

    # If condition and/or participant variables were not given, add a quasi condition/participant variable with
    # constant values
    if not condition_names:
        df_diff[_('Condition')] = _('single condition')
        condition_names = [_('Condition')]
    if not participant_name:
        df_diff[_('Participant')] = _('single participant')
        participant_name = [_('Participant')]

    # If any data is missing from a trial, drop the whole trial
    df_diff = df_diff.dropna(subset=error_name+RT_name+participant_name+condition_names)

    # Calculate RT and error rate statistics
    mean_correct_RT_table = pd.pivot_table(df_diff[df_diff[error_name[0]] == 0], values=RT_name[0],
                                           index=participant_name, columns=condition_names, aggfunc=np.mean)
    var_correct_RT_table = pd.pivot_table(df_diff[df_diff[error_name[0]] == 0], values=RT_name[0],
                                          index=participant_name, columns=condition_names, aggfunc=np.var)
    # TODO for the var function do we need a ddof=1 parameter?
    mean_percent_correct_table = 1 - pd.pivot_table(df_diff, values=error_name[0], index=participant_name,
                                                    columns=condition_names,
                                                    aggfunc=cs_stat_num.diffusion_edge_correction_mean)
    if case_unsensitive_index_sort:
        # default pivot_table() sort returns case-sensitive ordered indexes
        # we reorder the tables to be case-insensitive
        from pandas.api.types import is_string_dtype
        if is_string_dtype(mean_correct_RT_table.index):
            mean_correct_RT_table.sort_index(inplace=True, key=lambda x: x.str.lower())
            var_correct_RT_table.sort_index(inplace=True, key=lambda x: x.str.lower())
            mean_percent_correct_table.sort_index(inplace=True, key=lambda x: x.str.lower())

    previous_precision = pd.get_option('display.precision')
    pd.set_option('display.precision', 3)  # thousandth in error, milliseconds in RT, thousandths in diffusion parameters
    result += '\n\n' + _('Mean percent correct with edge correction') + _format_html_table(mean_percent_correct_table.
                                                                                           to_html(bold_rows=False))
    result += '\n\n' + _('Mean correct reaction time') + _format_html_table(mean_correct_RT_table.
                                                                            to_html(bold_rows=False))
    result += '\n\n' + _('Correct reaction time variance') + _format_html_table(var_correct_RT_table.
                                                                                to_html(bold_rows=False))

    # Recover diffusion parameters
    original_index = mean_percent_correct_table.index  # to recover index order later
    original_columns = mean_percent_correct_table.columns  # to recover column order later
    EZ_parameters = pd.concat([mean_percent_correct_table.stack(condition_names),
                               var_correct_RT_table.stack(condition_names),
                               mean_correct_RT_table.stack(condition_names)],
                              axis=1).apply(lambda x: cs_stat_num.diffusion_get_ez_params(*x),
                                            axis=1, result_type='expand')
    EZ_parameters.columns = ['drift rate', 'threshold', 'nondecision time']
    drift_rate_table = EZ_parameters['drift rate'].unstack(condition_names)
    threshold_table = EZ_parameters['threshold'].unstack(condition_names)
    nondecision_time_table = EZ_parameters['nondecision time'].unstack(condition_names)
    # stack() and unstack() may change the order of the indexes and columns, so we recover them
    drift_rate_table = drift_rate_table.reindex(index=original_index, columns=original_columns)
    threshold_table = threshold_table.reindex(index=original_index, columns=original_columns)
    nondecision_time_table = nondecision_time_table.reindex(index=original_index, columns=original_columns)
    result += '\n\n' + _('Drift rate') + _format_html_table(drift_rate_table.to_html(bold_rows=False))
    result += '\n\n' + _('Threshold') + _format_html_table(threshold_table.to_html(bold_rows=False))
    result += '\n\n' + _('Nondecision time') + _format_html_table(nondecision_time_table.to_html(bold_rows=False))
    pd.set_option('display.precision', previous_precision)

    return result


def safe_var_names(names):  # TODO not used at the moment. maybe could be deleted.
    """Change the variable names for R."""
    # TODO exclude unicode characters
    for i in range(len(names)):
        names[i] = str(names[i]).translate(string.maketrans(' -', '__'))  # use underscore instead of space or dash
        if names[i][0].isdigit():  # do not start with number
            names[i] = '_'+names[i]
        name_changed = False
        for j in range(i):
            while names[i] == names[j]:
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


def frequencies(pdf, var_name, meas_level):
    """Display frequency distribution of a variable.

    Parameters
    ----------
    pdf : pandas dataframe
    var_name : str
        name of the variable
    meas_level :
        measurement level of the variable

    Returns
    -------

    """

    freq = pd.DataFrame()
    freq[_('Value')] = freq.index
    freq[_('Freq')] = pdf[var_name].value_counts().sort_index()
    freq[_('Value')] = freq.index  # previous assignment gives nans for empty df
    if meas_level != 'nom':
        freq[_('Cum freq')] = freq[_('Freq')].cumsum()
    freq[_('Rel freq')] = pdf[var_name].value_counts(normalize=True).sort_index() * 100
    if meas_level != 'nom':
        freq[_('Cum rel freq')] = freq[_('Rel freq')].cumsum()
    text_result = _format_html_table(freq.to_html(formatters={_('Rel freq'): lambda x: '%.1f%%' % x,
                                                              _('Cum rel freq'): lambda x: '%.1f%%' % x},
                                                  bold_rows=False, index=False, classes="table_cs_pd"))
    return text_result


def proportions_ci(pdf, var_name):
    """Calculate the confidence intervals of proportions.

    Parameters
    ----------
    pdf : pandas dataframe
        It is assumed that nans are dropped.
    var_name : str
        name of the variable

    Returns
    -------
    str
    """
    from statsmodels.stats import proportion

    proportions = pdf[var_name].value_counts(normalize=True).sort_index()
    proportions_ci_np = proportion.multinomial_proportions_confint(proportions)
    proportions_ci_pd = pd.DataFrame(proportions_ci_np, index=proportions.index, columns=[_('low'), _('high')]) * 100
    text_result = '%s - %s\n%s' % (_('Relative frequencies'), _('95% confidence interval (multinomial proportions)'),
                                   _format_html_table(proportions_ci_pd.to_html(bold_rows=False,
                                                                                float_format=lambda x: '%.1f%%' % x,
                                                                                classes="table_cs_pd")))
    if (pdf[var_name].value_counts() < 5).any():
        text_result += '<warning>' + _('Some of the cells do not include at least 5 cases, so the confidence '
                                       'intervals may be invalid.') + '</warning>\n'
    return text_result


def print_var_stats(pdf, var_names, meas_levs, groups=None, statistics=[]):
    """
    Computes descriptive stats for variables and/or groups.

    Parameters
    ----------
    pdf : pandas dataframe
        It is assumed that missing cases are dropped
    var_names : list of str
        variable names to use
    groups : list of str
        grouping variable names
    meas_levs :

    statistics : list of str
        they can be numpy functions, such as 'mean, 'median', and they should be included in the stat_names list

    Now it only handles a single dependent variable and a single grouping variable.

    Returns
    -------

    """
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

    text_result = ''
    prec = None
    # Compute only variable statistics
    if not groups:  # for single variable or repeated measures variables
        # drop all data with NaN pair
        data = pdf[var_names]
        pdf_result = pd.DataFrame(columns=var_names)
        text_result += '<cs_h3>' + (_('Descriptives for the variables') if len(var_names) > 1 else
                                    _('Descriptives for the variable')) + '</cs_h3>'
        for var_name in var_names:
            if meas_levs[var_name] != 'nom':
                prec = cs_util.precision(data[var_name])+1
            for stat in statistics:
                pdf_result.loc[stat_names[stat], var_name] = '%0.*f' % \
                                                             (2 if stat == 'variation ratio' else prec,
                                                              stat_functions[stat](data[var_name].dropna()))
    # There is at least one grouping variable
    else:
        # missing groups and values will be dropped (though this is not needed since it is assumed that they have been
        # dropped)
        groups, grouped_data = _split_into_groups(pdf, var_names[0], groups)
        groups = [' : '.join(map(str, group)) for group in groups]
        pdf_result = pd.DataFrame(columns=groups)

        text_result += '<cs_h3>' + _('Descriptives for the groups') + '</cs_h3>'
        # Not sure if the precision can be controlled per cell with this method;
        # Instead we make a pandas frame with str cells
#        pdf_result = pd.DataFrame([np.mean(group_data.dropna()) for group_data in grouped_data], columns=[_('Mean')],
#        index=groups)
#        text_result += pdf_result.T.to_html()
        for group_label, group_data in zip(groups, grouped_data):
            if len(group_data):
                if meas_levs[var_names[0]] != 'nom':
                    prec = cs_util.precision(group_data) + 1
                for stat in statistics:
                    pdf_result.loc[stat_names[stat], group_label] = '%0.*f' % \
                                                                    (2 if stat == 'variation ratio' else prec,
                                                                     stat_functions[stat](group_data.dropna()))
            else:  # TODO can we remove this part?
                text_result += _('No data')
                for stat in statistics:
                    pdf_result.loc[stat_names[stat], group_label] = _('No data')
    text_result += _format_html_table(pdf_result.to_html(bold_rows=False, classes="table_cs_pd"))
    return text_result


def variable_estimation(data, statistics=[]):
    """
    Calculate the point and interval estimations of the required parameters.

    Parameters
    ----------
    data : pandas series
         It is assumed that nans are dropped.

    statistics : list of {'mean', 'std', 'median'}

    Returns
    -------
    str
        Table of the point and interval estimations
    """
    pdf_result = pd.DataFrame()
    population_param_text = ''
    for statistic in statistics:
        if statistic == 'mean':
            population_param_text += _('Present confidence interval values for the mean suppose normality.') + '\n'
            pdf_result.loc[_('Mean'), _('Point estimation')] = np.mean(data)
            pdf_result.loc[_('Mean'), _('95% confidence interval (low)')], \
            pdf_result.loc[_('Mean'), _('95% confidence interval (high)')] = DescrStatsW(data).tconfint_mean()
        if statistic == 'std':
            # Currently, sd calculation assumes that mean has already been calculated (and column names are alraedy given)
            stddev = np.std(data, ddof=1)
            lower, upper = cs_stat_num.stddev_ci(stddev, len(data), 0.95)
            pdf_result.loc[_('Standard deviation')] = [stddev, lower, upper]
        if statistic == 'median':
            pdf_result.loc[_('Median'), _('Point estimation')] = np.median(data)
            pdf_result.loc[_('Median'), _('95% confidence interval (low)')], \
            pdf_result.loc[_('Median'), _('95% confidence interval (high)')] = \
                cs_stat_num.quantile_ci(pd.DataFrame(data))
    pdf_result = pdf_result.fillna(_('Out of the data range'))
    prec = cs_util.precision(data) + 1
    population_param_text += _format_html_table(pdf_result.to_html(bold_rows=False, classes="table_cs_pd",
                                                                   float_format=lambda x: '%0.*f' % (prec, x)))
    return population_param_text


def confidence_interval_t(data):
    """Calculate the confidence interval of the mean.

    95%, two-sided CI based on t-distribution
    http://statsmodels.sourceforge.net/stable/_modules/statsmodels/stats/weightstats.html#DescrStatsW.tconfint_mean

    Parameters
    ----------
    data : pandas dataframe or pandas series
         include all the variables the CI is needed for

    Returns
    -------
    int
        the difference from the mean
    """
    # TODO is this solution slow? Should we write our own CI function?
    cil, cih = DescrStatsW(data).tconfint_mean()
    ci = (cih-cil)/2
    if isinstance(data, pd.Series):  # TODO do we still need this? do the callers always use a pdf argument?
        return ci  # FIXME this one is for series? The other is for dataframes?
    elif isinstance(data, pd.DataFrame):
        return pd.Series(ci, index=data.columns)
        # without var names the call from comp_group_graph_cum fails

### Variable pairs ###

def variable_pair_regression_coefficients(predictors, meas_lev, normality=None, homoscedasticity=None,
                                          multicollinearity=None, result=None):
    """
    Calculate point and interval estimates of regression parameters (slopes, and intercept) in a regression analysis.

    Parameters
    ----------
    predictors : list of str
        List of explanatory variable names.
    meas_lev : str
        Measurement level of the regressors
    normality : bool or None
        True if variables follow a multivariate normal distribution, False otherwise. None if normality couldn't be
        calculated or if the parameter was not specified.
    homoscedasticity : bool or None
        True if variables are homoscedastic, False otherwise. None if homoscedasticity couldn't be calculated or
        if the parameter was not specified.
    multicollinearity : bool or None
        True if multicollinearity is suspected (VIF>10), False otherwise. None if the parameter was not specified.
    result: statsmodels regression result object
        The result of the multiple regression analysis.

    Returns
    -------
    str
        Table of the point and interval estimations as html text
    """
    if meas_lev == "int":
        regression_coefficients = '<cs_h4>' + _('Regression coefficients') + '</cs_h4>'
        pdf_result = pd.DataFrame(columns=[_('Point estimation'), _('95% confidence interval')])

        # Warnings based on the results of the assumption tests
        if normality is None:
            regression_coefficients += '\n' + '<decision>' + _('Normality could not be calculated.') + ' ' +\
                                                   _('CIs may be biased.') + '</decision>'
        elif not normality:
            regression_coefficients += '\n' + '<decision>' \
                                       + _('Assumption of normality violated for CI calculations.') + ' ' + \
                                       _('CIs may be biased.') + '</decision>'
        else:
            regression_coefficients += '\n' + '<decision>' + _('Assumption of normality for CI calculations met.') + \
                                       '</decision>'

        if homoscedasticity is None:
            regression_coefficients += '\n' + '<decision>' + _('Homoscedasticity could not be calculated.') + ' ' + \
                                       _('CIs may be biased.') + '</decision>'
        elif not homoscedasticity:
            regression_coefficients += '\n' + '<decision>' \
                                       + _('Assumption of homoscedasticity violated for CI calculations.') + ' ' + \
                                       _('CIs may be biased.') + '</decision>'
        else:
            regression_coefficients += '\n' + '<decision>' + _('Assumption of homoscedasticity for CI '
                                                               'calculations met.') + '</decision>'

        if len(predictors) > 1:
            if multicollinearity is None:
                regression_coefficients += '\n' + '<decision>' + _('Multicollinearity could not be calculated.') + \
                                           ' ' + _('Point estimates and CIs may be inaccurate.') + '</decision>'
            elif multicollinearity:
                regression_coefficients += '\n' + '<decision>' \
                                           + _('Multicollinearity suspected.') + ' ' + \
                                           _('Point estimates and CIs may be inaccurate.') + '</decision>'
            else:
                regression_coefficients += '\n' + '<decision>' + _('Assumption of multicollinearity for'
                                                                   ' CI calculations met.') + '</decision>'

        # Gather point estimates and CIs into table
        cis = result.conf_int(alpha=0.05)
        pdf_result.loc[_('Intercept')] = \
            ['%0.3f' % (result.params[0]), '[%0.3f, %0.3f]' % (cis.loc['const', 0], cis.loc['const', 1])]
        for predictor in predictors:
            pdf_result.loc['Slope for %s' % predictor] = ['%0.3f' % (result.params[predictor]), '[%0.3f, %0.3f]' %
                                                          (cis.loc[predictor, 0], cis.loc[predictor, 1])]
    else:
        regression_coefficients = None

    if regression_coefficients:
        regression_coefficients += _format_html_table(pdf_result.to_html(bold_rows=False, escape=False,
                                                                         classes='table_cs_pd'))

    return regression_coefficients


def variable_pair_standard_effect_size(data, meas_lev, sample=True, normality=None, homoscedasticity=None):
    """Calculate standardized effect size measures.
    (Some stats are also calculated elsewhere, making the analysis slower, but separation is a priority.)

    Parameters
    ----------
    data : pandas dataframe
    meas_lev : str
        Measurement level of variables.
    sample : bool
        True for sample descriptives, False for population estimations.
    normality: bool or None
        True if variables follow a multivariate normal distribution, False otherwise. None if normality couldn't be
        calculated or if the parameter was not specified.
    homoscedasticity : bool or None
        True if variables are homoscedastic, False otherwise. None if homoscedasticity couldn't be calculated or
        if the parameter was not specified.

    Returns
    -------
    html text
    """
    pdf_result = pd.DataFrame()
    if sample:
        standardized_effect_size_result = '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>'
        if meas_lev in ['int', 'unk']:
            pdf_result.loc[_("Pearson's correlation"), _('Value')] = \
                '<i>r</i> = %0.3f' % stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])[0]
            pdf_result.loc[_("Spearman's rank-order correlation"), _('Value')] = \
                '<i>r<sub>s</sub></i> = %0.3f' % stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])[0]
        elif meas_lev == 'ord':
            pdf_result.loc[_("Spearman's rank-order correlation"), _('Value')] = \
                '<i>r<sub>s</sub></i> = %0.3f' % stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])[0]
        elif meas_lev == 'nom':
            try:
                cramersv = pingouin.chi2_independence(data, data.columns[0], data.columns[1])[2].loc[0, 'cramer']
                # TODO this should be faster when minimum scipy can be 1.7:
                # cramersv = stats.contingency.association(data.iloc[:, 0:1], method='cramer')) # new in scipy 1.7
                pdf_result.loc[_("Cramér's V measure of association"), _('Value')] = \
                    '&phi;<i><sub>c</sub></i> = %.3f' % cramersv
            except ZeroDivisionError:  # TODO could this be avoided?
                pdf_result.loc[_("Cramér's V measure of association"), _('Value')] = \
                    'cannot be computed (division by zero)'
    else:  # population estimations
        standardized_effect_size_result = '<cs_h4>' + _('Standardized effect sizes') + '</cs_h4>'
        pdf_result = pd.DataFrame(columns=[_('Point estimation'), _('95% confidence interval')])
        if meas_lev in ['int', 'unk']:
            df = len(data) - 2
            r, p = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])
            r_ci_low, r_ci_high = cs_stat_num.corr_ci(r, df + 2)
            pdf_result.loc[_("Pearson's correlation") + ', <i>r</i>'] = \
                ['%0.3f' % r, '[%0.3f, %0.3f]' % (r_ci_low, r_ci_high)]

            # Warnings based on the results of the assumption tests
            if normality is None:
                standardized_effect_size_result += '\n' + '<decision>' + _('Normality could not be calculated.') + ' ' +\
                                                   _('CIs may be biased.') + '</decision>'
            elif not normality:
                standardized_effect_size_result += '\n' + '<decision>' + \
                                                   _('Assumption of normality violated.') + ' ' + \
                                                   _('CIs may be biased.') + '</decision>'
            else:
                standardized_effect_size_result += '\n' + '<decision>' + \
                                                   _('Assumption of normality met.') + '</decision>'

            if homoscedasticity is None:
                standardized_effect_size_result += '\n' + '<decision>' + _('Homoscedasticity could not be calculated.') \
                                                   + ' ' + _('CIs may be biased.') + '</decision>'
            elif not homoscedasticity:
                standardized_effect_size_result += '\n' + '<decision>' \
                                           + _('Assumption of homoscedasticity violated.') + ' ' + \
                                           _('CIs may be biased.') + '</decision>'
            else:
                standardized_effect_size_result += '\n' + '<decision>' + _('Assumption of homoscedasticity met.') \
                                                   + '</decision>'

        if meas_lev in ['int', 'unk', 'ord']:
            df = len(data) - 2
            r, p = stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])
            r_ci_low, r_ci_high = cs_stat_num.corr_ci(r, df + 2)
            pdf_result.loc[_("Spearman's rank-order correlation") + ', <i>r<sub>s</sub></i>'] = \
                ['%0.3f' % (r), '[%0.3f, %0.3f]' % (r_ci_low, r_ci_high)]
        elif meas_lev == 'nom':
            standardized_effect_size_result = ''
    if standardized_effect_size_result:
        standardized_effect_size_result += _format_html_table(pdf_result.to_html(bold_rows=False, escape=False,
                                                                                 classes='table_cs_pd'))
    return standardized_effect_size_result


def multiple_variables_standard_effect_size(data, predictors, y, result, normality=None, homoscedasticity=None,
                                            multicollinearity=None, sample=True):
    """Calculate standardized effect size measures for multiple regression.

    Parameters
    ----------
    data : pandas dataframe
    predictors : list of str
        Name of the explanatory variables.
    y : list of str  # TODO is there a reason why this is a list? it is inconsistent with other interfaces
        Name of the dependent variable.
    result : statsmodels regression result object
        The result of the multiple regression analyses.
    normality: bool or None
        True if variables follow a multivariate normal distribution, False otherwise. None if normality couldn't be
        calculated or if the parameter was not specified.
    homoscedasticity : bool or None
        True if variables are homoscedastic, False otherwise. None if homoscedasticity couldn't be calculated or
        if the parameter was not specified.
    multicollinearity : bool or None
        True if possible multicollinearity was detected (VIF>10). None if the parameter was not specified.
    sample : bool
        True for sample descriptives, False for population estimations.

    Returns
    -------
    html text
    """
    # TODO validate

    if sample:
        standardized_effect_size_result = '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>' + '\n'
    else:
        standardized_effect_size_result = '<cs_h4>' + _('Standardized effect sizes') + '</cs_h4>' + '\n'
    # Warnings based on the results of the assumption tests
    # TODO warnings should be printed only with population properties?
    if normality is None:
        standardized_effect_size_result += '\n' + '<decision>' + _('Normality could not be calculated.') + ' ' + \
                                           _('CIs may be biased.') + '</decision>'
    elif not normality:
        standardized_effect_size_result += '\n' + '<decision>' + \
                                           _('Assumption of normality violated.') + ' ' + \
                                           _('CIs may be biased.') + '</decision>'
    else:
        standardized_effect_size_result += '\n' + '<decision>' + \
                                           _('Assumption of normality met.') + '</decision>'

    if homoscedasticity is None:
        standardized_effect_size_result += '\n' + '<decision>' + _('Homoscedasticity could not be calculated.') + \
                                           ' ' + _('CIs may be biased.') + '</decision>'
    elif not homoscedasticity:
        standardized_effect_size_result += '\n' + '<decision>' \
                                           + _('Assumption of homoscedasticity violated.') + ' ' + \
                                           _('CIs may be biased.') + '</decision>'
    else:
        standardized_effect_size_result += '\n' + '<decision>' + _('Assumption of homoscedasticity met.') + '</decision>'

    if multicollinearity is None:
        standardized_effect_size_result += '\n' + '<decision>' + _('Multicollinearity could not be calculated.') + \
                                           ' ' + _('CIs may be biased.') + '</decision>'
    elif not multicollinearity:
        standardized_effect_size_result += '\n' + '<decision>' \
                                           + _('Assumption of multicollinearity violated.') + ' ' + \
                                           _('CIs may be biased.') + '</decision>'
    else:
        standardized_effect_size_result += '\n' + '<decision>' + _('Assumption of multicollinearity met.') + '</decision>'

    # Calculate effect sizes for sample or population
    if sample:
        pdf_result_corr = pd.DataFrame()
        standardized_effect_size_result += "\n" + _('<i>R<sup>2</sup></i> = %0.3f' % result.rsquared) + '\n'
    else:  # population
        pdf_result_model = pd.DataFrame(columns=[_('Point estimation'), _('95% confidence interval')])
        pdf_result_corr = pd.DataFrame(columns=[_('Point estimation'), _('95% confidence interval')])

        ci = cs_stat_num.calc_r2_ci(result.rsquared_adj, len(predictors), len(data))
        pdf_result_model.loc[_('Adjusted <i>R<sup>2</sup></i>')] = \
            ['%0.3f' % result.rsquared_adj, '[%0.3f, %0.3f]' % (ci[0], ci[1])]

        pdf_result_model.loc[_('Log-likelihood')] = ['%0.3f' % result.llf, '']
        pdf_result_model.loc[_('AIC')] = ['%0.3f' % result.aic, '']
        pdf_result_model.loc[_('BIC')] = ['%0.3f' % result.bic, '']
        standardized_effect_size_result += _format_html_table(pdf_result_model.to_html(bold_rows=False, escape=False,
                                                                                       classes='table_cs_pd')) + '\n'

    standardized_effect_size_result += '\n' + _("Pearson's partial correlations")

    for predictor in predictors:
        predictors_other = predictors.copy()
        predictors_other.remove(predictor)

        if sample:
            pdf_result_corr.loc[predictor, _('Value')] = \
                '<i>r</i> = %0.3f' % pingouin.partial_corr(data, predictor, y, predictors_other)['r']
        else:
            partial_result = pingouin.partial_corr(data, predictor, y, predictors_other)
            pdf_result_corr.loc[predictor + ', <i>r</i>'] = \
                ['%0.3f' % (partial_result['r']), '[%0.3f, %0.3f]' % (partial_result['CI95%'][0][0],
                                                                      partial_result['CI95%'][0][1])]

    standardized_effect_size_result += _format_html_table(pdf_result_corr.to_html(bold_rows=False, escape=False,
                                                                                  classes='table_cs_pd'))

    return standardized_effect_size_result


def correlation_matrix(data, regressors):
    """Create Pearson's correlation matrix for assessment of multicollinearity.

    Parameters
    ----------
    data : pandas dataframe
    regressors : list of str
        list of explanatory variable names

    Returns
    -------
    html text
    """

    corr_table = data[regressors].corr()
    table = _('Pearson correlation matrix of explanatory variables')
    table += _format_html_table(corr_table.to_html(bold_rows=False, escape=False, classes='table_cs_pd')) + '\n'
    return table


def vif_table(data, var_names):
    """Calculate and display Variance Inflation Factors. Raises warning and displays corresponding
    auxiliary regression weights in case of suspected multicollineearity (VIF>10).

    Parameters
    ----------
    data : pandas dataframe
    var_names : list of str
        list of explanatory variable names

    Returns
    -------
    html text, bool
        Html text with variance inflation factors (VIFs) and beta weights from auxiliary regression in case of VIF > 10.
        Boolean is True when any VIF is greater than 10. False otherwise.
    """

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.api import add_constant
    from statsmodels.api import OLS

    regressors = add_constant(data[var_names])
    table = _('Variance inflation factors of explanatory variables and constant')
    vifs = pd.DataFrame([variance_inflation_factor(regressors.values, i) \
                         for i in range(regressors.shape[1])], index=regressors.columns, columns=[_('VIF')])
    table += _format_html_table(vifs.to_html(bold_rows=False, escape=False, classes='table_cs_pd')) + '\n'

    multicollinearity = False
    for regressor in var_names:
        if vifs.loc[regressor, _('VIF')] > 10:
            multicollinearity = True
            table += '\n' + '<decision>' + _('VIF > 10 in variable %s ') % regressor + '\n' + \
                     _('Possible multicollinearity.') + '\n</decision>'
            regressors_other = var_names.copy()
            regressors_other.remove(regressor)
            table += _('Beta weights when regressing %s on all other regressors.' % regressor)
            slopes = pd.DataFrame(OLS(data[regressor], add_constant(data[regressors_other])).fit().params,
                                  columns=[_('Slope on %s') % regressor])
            table += _format_html_table(slopes.to_html(bold_rows=False, escape=False, classes='table_cs_pd')) + '\n'
    return table, multicollinearity


def contingency_table(data_frame, x, y, count=False, percent=False, ci=False, margins=False):
    """Create contingency tables. Use for nominal data.
    It works for any number of x and y variables.

    Parameters
    ----------
    data_frame : pandas dataframe
        It is assumed that missing cases are removed.
    x : list of str
        list of variable names (for columns)
    y : list of str
        list of variable names (for rows)
    count : bool
    percent : bool
    ci : bool
        multinomial, goodman method
    margins : bool
        option for count and percent tables,  ci calculation ignores it

    Returns
    -------

    """
    text_result = ''
    if count:
        cont_table_count = pd.crosstab(index=[data_frame[ddd] for ddd in data_frame[y]],
                                       columns=[data_frame[ddd] for ddd in data_frame[x]], margins=margins,
                                       margins_name=_('Total'))
        text_result += '\n%s - %s\n%s\n' % (_('Contingency table'), _('Case count'),
                                            _format_html_table(cont_table_count.to_html(bold_rows=False,
                                                                                        classes="table_cs_pd")))
    if percent:
        # for the pd.crosstab() function normalize=True parameter does not work with multiple column variables
        # (neither pandas version 0.22 nor 1.0.3 works, though with different error messages), so make a workaround
        cont_table_count = pd.crosstab([data_frame[ddd] for ddd in data_frame[y]],
                                       [data_frame[ddd] for ddd in data_frame[x]], margins=margins,
                                       margins_name=_('Total'))
        # normalize by the total count
        cont_table_perc = cont_table_count / cont_table_count.iloc[-1, -1] * 100
        text_result += '\n%s - %s\n%s\n' % (_('Contingency table'), _('Percentage'),
                                            _format_html_table(cont_table_perc.to_html(bold_rows=False,
                                                                                       classes="table_cs_pd",
                                                                                       float_format=lambda x: '%.1f%%'
                                                                                                              % x)))
    if ci:
        from statsmodels.stats import proportion
        cont_table_count = pd.crosstab([data_frame[ddd] for ddd in data_frame[y]],
                                       [data_frame[ddd] for ddd in data_frame[x]])  # don't use margins
        cont_table_ci_np = proportion.multinomial_proportions_confint(cont_table_count.unstack())
        # add index and column names for the numpy results, and reformat (unstack) to the original arrangement
        cont_table_ci = pd.DataFrame(cont_table_ci_np, index=cont_table_count.unstack().index,
                                     columns=[_('low'), _('high')]).stack().unstack(level=list(range(len(x))) +
                                                                                          [len(x)+1]) * 100
        text_result += '%s - %s\n%s\n' % (_('Contingency table'),
                                          _('95% confidence interval (multinomial proportions)'),
                                            _format_html_table(cont_table_ci.to_html(bold_rows=False,
                                                                                     classes="table_cs_pd",
                                                                                     float_format=lambda x: '%.1f%%' %
                                                                                                            x)))
        if (cont_table_count < 5).values.any(axis=None):  # df.any(axis=None) doesn't work for some reason,
                                                          # so we use the np version
            text_result += '<warning>' + _('Some of the cells do not include at least 5 cases, so the confidence '
                                           'intervals may be invalid.') + '</warning>\n'

    """
    # Binomial CI with continuity correction
    n = cont_table_count.values.sum()
    margin_of_error = 1.96 * np.sqrt(cont_table_perc * (1 - cont_table_perc) / n) + 0.5 / n
    ct_low = cont_table_perc - margin_of_error
    ct_high = cont_table_perc + margin_of_error
    ct_ci = pd.concat([ct_low, ct_high], keys=[_('CI low'), _('CI high')])
    text_result += '\n%s\n%s\n' % (_('Proportions 95% CI'),
                                  _format_html_table(ct_ci.unstack(level=0).to_html(bold_rows=False, float_format=lambda x: '%.1f%%'%x)))

    # Binomial CI without continuity correction
    from statsmodels.stats import proportion
    pci = lambda x: proportion.proportion_confint(x, nobs=n, method='normal')
    cont_table_count.applymap(pci)
    """

    return text_result


### Compare variables ###


def repeated_measures_estimations(data, meas_level):
    """Draw means with CI for int vars, and medians for ord vars.
    """
    # TODO the same things are calculated in cs_chart.create_repeated_measures_population_chart()
    condition_estimations_pdf = pd.DataFrame()
    if meas_level in ['int', 'unk']:
        condition_estimations_pdf[_('Point estimation')] = data.mean()
        # APA format, but cannot be used the numbers if copied to spreadsheet
        #group_means_pdf[_('95% confidence interval')] = '['+ cils.map(str) + ', ' + cihs.map(str) + ']'
        cils, cihs = DescrStatsW(data).tconfint_mean()
        condition_estimations_pdf[_('95% CI (low)')] = cils
        condition_estimations_pdf[_('95% CI (high)')] = cihs
    if meas_level == 'ord':
        condition_estimations_pdf[_('Point estimation')] = data.median()
        cis_np = cs_stat_num.quantile_ci(data)
        condition_estimations_pdf[_('95% CI (low)')], condition_estimations_pdf[_('95% CI (high)')] = cis_np
        condition_estimations_pdf = condition_estimations_pdf.fillna(_('Out of the data range'))
    return condition_estimations_pdf


def repeated_measures_effect_size(pdf, var_names, factors, meas_level, sample=True):
    """

    Parameters
    ----------
    pdf : pandas dataframe
    var_names
    factors
    meas_level
    sample : bool
        Should the effect size for sample or population be calculated?

    Returns
    -------
    str or None
        None if effect size is not calculated

    """
    standardized_effect_size_result = '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>'

    if sample:  # Effects sizes for samples
        pdf_result = pd.DataFrame()
        if len(factors) < 2:  # Single (or no) factor
            if len(var_names) == 2:  # Two variables
                if meas_level in ['int', 'unk']:
                    pdf_result.loc[_("Cohen's d"), _('Value')] = \
                        pingouin.compute_effsize(pdf[var_names[0]], pdf[var_names[1]], paired=True, eftype='cohen')
                    pdf_result.loc[_("Eta-squared"), _('Value')] = \
                        pingouin.compute_effsize(pdf[var_names[0]], pdf[var_names[1]], paired=True, eftype='eta-square')
                else:  # ordinal or nominal variable
                    standardized_effect_size_result = None
            else:  # More than two variables
                standardized_effect_size_result = None
        else:  # Multiple factors
            standardized_effect_size_result = None

    else:  # Effect size estimations
        pdf_result = pd.DataFrame(columns=[_('Point estimation'), _('95% CI (low)'), _('95% CI (high)')])
        if len(factors) < 2:  # Single (or no) factor
            if len(var_names) == 2:  # Two variables
                if meas_level in ['int', 'unk']:
                    hedges = pingouin.compute_effsize(pdf[var_names[0]], pdf[var_names[1]], paired=True,
                                                      eftype='hedges')
                    hedges_ci = pingouin.compute_esci(stat=hedges, nx=len(pdf[var_names[0]]), ny=len(pdf[var_names[1]]),
                                                      paired=True, eftype='cohen', confidence=0.95, decimals=3)
                    pdf_result.loc[_("Hedges' g")] = hedges, *hedges_ci
                else:  # ordinal or nominal variable
                    standardized_effect_size_result = None
            else:  # More than two variables
                standardized_effect_size_result = None
        else:  # Multiple factors
            standardized_effect_size_result = None

    if not pdf_result.empty:
        standardized_effect_size_result += _format_html_table(pdf_result.to_html(bold_rows=False, escape=False,
                                                                                 float_format=lambda x: '%0.3f' % (x),
                                                                                 classes="table_cs_pd"))
    return standardized_effect_size_result


### Compare groups ###


def comp_group_graph_cum(data_frame, meas_level, var_names, groups, group_levels):
    pass


def comp_group_estimations(pdf, meas_level, var_names, groups):
    """Draw means with CI for int vars, and medians for ord vars.

    Parameters
    ----------
    pdf : pandas dataframe
        it is assumed that missing cases are removed.
    meas_level
    var_names
    groups

    Returns
    -------

    """
    group_estimations_pdf = pd.DataFrame()
    if meas_level in ['int', 'unk']:
        pdf = pdf[[var_names[0]] + groups]
        means = pdf.groupby(groups, sort=True).aggregate(np.mean)[var_names[0]]
        # TODO can we use directly DescrStatsW instead of confidence_interval_t? (later we only use cil and cih)
        cis = pdf.groupby(groups, sort=True).aggregate(confidence_interval_t)[var_names[0]]
        group_estimations_pdf[_('Point estimation')] = means
        # APA format, but cannot be used the numbers if copied to spreadsheet
        #group_means_pdf[_('95% confidence interval')] = '['+ (means-cis).map(str) + ', ' + (means+cis).map(str) + ']'
        group_estimations_pdf[_('95% CI (low)')] = means - cis
        group_estimations_pdf[_('95% CI (high)')] = means + cis
        if len(groups) == 1 and len(set(pdf[groups[0]])) == 2:  # when we have two groups
            # TODO same assumptions apply as for t-test?
            # CI http://onlinestatbook.com/2/estimation/difference_means.html
            # However, there are other computational methods:
            # http://dept.stat.lsa.umich.edu/~kshedden/Python-Workshop/stats_calculations.html
            # http://www.statisticslectures.com/topics/ciindependentsamplest/
            dummy_groups, [var1, var2] = _split_into_groups(pdf, var_names[0], groups[0])
            var1 = var1.dropna()
            var2 = var2.dropna()
            df = len(var1) + len(var2) - 2
            mean_diff = np.mean(var1) - np.mean(var2)
            sse = np.sum((np.mean(var1) - var1) ** 2) + np.sum((np.mean(var2) - var2) ** 2)
            mse = sse / df
            nh = 2.0 / (1.0 / len(var1) + 1.0 / len(var2))
            s_m1m2 = np.sqrt(2 * mse / nh)
            t_cl = stats.t.ppf(1 - (0.05 / 2), df)  # two-tailed
            lci = mean_diff - t_cl * s_m1m2
            hci = mean_diff + t_cl * s_m1m2
            group_estimations_pdf.loc[_('Difference between the two groups:')] = [mean_diff, lci, hci]
    elif meas_level == 'ord':
        pdf = pdf[[var_names[0]] + groups]
        group_estimations_pdf[_('Point estimation')] = pdf.groupby(groups, sort=True).aggregate(np.median)[var_names[0]]
        cis = pdf.groupby(groups, group_keys=False, sort=True).apply(lambda x: cs_stat_num.quantile_ci(x)[:, 0])
            # TODO this solution works, but a more reasonable code would be nicer
        # APA format, but cannot be used the numbers if copied to spreadsheet
        #group_means_pdf[_('95% confidence interval')] = '['+ (means-cis).map(str) + ', ' + (means+cis).map(str) + ']'
        group_estimations_pdf[_('95% CI (low)')] = np.concatenate(cis.values).reshape((-1, 2))[:, 0]
        group_estimations_pdf[_('95% CI (high)')] = np.concatenate(cis.values).reshape((-1, 2))[:, 1]
            # TODO this solution works, but a more reasonable code would be nicer
        group_estimations_pdf = group_estimations_pdf.fillna(_('Out of the data range'))
    return group_estimations_pdf


def compare_groups_effect_size(pdf, dependent_var_name, groups, meas_level, sample=True):
    """

    Parameters
    ----------
    pdf : pandas dataframe
        It assumes that missing cases are dropped.
    dependent_var_name
    groups
    meas_level
    sample : bool

    Returns
    -------
    str or None
        None if effect size is not calculated
    """

    standardized_effect_size_result = '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>'

    if sample:
        pdf_result = pd.DataFrame()
        if meas_level in ['int', 'unk']:
            if len(groups) == 1:
                group_levels = sorted(set(pdf[groups + [dependent_var_name[0]]][groups[0]]))
                if len(group_levels) == 2:
                    groups, grouped_data = _split_into_groups(pdf, dependent_var_name[0], groups)
                    pdf_result.loc[_("Cohen's d"), _('Value')] = \
                        pingouin.compute_effsize(grouped_data[0], grouped_data[1], paired=False, eftype='cohen')
                    pdf_result.loc[_("Eta-squared"), _('Value')] = \
                        pingouin.compute_effsize(grouped_data[0], grouped_data[1], paired=False, eftype='eta-square')
                else:
                    standardized_effect_size_result = None
            else:
                standardized_effect_size_result = None
        elif meas_level == 'nom':
            if len(groups) == 1:
                data = pdf[[dependent_var_name[0], groups[0]]]
                try:
                    cramersv = pingouin.chi2_independence(data, dependent_var_name[0], groups[0])[2].loc[0, 'cramer']
                    # TODO this should be faster, when minimum scipy can be 1.7:
                    #cramersv = stats.contingency.association(data.iloc[:, 0:1], method='cramer')) # new in scipy 1.7
                    pdf_result.loc[_("Cramér's V measure of association"), _('Value')] = \
                        '&phi;<i><sub>c</sub></i> = %.3f' % cramersv
                except ZeroDivisionError:  # TODO could this be avoided?
                    pdf_result.loc[_("Cramér's V measure of association"), _('Value')] = \
                        'cannot be computed (division by zero)'
            else:
                standardized_effect_size_result = None
        else:  # Ordinal variable
            standardized_effect_size_result = None

    else:  # population estimations
        pdf_result = pd.DataFrame(columns=[_('Point estimation'), _('95% CI (low)'), _('95% CI (high)')])
        if meas_level in ['int', 'unk']:
            if len(groups) == 1:
                group_levels = sorted(set(pdf[groups + [dependent_var_name[0]]][groups[0]]))
                if len(group_levels) == 2:
                    groups, grouped_data = _split_into_groups(pdf, dependent_var_name[0], groups)
                    hedges = pingouin.compute_effsize(grouped_data[0], grouped_data[1], paired=False, eftype='hedges')
                    hedges_ci = pingouin.compute_esci(stat=hedges, nx=len(grouped_data[0]), ny=len(grouped_data[0]),
                                                      paired=False, eftype='cohen', confidence=0.95, decimals=3)
                    pdf_result.loc[_("Hedges' g")] = hedges, *hedges_ci
                else:  # more than 2 groups
                    pdf_result = pd.DataFrame()
                    from statsmodels.formula.api import ols
                    from statsmodels.stats.anova import anova_lm
                    # FIXME https://github.com/cogstat/cogstat/issues/136
                    anova_model = ols(str('Q("%s") ~ C(Q("%s"))' % (dependent_var_name[0], groups[0])), data=pdf).fit()
                    # Type I is run, and we want to run type III, but for a one-way ANOVA different types give the
                    # same results
                    anova_result = anova_lm(anova_model)
                    # http://en.wikipedia.org/wiki/Effect_size#Omega-squared.2C_.CF.892
                    omega2 = (anova_result['sum_sq'][0] - (anova_result['df'][0] * anova_result['mean_sq'][1])) / \
                             ((anova_result['sum_sq'][0]+anova_result['sum_sq'][1]) + anova_result['mean_sq'][1])
                    pdf_result.loc[_('Omega-squared'), _('Value')] = '&omega;<sup>2</sup> = %0.3g' % omega2
            else:  # More than 1 grouping variables
                standardized_effect_size_result = None
        else:  # Ordinal or nominal
            standardized_effect_size_result = None

    if not pdf_result.empty:
        standardized_effect_size_result += _format_html_table(pdf_result.to_html(bold_rows=False, escape=False,
                                                                                 float_format=lambda x: '%0.3f' % (x),
                                                                                 classes="table_cs_pd"))

    return standardized_effect_size_result

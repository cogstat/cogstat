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

try:
    from statsmodels.graphics.mosaicplot import mosaic
except:
    pass
from statsmodels.stats.weightstats import DescrStatsW
import pandas as pd

'''
# r is not needed for some time, but may be necessary at some later point again, so keep the code
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr  # requires at least rpy 2.1.0; use instead:
    #from rpy2.robjects import r # http://stackoverflow.com/questions/2128806/python-rpy2-cant-import-a-bunch-of-packages
except:
    pass
'''

run_power_analysis = False  # should this analysis included?
# - Sensitivity power analysis for one - sample t - test, two - sample t-test, paired samples t-test, Chi-square test, one-way ANOVA

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext


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
    result += '%s\n' % print_pivot_page(pdf, page_names)
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
                              axis=1).apply(lambda x: cs_stat_num.diffusion_get_ez_params(*x),
                                            axis=1, result_type='expand')
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


def display_variable_raw_data(pdf, var_name):
    """Display n of valid cases
    """
    data = pdf[var_name].dropna()
    text_result = _('N of valid cases: %g') % len(data) + '\n'
    missing_cases = len(pdf[var_name])-len(data)
    text_result += _('N of missing cases: %g') % missing_cases + '\n'
    return text_result

def frequencies(pdf, var_name, meas_level):
    """Frequencies
    
    arguments:
    var_name (str): name of the variable
    meas_level: measurement level of the variable
    """

    freq = pd.DataFrame()
    freq[_('Value')] = freq.index
    freq[_('Freq')] = pdf[var_name].value_counts(dropna=False).sort_index()
    freq[_('Value')] = freq.index  # previous assignment gives nans for empty df
    if meas_level != 'nom':
        freq[_('Cum freq')] = freq[_('Freq')].cumsum()
    freq[_('Rel freq')] = pdf[var_name].value_counts(normalize=True, dropna=False).sort_index() * 100
    if meas_level != 'nom':
        freq[_('Cum rel freq')] = freq[_('Rel freq')].cumsum()
    text_result = _format_html_table(freq.to_html(formatters={_('Rel freq'): lambda x: '%.1f%%' % x,
                                                              _('Cum rel freq'): lambda x: '%.1f%%' % x},
                                                  bold_rows=False, index=False, classes="table_cs_pd"))
    return text_result


def proportions_ci(pdf, var_name):
    """Proportions confidence intervals

    arguments:
    var_name (str): name of the variable
    """
    from statsmodels.stats import proportion

    proportions = pdf[var_name].value_counts(normalize=True, dropna=False).sort_index()
    proportions_ci_np = proportion.multinomial_proportions_confint(proportions)
    proportions_ci_pd = pd.DataFrame(proportions_ci_np, index=proportions.index, columns=[_('low'), _('high')]) * 100
    text_result = '%s - %s\n%s' % (_('Relative frequencies'), _('95% confidence interval (multinomial proportions)'),
                                     _format_html_table(proportions_ci_pd.to_html(bold_rows=False,
                                                                                  float_format=lambda x: '%.1f%%' % x,
                                                                                  classes="table_cs_pd")))
    if (pdf[var_name].value_counts(dropna=False) < 5).any():
        text_result += '<warning>' + _('Some of the cells does not include at least 5 cases, so the confidence intervals may be invalid.') + '</warning>\n'
    return text_result


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
    if not groups:
        # drop all data with NaN pair
        data = pdf[var_names].dropna()
        pdf_result = pd.DataFrame(columns=var_names)
        text_result += f"<cs_h3>{_('Descriptives for the variables') if len(var_names) > 1 else _('Descriptives for the variable')}</cs_h3>"
        for var_name in var_names:
            if meas_levs[var_name] != 'nom':
                prec = cs_util.precision(data[var_name])+1
            for stat in statistics:
                pdf_result.loc[stat_names[stat], var_name] = '%0.*f' % \
                                                             (2 if stat == 'variation ratio' else prec,
                                                              stat_functions[stat](data[var_name].dropna()))
    # There is at least one grouping variable
    else:
        # missing groups and values will be dropped

        groups, grouped_data = _split_into_groups(pdf, var_names[0], groups)
        groups = [' : '.join(map(str, group)) for group in groups]
        pdf_result = pd.DataFrame(columns=groups)

        text_result += f"<cs_h3>{_('Descriptives for the groups')}</cs_h3>"
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

    :param data:
    :param statistics:
    :return:
    """
    pdf_result = pd.DataFrame()
    population_param_text = ''
    for statistic in statistics:
        if statistic == 'mean':
            population_param_text += _('Present confidence interval values for the mean suppose normality.') + '\n'
            pdf_result.loc[_('Mean'), _('Point estimation')] = np.mean(data.dropna())
            pdf_result.loc[_('Mean'), _('95% confidence interval (high)')], \
            pdf_result.loc[_('Mean'), _('95% confidence interval (low)')] = DescrStatsW(data).tconfint_mean()
        if statistic == "std":
            pdf_result.loc[_('Standard deviation')] = [np.std(data.dropna(), ddof=1), '', '']
        if statistic == 'median':
            pdf_result.loc[_('Median'), _('Point estimation')] = np.median(data.dropna())
            pdf_result.loc[_('Median'), _('95% confidence interval (low)')], \
            pdf_result.loc[_('Median'), _('95% confidence interval (high)')] = \
                cs_stat_num.median_ci(pd.DataFrame(data.dropna()))
    pdf_result = pdf_result.fillna(_('Out of the data range'))
    prec = cs_util.precision(data) + 1
    population_param_text += _format_html_table(pdf_result.to_html(bold_rows=False,
                                                                  classes="table_cs_pd",
                                                                  float_format=lambda x: '%0.*f' % (prec, x)))
    return population_param_text


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


def variable_pair_standard_effect_size(data, meas_lev, sample=True):
    """
    (Some stats are also calculated elsewhere, making the analysis slower, but separation is a priority.)

    :param data:
    :param meas_lev:
    :param sample: True for sample descriptives, False for population estimations
    :return:
    """
    pdf_result = pd.DataFrame()
    standardized_effect_size_result = _('Standardized effect size')
    if sample:
        if meas_lev in ['int', 'unk']:
            pdf_result.loc[_("Pearson's correlation"), _('Value')] = \
                '<i>r</i> = %0.3f' % stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])[0]
            pdf_result.loc[_("Spearman's rank-order correlation"), _('Value')] = \
                '<i>r<sub>s</sub></i> = %0.3f' % stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])[0]
        elif meas_lev == 'ord':
            pdf_result.loc[_("Spearman's rank-order correlation"), _('Value')] = \
                '<i>r<sub>s</sub></i> = %0.3f' % stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])[0]
        elif meas_lev == 'nom':
            cont_table_data = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
            chi2, p, dof, expected = stats.chi2_contingency(cont_table_data.values)
            try:
                cramersv = (chi2 / (cont_table_data.values.sum() * (min(cont_table_data.shape) - 1))) ** 0.5
                pdf_result.loc[_("Cramér's V measure of association"), _('Value')] = \
                    '&phi;<i><sub>c</sub></i> = %.3f' % cramersv
            except ZeroDivisionError:  # TODO could this be avoided?
                pdf_result.loc[_("Cramér's V measure of association"), _('Value')] = \
                    'cannot be computed (division by zero)'
    else:  # population estimations
        pdf_result = pd.DataFrame(columns=[_('Point estimation'), _('95% confidence interval')])
        if meas_lev in ['int', 'unk']:
            df = len(data) - 2
            r, p = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])
            r_ci_low, r_ci_high = cs_stat_num.corr_ci(r, df + 2)
            pdf_result.loc[_("Pearson's correlation") + ', <i>r</i>'] = \
                ['%0.3f' % (r), '[%0.3f, %0.3f]' % (r_ci_low, r_ci_high)]
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
                                                                                 classes="table_cs_pd"))
    return standardized_effect_size_result


def contingency_table(data_frame, x, y, count=False, percent=False, ci=False, margins=False):
    """ Create contingency tables. Use for nominal data.
    It works for any number of x and y variables.
    :param data_frame:
    :param x: list of variable names (for columns)
    :param y: list of variable names (for rows)
    :param count:
    :param percent:
    :param ci: multinomial, goodman method
    :param margins:option for count and percent tables,  ci calculation ignores it

    :return:
    """
    text_result=''
    if count:
        cont_table_count = pd.crosstab(index=[data_frame[ddd] for ddd in data_frame[y]], columns=[data_frame[ddd] for ddd in data_frame[x]], margins=margins, margins_name=_('Total'))
        text_result += '\n%s - %s\n%s\n' % (_('Contingency table'), _('Case count'),
                                            _format_html_table(cont_table_count.to_html(bold_rows=False,
                                                                                        classes="table_cs_pd")))
    if percent:
        # for the pd.crosstab() function normalize=True parameter does not work with mutliple column variables (neither pandas version 0.22 nor 1.0.3 works, though with different error messages), so make a workaround
        cont_table_count = pd.crosstab([data_frame[ddd] for ddd in data_frame[y]], [data_frame[ddd] for ddd in data_frame[x]], margins=margins, margins_name=_('Total'))
        # normalize by the total count
        cont_table_perc = cont_table_count / cont_table_count.iloc[-1, -1] * 100
        text_result += '\n%s - %s\n%s\n' % (_('Contingency table'), _('Percentage'),
                                            _format_html_table(cont_table_perc.to_html(bold_rows=False,
                                                                                       classes="table_cs_pd",
                                                                                       float_format=lambda x: '%.1f%%' % x)))
    if ci:
        from statsmodels.stats import proportion
        cont_table_count = pd.crosstab([data_frame[ddd] for ddd in data_frame[y]], [data_frame[ddd] for ddd in data_frame[x]])  # don't use margins
        cont_table_ci_np = proportion.multinomial_proportions_confint(cont_table_count.unstack())
        # add index and column names for the numpy results, and reformat (unstack) to the original arrangement
        cont_table_ci = pd.DataFrame(cont_table_ci_np, index=cont_table_count.unstack().index, columns=[_('low'), _('high')]).stack().unstack(level=list(range(len(x))) + [len(x)+1]) * 100
        text_result += '%s - %s\n%s\n' % (_('Contingency table'), _('95% confidence interval (multinomial proportions)'),
                                            _format_html_table(cont_table_ci.to_html(bold_rows=False,
                                                                                     classes="table_cs_pd",
                                                                                     float_format=lambda x: '%.1f%%' % x)))
        if (cont_table_count < 5).values.any(axis=None):  # df.any(axis=None) doesn't work for some reason, so we use the np version
            text_result += '<warning>' + _('Some of the cells does not include at least 5 cases, so the confidence intervals may be invalid.') + '</warning>'


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
        cis, cils, cihs = confidence_interval_t(data, ci_only=False)
        condition_estimations_pdf[_('95% CI (low)')] = cils
        condition_estimations_pdf[_('95% CI (high)')] = cihs
    if meas_level == 'ord':
        condition_estimations_pdf[_('Point estimation')] = data.median()
        cis_np = cs_stat_num.median_ci(data)
        condition_estimations_pdf[_('95% CI (low)')], condition_estimations_pdf[_('95% CI (high)')] = cis_np
        condition_estimations_pdf = condition_estimations_pdf.fillna(_('Out of the data range'))
    return condition_estimations_pdf


### Compare groups ###


def comp_group_graph_cum(data_frame, meas_level, var_names, groups, group_levels):
    pass
def comp_group_estimations(data_frame, meas_level, var_names, groups):
    """Draw means with CI for int vars, and medians for ord vars.
    """
    group_estimations_pdf = pd.DataFrame()
    if meas_level in ['int', 'unk']:
        pdf = data_frame.dropna(subset=[var_names[0]])[[var_names[0]] + groups]
        means = pdf.groupby(groups, sort=True).aggregate(np.mean)[var_names[0]]
        cis = pdf.groupby(groups, sort=True).aggregate(confidence_interval_t)[var_names[0]]
        group_estimations_pdf[_('Point estimation')] = means
        # APA format, but cannot be used the numbers if copied to spreadsheet
        #group_means_pdf[_('95% confidence interval')] = '['+ (means-cis).map(str) + ', ' + (means+cis).map(str) + ']'
        group_estimations_pdf[_('95% CI (low)')] = means - cis
        group_estimations_pdf[_('95% CI (high)')] = means + cis
    elif meas_level == 'ord':
        pdf = data_frame.dropna(subset=[var_names[0]])[[var_names[0]] + groups]
        group_estimations_pdf[_('Point estimation')] = pdf.groupby(groups, sort=True).aggregate(np.median)[var_names[0]]
        cis = pdf.groupby(groups, group_keys=False, sort=True).apply(lambda x: cs_stat_num.median_ci(x)[:, 0])
            # TODO this solution works, but a more reasonable code would be nicer
        # APA format, but cannot be used the numbers if copied to spreadsheet
        #group_means_pdf[_('95% confidence interval')] = '['+ (means-cis).map(str) + ', ' + (means+cis).map(str) + ']'
        group_estimations_pdf[_('95% CI (low)')] = np.concatenate(cis.values).reshape((-1, 2))[:, 0]
        group_estimations_pdf[_('95% CI (high)')] = np.concatenate(cis.values).reshape((-1, 2))[:, 1]
            # TODO this solution works, but a more reasonable code would be nicer
        group_estimations_pdf = group_estimations_pdf.fillna(_('Out of the data range'))
    return group_estimations_pdf


def compare_groups_effect_size(pdf, dependent_var_name, groups, meas_level, sample=True):

    pdf_result = pd.DataFrame()
    standardized_effect_size_result = _('Standardized effect size')

    if sample:
        if meas_level == 'nom':
            if len(groups) == 1:
                data = pdf[[dependent_var_name[0], groups[0]]]
                cont_table_data = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
                chi2, p, dof, expected = stats.chi2_contingency(cont_table_data.values)
                try:
                    cramersv = (chi2 / (cont_table_data.values.sum() * (min(cont_table_data.shape) - 1))) ** 0.5
                    pdf_result.loc[_("Cramér's V measure of association"), _('Value')] = \
                        '&phi;<i><sub>c</sub></i> = %.3f' % cramersv
                except ZeroDivisionError:  # TODO could this be avoided?
                    pdf_result.loc[_("Cramér's V measure of association"), _('Value')] = \
                        'cannot be computed (division by zero)'

    else:  # population estimations
        if meas_level in ['int', 'unk']:
            if len(groups) == 1:
                from statsmodels.formula.api import ols
                from statsmodels.stats.anova import anova_lm
                # FIXME If there is a variable called 'C', then patsy is confused whether C is the variable or the categorical variable
                # http://gotoanswer.stanford.edu/?q=Statsmodels+Categorical+Data+from+Formula+%28using+pandas%
                # http://stackoverflow.com/questions/22545242/statsmodels-categorical-data-from-formula-using-pandas
                # http://stackoverflow.com/questions/26214409/ipython-notebook-and-patsy-categorical-variable-formula
                data = pdf.dropna(subset=[dependent_var_name[0], groups[0]])
                anova_model = ols(str(dependent_var_name[0]+' ~ C('+groups[0]+')'), data=data).fit()
                # Type I is run, and we want to run type III, but for a one-way ANOVA different types give the same results
                anova_result = anova_lm(anova_model)
                # http://en.wikipedia.org/wiki/Effect_size#Omega-squared.2C_.CF.892
                omega2 = (anova_result['sum_sq'][0] - (anova_result['df'][0] * anova_result['mean_sq'][1]))/\
                         ((anova_result['sum_sq'][0]+anova_result['sum_sq'][1]) +anova_result['mean_sq'][1])
                pdf_result.loc[_('Omega squared'), _('Value')] = '&omega;<sup>2</sup> = %0.3g' % omega2
    standardized_effect_size_result += _format_html_table(pdf_result.to_html(bold_rows=False, escape=False,
                                                                             classes="table_cs_pd"))

    return standardized_effect_size_result


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

    # Sensitivity power analysis
    if run_power_analysis:
        from statsmodels.stats.power import FTestAnovaPower
        power_analysis = FTestAnovaPower()
        text_result += _(
            'Sensitivity power analysis. Minimal effect size to reach 95%% power (effect size is in %s):') % _('f') + \
                       ' %0.2f\n' % power_analysis.solve_power(effect_size=None, nobs=len(data), alpha=0.05, power=0.95,
                                                               k_groups=len(set(data[grouping_name])))

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

    # http://statsmodels.sourceforge.net/stable/stats.html#multiple-tests-and-multiple-comparison-procedures
    if anova_result['PR(>F)'][0] < 0.05:  # post-hoc
        post_hoc_res = sm.stats.multicomp.pairwise_tukeyhsd(np.array(data[var_name]), np.array(data[grouping_name]),
                                                            alpha=0.05)
        text_result += '\n'+_('Groups differ. Post-hoc test of the means.')+'\n'
        text_result += ('<fix_width_font>%s\n</fix_width_font>' % post_hoc_res).replace(' ', '\u00a0')
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


def chi_square_test(pdf, var_name, grouping_name):
    """Chi-Square test
    Cramer's V: http://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    
    Arguments:
    var_name (str):
    grouping_name (str):
    """
    text_result = ''
    cont_table_data = pd.crosstab(pdf[grouping_name], pdf[var_name])
    chi2, p, dof, expected = stats.chi2_contingency(cont_table_data.values)
    try:
        cramersv = (chi2 / (cont_table_data.values.sum()*(min(cont_table_data.shape)-1)))**0.5
        cramer_result = _('Cramér\'s V measure of association: ')+'&phi;<i><sub>c</sub></i> = %.3f\n' % cramersv
    except ZeroDivisionError:  # TODO could this be avoided?
        cramer_result = _('Cramér\'s V measure of association cannot be computed (division by zero).')
    chi_result = ''

    # Sensitivity power analysis
    if run_power_analysis:
        from statsmodels.stats.power import GofChisquarePower
        power_analysis = GofChisquarePower()
        chi_result = _('Sensitivity power analysis. Minimal effect size to reach 95%% power (effect size is in %s):') % _(
            'w') + ' %0.2f\n' % power_analysis.solve_power(effect_size=None, nobs=cont_table_data.values.sum(), alpha=0.05,
                                                           power=0.95, n_bins=cont_table_data.size)

    chi_result += _("Result of the Pearson's Chi-square test: ") + \
                  '</i>&chi;<sup>2</sup></i>(%g, <i>N</i> = %d) = %.3f, %s' % \
                  (dof, cont_table_data.values.sum(), chi2, cs_util.print_p(p))
    return cramer_result, chi_result

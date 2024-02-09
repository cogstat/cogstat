# -*- coding: utf-8 -*-

"""
This module contains functions for hypothesis tests and for related power analyses.

Arguments are the pandas data frame (pdf), and parameters (among others they
are usually variable names).
Output is text (html and some custom notations).

Mostly scipy.stats, statsmodels, and pingouin are used to generate the results.
"""

import gettext
import os
import re

import numpy as np
import pandas as pd
import scikit_posthocs
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.contingency_tables import cochrans_q
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.weightstats import DescrStatsW
import pingouin

from . import cogstat_config as csc
from . import cogstat_stat_num as cs_stat_num
from . import cogstat_stat as cs_stat

if csc.versions['r']:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext

warn_unknown_variable = '<cs_warning>'+_('The properties of the variables are not set. Set them in your data source.') \
                        + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                        % 'https://doc.cogstat.org/Handling-data' \
                        + '\n</cs_warning>'  # TODO maybe this shouldn't be repeated, it's enough to show it at import

non_data_dim_precision = 2


def print_p(p, style='apa'):
    """
    Make an output according to the appropriate rules.

    Currently, APA rule is supported:

    - if p < 0.001, then print 'p < .001'
    - otherwise 'p = value' with 3 decimal places precision
    - leading zero is not displayed

    Parameters
    ----------
    p: float
        The p value to be displayed
    style: {'apa'}
        The format in which the value should be displayed

    Returns
    -------
    str
        p value in appropriate format
    """
    if style == 'apa':
        if p < 0.001:
            return '<i>p</i> &lt; .001'
        else:
            return '<i>p</i> = ' + ('%0.3f' % p).lstrip('0')


def print_sensitivity_effect_sizes(effect_sizes_95=None, effect_sizes_80=None):
    """
    Print the effect sizes of the sensitivity power analysis results. Power can be 95% and 80%. Multiple effect sizes
    can be provided in a dictionary.

    Parameters
    ----------
    effect_sizes_95: dict
        effect sizes for 95% power
        keys are the effect size names, values are the effect sizes
    effect_sizes_80: dict
        effect sizes for 80% power

    Returns
    -------
    str
        String to be added to the output
    """
    if effect_sizes_95 is None:
        effect_sizes_95 = {}
    if effect_sizes_80 is None:
        effect_sizes_80 = {}

    effect_sizes_95_str = ''
    for effect_size_name in effect_sizes_95:
        if effect_sizes_95[effect_size_name]:
            effect_sizes_95_str += _('Minimal effect size in %s') % effect_size_name + ': %0.2f. ' % \
                                   effect_sizes_95[effect_size_name]

    effect_sizes_80_str = ''
    for effect_size_name in effect_sizes_80:
        if effect_sizes_80[effect_size_name]:
            effect_sizes_80_str += _('Minimal effect size in %s') % effect_size_name + ': %0.2f. ' % \
                                   effect_sizes_80[effect_size_name]

    if effect_sizes_95_str or effect_sizes_80_str:
        text_result = _('Sensitivity power analysis.') + '\n'
        if effect_sizes_95_str:
            text_result += _('Minimal effect size to reach 95% power with the present '
                           'sample size for the present hypothesis test.') + ' ' + effect_sizes_95_str + '\n'
        if effect_sizes_80_str:
            text_result += _('Minimal effect size to reach 80% power with the present '
                             'sample size for the present hypothesis test.') + ' ' + effect_sizes_80_str + '\n'
    else:
        text_result = _('Sensitivity power could not be calculated.') + '\n'
    return text_result


### Single variables ###


def normality_test(pdf, data_measlevs, var_name, group_name='', group_value=''):
    """Check normality.

    Parameters
    ----------
    pdf : pandas dataframe
    data_measlevs
    var_name : str
        Name of the variable to be checked.
    group_name : str
        Name of the grouping variable if part of var_name should be checked. Otherwise ''.
    group_value : str
        Name of the group in group_name, if grouping is used.

    Returns
    -------
    bool
        Is the variable normal (False if normality is violated)
    str
    html, APA format, text result
    matplotlib image
        histogram with normal distribution
    matplotlib image
        QQ plot
    """
    text_result = ''
    if group_name:
        data = pdf[pdf[group_name] == group_value][var_name].dropna()
    else:
        data = pdf[var_name]

    if data_measlevs[var_name] in ['nom', 'ord']:
        return False, '<cs_decision>' + _('Normality can be checked only for interval variables.') + '\n</cs_decision>'
    if len(set(data)) == 1:
        return False, _('Normality cannot be checked for constant variable in %s%s.') % \
               (var_name, ' (%s: %s)' % (group_name, group_value) if group_name else '') + '\n'
    # TODO do we need this?
    #        if len(data)<7:
    #            return False, _(u'Sample size must be greater than 7 for normality test.\n'), None, None

    # http://statsmodels.sourceforge.net/stable/generated/statsmodels.stats.diagnostic.kstest_normal.html#
    # statsmodels.stats.diagnostic.kstest_normal
    # text_result += _('Testing normality with the Kolmogorov–Smirnov test:')+': <i>D</i> = %0.3g, <i>p</i> =
    #                %0.3f \n' %sm.stats.kstest_normal(data)
    # text_result += _('Testing normality with the Lillifors test')+': <i>D</i> = %0.3g, <i>p</i> =
    #                %0.3f \n' %sm.stats.lillifors(data)
    # A, p = sm.stats.normal_ad(data)
    # text_result += _('Anderson–Darling normality test in variable %s%s') %(var_name, ' (%s: %s)' %
    #               (group_name, group_value) if group_name else '') + ': <i>A<sup>2</sup></i> =
    #               %0.3g, %s\n' %(A, print_p(p))
    # text_result += _('Testing normality with the Anderson–Darling test: <i>A<sup>2</sup></i> = %0.3g,
    #                critical values: %r, sig_levels: %r \n') %stats.anderson(data, dist='norm')
    # text_result += _("Testing normality with the D'Agostin and Pearson method")+': <i>k2</i> = %0.3g, <i>p</i> =
    #                %0.3f \n' %stats.normaltest(data)
    # text_result += _('Testing normality with the Kolmogorov–Smirnov test')+': <i>D</i> = %0.3g, <i>p</i> = %0.3f \n' %
    #               stats.kstest(data, 'norm')
    if len(data) < 3:
        # translators: the first %s includes the name of the variable, the second %s includes the optional grouping variable and level names in parentheses
        return False, _('Too small sample to test normality in variable %s%s.\n' %
                        (var_name, ' (%s: %s)' % (group_name, group_value) if group_name else ''))
    else:
        w, p = stats.shapiro(data)
        text_result += _('Shapiro–Wilk normality test in variable %s%s') % \
                       (var_name, ' (%s: %s)' % (group_name, group_value) if group_name else '') + \
                       ': <i>W</i> = %0.*f, %s\n' % (non_data_dim_precision, w, print_p(p))

    # Decide about normality
    norm = False if p < 0.05 else True

    return norm, text_result


def one_t_test(pdf, data_measlevs, var_name, test_value=0):
    """Calculate one sample t-test.

    Parameters
    ----------
    pdf : pandas dataframe
        It is sufficient to include only the relevant variable. It is assumed that nans are dropped.
    var_name : str
        Name of the variable to test.
    test_value : numeric
        Test against this value.

    Returns
    -------
    str
        Result in APA format.
    matplotlib chart
        Bar chart with mean and confidence interval.
    """
    text_result = ''
    data = pdf[var_name]
    if data_measlevs[var_name] in ['int', 'unk']:
        if data_measlevs[var_name] == 'unk':
            text_result += warn_unknown_variable
        if len(set(data)) == 1:
            return _('One sample t-test cannot be run for constant variable') + '.\n', None

        descr = DescrStatsW(data)
        t, p, df = descr.ttest_mean(float(test_value))
        # Or we could use confidence_interval_t
        cil, cih = descr.tconfint_mean()
        ci = (cih - cil) / 2
        # prec = cs_util.precision(data) + 1
        # ci_text = '[%0.*f, %0.*f]' %(prec, cil, prec, cih)
        text_result = ''

        # Sensitivity power analysis
        # statsmodels may fail, see its API documentation
        # d: (mean divided by the standard deviation)
        try:
            from statsmodels.stats.power import TTestPower
            power_analysis = TTestPower()
            effect_size_95 = {_('d'): power_analysis.solve_power(effect_size=None, nobs=len(data), alpha=0.05,
                                                                 power=0.95, alternative='two-sided')}
            effect_size_80 = {_('d'): power_analysis.solve_power(effect_size=None, nobs=len(data), alpha=0.05,
                                                                 power=0.80, alternative='two-sided')}
        except ValueError:
            effect_size_95 = {_('d'): None}
            effect_size_80 = {_('d'): None}
        text_result += print_sensitivity_effect_sizes(effect_size_95, effect_size_80)

        text_result += _('One sample t-test against %g') % \
                       float(test_value) + ': <i>t</i>(%d) = %0.*f, %s\n' % (df, non_data_dim_precision, t, print_p(p))
        # Bayesian t-test
        bf10 = pingouin.bayesfactor_ttest(t, len(data), paired=True)
        text_result += _('Result of the Bayesian one sample t-test') + \
                       ': BF<sub>10</sub> = %0.*f, BF<sub>01</sub> = %0.*f\n' % \
                       (non_data_dim_precision, bf10, non_data_dim_precision, 1/bf10)
    else:
        text_result += _('One sample t-test is computed only for interval variables.')
    return text_result, ci


def wilcox_sign_test(pdf, data_measlevs, var_name, value=0):
    """Calculate Wilcoxon signed-rank test.

    Parameters
    ---------
    pdf : pandas dataframe

    var_name : str

    value : numeric

    Returns
    -------

    """

    text_result = ''
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
            text_result += _('Result of Wilcoxon signed-rank test')+': <i>W</i> = %0.3g, %s\n' % (v, print_p(p))
        '''
        T, p = stats.wilcoxon(np.array(pdf[var_name] - float(value)), correction=True)
        # we need to convert the pandas dataframe to numpy array because pdf cannot be always handled
        # correction=True in order to work like the R wilcox.test
        text_result += _('Result of Wilcoxon signed-rank test') + \
                       ': <i>T</i> = %0.*f, %s\n' % (non_data_dim_precision, T, print_p(p))
    else:
        text_result += _('Wilcoxon signed-rank test is computed only for interval or ordinal variables.')
    return text_result


### Variable pair ###

def homoscedasticity(data, predictors, predicted):
    """Check homoscedasticity

    Parameters
    ----------
    data : pandas dataframe
    predictors : list of str
        Names of the predictor variables.
    predicted : str
        Name of the predicted or dependent variable.

    Returns
    -------
    bool or None
        False if data is heteroscedastic, None if homoscedasticity cannot be caluclated.
    html text
        Output in APA format.
    """

    text_result = ''
    if len(data) < 3:
        return None, _('Too small sample to test homoscedasticity in variables: %s') % ', '.join(predictors + [predicted]) + '.\n'
    else:
        x = sm.add_constant(data[predictors])
        y = data[predicted]

        # With one predictor x is sorted in the main analysis which produces erroneous results in the tests below,
        # so here we run the regression model again with unsorted x.
        residual_unsorted = sm.regression.linear_model.OLS(y, x).fit().resid

        koenker = sm.stats.diagnostic.het_breuschpagan(residual_unsorted, x, robust=True)
        lm_koenker = koenker[0]
        p_koenker = koenker[1]

        try:
            white = sm.stats.diagnostic.het_white(residual_unsorted, x)
            lm_white = white[0]
            p_white = white[1]
        except AssertionError:
            lm_white = None
            p_white = None

        homoscedasticity = False if p_koenker < 0.05 or p_white < 0.05 else True

        text_result += _("Koenker's studentized score test") \
                       + ": <i>LM</i> = %0.*f, %s\n" % (non_data_dim_precision, lm_koenker, print_p(p_koenker))
        if lm_white:
            text_result += _("White's test") \
                           + ': <i>LM</i> = %0.*f, %s\n' % (non_data_dim_precision, lm_white, print_p(p_white))
        else:
            text_result += _("White's test could not be run") + '.'

        return homoscedasticity, text_result


def multivariate_normality(data, var_names):
    """Henze-Zirkler test of multivariate normality.

        Parameters
        ----------
        data : pandas dataframe
            It is sufficient to include only the relevant variables. It is assumed that nans are dropped.
        var_names : list of str
            Names of the variables to test.

        Returns
        -------
        bool or None
            True if normality is true. None if normality cannot be calculated.
        html text
            Output in APA format.

        """

    text_result = ''

    if len(data) < 3:
        return None, _('Too small sample to test normality in variables: %s') % ', '.join(var_names) + '.\n'
    if len(data) > 1000:
        # this is a limitation of the pingouin module; calculation would require too much memory
        # https://github.com/raphaelvallat/pingouin/issues/121
        return None, _('Currently, testing normality in variables (%s) is not possible with more than 1000 cases') % \
                     ', '.join(var_names) + '.\n'
    else:
        hz, p, sig = pingouin.multivariate_normality(data, alpha=.05)
        text_result += _('Henze-Zirkler test of multivariate normality in variables %s') % ', '.join(var_names) + \
                       ': <i>W</i> = %0.*f, %s\n' % (non_data_dim_precision, hz, print_p(p))

    return sig, text_result


def variable_pair_hyp_test(data, x, y, meas_lev, normality=None, homoscedasticity=None):
    """
    Run relevant hypothesis tests.

    Parameters
    ----------
    data : pandas dataframe
        the dataframe that includes the relevant two variables
    x : str
        name of the x variable in data
    y : str
        name of the y variable in data
    meas_lev : {'int', 'ord', 'nom'}
        lowest measurement level of the data
    normality: bool or None
        True when variables follow a multivariate normal distribution, False otherwise. None if normality couldn't be
        calculated or if the parameter was not specified.
    homoscedasticity: bool or None
        True when homoscedasticity is true, False otherwise. None if homoscedasticity could not be calculated or if
        the parameter was not specified.

    Returns
    -------
    str
        Hypothesis test results in html format
    """
    if meas_lev == 'int':
        population_result = '<cs_decision>' + _('Testing if correlation differs from 0') + '.</cs_decision>\n'
        df = len(data) - 2

        if normality and homoscedasticity:
            population_result += '<cs_decision>'+_('Interval variables') + '. ' + _('Normality not violated') + \
                                 '. ' + _('Homoscedasticity not violated') + '. >> ' + \
                                 _("Running Pearson's and Spearman's correlation") + '.\n</cs_decision>'

            r, p = stats.pearsonr(data[x], data[y])
            population_result += _("Pearson's correlation") + \
                                 ': <i>r</i>(%d) = %0.*f, %s\n' % \
                                 (df, non_data_dim_precision, r, print_p(p))
            # Bayesian test
            bf10 = pingouin.bayesfactor_pearson(r, len(data))
            population_result += _('Bayes Factor for Pearson correlation') + \
                           ': BF<sub>10</sub> = %0.*f, BF<sub>01</sub> = %0.*f\n' % \
                           (non_data_dim_precision, bf10, non_data_dim_precision, 1/bf10)

            r, p = stats.spearmanr(data[x], data[y])
            population_result += _("Spearman's rank-order correlation") + \
                                 ': <i>r<sub>s</sub></i>(%d) = %0.*f, %s' % \
                                 (df, non_data_dim_precision, r, print_p(p))

        elif normality is None or homoscedasticity is None:
            population_result += '<cs_decision>'+_('Interval variables') + '. ' \
                                 + _('Assumptions of hypothesis tests could not be tested') + '. ' + \
                                 _('Hypothesis tests may be inaccurate') + '. >> ' + \
                                 _("Running Pearson's and Spearman's correlation") + '.\n</cs_decision>'

            r, p = stats.pearsonr(data[x], data[y])
            population_result += _("Pearson's correlation") + \
                                 ': <i>r</i>(%d) = %0.*f, %s\n' % \
                                 (df, non_data_dim_precision, r, print_p(p))
            # Bayesian test
            bf10 = pingouin.bayesfactor_pearson(r, len(data))
            population_result += _('Bayes Factor for Pearson correlation') + \
                           ': BF<sub>10</sub> = %0.*f, BF<sub>01</sub> = %0.*f\n' % \
                           (non_data_dim_precision, bf10, non_data_dim_precision, 1/bf10)

            r, p = stats.spearmanr(data[x], data[y])
            population_result += _("Spearman's rank-order correlation") + \
                                 ': <i>r<sub>s</sub></i>(%d) = %0.*f, %s' % \
                                 (df, non_data_dim_precision, r, print_p(p))

        else:
            violations = ''

            if not normality:
                violations += _('Normality violated') + '. '
            if not homoscedasticity:
                violations += _('Homoscedasticity violated') + '. '

            population_result += '<cs_decision>'+_('Interval variables') + '. ' + _(violations) + ' >> ' + \
                                 _("Running Spearman's correlation") + '.\n</cs_decision>'

            r, p = stats.spearmanr(data[x], data[y])
            population_result += _("Spearman's rank-order correlation") + \
                                 ': <i>r<sub>s</sub></i>(%d) = %0.*f, %s' % \
                                 (df, non_data_dim_precision, r, print_p(p))

    elif meas_lev == 'ord':
        population_result = '<cs_decision>' + _('Testing if correlation differs from 0') + '.</cs_decision>\n'
        population_result += '<cs_decision>'+_('Ordinal variables.')+' >> '+_("Running Spearman's correlation.") + \
                             '\n</cs_decision>'
        df = len(data) - 2
        r, p = stats.spearmanr(data[x], data[y])
        population_result += _("Spearman's rank-order correlation") + \
                             ': <i>r<sub>s</sub></i>(%d) = %0.*f, %s' % \
                             (df, non_data_dim_precision, r, print_p(p))
    elif meas_lev == 'nom':
        population_result = '<cs_decision>' + _('Testing if variables are independent') + '.</cs_decision>\n'
        # TODO enable the following warning
        #if not(self.data_measlevs[x] == 'nom' and self.data_measlevs[y] == 'nom'):
        #    population_result += '<cs_warning>' + _('Not all variables are nominal. Consider comparing groups.') + \
        #                         '</cs_warning>\n'
        population_result += '<cs_decision>' + _('Nominal variables.') + ' >> ' + _('Running Cramér\'s V.') + \
                             '\n</cs_decision>'
        population_result += chi_squared_test(data, x, y)
    return population_result


def multiple_regression_hyp_tests(data, result, predictors, normality, homoscedasticity, multicollinearity):
    """Hypothesis tests for model and regressor slopes in multiple linear regression.

    Parameters
    ----------
    data : pandas dataframe
    result : statsmodels regression result object
        The result of the multiple regression analysis.
    predictors : list of str
        List of explanatory variable names.
    normality : bool or None
        True when variables follow a multivariate normal distribution, False otherwise. None if normality couldn't be
        calculated or if the parameter was not specified.
    homoscedasticity : bool or None
        True when homoscedasticity is true, False otherwise. None if homoscedasticity could not be calculated or if
        the parameter was not specified.
    multicollinearity : bool or None
        True when multicollinearity is suspected (VIF>10), False otherwise. None if the parameter was not specified.

    Returns
    -------
    str
        html text
    """

    if normality and homoscedasticity and not multicollinearity:
        output = '<cs_decision>' + _('Interval variables') + '. ' + _('More than two variables') + '. ' + \
                 _('Normality met') + '. ' + _('Homoscedasticity met') + '. ' + _('No multicollinearity') + '. >> ' + \
                 _('Running model F-test and tests for regressor slopes') + '.\n</cs_decision>'

    elif normality is None or homoscedasticity is None or multicollinearity is None:
        output = '<cs_decision>' + _('Interval variables') + '. ' + _('More than two variables') + '. ' + \
                 _('Assumptions of hypothesis tests could not be tested') + '. ' + \
                 _('Hypothesis tests may be inaccurate') + '. >> ' \
                 + _('Running model F-test and tests for regressor slopes') + '.\n</cs_decision>'

    else:
        violations = ''

        if not normality:
            violations += _('Normality violated') + '. '
        if not homoscedasticity:
            violations += _('Homoscedasticity violated') + '. '
        if multicollinearity:
            violations += _('Multicollinearity suspected') + '. '

        output = '<cs_decision>' + _('Interval variables') + '. ' + violations + ' >> ' + \
                 _('Hypothesis tests may be inaccurate') + '. ' + \
                 _('Running model F-test and tests for regressor slopes') + '.\n</cs_decision>'

    output += _('Model F-test') + \
                         ': <i>F</i>(%d,%d) = %0.*f, %s' % \
                         (result.df_model, result.df_resid, non_data_dim_precision, result.fvalue,
                          print_p(result.f_pvalue)) + '\n'
    output += _('Regressor slopes') + ':\n'
    for predictor in predictors:
        output += predictor + ': <i>t</i>(%d) = %0.*f, %s' % (len(data) - len(predictors) - 1, non_data_dim_precision,
                                                              result.tvalues[predictor],
                                                              print_p(result.pvalues[predictor])) + '\n'

    # TODO hypothesis tests for partial correlation coefficients.
    # Pingouin doesn't use t-tests and doesn't give test statistics.

    return output
  
  
def ordered_logistic_regression_hyp_tests(data, result, predictors, multicollinearity):
    """Hypothesis tests for model and regressor slopes in ordinal logistic regression with multiple variable.

    Parameters
    ----------
    data : pandas dataframe
    result : statsmodels regression result object
        The result of the multiple regression analysis.
    predictors : list of str
        List of explanatory variable names.
    multicollinearity : bool or None
        True when multicollinearity is suspected (VIF>10), False otherwise. None if the parameter was not specified.

    Returns
    -------
    str
        html text
    """

    if not multicollinearity:
        output = '<decision>' + ('Interval variables. More than two variables.') + ' ' + \
                 ('No multicollinearity.') + ' >> ' + '\n' + \
                 '\n</decision>'

    elif multicollinearity is None:
        output = '<decision>' + ('Interval variables. More than two variables.') + ' ' \
                             + (
            'Assumptions of hypothesis tests could not be tested. Hypothesis tests may be inaccurate.') + ' >> ' \
                             '\n</decision>'

    else:
        violations = ''
        if multicollinearity:
            violations += ('Multicollinearity suspected.') + ' '

        output = '<decision>' + ('Interval variables.') + ' ' + violations + ' >> ' + \
                 ('Hypothesis tests may be inaccurate.') 


    output += ('Regressor slopes:') + '\n'
    for predictor in predictors:
        output += predictor + ': <i>t</i>(%d) = %0.*f, %s' % (len(data) - len(predictors) - 1, non_data_dim_precision,
                                                              result.tvalues[predictor],
                                                              print_p(result.pvalues[predictor])) + '\n'

    # TODO hypothesis tests for partial correlation coefficients.
    #  Pingouin doesn't use t-tests and doesn't give test statistics.

    return output
  
  
def ordered_regression_model_fit(data, predictors, y, result, multicollinearity=None, sample=True):
    
        orderedreg_model_fit = ''


    # Calculate effect sizes for sample or population
        pdf_model_result_fit = pd.DataFrame(columns=[('Loglikelihood') , ('Chi-Square'), ('Degrees of freedom'), ('P value')])  
        
        pdf_model_result_fit.loc[("Model with intercept")] = \
            ['%0.3f' % result.llnull, '','','']
        
        pdf_model_result_fit.loc[('Full model')] = ['%0.3f' % result.llf, '%0.3f' % result.llr , '%0.3f' % result.df_model, '%0.3f' % result.llr_pvalue]
        #orderedreg_model_fit += cs_stat._format_html_table(pdf_model_result_fit.to_html(bold_rows=False, escape=False,
                                                                                       #classes='table_cs_pd')) + '\n'



        orderedreg_model_fit += pdf_model_result_fit.to_html(bold_rows=False, escape=False).replace('\n', '')

        return orderedreg_model_fit
      
      
### Reliability analyses ###

def reliability_interrater_assumptions(data, data_long, var_names, meas_lev):
    """
    Testing assumptions of ICC calculations in interrater relability analysis.

    Parameters
    ----------
    data : pandas dataframe
        Original wide format data.
    data_long : pandas dataframe
        Long format data.
    var_names : list of str
        Names of variables to be analysed.
    meas_lev : list of str
        Measurement levels of variables.

    Returns
    -------
    list of str
        List of variables in which normality is violated.
    str
        HTML text summary of normality test.
    float
        P-value of Levene test.
    str
        HTML text summary of homoscedasticity test.
    """

    non_normal_vars = []
    norm_text = ''
    for var_name in var_names:
        norm, text_result = normality_test(data, data_measlevs=meas_lev, var_name=var_name)
        norm_text += text_result
        if not norm:
            non_normal_vars.append(var_name)
    var_hom_p, var_text_result = levene_test(data_long, var_name='value', group_name='variable')

    return non_normal_vars, norm_text, var_hom_p, var_text_result

def reliability_interrater_hyp_test(hyp_test_table, non_normal_vars, var_hom_p):
    """
    Hypothesis test output for ICC values with warnings in case of violated assumptions. Testing against 0.

    Parameters
    ----------
    hyp_test_table : pandas dataframe
        Three rows with the three ICC tests
        Columns are df1, df2, F, p
    non_normal_vars : list of str
        List of variables where normality has been violated.
    var_hom_p : float
        P-value of Levene test of heteroscedasticity.

    Returns
    -------
    str
        HTML text
    """

    # TODO is it useful to test an intraclass correlation against 0? What test value should be used?
    hypothesis_test_result = '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>'
    hypothesis_test_result += '<cs_decision>' + _('Testing if ICC differs from 0') + '.</cs_decision>' + '\n'

    if not non_normal_vars:
        hypothesis_test_result += '<cs_decision>' + _('Assumption of normality met') + '. </cs_decision>'
    else:
        hypothesis_test_result += '<cs_decision>' + \
                                  _('Assumption of normality violated in variable(s) %s'
                                    % ', ' .join(non_normal_vars)) + '. </cs_decision>'
    if var_hom_p < 0.05:
        hypothesis_test_result += '<cs_decision>' + _('Assumption of homogeneity of variances violated') + \
                                  '. </cs_decision>'
    else:
        hypothesis_test_result += '<cs_decision>' + _('Assumption of homogeneity of variances met') + '. </cs_decision>'

    if (var_hom_p < 0.05) or (len(non_normal_vars) != 0):
        hypothesis_test_result += '<cs_decision>' + _('Hypothesis tests may be inaccurate') + '.</cs_decision>' + '\n'
    else:
        hypothesis_test_result += '\n'

    hypothesis_test_result += '<cs_decision>' + _('Running F-tests') + '.</cs_decision>' + '\n'
    for hyp_test_index, hyp_test_row in hyp_test_table.iterrows():
        hypothesis_test_result += _('F-test for %s') % hyp_test_index + ': <i>F</i>(%d, %d) = %0.*f, %s\n' \
                                  % (hyp_test_row['df1'], hyp_test_row['df2'], non_data_dim_precision,
                                     hyp_test_row['F'], print_p(hyp_test_row['pval']))

    return hypothesis_test_result


### Compare variables ###


def decision_repeated_measures(data, meas_level, factors, var_names, data_measlevs):
    """

    Parameters
    ----------
    data : pandas dataframe
    meas_level
    factors
    var_names
    data_measlevs

    Returns
    -------

    """
    result_ht = '<cs_decision>'
    if meas_level in ['int', 'unk']:
        result_ht += _('Testing if the means are the same.') + '</cs_decision>\n'
    elif meas_level == 'ord':
        result_ht += _('Testing if the medians are the same.') + '</cs_decision>\n'
    elif meas_level == 'nom':
        result_ht += _('Testing if the distributions are the same.') + '</cs_decision>\n'
    if len(factors) == 1:  # one-way comparison
        if len(var_names) < 2:
            result_ht += _('At least two variables required.')
        elif len(var_names) == 2:
            result_ht += '<cs_decision>' + _('Two variables. ') + '</cs_decision>'

            if meas_level == 'int':
                result_ht += '<cs_decision>' + _('Interval variables.') + ' >> ' + _(
                    'Choosing paired t-test or paired Wilcoxon test depending on the assumptions.') + '\n</cs_decision>'

                result_ht += '<cs_decision>' + _('Checking for normality.') + '\n</cs_decision>'
                non_normal_vars = []
                # TODO is this variable name localizable? If not, any other solution to localize it?
                temp_diff_var_name = 'Difference of %s and %s' % tuple(var_names)
                data[temp_diff_var_name] = data[var_names[0]] - data[var_names[1]]
                norm, text_result = normality_test(data, {temp_diff_var_name: 'int'}, temp_diff_var_name)
                result_ht += text_result
                if not norm:
                    non_normal_vars.append(temp_diff_var_name)

                if not non_normal_vars:
                    result_ht += '<cs_decision>' + _(
                        'Normality is not violated. >> Running paired t-test.') + '\n</cs_decision>'
                    result_ht += paired_t_test(data, var_names)
                else:  # TODO should the descriptive be the mean or the median?
                    result_ht += '<cs_decision>' + _('Normality is violated in variable(s): %s.') % ', '. \
                        join(non_normal_vars) + ' >> ' + _('Running paired Wilcoxon test.') + '\n</cs_decision>'
                    result_ht += paired_wilcox_test(data, var_names)
            elif meas_level == 'ord':
                result_ht += '<cs_decision>' + _('Ordinal variables.') + ' >> ' + _(
                    'Running paired Wilcoxon test.') + '\n</cs_decision>'
                result_ht += paired_wilcox_test(data, var_names)
            else:  # nominal variables
                if len(set(data.values.ravel())) == 2:
                    result_ht += '<cs_decision>' + _('Nominal dichotomous variables.') + ' >> ' + _(
                        'Running McNemar test.') \
                                 + '\n</cs_decision>'
                    result_ht += mcnemar_test(data, var_names)
                else:
                    result_ht += '<cs_decision>' + _('Nominal non dichotomous variables.') + ' >> ' + \
                                 _('Sorry, not implemented yet.') + '\n</cs_decision>'
        else:
            result_ht += '<cs_decision>' + _('More than two variables. ') + '</cs_decision>'
            if meas_level in ['int', 'unk']:
                result_ht += '<cs_decision>' + _('Interval variables.') + ' >> ' + \
                             _('Choosing repeated measures ANOVA or Friedman test depending on the assumptions.') + \
                             '\n</cs_decision>'

                result_ht += '<cs_decision>' + _('Checking for normality.') + '\n</cs_decision>'
                non_normal_vars = []
                for var_name in var_names:
                    norm, text_result = normality_test(data, data_measlevs, var_name)
                    result_ht += text_result
                    if not norm:
                        non_normal_vars.append(var_name)

                if not non_normal_vars:
                    result_ht += '<cs_decision>' + _('Normality is not violated.') + ' >> ' + \
                                 _('Running repeated measures one-way ANOVA.') + '\n</cs_decision>'
                    result_ht += repeated_measures_anova(data, var_names, factors)
                else:
                    result_ht += '<cs_decision>' + _('Normality is violated in variable(s): %s.') % ', '. \
                        join(non_normal_vars) + ' >> ' + _('Running Friedman test.') + '\n</cs_decision>'
                    result_ht += friedman_test(data, var_names)
            elif meas_level == 'ord':
                result_ht += '<cs_decision>' + _('Ordinal variables.') + ' >> ' + _(
                    'Running Friedman test.') + '\n</cs_decision>'
                result_ht += friedman_test(data, var_names)
            else:
                if len(set(data.values.ravel())) == 2:
                    result_ht += '<cs_decision>' + _('Nominal dichotomous variables.') + ' >> ' + _(
                        "Running Cochran's Q test.") + \
                                 '\n</cs_decision>'
                    result_ht += cochran_q_test(data, var_names)
                else:
                    result_ht += '<cs_decision>' + _('Nominal non dichotomous variables.') + ' >> ' \
                                 + _('Sorry, not implemented yet.') + '\n</cs_decision>'
    else:  # two- or more-ways comparison
        if meas_level in ['int', 'unk']:
            result_ht += '<cs_decision>' + _('Interval variables with several factors.') + ' >> ' + \
                         _('Choosing repeated measures ANOVA.') + \
                         '\n</cs_decision>'
            result_ht += repeated_measures_anova(data, var_names, factors)
        elif meas_level == 'ord':
            result_ht += '<cs_decision>' + _('Ordinal variables with several factors.') + ' >> ' \
                         + _('Sorry, not implemented yet.') + '\n</cs_decision>'
        elif meas_level == 'nom':
            result_ht += '<cs_decision>' + _('Nominal variables with several factors.') + ' >> ' \
                         + _('Sorry, not implemented yet.') + '\n</cs_decision>'
    return result_ht


def paired_t_test(pdf, var_names):
    """Calculate paired sample t-test.

    Parameters
    ----------
    pdf : pandas dataframe
    var_names : list of str
        two variable names to compare

    Returns
    -------
    str
    """
    # Not available in statsmodels
    if len(var_names) != 2:
        return _('Paired t-test requires two variables.')

    variables = pdf[var_names]
    text_result = ''

    # Sensitivity power analysis
    # statsmodels may fail, see its API documentation
    # d: (mean divided by the standard deviation of the differences)
    try:
        from statsmodels.stats.power import TTestPower
        power_analysis = TTestPower()
        effect_size_95 = {_('d'): power_analysis.solve_power(effect_size=None, nobs=len(variables), alpha=0.05,
                                                             power=0.95, alternative='two-sided')}
        effect_size_80 = {_('d'): power_analysis.solve_power(effect_size=None, nobs=len(variables), alpha=0.05,
                                                             power=0.80, alternative='two-sided')}
    except ValueError:
        effect_size_95 = {_('d'): None}
        effect_size_80 = {_('d'): None}
    text_result += print_sensitivity_effect_sizes(effect_size_95, effect_size_80)

    df = len(variables) - 1
    t, p = stats.ttest_rel(variables.iloc[:, 0], variables.iloc[:, 1])
    text_result += _('Result of paired samples t-test') + \
                   ': <i>t</i>(%d) = %0.*f, %s\n' % (df, non_data_dim_precision, t, print_p(p))
    # Bayesian t-test
    bf10 = pingouin.bayesfactor_ttest(t, len(variables), paired=True)
    text_result += _('Result of the Bayesian paired two-samples t-test') + \
                   ': BF<sub>10</sub> = %0.*f, BF<sub>01</sub> = %0.*f\n' % \
                   (non_data_dim_precision, bf10, non_data_dim_precision, 1/bf10)

    return text_result


def paired_wilcox_test(pdf, var_names):
    """Calculate paired Wilcoxon Signed Rank test.
    http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test

    Parameters
    ---------
    pdf : pandas dataframe
    var_names: list of str
        two variable names to compare

    Returns
    -------
    """
    # Not available in statsmodels
    text_result = ''
    if len(var_names) != 2:
        return _('Paired Wilcoxon test requires two variables.')

    try:
        T, p = stats.wilcoxon(pdf[var_names[0]], pdf[var_names[1]])
        text_result += _('Result of Wilcoxon signed-rank test') + \
                       ': <i>T</i> = %0.*f, %s\n' % (non_data_dim_precision, T, print_p(p))
        # The test does not use df, despite some of the descriptions on the net.
        # So there's no need to display df.
    except ValueError:
        text_result += 'Wilcoxon signed-rank test do not work if the difference of the two variables is 0 in all cases.'

    return text_result


def mcnemar_test(pdf, var_names):
    mcnemar_result = mcnemar(pd.crosstab(pdf[var_names[0]], pdf[var_names[1]]), exact=False)
    return _('Result of the McNemar test') + ': &chi;<sup>2</sup>(1, <i>N</i> = %d) = %0.*f, %s\n' % \
           (len(pdf[var_names[0]]), non_data_dim_precision, mcnemar_result.statistic, print_p(mcnemar_result.pvalue))


def cochran_q_test(pdf, var_names):
    q, p, df = cochrans_q(pdf[var_names], return_object=False)
        # Note that df is not documented as of statsmodels 0.11.1
    return _("Result of Cochran's Q test") + ': <i>Q</i>(%d, <i>N</i> = %d) = %0.*f, %s\n' % \
           (df, len(pdf[var_names[0]]), non_data_dim_precision, q, print_p(p))


def repeated_measures_anova(pdf, var_names, factors=None):
    """

    Parameters
    ---------
    pdf : pandas dataframe
        It is assumed that NaNs are dropped.
    var_names
    factors

    Returns
    -------

    """

    if factors is None:
        factors = []
    if len(factors) == 1:  # one-way comparison
        # Mauchly's test for sphericity
        spher, w, chisq, dof, wp = pingouin.sphericity(pdf[var_names])
        text_result = _("Result of Mauchly's test to check sphericity") + \
                      ': <i>W</i> = %0.*f, %s. ' % (non_data_dim_precision, w, print_p(wp))

        # ANOVA
        aov = pingouin.rm_anova(pdf[var_names], correction=True)
        if wp < 0.05:  # sphericity is violated
            p = aov.loc[0, 'p-GG-corr']
            text_result += '\n<cs_decision>'+_('Sphericity is violated.') + ' >> ' \
                           + _('Using Greenhouse–Geisser correction.') + '\n</cs_decision>' + \
                           _('Result of repeated measures ANOVA') + ': <i>F</i>(%0.3g, %0.3g) = %0.*f, %s\n' \
                           % (aov.loc[0, 'ddof1'] * aov.loc[0, 'eps'], aov.loc[0, 'ddof2'] * aov.loc[0, 'eps'],
                              non_data_dim_precision, aov.loc[0, 'F'], print_p(aov.loc[0, 'p-GG-corr']))
        else:  # sphericity is not violated
            p = aov.loc[0, 'p-unc']
            text_result += '\n<cs_decision>'+_('Sphericity is not violated. ') + '\n</cs_decision>' + \
                           _('Result of repeated measures ANOVA') + ': <i>F</i>(%d, %d) = %0.*f, %s\n' \
                           % (aov.loc[0, 'ddof1'], aov.loc[0, 'ddof2'],
                              non_data_dim_precision, aov.loc[0, 'F'], print_p(aov.loc[0, 'p-unc']))

        # Post-hoc tests
        if p < 0.05:
            pht = cs_stat_num.pairwise_ttest(pdf[var_names], var_names).sort_index()
            text_result += '\n' + _('Comparing variables pairwise with the Holm–Bonferroni correction:')
            #print pht
            pht['text'] = pht.apply(lambda x: '<i>t</i> = %0.*f, %s' %
                                              (non_data_dim_precision, x['t'], print_p(x['p (Holm)'])), axis=1)

            pht_text = pht[['text']]
            text_result += pht_text.to_html(bold_rows=True, escape=False, header=False).replace('\n', '')

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

        pdf_temp = pdf[var_names]
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
            text_result += (': <i>F</i>(%d, %d) = %0.*f, %s\n' %
                            (row['Num DF'], row['Den DF'],
                             non_data_dim_precision, row['F Value'], print_p(row['Pr > F'])))

        # TODO post hoc - procedure for any number of factors (i.e., not only for two factors)
    #print(text_result)

    return text_result


def friedman_test(pdf, var_names):
    """Friedman t-test

    Parameters
    ----------
    pdf : pandas dataframe
    var_names : list of str

    """
    # Not available in statsmodels
    text_result = ''
    if len(var_names) < 2:
        return _('Friedman test requires at least two variables.')

    variables = pdf[var_names]
    chi2, p = stats.friedmanchisquare(*[np.array(var) for var in variables.T.values])
    df = len(var_names) - 1
    n = len(variables)
    text_result += _('Result of the Friedman test: ') + '&chi;<sup>2</sup>(%d, <i>N</i> = %d) = %0.*f, %s\n' % \
                   (df, n, non_data_dim_precision, chi2, print_p(p))  # χ2(1, N=90)=0.89, p=.35
    if p < 0.05:
        # Run the post hoc tests
        text_result += '\n' + _('Variables differ. Running post-hoc pairwise comparison.') + '\n'
        text_result += _("Results of Durbin-Conover test (p values).") + '\n'
        posthoc_result = scikit_posthocs.posthoc_durbin(variables)
        text_result += posthoc_result.to_html(float_format=lambda x: '%.3f' % x).replace('\n', '')

    return text_result


### Compare groups ###


def decision_one_grouping_variable(df, meas_level, data_measlevs, var_names, groups, group_levels,
                                   single_case_slope_SE, single_case_slope_trial_n):
    result_ht = '<cs_decision>'
    if meas_level in ['int', 'unk']:
        result_ht += _('Testing if the means are the same.') + '</cs_decision>\n'
    elif meas_level == 'ord':
        result_ht += _('Testing if the medians are the same.') + '</cs_decision>\n'
    elif meas_level == 'nom':
        result_ht += _('Testing if the distributions are the same.') + '</cs_decision>\n'

    result_ht += '<cs_decision>' + _('One grouping variable. ') + '</cs_decision>'
    if len(group_levels) == 1:
        result_ht += _('There is only one group. At least two groups required.') + '\n</cs_decision>'

    # Compare two groups
    elif len(group_levels) == 2:
        result_ht += '<cs_decision>' + _('Two groups. ') + '</cs_decision>'
        if meas_level == 'int':
            group_levels, [var1, var2] = cs_stat._split_into_groups(df, var_names[0], groups)
            if len(var1) == 1 or len(var2) == 1:  # Single case vs control group
                result_ht += '<cs_decision>' + _('One group contains only one case. >> Choosing modified t-test.') + \
                             '\n</cs_decision>'
                result_ht += '<cs_decision>' + _('Checking for normality.') + '\n</cs_decision>'
                group = group_levels[1] if len(var1) == 1 else group_levels[0]
                norm, text_result = normality_test(df, data_measlevs, var_names[0], group_name=groups[0],
                                                   group_value=group[0])
                result_ht += text_result
                if not norm:
                    result_ht += '<cs_decision>' + _('Normality is violated in variable ') + var_names[0] + ', ' + \
                                 _('group ') + str(group) + '.\n</cs_decision>'
                    result_ht += '<cs_decision>>> ' + _('Running Brunner–Munzel test.') + '\n</cs_decision>'
                    result_ht += brunner_munzel_test(df, var_names[0], groups[0])
                else:
                    result_ht += '<cs_decision>' + _('Normality is not violated. >> Running modified t-test.') + \
                                 '\n</cs_decision>'
                    result_ht += single_case_task_extremity(df, var_names[0], groups[0], single_case_slope_SE if
                                 single_case_slope_SE else None, single_case_slope_trial_n)
            else:
                result_ht += '<cs_decision>' + _('Interval variable.') + ' >> ' + \
                             _("Choosing two sample t-test, Brunner-Munzel test or Welch's t-test depending on "
                               "assumptions.") + '\n</cs_decision>'
                result_ht += '<cs_decision>' + _('Checking for normality.') + '\n</cs_decision>'
                non_normal_groups = []
                for group in group_levels:
                    norm, text_result = normality_test(df, data_measlevs, var_names[0], group_name=groups[0],
                                                       group_value=group[0])
                    result_ht += text_result
                    if not norm:
                        non_normal_groups.append(group)
                result_ht += '<cs_decision>' + _('Checking for homogeneity of variance across groups.') + '\n</cs_decision>'
                homogeneity_vars = True
                p, text_result = levene_test(df, var_names[0], groups[0])
                result_ht += text_result
                if p < 0.05:
                    homogeneity_vars = False

                if not (non_normal_groups) and homogeneity_vars:
                    result_ht += '<cs_decision>' + \
                                 _('Normality and homogeneity of variance are not violated. >> Running two sample '
                                   't-test.') + '\n</cs_decision>'
                    result_ht += independent_t_test(df, var_names[0], groups[0])
                elif non_normal_groups:
                    result_ht += '<cs_decision>' + _('Normality is violated in variable %s, group(s) %s.') % \
                                 (var_names[0], ', '.join(map(str, non_normal_groups))) + ' >> ' + \
                                 _('Running Brunner-Munzel test.') + '\n</cs_decision>'
                    result_ht += brunner_munzel_test(df, var_names[0], groups[0])
                elif not homogeneity_vars:
                    result_ht += '<cs_decision>' + _('Homogeneity of variance violated in variable %s.') % \
                                 var_names[0] + ' >> ' + _("Running Welch's t-test.") + '\n</cs_decision>'
                    result_ht += welch_t_test(df, var_names[0], groups[0])

        elif meas_level == 'ord':
            result_ht += '<cs_decision>' + _('Ordinal variable.') + ' >> ' + _(
                'Running Bruner-Munzel test.') + '</cs_decision>\n'
            result_ht += brunner_munzel_test(df, var_names[0], groups[0])
        elif meas_level == 'nom':
            result_ht += '<cs_decision>' + _('Nominal variable.') + ' >> ' + _(
                'Running chi-squared test.') + ' ' + '</cs_decision>\n'
            chi_result = chi_squared_test(df, var_names[0], groups[0])
            result_ht += chi_result

    # Compare more than two groups
    elif len(group_levels) > 2:
        result_ht += '<cs_decision>' + _('More than two groups.') + ' </cs_decision>'
        if meas_level == 'int':
            result_ht += '<cs_decision>' + _('Interval variable.') + ' >> ' + \
                         _('Choosing one-way ANOVA or Kruskal–Wallis test depending on the assumptions.') + \
                         '</cs_decision>\n'

            result_ht += '<cs_decision>' + _('Checking for normality.') + '\n</cs_decision>'
            non_normal_groups = []
            for group in group_levels:
                norm, text_result = normality_test(df, data_measlevs, var_names[0], group_name=groups[0],
                                                   group_value=group)
                result_ht += text_result
                if not norm:
                    non_normal_groups.append(group)
            result_ht += '<cs_decision>' + _('Checking for homogeneity of variance across groups.') + '\n</cs_decision>'
            homogeneity_vars = True
            p, text_result = levene_test(df, var_names[0], groups[0])
            result_ht += text_result
            if p < 0.05:
                homogeneity_vars = False

            if not (non_normal_groups) and homogeneity_vars:
                result_ht += '<cs_decision>' + \
                             _('Normality and homogeneity of variance are not violated. >> Running one-way ANOVA.') \
                             + '\n</cs_decision>'
                anova_result = one_way_anova(df, var_names[0], groups[0])
                result_ht += anova_result

            if non_normal_groups:
                result_ht += '<cs_decision>' + _('Normality is violated in variable %s, group(s) %s. ') % \
                             (var_names[0], ', '.join(map(str, non_normal_groups))) + '</cs_decision>'
            if not homogeneity_vars:
                result_ht += '<cs_decision>' + _('Homogeneity of variance violated in variable %s.') % var_names[0] + \
                             '</cs_decision>'
            if non_normal_groups or (not homogeneity_vars):
                result_ht += '<cs_decision>' + '>> ' + _('Running Kruskal–Wallis test.') + '\n</cs_decision>'
                result_ht += kruskal_wallis_test(df, var_names[0], groups[0])

        elif meas_level == 'ord':
            result_ht += '<cs_decision>' + _('Ordinal variable.') + ' >> ' + _('Running Kruskal–Wallis test.') + \
                         '</cs_decision>\n'
            result_ht += kruskal_wallis_test(df, var_names[0], groups[0])
        elif meas_level == 'nom':
            result_ht += '<cs_decision>' + _('Nominal variable.') + ' >> ' + _('Running chi-squared test.') + \
                         '</cs_decision>\n'
            chi_result = chi_squared_test(df, var_names[0], groups[0])
            result_ht += chi_result
    return result_ht


def decision_several_grouping_variables(df, meas_level, var_names, groups):
    """

    Parameters
    ----------
    df : pandas dataframe
        It is assumed that cases with NaNs are dropped.
    meas_level
    var_names
    groups

    Returns
    -------

    """
    result_ht = '<cs_decision>'
    if meas_level in ['int', 'unk']:
        result_ht += _('Testing if the means are the same.') + '</cs_decision>\n'
    elif meas_level == 'ord':
        result_ht += _('Testing if the medians are the same.') + '</cs_decision>\n'
    elif meas_level == 'nom':
        result_ht += _('Testing if the distributions are the same.') + '</cs_decision>\n'

    result_ht += '<cs_decision>' + _('At least two grouping variables.') + ' </cs_decision>'
    if meas_level == 'int':
        #group_levels, vars = cs_stat._split_into_groups(df, var_names[0], groups)
        result_ht += '<cs_decision>' + _('Interval variable.') + ' >> ' + \
                     _("Choosing factorial ANOVA.") + '\n</cs_decision>'
        result_ht += multi_way_anova(df, var_names[0], groups)

    elif meas_level == 'ord':
        result_ht += '<cs_decision>' + _('Ordinal variable.') + ' >> ' + \
                     _('Sorry, not implemented yet.') + '</cs_decision>\n'
    elif meas_level == 'nom':
        result_ht += '<cs_decision>' + _('Nominal variable.') + ' >> ' + \
                     _('Sorry, not implemented yet.') + ' ' + '</cs_decision>\n'
    return result_ht


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

    dummy_groups, var_s = cs_stat._split_into_groups(pdf, var_name, group_name)
    w, p = stats.levene(*var_s)
    text_result += _('Levene test') + ': <i>W</i> = %0.*f, %s\n' % (non_data_dim_precision, w, print_p(p))

    return p, text_result


def independent_t_test(pdf, var_name, grouping_name):
    """Independent samples t-test

    arguments:
    var_name (str):
    grouping_name (str):
    """
    from statsmodels.stats.weightstats import ttest_ind
    text_result = ''

    dummy_groups, [var1, var2] = cs_stat._split_into_groups(pdf, var_name, grouping_name)
    t, p, df = ttest_ind(var1, var2)

    # Sensitivity power analysis
    # d: (difference between the two means divided by the standard deviation)
    try:
        # statsmodels may fail, see its API documentation
        from statsmodels.stats.power import TTestIndPower
        power_analysis = TTestIndPower()
        effect_size_95 = {_('d'): power_analysis.solve_power(effect_size=None, nobs1=len(var1), alpha=0.05, power=0.95,
                                                             ratio=len(var2) / len(var1), alternative='two-sided')}
        effect_size_80 = {_('d'): power_analysis.solve_power(effect_size=None, nobs1=len(var1), alpha=0.05, power=0.80,
                                                             ratio=len(var2) / len(var1), alternative='two-sided')}
    except ValueError:
        effect_size_95 = {_('d'): None}
        effect_size_80 = {_('d'): None}
    text_result += print_sensitivity_effect_sizes(effect_size_95, effect_size_80)

    text_result += _('Result of independent samples t-test:') + ' <i>t</i>(%0.3g) = %0.*f, %s\n' % \
                   (df, non_data_dim_precision, t, print_p(p))
    # Bayesian t-test
    bf10 = pingouin.bayesfactor_ttest(t, len(var1), len(var2))
    text_result += _('Result of the Bayesian independent two-samples t-test') + \
                   ': BF<sub>10</sub> = %0.*f, BF<sub>01</sub> = %0.*f\n' % \
                   (non_data_dim_precision, bf10, non_data_dim_precision, 1/bf10)

    return text_result


def single_case_task_extremity(pdf, var_name, grouping_name, se_name=None, n_trials=None):
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
    group_levels, [var1, var2] = cs_stat._split_into_groups(pdf, var_name, grouping_name)
    if not se_name:  # Simple performance score
        try:
            if len(var1) == 1:
                ind_data = var1
                group_data = var2
            else:
                ind_data = var2
                group_data = var1
            t, p, df = cs_stat_num.modified_t_test(ind_data, group_data)
            text_result += _('Result of the modified independent samples t-test:') + \
                           ' <i>t</i>(%0.3g) = %0.*f, %s\n' % (df, non_data_dim_precision, t, print_p(p))
        except ValueError:
            text_result += _('One of the groups should include only a single data.')
    else:  # slope performance
        group_levels, [se1, se2] = cs_stat._split_into_groups(pdf, se_name, grouping_name)
        if len(var1) == 1:
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
        text_result += _('Result of slope test with %s:') % (test) + \
                       ' <i>t</i>(%0.3g) = %0.*f, %s\n' % (df, non_data_dim_precision, t, print_p(p))
    return text_result


def welch_t_test(pdf, var_name, grouping_name):
    """ Welch's t-test

    :param pdf: pandas data frame
    :param var_name: name of the dependent variable
    :param grouping_name: name of the grouping variable
    :return: html text with APA format result
    """
    dummy_groups, [var1, var2] = cs_stat._split_into_groups(pdf, var_name, grouping_name)
    t, p = stats.ttest_ind(var1.dropna(), var2.dropna(), equal_var=False)
    # http://msemac.redwoods.edu/~darnold/math15/spring2013/R/Activities/WelchTTest.html
    n1 = len(var1)
    n2 = len(var2)
    A = np.std(var1)/n1
    B = np.std(var2)/n2
    df = (A+B)**2/(A**2/(n1-1)+B**2/(n2-1))
    return _("Result of Welch's unequal variances t-test:") + \
           ' <i>t</i>(%0.3g) = %0.*f, %s\n' % (df, non_data_dim_precision, t, print_p(p))


def mann_whitney_test(pdf, var_name, grouping_name):
    """Mann–Whitney test

    Parameters
    ----------
    pdf: pandas dataframe
    var_name : str
    grouping_name : str
    """
    # Not available in statsmodels

    dummy_groups, [var1, var2] = cs_stat._split_into_groups(pdf, var_name, grouping_name)
    u, p = stats.mannwhitneyu(var1, var2, alternative='two-sided')
    text_result = _('Result of independent samples Mann–Whitney rank test: ') + '<i>U</i> = %0.*f, %s\n' % \
                   (non_data_dim_precision, u, print_p(p))

    return text_result


def brunner_munzel_test(pdf, var_name, grouping_name):
    """Brunner-Munzel test

    Parameters
    ----------
    pdf : pandas dataframe
    var_name : str
    grouping_name : str

    Returns
    -------
    str
        results of the test
    """

    # alternative implementation:
    # https://www.statsmodels.org/stable/generated/statsmodels.stats.nonparametric.rank_compare_2indep.html

    # TODO use the permutation Brunner-Munzel test when the sample size is smaller than 10 in any group
    # https://github.com/trevismd/permutations-stats

    dummy_groups, [var1, var2] = cs_stat._split_into_groups(pdf, var_name, grouping_name)
    w, p = stats.brunnermunzel(var1, var2, alternative='two-sided')
    text_result = _('Result of the Brunner-Munzel test: ') + '<i>W</i> = %0.*f, %s\n' % \
                   (non_data_dim_precision, w, print_p(p))
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
    data = pdf[[var_name] + [grouping_name]]

    # Sensitivity power analysis
    # 1. Calculate effect size in F
    # statsmodels may fail, see its API documentation
    effect_size_95 = {}
    effect_size_80 = {}
    try:
        from statsmodels.stats.power import FTestAnovaPower
        power_analysis = FTestAnovaPower()
        effect_size_95[_('f')] = power_analysis.solve_power(effect_size=None, nobs=len(data), alpha=0.05, power=0.95,
                                                            k_groups=len(set(data[grouping_name])))
        effect_size_80[_('f')] = power_analysis.solve_power(effect_size=None, nobs=len(data), alpha=0.05, power=0.80,
                                                            k_groups=len(set(data[grouping_name])))
    except ValueError:
        effect_size_95[_('f')] = None
        effect_size_80[_('f')] = None

    # 2. Calculate effect size in eta-square
    # pingouin may fail in calculating the effect size, see its API documentation
    try:
        eta_square_95 = pingouin.power_anova(eta=None, n=len(data), alpha=0.05, power=0.95,
                                             k=len(set(data[grouping_name])))
        eta_square_80 = pingouin.power_anova(eta=None, n=len(data), alpha=0.05, power=0.80,
                                             k=len(set(data[grouping_name])))
    except TypeError:  # in pingouin 0.5.2 eta was renamed to eta_squared
        eta_square_95 = pingouin.power_anova(eta_squared=None, n=len(data), alpha=0.05, power=0.95,
                                             k=len(set(data[grouping_name])))
        eta_square_80 = pingouin.power_anova(eta_squared=None, n=len(data), alpha=0.05, power=0.80,
                                             k=len(set(data[grouping_name])))
    effect_size_95[_('eta-square')] = None if np.isnan(eta_square_95) else eta_square_95
    effect_size_80[_('eta-square')] = None if np.isnan(eta_square_80) else eta_square_80

    # 3. Create output text
    text_result += print_sensitivity_effect_sizes(effect_size_95, effect_size_80)

    # FIXME https://github.com/cogstat/cogstat/issues/136
    anova_model = ols(str('Q("%s") ~ C(Q("%s"))' % (var_name, grouping_name)), data=data).fit()
    # Type I is run, and we want to run type III, but for a one-way ANOVA different types give the same results
    anova_result = anova_lm(anova_model)
    text_result += _('Result of one-way ANOVA: ') + '<i>F</i>(%d, %d) = %0.*f, %s\n' % \
                   (anova_result['df'][0], anova_result['df'][1], non_data_dim_precision, anova_result['F'][0],
                    print_p(anova_result['PR(>F)'][0]))

    # http://statsmodels.sourceforge.net/stable/stats.html#multiple-tests-and-multiple-comparison-procedures
    if anova_result['PR(>F)'][0] < 0.05:  # post-hoc
        post_hoc_res = sm.stats.multicomp.pairwise_tukeyhsd(np.array(data[var_name]), np.array(data[grouping_name]),
                                                            alpha=0.05)
        text_result += '\n' + _('Groups differ. Post-hoc test of the means.') + '\n'
        text_result += ('<cs_fix_width_font>%s\n</cs_fix_width_font>' % post_hoc_res).replace(' ', '\u00a0')
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


def multi_way_anova(pdf, var_name, grouping_names):
    """Two-way ANOVA

    Parameters
    pdf : pandas dataframe
        It is assumed that missing cases are dropped.
    var_name : str
    grouping_names : list of str

    Returns
    -------

    """
    # http://statsmodels.sourceforge.net/stable/examples/generated/example_interactions.html#one-way-anova
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    import patsy
    anova_model = ols(str('%s ~ %s' % (var_name, ' * '.join([f'patsy.builtins.C({group_name}, patsy.builtins.Sum)'
                                                             for group_name in grouping_names]))), data=pdf).fit()
    anova_result = anova_lm(anova_model, typ=3)
    text_result = _('Result of multi-way ANOVA') + ':\n'

    # Main effects
    for group_i, group in enumerate(grouping_names):
        text_result += _('Main effect of %s: ' % group) + '<i>F</i>(%d, %d) = %0.*f, %s\n' % \
                       (anova_result['df'][group_i+1], anova_result['df'][-1], non_data_dim_precision,
                        anova_result['F'][group_i+1], print_p(anova_result['PR(>F)'][group_i + 1]))

    # Interaction effects
    for interaction_line in range(group_i+2, len(anova_result)-1):
        text_result += _('Interaction of %s: ') % \
                       (' and '.join([a[1:-21] for a in re.findall('\(.*?\)', anova_result.index[interaction_line])])) + \
                       '<i>F</i>(%d, %d) = %0.*f, %s\n' % \
                       (anova_result['df'][interaction_line], anova_result['df'][-1], non_data_dim_precision,
                        anova_result['F'][interaction_line], print_p(anova_result['PR(>F)'][interaction_line]))

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
        text_result += ('<cs_fix_width_font>%s\n</cs_fix_width_font>' % post_hoc_res).replace(' ', u'\u00a0')
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
    """Kruskal–Wallis test

    Arguments:
    var_name (str):
    grouping_name (str):
    """
    # Not available in statsmodels
    text_result = ''

    dummy_groups, variables = cs_stat._split_into_groups(pdf, var_name, grouping_name)
    variables = [variable for variable in variables]
    try:
        H, p = stats.kruskal(*variables)
        df = len(dummy_groups)-1
        n = len(pdf[var_name].dropna())  # TODO Is this OK here?
        text_result += _('Result of the Kruskal–Wallis test: ')+'&chi;<sup>2</sup>(%d, <i>N</i> = %d) = %0.*f, %s\n' % \
                                                                (df, n, non_data_dim_precision, H, print_p(p))  # χ2(1, N=90)=0.89, p=.35
        if p < 0.05:
            # Run the post hoc tests
            text_result += '\n' + _('Groups differ. Post-hoc test of the means.') + '\n'
            text_result += _("Results of Dunn's test (p values).") + '\n'
            posthoc_result = scikit_posthocs.posthoc_dunn(pdf, val_col=var_name, group_col=grouping_name)
            text_result += posthoc_result.to_html(float_format=lambda x: '%.3f' % x).replace('\n', '')

    except Exception as e:
        text_result += _('Result of the Kruskal–Wallis test: ')+str(e)

    return text_result


def chi_squared_test(pdf, var_name, grouping_name):
    """Run Chi-squared hypothesis test and related power analysis.

    Cramer's V: http://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

    Parameters
    ----------
    pdf : pandas DataFrame
    var_name : str
        name one of the nominal variables
    grouping_name : str
        name of the other nominal variable

    Returns
    -------
    str
        html with the hypothesis tests and power analysis results
    """
    text_result = ''
    cont_table_data = pd.crosstab(pdf[grouping_name], pdf[var_name])

    # Sensitivity power analysis
    # statsmodels may fail, see its API documentation # TODO after changing to pingouin, do we still need "try:"?
    try:
        # For a test of independence, df = (Rows − 1)×(Cols − 1)
        # https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Definition
        effect_size_95 = {_('w'): pingouin.power_chi2(dof=(cont_table_data.shape[0] - 1) *
                                                          (cont_table_data.shape[1] - 1),
                                                      w=None, n=cont_table_data.values.sum(), power=0.95, alpha=0.05)}
        effect_size_80 = {_('w'): pingouin.power_chi2(dof=(cont_table_data.shape[0] - 1) *
                                                          (cont_table_data.shape[1] - 1),
                                                      w=None, n=cont_table_data.values.sum(), power=0.80, alpha=0.05)}

        """An alternative method is to use statsmodels' solution, but its result is not consistent with the G*Power
        result and with pingouin's solution. See https://github.com/cogstat/cogstat/issues/207. 
        
        from statsmodels.stats.power import GofChisquarePower
        power_analysis = GofChisquarePower()
        print(power_analysis.solve_power(effect_size=None, nobs=cont_table_data.values.sum(),
                                                   alpha=0.05, power=0.95,
                                                   n_bins=(cont_table_data.shape[0] - 1) *
                                                          (cont_table_data.shape[1] - 1) + 1))"""
    except ValueError:
        effect_size_95 = {_('w'): None}
        effect_size_80 = {_('w'): None}
    text_result += print_sensitivity_effect_sizes(effect_size_95, effect_size_80)

    # Hypothesis test
    chi2, p, dof, expected = stats.chi2_contingency(cont_table_data.values)
    text_result += _("Result of the Pearson's chi-squared test: ") + \
                  '</i>&chi;<sup>2</sup></i>(%g, <i>N</i> = %d) = %.*f, %s' % \
                  (dof, cont_table_data.values.sum(), non_data_dim_precision, chi2, print_p(p))

    return text_result


def decision_mixed_design(data, meas_level, var_names, factors, grouping_variables):
    """

    Parameters
    ----------
    data : pandas dataframe
    meas_level :
    var_names :
    factors :
    grouping_variables : list of str

    Returns
    -------

    """

    result_ht = '<cs_decision>'
    if meas_level in ['int', 'unk']:
        result_ht += _('Testing if the means are the same.') + '</cs_decision>\n'
    elif meas_level == 'ord':
        result_ht += _('Testing if the medians are the same.') + '</cs_decision>\n'
    elif meas_level == 'nom':
        result_ht += _('Testing if the distributions are the same.') + '</cs_decision>\n'

    result_ht += '<cs_decision>' + _('Repeated measures and grouping variables.') + ' </cs_decision>'
    if meas_level == 'int':
        result_ht += '<cs_decision>' + _('Interval variable.') + ' >> ' + \
                     _("Choosing mixed ANOVA.") + '\n</cs_decision>'
        result_ht += mixed_anova(data, var_names, factors, grouping_variables)
    elif meas_level == 'ord':
        result_ht += '<cs_decision>' + _('Ordinal variable.') + ' >> ' + \
                     _('Sorry, not implemented yet.') + '</cs_decision>\n'
    elif meas_level == 'nom':
        result_ht += '<cs_decision>' + _('Nominal variable.') + ' >> ' + \
                     _('Sorry, not implemented yet.') + ' ' + '</cs_decision>\n'
    return result_ht


def mixed_anova(pdf, var_names, factors, grouping_variables):
    """

    Parameters
    ----------
    pdf : pandas dataframe
    var_names :
    factors :
    grouping_variables : list of str

    Returns
    -------

    """

    # at the moment, not available in Python (pingouin can run 2-way ANOVA, statsmodels cannot run mixed ANOVA)
    if csc.versions['r']:
        pandas2ri.activate()
        base = importr('base')
        ez = importr('ez')

        # Prepare the dataset for the ANOVA
        # new temporary names are needed to set the independent factors in the long format
        # (alternatively, one might set it later in the long format directly)
        temp_var_names = ['']
        for factor in factors:
            # TODO this will not work if the factor name includes the current separator (_)
            temp_var_names = [previous_var_name + '_' + factor[0] + str(i)
                              for previous_var_name in temp_var_names for i in range(factor[1])]
        temp_var_names = [temp_var_name[1:] for temp_var_name in temp_var_names]
        # print(temp_var_names)

        pdf_temp = pdf[var_names]
        pdf_temp.columns = temp_var_names
        pdf_temp = pdf_temp.assign(ID=pdf_temp.index)
        pdf_long = pd.melt(pdf_temp, id_vars='ID', value_vars=temp_var_names)
        pdf_long = pd.concat([pdf_long, pdf_long['variable'].str.split('_', expand=True).
                             rename(columns={i: factors[i][0] for i in range(len(factors))})], axis=1)
        pdf_long = pdf_long.join(pdf[grouping_variables], on='ID')
        # replace spaces for R
        pdf_long.columns = [column.replace(' ', '_') for column in pdf_long.columns]
        rdf_long = pandas2ri.py2rpy(pdf_long)

        # Run ANOVA
        anova_r = ez.ezANOVA(data=rdf_long, dv=base.as_symbol('value'), wid=base.as_symbol('ID'),
                             within=base.as_symbol(robjects.StrVector([factor[0].replace(' ', '_')
                                                                       for factor in factors])),
                             between=base.as_symbol(robjects.StrVector(grouping_variables)), type=3)
        anova_table = pandas2ri.rpy2py_dataframe(anova_r[0]).reset_index(drop=True)
        #sphericity_check_table = pandas2ri.rpy2py_dataframe(anova_r[1])
        #sphericity_correction_table = pandas2ri.rpy2py_dataframe(anova_r[2])

        # TODO reverse _s to spaces
        text_result = _('Result of multi-way ANOVA') + ':\n'
        for index, effect in anova_table.iterrows():
            if index < len(factors) + len(grouping_variables):  # Main effects
                text_result += _('Main effect of %s: ' % effect['Effect'])
            else:  # Interaction effects
                text_result += _('Interaction of %s: ') % (' and '.join(effect['Effect'].split(':')))
            text_result += '<i>F</i>(%d, %d) = %0.*f, %s\n' % \
                           (effect['DFn'], effect['DFd'], non_data_dim_precision, effect['F'], print_p(effect['p']))

        return text_result
    else:
        return _('Sorry, R is not available to run %.') % _('mixed ANOVA')

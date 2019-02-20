# -*- coding: utf-8 -*-

"""
This module contains functions for statistical analysis that cannot be found in
other packages.

Arguments are the pandas data frame (pdf) and parameters.
Output is the result of the numerical analysis in numerical form.
"""

import numpy as np
from scipy import stats
import pandas as pd

### Variable pairs ###


def corr_ci(r, n, confidence=0.95):
    """ Compute confidence interval for Spearman or Pearson correlation coefficients based on Fisher transformation
    https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Using_the_Fisher_transformation
    :param r: correlation coefficient
    :param n: sample size
    :param confidence: confidence, default is 0.95
    :return: low and high
    """
    delta = stats.norm.ppf(1.0 - (1 - confidence) / 2) / np.sqrt(n - 3)
    lower = np.tanh(np.arctanh(r) - delta)
    upper = np.tanh(np.arctanh(r) + delta)
    return lower, upper


def modified_t_test(ind_data, group_data):
    """Compare a single case to a group.

    More information:
    Crawford, J.R. & Howell, D.C. (1998). Comparing an individual's test score
    against norms derived from small samples. The Clinical Neuropsychologist,
    12, 482-486.

    :param ind_data: single row pandas data frame for the single case
    :param group_data: several rows of pandas data frame for the control grop values
    :return tstat: test statistics
    :return pvalue: p value of the test
    :return df: degrees fo freedom
    """

    group_data_n = len(group_data)
    tstat = (ind_data.iloc[0] - np.mean(group_data)) / (np.std(group_data) * np.sqrt((group_data_n+1.0)/group_data_n))
    df = group_data_n-1
    pvalue = stats.t.sf(np.abs(tstat), df)*2  # two-sided
    return tstat, pvalue, df


def slope_extremity_test(n_trials, case_slope, case_SE, control_slopes, control_SEs):
    """
    This function checks the extremity of a single case performance expressed as a slope compared to the control data.

    More information:
    Crawford, J. R., & Garthwaite, P. H. (2004). Statistical Methods for Single-Case Studies in Neuropsychology:
    Comparing the Slope of a Patient’s Regression Line with those of a Control Sample. Cortex, 40(3), 533–548.
    http://doi.org/10.1016/S0010-9452(08)70145-X

    :param n_trials: number of trials the slopes rely on
    :param case_slope, case_SE: single row pandas data frames with the slope and the standard error of the single case
    :param control_slopes, control_SEs: single row pandas data frames with the slope and the standard error of the control cases

    Returns the appropriate test statistic value, the degree of freedom, the p-value, and the chosen test type (string)
    """

    beta_mean = control_slopes.mean(axis=0)
    s_square_mean = (control_SEs ** 2).mean(axis=0)
    u_square = control_slopes.var(axis=0)
    sigma_square = s_square_mean - u_square
    n_control = float(control_slopes.count())

    cond_1 = ((control_SEs ** 2) <= (sigma_square / 10)).all()
    cond_2 = (case_SE ** 2) <= (sigma_square / 10)
    cond_5 = u_square > s_square_mean

    def test_a(n_control, n_trials, control_SEs, s_square_mean):
        """Testing for equal variances in the control sample"""
        g = 1 + (n_control + 1) / (3 * n_control * (n_trials - 2))
        sum_ln_se = np.log((control_SEs ** 2)).sum()
        chi2 = (n_trials - 2) * ((n_control * np.log(s_square_mean) - sum_ln_se)) / g
        df = n_control - 1
        p = 1 - stats.chi2.cdf(chi2, df)
        return p

    cond_3 = test_a(n_control=n_control, n_trials=n_trials, control_SEs=control_SEs, s_square_mean=s_square_mean) < 0.05

    def test_b(n_control, n_trials, case_SE, s_square_mean):
        """Comparing the variance of the patient with those of the control sample"""
        case_numerator = (case_SE ** 2) > s_square_mean
        F = (case_SE ** 2) / s_square_mean if case_numerator else s_square_mean / (case_SE ** 2)
        df_1 = n_trials - 2 if case_numerator else n_control * (n_trials - 2)
        df_2 = n_control * (n_trials - 2) if case_numerator else n_trials - 2
        p = 1 - stats.f.cdf(F, df_1, df_2)
        return p

    cond_4 = test_b(n_control=n_control, n_trials=n_trials, case_SE=case_SE, s_square_mean=s_square_mean) < 0.05

    def test_c(case_slope, beta_mean, u_square, n_control):
        """Comparing slopes whose variances are the same for patient and controls"""
        t = (case_slope - beta_mean) / (np.sqrt(u_square) * np.sqrt((n_control + 1) / n_control))
        df = n_control - 1
        p = 1 - stats.t.cdf(abs(t), df)
        return t, df, p

    def test_d1(case_slope, beta_mean, n_control, s_square_mean, case_SE, u_square, n_trials):
        t = (case_slope - beta_mean) / np.sqrt(u_square * ((n_control + 1) / n_control) - s_square_mean + case_SE ** 2)
        df = (u_square * ((n_control + 1) / n_control) - s_square_mean + case_SE ** 2) ** 2 / (
                    (1 / (n_control - 1)) * (u_square * ((n_control + 1) / n_control)) ** 2 + (
                        s_square_mean ** 2 / (n_control * (n_trials - 2))) + (case_SE ** 2 ** 2 / (n_trials - 2)))
        p = 1 - stats.t.cdf(abs(t), df)
        return t, df, p

    def test_d2(case_slope, beta_mean, case_SE, s_square_mean, n_control, n_trials):
        t = (case_slope - beta_mean) / np.sqrt(case_SE ** 2 + s_square_mean / n_control)
        df = (case_SE ** 2 + s_square_mean / n_control) ** 2 / (
                    case_SE ** 2 ** 2 / (n_trials - 2) + (s_square_mean ** 2 / (n_control ** 3 * (n_trials - 2))))
        p = 1 - stats.t.cdf(abs(t), df)
        return t, df, p

    if cond_1:
        if cond_2:
            test = 'Test c'
            t, df, p = test_c(case_slope=case_slope, beta_mean=beta_mean, u_square=u_square, n_control=n_control)
            if cond_4:
                if cond_5:
                    test = 'Test d.1'
                    t, df, p = test_d1(case_slope=case_slope, beta_mean=beta_mean, n_control=n_control,
                                       s_square_mean=s_square_mean, case_SE=case_SE, u_square=u_square,
                                       n_trials=n_trials)
                else:
                    test = 'Test d.2'
                    t, df, p = test_d2(case_slope=case_slope, beta_mean=beta_mean, case_SE=case_SE,
                                       s_square_mean=s_square_mean, n_control=n_control, n_trials=n_trials)
            else:
                test = 'Test c'
                t, df, p = test_c(case_slope=case_slope, beta_mean=beta_mean, u_square=u_square, n_control=n_control)
    else:
        if cond_3:
            test = 'Consider reformulate your question with correlation or use Bayesian methods'
            t, df, p = [None, None, None]
        else:
            if cond_4:
                if cond_5:
                    test = 'Test d.1'
                    t, df, p = test_d1(case_slope=case_slope, beta_mean=beta_mean, n_control=n_control,
                                       s_square_mean=s_square_mean, case_SE=case_SE, u_square=u_square,
                                       n_trials=n_trials)
                else:
                    test = 'Test d.2'
                    t, df, p = test_d2(case_slope=case_slope, beta_mean=beta_mean, case_SE=case_SE,
                                       s_square_mean=s_square_mean, n_control=n_control, n_trials=n_trials)
            else:
                test = 'Test c'
                t, df, p = test_c(case_slope=case_slope, beta_mean=beta_mean, u_square=u_square, n_control=n_control)

    return t, df, p, test


def repeated_measures_anova(data, dep_var, indep_var=None, id_var=None, wide=True):
    """
    Standard one-way repeated measures ANOVA
    
    ### Arguments:
    data: pandas DataFrame
    dep_var: dependent variable - label (long format) or a list of labels (wide format)
    indep_var: label of the independent variable (only necessary if data is in long format)
    id_var: label of the variable which contains the participants' identifiers. Default assumes that the table index
            contains the identifiers.
    wide: whether the data is in wide format
    
    ### Returns: [DFn, DFd, F, pF, W, pW], corr_table
    DFn, DFd: degrees of freedom used when calculating p(F)
    F: F statistic (uncorrected)
    pF: p value of the F statistic (uncorrected)
    W: statistic of Mauchly's test for sphericity
    pW: p value of Mauchly's test for sphericity (sphericity is violated if p(W)<0.05)
    corr_table: numpy array, contains Greenhouse & Geisser's, Huhyn & Feldt's, and the "lower-bound" epsilons,
                and the corrected p values of F (degrees of freedom should be multiplied with the epsilon to get
                the corrected df values)
    """
    ### Reshaping data
    if wide:
        if not id_var:
            data = data.assign(ID=data.index)
            id_var = 'ID'
        data = pd.melt(data, id_vars=id_var, value_vars=dep_var, var_name='condition', value_name='measured')
        dep_var = 'measured'
        indep_var = 'condition'

    ### one-way ANOVA
    n = len(set(data[id_var]))
    k = len(set(data[indep_var]))
    DFn = (k-1)
    DFd = (k-1)*(n-1)
    q_eff = []
    q_err = []
    for j, var in enumerate(list(set(data[indep_var]))):
        subset_j = data[data[indep_var] == var]
        q_eff.insert(j, n*np.square(np.mean(subset_j[dep_var])-np.mean(data[dep_var])))
        for i, sub in enumerate(list(set(data[id_var]))):
            subset_i = data[data[id_var] == sub]
            subset_ij = subset_j[subset_j[id_var] == sub]
            q_err.insert(j+i, np.square(np.mean(subset_ij[dep_var]) -np.mean(subset_i[dep_var]) - np.mean(subset_j[dep_var]) + np.mean(data[dep_var])))
    # F-statistic    
    F = (sum(q_eff)/DFn)/(sum(q_err)/DFd)
    pF = 1-stats.f.cdf(F, DFn, DFd)

    ### Mauchly's test for sphericity & Degree of freedom corrections
    # Calculating sample covariances
    table = np.empty((0, n))
    for j, var in enumerate(list(set(data[indep_var]))):
        subset_j = data[data[indep_var] == var]
        row = []
        for i, sub in enumerate(list(set(data[id_var]))):
            subset_ij = subset_j[subset_j[id_var] == sub]
            row.insert(i, np.mean(subset_ij[dep_var].values))
        table = np.vstack([table, np.asarray(row)])    
    samp_table = np.cov(table)
    samp_means = samp_table.mean(axis=1)
    # Estimating population covariances
    pop_table = np.empty((0, k))
    for x in range(k):
        row = []
        for y in range(k):
            row.insert(y, samp_table[x][y]-samp_means[x]-samp_means[y]+samp_table.mean())
        pop_table = np.vstack([pop_table, np.asarray(row)])
    # Mauchly's W statistic
    W = np.prod([x for x in list(np.linalg.eigvals(pop_table)) if x > 0.00000000001])/np.power(np.trace(pop_table)/(k-1), (k-1)) # uses the pseudo-determinant (discards all near-zero eigenvalues)
    dfW = int((0.5*k*(k-1))-1)
    fW = float(2*np.square(k-1)+(k-1)+2)/float(6*(k-1)*(n-1))
    chiW = (fW-1)*(n-1)*np.log(W)
    pW = 1-stats.chi2.cdf(chiW, dfW)

    # Greenhouse & Geisser's epsilon
    GG = np.square(np.trace(pop_table))/(np.sum(np.square(pop_table))*(k-1))
    # Huynh & Feldt's epsilon
    HF = (n*(k-1)*GG-2)/((k-1)*(n-1-(k-1)*GG))
    # Lower-bound epsilon
    LB = 1/float(k-1)
    # Correction
    corr_list = [GG, HF, LB]
    corr_table = np.empty((0, 2))
    for epsilon in corr_list:
        F_corr = (sum(q_eff)/(DFn*epsilon))/(sum(q_err)/(DFd*epsilon))
        pF_corr = 1-stats.f.cdf(F_corr, DFn*epsilon, DFd*epsilon)
        corr_table = np.vstack([corr_table, np.array([epsilon, pF_corr])])

    return [DFn, DFd, F, pF, W, pW], corr_table


def pairwise_ttest(data, dep_var, indep_var=None, id_var=None, wide=True, paired=True):
    """
    Posthoc pairwise t-tests for ANOVA
    
    ### Arguments:
    data: pandas DataFrame
    dep_var: dependent variable - label (long format) or a list of labels (wide format)
    indep_var: label of the independent variable (only necessary if data is in long format)
    id_var: label of the variable which contains the participants' identifiers. Default assumes that the table index
            contains the identifiers.
    wide: whether the data is in wide format
    paired: whether the samples are related
    
    ### Returns: pandas DataFrame with the t-statistics and associated p values (corrected and uncorrected) of each
                pairings
    """
    ### Reshaping data
    if wide:
        if not id_var:
            data = data.assign(ID=data.index)
            id_var = 'ID'
        data = pd.melt(data, id_vars=id_var, value_vars=dep_var, var_name='condition', value_name='measured')
        dep_var = 'measured'
        indep_var = 'condition'
    # Selecting test
    if paired:
        test = stats.ttest_rel
    else:
        test = stats.ttest_ind

    # Pairwise t-tests
    table = np.empty((0, 2))
    pairings = []
    for f in list(set(data[indep_var])):
        for f2 in list(set(data[indep_var])):
            if f != f2 and (f2, f) not in pairings:
                subset_f = data[data[indep_var] == f]
                subset_f2 = data[data[indep_var] == f2]
                table = np.vstack([table, np.asarray(test(subset_f[dep_var], subset_f2[dep_var]))])
                pairings.append((f, f2))

    # Corrections
    fam_size = (np.square(len(set(data[indep_var])))-len(set(data[indep_var])))/2
    bonf_list = []
    holm_list = []
    sorted_p = sorted(list(table[:, 1]))
    for p in table[:, 1]:
        bonf_list.append(min(p*fam_size, 1))
        holm_list.append(min(p*(fam_size-sorted_p.index(p)), 1))
    table = np.hstack([table, np.asarray(zip(bonf_list, holm_list))])
    table = pd.DataFrame(table, index=pd.MultiIndex.from_tuples(pairings), columns=['t', 'p', 'p (Bonf)', 'p (Holm)'])
    return table

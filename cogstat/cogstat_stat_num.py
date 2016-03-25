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
    :param confidence: sample size
    :return: low and high
    """
    delta = stats.norm.ppf(1.0 - (1 - confidence) / 2) / np.sqrt(n - 3)
    lower = np.tanh(np.arctanh(r) - delta)
    upper = np.tanh(np.arctanh(r) + delta)
    return lower, upper


def modified_t_test(x1, x2):
    """Compare a single case to a group.

    Crawford, J.R. & Howell, D.C. (1998). Comparing an individual's test score
    against norms derived from small samples. The Clinical Neuropsychologist,
    12, 482-486.

    :param x1, x2: data of two groups. One of them includes a single data,
    the other one includes multiple values
    :return tstat: test statistics
    :return pvalue: p value of the test
    :return df: degrees fo freedom
    """

    if len(x1) == 1:
        ind_data = x1
        group_data = x2.dropna()
    elif len(x2) == 1:
        ind_data = x2
        group_data = x1.dropna()
    else:
        raise ValueError('one of the groups should include only a single data')
    group_data_n = len(group_data)
    tstat = (ind_data.iloc[0] - np.mean(group_data)) / (np.std(group_data) * np.sqrt((group_data_n+1)/group_data_n))
    df = group_data_n-1
    pvalue = stats.t.sf(np.abs(tstat), df)*2  # two-sided
    return tstat, pvalue, df


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
    table = np.empty((0,n))
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
    corr_table = np.empty((0,2))
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
            if f != f2 and '%s - %s' % (f2, f) not in pairings:
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

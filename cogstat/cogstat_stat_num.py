# -*- coding: utf-8 -*-

"""
This module contains functions for statistical analysis that cannot be found in
other packages.

Arguments are the pandas data frame (pdf) and parameters.
Output is the result of the numerical analysis in numerical form.
"""

import numpy as np
from scipy import stats

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

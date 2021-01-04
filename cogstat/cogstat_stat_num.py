# -*- coding: utf-8 -*-

"""
This module contains functions for statistical analysis that cannot be found in
other packages.

Arguments are the pandas data frame (pdf) and parameters.
Output is the result of the numerical analysis in numerical form.
"""

from zipfile import ZipFile
import json
from tempfile import TemporaryDirectory
import struct
import os
import os.path

import numpy as np
from scipy import stats
import pandas as pd


def median_ci(data, quantile=0.5):
    """
    Calculate confidence interval of quantiles.

    Calculation is based on:
    https://www-users.york.ac.uk/~mb55/intro/cicent.htm

    Parameters
    ----------
    data : pandas.DataFrame
    quantile : float
        Quantile for which CI will be calculated. The default value is median (0.5).

    Returns
    -------
    np.array
        Lower and upper CI of the quantile for all variables.

    """
    n = len(data)
    lower_limit = (n * quantile) - (1.96 * np.sqrt(n * quantile * (1 - quantile)))
    upper_limit = 1 + (n * quantile) + (1.96 * np.sqrt(n * quantile * (1 - quantile)))
    median_ci_np = np.sort(data, axis=0)[[max(0, int(np.round(lower_limit-1))),
                                          min(n-1, int(np.round(upper_limit-1)))], :]
    if lower_limit < 1:
        median_ci_np[0] = np.nan
    if upper_limit > n + 1:
        median_ci_np[1] = np.nan
    return median_ci_np


def corr_ci(r, n, confidence=0.95):
    """
    Compute confidence interval for Spearman or Pearson correlation coefficients based on Fisher transformation.

    https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Using_the_Fisher_transformation

    Parameters
    ----------
    r : float
        Correlation coefficient
    n : int
        Sample size
    confidence : float
        Confidence, default is 0.95.

    Returns
    -------
    (float, float)
        Lower and upper limit of the CI
    """
    delta = stats.norm.ppf(1.0 - (1 - confidence) / 2) / np.sqrt(n - 3)
    lower = np.tanh(np.arctanh(r) - delta)
    upper = np.tanh(np.arctanh(r) + delta)
    return lower, upper


def modified_t_test(ind_data, group_data):
    """
    Compare a single case to a group.

    More information:
    Crawford, J.R. & Howell, D.C. (1998). Comparing an individual's test score
    against norms derived from small samples. The Clinical Neuropsychologist,
    12, 482-486.

    Parameters
    ----------
    ind_data : pandas.DataFrame
        Single row pandas data frame for the single case
    group_data : pandas.DataFrame
        Several rows of pandas data frame for the control group values

    Returns
    -------
    float, float, int

        - test statistics
        - p value of the test
        - degrees fo freedom
    """

    group_data_n = len(group_data)
    tstat = (ind_data.iloc[0] - np.mean(group_data)) / (np.std(group_data) * np.sqrt((group_data_n+1.0)/group_data_n))
    df = group_data_n-1
    pvalue = stats.t.sf(np.abs(tstat), df)*2  # two-sided
    return tstat, pvalue, df


def slope_extremity_test(n_trials, case_slope, case_SE, control_slopes, control_SEs):
    """
    Check the extremity of a single case performance expressed as a slope compared to the control data.

    More information:
    Crawford, J. R., & Garthwaite, P. H. (2004). Statistical Methods for Single-Case Studies in Neuropsychology:
    Comparing the Slope of a Patient’s Regression Line with those of a Control Sample. Cortex, 40(3), 533–548.
    http://doi.org/10.1016/S0010-9452(08)70145-X

    Parameters
    ----------
    n_trials : int
        Number of trials the slopes rely on.
    case_slope : pandas.DataFrame
        Single row pandas data frames with the slope of the single case.
    case_SE : pandas.DataFrame
        Single row pandas data frames with the standard error of the single case.
    control_slopes : pandas.DataFrame
        Single row pandas data frames with the slope of the control cases.
    control_SEs : pandas.DataFrame
        Single row pandas data frames with the standard error of the control cases.

    Returns
    -------
    float, int, float, str

        - test statistic value
        - degree of freedom
        - p-value
        - the chosen test type
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
    Perform standard one-way repeated measures ANOVA.
    
    Parameters
    ----------
    data : pandas DataFrame
    dep_var : str or list of str
        Dependent variable - label (long format) or a list of labels (wide format)
    indep_var : str
        Label of the independent variable (only necessary if data is in long format)
    id_var: str
        Label of the variable which contains the participants' identifiers. Default assumes that the table index
        contains the identifiers.
    wide : bool
        Whether the data is in wide format.
    
    Returns
    -------
    list of [int, int, float, float, sloat, float], numpy.array

        - [DFn, DFd, F, pF, W, pW], corr_table
        - DFn, DFd: degrees of freedom used when calculating p(F)
        - F: F statistic (uncorrected)
        - pF: p value of the F statistic (uncorrected)
        - W: statistic of Mauchly's test for sphericity
        - pW: p value of Mauchly's test for sphericity (sphericity is violated if p(W)<0.05)
        - corr_table: contains Greenhouse & Geisser's, Huhyn & Feldt's, and the "lower-bound" epsilons, and the
          corrected p values of F (degrees of freedom should be multiplied with the epsilon to get the corrected df
          values)
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
            q_err.insert(j+i, np.square(np.mean(subset_ij[dep_var]) - np.mean(subset_i[dep_var]) -
                                        np.mean(subset_j[dep_var]) + np.mean(data[dep_var])))
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
    W = np.prod([x for x in list(np.linalg.eigvals(pop_table)) if x > 0.00000000001]) / \
        np.power(np.trace(pop_table) / (k-1), (k-1))  # uses the pseudo-determinant (discards all near-zero eigenvalues)
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
    Calculate posthoc pairwise t-tests for ANOVA.
    
    Parameters
    ----------
    data : pandas DataFrame
    dep_var : str or list of str
        Dependent variable - label (long format) or a list of labels (wide format)
    indep_var : str
        Label of the independent variable (only necessary if data is in long format)
    id_var: str
        Label of the variable which contains the participants' identifiers. Default assumes that the table index
        contains the identifiers.
    wide : bool
        Whether the data is in wide format.
    paired : bool
        Whether the samples are related.

    Returns
    -------
    pandas.DataFrame
        t-statistics and associated p values (corrected and uncorrected) of each pairings
    """
    # TODO keep the order of the dep_vars
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
    table = np.hstack([table, np.asarray(list(zip(bonf_list, holm_list)))])
    table = pd.DataFrame(table, index=pd.MultiIndex.from_tuples(pairings), columns=['t', 'p', 'p (Bonf)', 'p (Holm)'])
    return table


def diffusion_edge_correction_mean(data):
    """
    For behavioral data calculate mean error rate with edge correction for the EZ  diffusion analysis.


    See more details at: Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007). An EZ-diffusion
    model for response time and accuracy. Psychonomic Bulletin & Review, 14(1), 3–22. https://doi.org/10.3758/BF03194023

    Parameters
    ----------
    data :
        Values (error rates) to be corrected

    Returns
    -------

        Corrected mean error rate
    """
    mean = np.mean(data)
    if mean == 0:
        mean = 1.0/(len(data)*2)
    elif mean == 1:
        mean = 1-1.0/(len(data)*2)
    elif mean == 0.5:
        mean = 1-1.0/(len(data)*2)
    return mean


def diffusion_get_ez_params(Pc, VRT, MRT, s=0.1):
    """
    Recover the diffusion parameters for behavioral data with the EZ method.

    See more details at: Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P. (2007).
    An EZ-diffusion model for response time and accuracy. Psychonomic Bulletin & Review, 14(1), 3–22.
    https://doi.org/10.3758/BF03194023

    Parameters
    ----------
    Pc : float
        Percent correct
    VRT : float
        Correct reaction time variance
    MRT : float
        Mean correct reaction time
    s : float


    Returns
    -------
    float, float, float
        - drift rate
        - threshold
        - nondecision time

    Examples
    --------
    Example of the paper Wagenmakers, van der Maas, Grasman (2007).

    It should return 0.09993853, 0.1399702, 0.30003

    >>> diffusion_get_ez_params(0.802, .112, .723)
    """
    # This has been handled by the edge correction
    #if Pc == 0 or Pc == 0.5 or Pc == 1:
        #pass
        #print 'Oops, invalid Pc value: %s!'%Pc
    v = np.sign(Pc-.5)*s*((np.log(Pc/(1-Pc)))*((np.log(Pc/(1-Pc)))*Pc**2 -
                                               (np.log(Pc/(1-Pc)))*Pc + Pc - .5)/VRT)**(0.25)  # This gives drift rate.
    a = s**2*np.log(Pc/(1-Pc))/v  # This gives boundary separation.
    ter = MRT - (a/(2*v)) * (1-np.exp(-v*a/s**2))/(1+np.exp(-v*a/s**2))  # This gives nondecision time.

    return v, a, ter


def read_jasp_file(path):
    """
    Open JASP file.

    The code is based on the jasp import filter in jamovi:
    https://github.com/jamovi/jamovi/blob/master/server/jamovi/server/formatio/jasp.py

    Parameters
    ----------
    path : str
        Path of the jasp file.
    Returns
    -------
    pandas.DataFrame and list of {'nom', 'ord', 'int'}
        Returns the values, variable names and measurement levels
    """

    with ZipFile(path, 'r') as zip:

        meta_dataset = json.loads(zip.read('metadata.json').decode('utf-8'))['dataSet']

        # Set variable names and measurement types
        column_names = [meta_column['name'] for meta_column in meta_dataset['fields']]
        jasp_to_cs_measurement_levels = {'Nominal': 'nom', 'NominalText': 'nom', 'Ordinal': 'ord', 'Continuous': 'int'}
        meas_levs = [jasp_to_cs_measurement_levels[meta_column['measureType']] for meta_column in meta_dataset['fields']]

        # TODO labels
        """
        try:
            xdata_content = zip.read('xdata.json').decode('utf-8')
            xdata = json.loads(xdata_content)

            for column_name in column_names:
                if column_name in xdata:
                    meta_labels = xdata[column_name]['labels']
                    for meta_label in meta_labels:
                        TODO_store_levels(meta_label[0], meta_label[1], str(meta_label[0]))
        except Exception:
            pass
        """

        row_count = meta_dataset['rowCount']
        pdf = pd.DataFrame(columns=column_names, index=range(row_count))

        with TemporaryDirectory() as dir:
            zip.extract('data.bin', dir)
            data_path = os.path.join(dir, 'data.bin')
            data_file = open(data_path, 'rb')

            for col_i, meas_lev in enumerate(meas_levs):
                if meas_lev == 'int':
                    for i in range(row_count):
                        byts = data_file.read(8)
                        value = struct.unpack('<d', byts)
                        pdf.iloc[i, col_i] = value[0]
                else:
                    for i in range(row_count):
                        byts = data_file.read(4)
                        value = struct.unpack('<i', byts)
                        pdf.iloc[i, col_i] = value[0]
            data_file.close()

        return (pdf, meas_levs)


def read_jamovi_file(path):
    """
    Open jamovi file.

    The code is based on jamovi file import:
    https://github.com/jamovi/jamovi/blob/master/server/jamovi/server/formatio/omv.py

    Parameters
    ----------
    path : str
        Path of the jasp file.
    Returns
    -------
    pandas.DataFrame and list of {'nom', 'ord', 'int'}
        Returns the values, variable names and measurement levels

    """

    def _read_string_from_table(stream, pos):
        _buffer = bytearray(512)
        final_pos = stream.seek(pos)
        if pos != final_pos:
            return ''
        stream.readinto(_buffer)
        try:
            end = _buffer.index(bytes(1))  # find string terminator
            return _buffer[0:end].decode('utf-8', errors='ignore')
        except ValueError:
            return _buffer.decode('utf-8', errors='ignore')

    with ZipFile(path, 'r') as zip:
        meta_dataset = json.loads(zip.read('metadata.json').decode('utf-8'))['dataSet']

        # TODO
        """
        if 'transforms' in meta_dataset:
            for meta_transform in meta_dataset['transforms']:
                name = meta_transform['name']
                transform = data.append_transform(name)
                measure_type_str = meta_transform.get('measureType', 'None')
                transform.measure_type = MeasureType.parse(measure_type_str)
        """

        column_names = [meta_column['name'] for meta_column in meta_dataset['fields']]
        row_count = meta_dataset['rowCount']
        pdf = pd.DataFrame(columns=column_names, index=range(row_count))

        jamovi_to_cs_measurement_levels = {'ID': 'nom', 'Nominal': 'nom', 'NominalText': 'nom',
                                           'Ordinal': 'ord', 'Continuous': 'int'}
        meas_levs = [jamovi_to_cs_measurement_levels[meta_column['measureType']]
                     for meta_column in meta_dataset['fields']]

        # TODO
        #missing_values = meta_column.get('missingValues', [])

        # TODO labels
        """
        try:
            xdata_content = zip.read('xdata.json').decode('utf-8')
            xdata = json.loads(xdata_content)

            for column in data:
                if column.name in xdata:
                    try:
                        meta_labels = xdata[column.name]['labels']
                        if meta_labels:
                            for meta_label in meta_labels:
                                import_value = meta_label[1]
                                if len(meta_label) > 2:
                                    import_value = meta_label[2]
                                column.append_level(meta_label[0], meta_label[1],  import_value)
                    except Exception:
                        pass
        except Exception:
            pass
        """

        with TemporaryDirectory() as dir:
            zip.extract('data.bin', dir)
            data_file = open(os.path.join(dir, 'data.bin'), 'rb')

            try:
                zip.extract('strings.bin', dir)
                string_table_present = True
                string_table = open(os.path.join(dir, 'strings.bin'), 'rb')
            except Exception:
                string_table_present = False

            BUFF_SIZE = 65536
            buff = memoryview(bytearray(BUFF_SIZE))

            for column_i, column in enumerate(meta_dataset['fields']):

                if column['dataType'] == 'Decimal':
                    elem_fmt = '<d'
                    elem_width = 8
                    transform = None
                elif column['dataType'] == 'Text' and column['measureType'] == 'ID':
                    elem_fmt = '<i'
                    elem_width = 4
                    if string_table_present:
                        def transform(x):
                            if x == -2147483648:
                                return ''
                            else:
                                return _read_string_from_table(string_table, x)
                    else:
                        def transform(x):
                            if x == -2147483648:
                                return ''
                            else:
                                return str(x)
                else:
                    elem_fmt = '<i'
                    elem_width = 4
                    transform = None

                for row_offset in range(0, row_count, int(BUFF_SIZE / elem_width)):
                    n_bytes_to_read = min(elem_width * (row_count - row_offset), BUFF_SIZE)
                    buff_view = buff[0:n_bytes_to_read]
                    data_file.readinto(buff_view)

                    # TODO
                    """
                    if transform:
                        for i, values in enumerate(struct.iter_unpack(elem_fmt, buff_view)):
                            pdf.iloc[row_offset + i, column_i] = transform(values[0])
                    else:
                    """
                    for i, values in enumerate(struct.iter_unpack(elem_fmt, buff_view)):
                        pdf.iloc[row_offset + i, column_i] = values[0]

            data_file.close()
            if string_table_present:
                string_table.close()

    return pdf, meas_levs
# -*- coding: utf-8 -*-

import unittest
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
print(sys.path)
from pathlib import Path
import numpy as np
import pandas as pd
from cogstat import cogstat as cs

print(cs.__file__)
print(cs.__version__)
print(os.path.abspath(cs.__file__))

"""
- All statistical value should be tested at least once.
- All leafs of the decision tree should be tested once.
- Tests shouldn't give p<0.001 results, because exact values cannot be tested. 
- No need to test the details of the statistical methods imported from other modules, 
because that is the job of that specific module.
- All data variables should be used with 3 digits decimal precision, to ensure that copying 
the data for validation no additional rounding happens.
- Validated results should be added as a comment here: validating software, version, optionally information about how thw analysis can be run, result, any comments
"""

#cs.output_type = 'do not format'

np.random.seed(555)
# https://docs.scipy.org/doc/numpy/reference/routines.random.html
# Make sure to use round function to have the same precision of the data when copied to other software
data_np = np.vstack((
    np.round(np.random.normal(loc=3, scale=3, size=30), 3),
    np.round(np.random.lognormal(mean=3, sigma=3, size=30), 3),
    np.random.randint(3, size=30),
    np.random.randint(3, size=30),
    np.round(np.random.normal(loc=3, scale=3, size=30), 3),
    np.round(np.random.lognormal(mean=1.4, sigma=0.6, size=30), 3),
    np.round(np.random.normal(loc=6, scale=3, size=30), 3),
    np.round(np.random.normal(loc=7, scale=6, size=30), 3),
    np.random.randint(2, size=30),
    np.random.randint(2, size=30),
    np.random.randint(2, size=30),
    np.concatenate((np.round(np.random.normal(loc=3, scale=3, size=15), 3),
                    np.round(np.random.normal(loc=4, scale=3, size=15), 3))),
    np.array([1]*15+[2]*15),
    np.array([1]+[2]*29),
    np.concatenate((np.round(np.random.normal(loc=3, scale=3, size=15), 3),
                    np.round(np.random.lognormal(mean=1.5, sigma=2.0, size=15), 3))),
    np.concatenate((np.round(np.random.normal(loc=3, scale=3, size=15), 3),
                    np.round(np.random.normal(loc=3, scale=7, size=15), 3))),
    np.array([1]*10+[2]*8+[3]*12),
    np.concatenate((np.round(np.random.normal(loc=3, scale=3, size=10), 3),
                    np.round(np.random.normal(loc=3, scale=3, size=8), 3),
                    np.round(np.random.normal(loc=6, scale=3, size=12), 3)))
    ))
data_pd = pd.DataFrame(data_np.T, columns=
                       ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r'])
data = cs.CogStatData(data=data_pd, measurement_levels=['int', 'int', 'nom', 'nom', 'int', 'int', 'int', 'int', 'nom',
                                                       'nom', 'nom', 'int', 'nom', 'nom', 'int', 'int', 'int', 'int'])

#pd.set_option('display.expand_frame_repr', False)
#print (data_pd)

class CogStatTestCase(unittest.TestCase):
    """Unit tests for CogStat."""

    def test_explore_variables(self):
        """Test explore variables"""

        # Int variable
        result = data.explore_variable('a', 1, 2.0)
        #for i, res in enumerate(result): print(i, res)
        self.assertTrue('N of valid cases: 30' in result[2])
        self.assertTrue('N of missing cases: 0' in result[2])
            # jamovi 2.0.0.0 3.14
        self.assertTrue('<td>Mean</td>      <td>3.1438</td>' in result[4])
            # LibreOffice 7.1.5.2 stdev.p() 3.21518743552465
        self.assertTrue('<td>Standard deviation</td>      <td>3.2152</td>' in result[4])

            # jamovi 2.0.0.0 0.359
        self.assertTrue('<td>Skewness</td>      <td>0.3586</td>' in result[4])
            # jamovi 2.0.0.0 0.0446
        self.assertTrue('<td>Kurtosis</td>      <td>0.0446</td>' in result[4])
            # jamovi 2.0.0.0 12.8
        self.assertTrue('<td>Range</td>      <td>12.7840</td>' in result[4])
            # jamovi 2.0.0.0 9.98
        self.assertTrue('<td>Maximum</td>      <td>9.9810</td>' in result[4])
            # jamovi 2.0.0.0 4.39
        self.assertTrue('<td>Upper quartile</td>      <td>4.3875</td>' in result[4])
            # jamovi 2.0.0.0 2.85
        self.assertTrue('<td>Median</td>      <td>2.8545</td>' in result[4])
            # jamovi 2.0.0.0 1.42
        self.assertTrue('<td>Lower quartile</td>      <td>1.4190</td>' in result[4])
            # jamovi 2.0.0.0 -2.80
        self.assertTrue('<td>Minimum</td>      <td>-2.8030</td>' in result[4])
        # Shapiro–Wilk normality
            # jamovi 2.0.0.0 0.959
        self.assertTrue('<i>W</i> = 0.96' in result[6])  # <i>W</i> = 0.959
            # jamovi 2.0.0.0 0.287
        self.assertTrue('<i>p</i> = .287' in result[6])

        # Population estimation and one sample t-test
            # jamovi 2.0.0.0 CI 1.97, 4.31 - based on Z-value
            # https://www.statskingdom.com/confidence-interval-calculator.html To use the t-value based solution, do not
            # use the population SD
            # Mean confidence interval: [1.922672 , 4.364861]
        self.assertTrue('<td>Mean</td>      <td>3.1438</td>      <td>1.9227</td>      <td>4.3649</td>' in result[9])
            # jamovi 2.0.0.0 3.27 - SD estimates population SD
            # https: // www.statskingdom.com / confidence - interval - calculator.html To use the t-value based solution, do not
            # use the population SD
            # Standard deviation confidence interval: [2.604372 , 4.396115]
        self.assertTrue('<td>Standard deviation</td>      <td>3.2702</td>      <td>2.6044</td>      <td>4.3961</td>' in result[9])
        # Sensitivity power analysis
            # G*Power 3.1.9.6: 0.6811825
            # jamovi v1.2.19.0, jpower 0.1.2: 0.681
        self.assertTrue('(effect size is in d): 0.68' in result[11])
            # Note that the test value is 2 here.
            # jamovi 2.0.0.0 1.92, 0.065
        self.assertTrue('t</i>(29) = 1.92' in result[11])
        self.assertTrue('p</i> = .065' in result[11])

        # Wilcoxon signed-rank test for non-normal interval variable
        result = data.explore_variable('b', 0, 20.0)
            # jamovi 2.0.0.0 W(!) 262, p 0.556
            # TODO not sure yet why the difference, see https://github.com/cogstat/cogstat/issues/31
        self.assertTrue('T</i> = 203' in result[11])
        self.assertTrue('p</i> = .551' in result[11])

        # Ord variable
        data.data_measlevs['a'] = 'ord'
        result = data.explore_variable('a', 1, 2.0)
        self.assertTrue('N of valid cases: 30' in result[2])
        self.assertTrue('N of missing cases: 0' in result[2])
        self.assertTrue('<td>Maximum</td>      <td>9.9810</td>' in result[4])
        self.assertTrue('<td>Upper quartile</td>      <td>4.3875</td>' in result[4])
        self.assertTrue('<td>Median</td>      <td>2.8545</td>' in result[4])
        self.assertTrue('<td>Lower quartile</td>      <td>1.4190</td>' in result[4])
        self.assertTrue('<td>Minimum</td>      <td>-2.8030</td>' in result[4])
        # TODO median CI
        # Wilcoxon signed-rank test
            # jamovi 2.0.0.0 W(!) 320, p 0.073
            # TODO https://github.com/cogstat/cogstat/issues/31
        self.assertTrue('T</i> = 145' in result[9])
        self.assertTrue('p</i> = .074' in result[9])
        data.data_measlevs['a'] = 'int'

        # Nominal variable
        #result = data.explore_variable('c')
        # TODO variation ratio
        # TODO multinomial proportion CI

    def test_explore_variable_pairs(self):
        """Test explore variable pairs"""

        # Int variables
        result = data.explore_variable_pair('a', 'b')
        self.assertTrue('N of valid pairs: 30' in result[1])
        self.assertTrue('N of missing pairs: 0' in result[1])
            # jamovi 2.0.0.0 -0.141
        self.assertTrue('<i>r</i> = -0.141' in result[4])
            # jamovi 2.0.0.0 [-0.477, 0.231]
        self.assertTrue('[-0.477, 0.231]' in result[6])
            # jamovi 2.0.0.0 p 0.456
        self.assertTrue("Pearson's correlation: <i>r</i>(28) = -0.14, <i>p</i> = .456" in result[7])  # <i>r</i>(28) = -0.141
            # jamovi 2.0.0.0 A -21.8 Intercept 300.5
        self.assertTrue('y = -21.811x + 300.505' in result[3])
            # jamovi 2.0.0.0 -0.363
        self.assertTrue('-0.363' in result[4])
            # TODO validate
        self.assertTrue('[-0.640, -0.003]' in result[6])
            # jamovi 2.0.0.0 p 0.049 (0.04919 with more precision) TODO
        self.assertTrue("Spearman's rank-order correlation: <i>r<sub>s</sub></i>(28) = -0.36, <i>p</i> = .048" in result[7])  # <i>r<sub>s</sub></i>(28) = -0.363

        # Ord variables
        data.data_measlevs['a'] = 'ord'
        data.data_measlevs['b'] = 'ord'
        result = data.explore_variable_pair('a', 'b')
        self.assertTrue('-0.363' in result[4])
        self.assertTrue('[-0.640, -0.003]' in result[5])
        self.assertTrue("Spearman's rank-order correlation: <i>r<sub>s</sub></i>(28) = -0.36, <i>p</i> = .048" in result[6])  # <i>r<sub>s</sub></i>(28) = -0.363
        data.data_measlevs['a'] = 'int'
        data.data_measlevs['b'] = 'int'

        # Nom variables
        result = data.explore_variable_pair('c', 'd')
        self.assertTrue('N of valid pairs: 30' in result[1])
        self.assertTrue('N of missing pairs: 0' in result[1])
        # Cramer's V
            # jamovi 2.0.0.0 0.372
        self.assertTrue('<sub>c</sub></i> = 0.372' in result[4])
        # Sensitivity power analysis
            # G*Power 3.1.9.6, Goodness of fit test, df=4: Contingency tables: 0.7868005
            #  TODO GPower gives 0.8707028 with df of 8; Seems like statsmodels GofChisquarePower calculates power
            #  with df=8; should we use 4 or 8 df? https://github.com/cogstat/cogstat/issues/134
        self.assertTrue('(effect size is in w): 0.87' in result[6])
        # Chi-squared
            # jamovi v1.2.19.0: X2, df, p, N: 8.31, 4, 0.081, 30
        self.assertTrue('(4, <i>N</i> = 30) = 8.31' in result[6])  # (4, <i>N</i> = 30) = 8.312
        self.assertTrue('<i>p</i> = .081' in result[6])

    def test_diffusion(self):
        """Test diffusion analysis"""
        data_diffusion = cs.CogStatData(data=str(Path('data/diffusion.csv')))
        result = data_diffusion.diffusion(error_name=['Error'], RT_name=['RT_sec'], participant_name=['Name'], condition_names=['Num1', 'Num2'])
        # Drift rate
        self.assertTrue('<td>zsiraf</td>      <td>0.190</td>      <td>0.276</td>      <td>0.197</td>      <td>0.235</td>      <td>0.213</td>' in result[1])
        # Threshold
        self.assertTrue('<td>zsiraf</td>      <td>0.178</td>      <td>0.096</td>      <td>0.171</td>      <td>0.112</td>      <td>0.088</td>' in result[1])
        # Nondecision time
        self.assertTrue('<td>zsiraf</td>      <td>0.481</td>      <td>0.590</td>      <td>0.483</td>      <td>0.561</td>      <td>0.522</td>' in result[1])

    def test_compare_variables(self):
        """Test compare variables"""

        # 2 Int variables
        result = data.compare_variables(['a', 'e'])
        self.assertTrue('N of valid cases: 30' in result[1])
        self.assertTrue('N of missing cases: 0' in result[1])
        # Cohen's d
            # CS formula: https://pingouin-stats.org/generated/pingouin.compute_effsize.html
            # Based on the formula, calculated in LO Calc 6.4: 0.030004573510063
            # jamovi v1.2.19.0: 0.0202; formula: https://github.com/jamovi/jmv/blob/master/R/ttestps.b.R#L54-L66
        self.assertTrue("<td>Cohen's d</td>      <td>0.030</td>" in result[3])
        # eta-squared
            # CS formula: https://pingouin-stats.org/generated/pingouin.convert_effsize.html
            # Based on the formula, calculated in LO Calc 6.4: 0.0002250179634
            # jamovi v1.2.19.0: 0.000
        self.assertTrue('<td>Eta-squared</td>      <td>0.000</td>' in result[3])
        # Sample means
            # jamovi 2.0.0.0 3.14, 3.05
        self.assertTrue('<td>3.1438</td>      <td>3.0502</td>' in result[3])
        # Hedges'g (with CI)
            # CS formula: https://pingouin-stats.org/generated/pingouin.compute_effsize.html
            # https://pingouin-stats.org/generated/pingouin.compute_esci.html
            # Note that the latter (CI) method has changed in v0.3.5 https://pingouin-stats.org/changelog.html
            # Based on the formula, calculated in LO Calc 7.0: 0.029614903724218, -0.34445335392457, 0.403683161373007
            # Note that the last value is 0.404 in LO, not .403 as in pingouin TODO
        self.assertTrue("<td>Hedges' g</td>      <td>0.030</td>      <td>-0.344</td>      <td>0.403</td>" in result[5])
            # jamovi 2.0.0.0 0.954 0.215
        self.assertTrue('<i>W</i> = 0.95, <i>p</i> = .215' in result[7])  # <i>W</i> = 0.954
        # Sensitivity power analysis
            # G*Power 3.1.9.6: 0.6811825
            # jamovi v1.2.19.0, jpower 0.1.2: 0.681
        self.assertTrue('(effect size is in d): 0.68' in result[7])
        # Paired samples t-test
            # jamovi v1.2.19.0: t, df, p: 0.110, 29.0, 0.913
        self.assertTrue('<i>t</i>(29) = 0.11, <i>p</i> = .913' in result[7])

        # 2 Int variables - non-normal
        result = data.compare_variables(['e', 'f'])
            # jamovi 2.0.0.0 0.915, 0.019
        self.assertTrue('<i>W</i> = 0.91, <i>p</i> = .019' in result[7])  # <i>W</i> = 0.915
        # Wilcoxon signed-rank test
            # jamovi 2.0.0.0 110, 0.011 (0.01060 with more precision) TODO https://github.com/cogstat/cogstat/issues/31
        #print(result[7])
        self.assertTrue('<i>T</i> = 110.00, <i>p</i> = .012' in result[7])

        # 3 Int variables
        result = data.compare_variables(['a', 'e', 'g'])
            # jamovi 2.0.0.0 3.14, 3.05, 5.73
        self.assertTrue('<td>3.1438</td>      <td>3.0502</td>      <td>5.7295</td>' in result[3])
            # TODO validate
        self.assertTrue('a: <i>W</i> = 0.96, <i>p</i> = .287' in result[7])  # <i>W</i> = 0.959
        self.assertTrue('e: <i>W</i> = 0.97, <i>p</i> = .435' in result[7])  # <i>W</i> = 0.966
        self.assertTrue('g: <i>W</i> = 0.95, <i>p</i> = .133' in result[7])  # <i>W</i> = 0.946
            # jamovi 2.0.0.0 0.975 0.703
        self.assertTrue('sphericity: <i>W</i> = 0.98, <i>p</i> = .703' in result[7])  # <i>W</i> = 0.975
            # jamovi 2.0.0.0 6.16 0.004
        self.assertTrue('<i>F</i>(2, 58) = 6.17, <i>p</i> = .004' in result[7])
            # jamovi 2.0.0.0 (Holm correction) 0.110, 0.913; 3.167, 0.011; 2.883, 0.015
        self.assertTrue('0.11, <i>p</i> = .913' in result[7])  # TODO keep the order of the variables, and have a fixed sign
        self.assertTrue('3.17, <i>p</i> = .011' in result[7])
        self.assertTrue('2.88, <i>p</i> = .015' in result[7])

        # 3 Int variables, sphericity violated
        result = data.compare_variables(['a', 'e', 'h'])
        self.assertTrue('<td>3.1438</td>      <td>3.0502</td>      <td>6.5786</td>' in result[3])
        self.assertTrue('a: <i>W</i> = 0.96, <i>p</i> = .287' in result[7])  # <i>W</i> = 0.959
        self.assertTrue('e: <i>W</i> = 0.97, <i>p</i> = .435' in result[7])  # <i>W</i> = 0.966
        self.assertTrue('h: <i>W</i> = 0.98, <i>p</i> = .824' in result[7])
            # jamovi 2.0.0.0  0.793, 0.039
        self.assertTrue('sphericity: <i>W</i> = 0.79, <i>p</i> = .039' in result[7])  # <i>W</i> = 0.793
            # jamovi 2.0.0.0 1.66, 48.04, 6.16, 0.007
        self.assertTrue('<i>F</i>(1.66, 48) = 6.16, <i>p</i> = .007' in result[7])
            # jamovi 2.0.0.0 (Holm correction) 0.110, 0.913; 2.678, 0.026; 2.809, 0.026
        self.assertTrue('0.11, <i>p</i> = .913' in result[7])  # TODO keep the order of the variables, and have a fixed sign
        self.assertTrue('2.68, <i>p</i> = .026' in result[7])
        self.assertTrue('2.81, <i>p</i> = .026' in result[7])

        # 3 Int variables, non-normal
        result = data.compare_variables(['a', 'e', 'f'])
        self.assertTrue('<td>3.1438</td>      <td>3.0502</td>      <td>5.3681</td>' in result[3])
        self.assertTrue('a: <i>W</i> = 0.96, <i>p</i> = .287' in result[7])  # <i>W</i> = 0.959
        self.assertTrue('e: <i>W</i> = 0.97, <i>p</i> = .435' in result[7])  # <i>W</i> = 0.966
        self.assertTrue('f: <i>W</i> = 0.82, <i>p</i> &lt; .001' in result[7])  # <i>W</i> = 0.818
            # jamovi 2.0.0.0 df 2, khi2 6.47, p 0.039
        self.assertTrue('&chi;<sup>2</sup>(2, <i>N</i> = 30) = 6.47, <i>p</i> = .039' in result[7])

        # 2 × 2 Int variables
        result = data.compare_variables(['a', 'b', 'e', 'f'], factors=[['first', 2], ['second', 2]])
            # jamovi 2.0.0.0 6.06, 0.020; 6.29, 0.018; 6.04, 0.020
        self.assertTrue('Main effect of first: <i>F</i>(1, 29) = 6.06, <i>p</i> = .020' in result[7])
        self.assertTrue('Main effect of second: <i>F</i>(1, 29) = 6.29, <i>p</i> = .018' in result[7])
        self.assertTrue('Interaction of factors first, second: <i>F</i>(1, 29) = 6.04, <i>p</i> = .020' in result[7])

        # 2 Ord variables
        data.data_measlevs['a'] = 'ord'
        data.data_measlevs['e'] = 'ord'
        data.data_measlevs['f'] = 'ord'
        result = data.compare_variables(['e', 'f'])
        self.assertTrue('<td>2.3895</td>      <td>4.2275</td>' in result[3])
        # Wilcoxon signed-rank test
            # jamovi 2.0.0.0 110, 0.011 TODO
        self.assertTrue('<i>T</i> = 110.00, <i>p</i> = .012' in result[6])

        # 3 Ord variables
        result = data.compare_variables(['a', 'e', 'f'])
        self.assertTrue('<td>2.8545</td>      <td>2.3895</td>      <td>4.2275</td>' in result[3])
            # jamovi 2.0.0.0 6.47, 0.039
        self.assertTrue('&chi;<sup>2</sup>(2, <i>N</i> = 30) = 6.47, <i>p</i> = .039' in result[6])
        data.data_measlevs['a'] = 'int'
        data.data_measlevs['e'] = 'int'
        data.data_measlevs['f'] = 'int'

        # 2 Nom variables
        result = data.compare_variables(['i', 'j'])
        # TODO on Linux the row labels are 0.0 and 1.0 instead of 0 and 1
        self.assertTrue('<td>0.0</td>      <td>4</td>      <td>9</td>      <td>13</td>    </tr>    <tr>      <td>1.0</td>      <td>9</td>' in result[3])
        # McNemar
            # jamovi 2.0.0.0 0.00, 1.000 TODO https://github.com/cogstat/cogstat/issues/55
        self.assertTrue('&chi;<sup>2</sup>(1, <i>N</i> = 30) = 0.06, <i>p</i> = .814' in result[5])  # &chi;<sup>2</sup>(1, <i>N</i> = 30) = 0.0556

        # 3 Nom variables
        result = data.compare_variables(['i', 'j', 'k'])
        # Cochran's Q
            # TODO validate
        self.assertTrue('<i>Q</i>(2, <i>N</i> = 30) = 0.78, <i>p</i> = .676' in result[7])  # <i>Q</i>(2, <i>N</i> = 30) = 0.783

    def test_compare_groups(self):
        """Test compare groups"""

        # 2 Int groups
        result = data.compare_groups('l', ['m'])
            # jamovi 2.0.0.0 2.53,4.58
        self.assertTrue('<td>2.5316</td>      <td>4.5759</td>' in result[3])
        # Cohen's d
            # CS formula: https://pingouin-stats.org/generated/pingouin.compute_effsize.html
            # Based on the formula, calculated in LO Calc 6.4: -0.704171924382848
            # jamovi v1.2.19.0: 0.0704
        self.assertTrue("<td>Cohen's d</td>      <td>-0.704</td>" in result[3])
        # eta-squared
            # CS formula: https://pingouin-stats.org/generated/pingouin.convert_effsize.html
            # Based on the formula, calculated in LO Calc 6.4: 0.110292204104377
            # jamovi v1.2.19.0: 0.117 # TODO why the difference?
        self.assertTrue('<td>Eta-squared</td>      <td>0.110</td>' in result[3])
        # Hedges'g (with CI)
            # CS formula: https://pingouin-stats.org/generated/pingouin.compute_effsize.html
            # https://pingouin-stats.org/generated/pingouin.compute_esci.html
            # Note that the latter (CI) method has changed in v0.3.5 https://pingouin-stats.org/changelog.html
            # Based on the formula, calculated in LO Calc 7.0: -0.685140250750879, -1.45474443187683, 0.084463930375068
        self.assertTrue('<td>Difference between the two groups:</td>      <td>-2.0443</td>      <td>-4.2157</td>      <td>0.1272</td>' in result[5])
        self.assertTrue("<td>Hedges' g</td>      <td>-0.685</td>      <td>-1.455</td>      <td>0.084</td>" in result[6])
            # TODO validate
        self.assertTrue('(m: 1.0): <i>W</i> = 0.96, <i>p</i> = .683' in result[8])  # <i>W</i> = 0.959
        self.assertTrue('(m: 2.0): <i>W</i> = 0.98, <i>p</i> = .991' in result[8])  # <i>W</i> = 0.984
        self.assertTrue('<i>W</i> = 0.30, <i>p</i> = .585' in result[8])  # <i>W</i> = 0.305
        # Sensitivity power analysis
            # G*Power 3.1.9.6: 1.3641059
            # jamovi v1.2.19.0, jpower 0.1.2: 1.36
        self.assertTrue('(effect size is in d): 1.36' in result[8])
        # independent samples t-test
            # jamovi v1.2.19.0: t, df, p: -1.93, 28.0, 0.064
        self.assertTrue('<i>t</i>(28) = -1.93, <i>p</i> = .064' in result[8])

        # Non-normal group
        result = data.compare_groups('o', ['m'])
        self.assertTrue('(m: 2.0): <i>W</i> = 0.81, <i>p</i> = .005' in result[8])  # <i>W</i> = 0.808
        # Mann-Whitney
            # jamovi 2.0.0.0 51.0, 0.010 TODO
        self.assertTrue('<i>U</i> = 51.00, <i>p</i> = .011' in result[8])

        # Heteroscedastic groups
        result = data.compare_groups('p', ['m'])
        # Welch's t-test
            # jamovi 2.0.0.0 0.119, 0.907
        self.assertTrue('<i>t</i>(25.3) = 0.12, <i>p</i> = .907' in result[8])  # <i>t</i>(25.3) = 0.119


        # TODO single case vs. group

        # 3 Int groups
        result = data.compare_groups('r', ['q'])
            # jamovi 2.0.0.0 3.29, 5.04, 7.24
        self.assertTrue('<td>3.2869</td>      <td>5.0400</td>      <td>7.2412</td>' in result[3])
            # TODO validate
        self.assertTrue('&omega;<sup>2</sup> = 0.167' in result[6])
        # Levene's test for homogeneity
            # jamovi 2.0.0.0 0.495, 0.615 TODO https://github.com/cogstat/cogstat/issues/56
        self.assertTrue('<i>W</i> = 0.68, <i>p</i> = .517' in result[8])  # <i>W</i> = 0.675
        # Sensitivity power analysis
            # G*Power 3.1.9.6: 0.7597473
        self.assertTrue('(effect size is in f): 0.76' in result[8])
            # jamovi 2.0.0.0 4.00, 0.030
        self.assertTrue('<i>F</i>(2, 27) = 4.00, <i>p</i> = .030' in result[8])
        # TODO post-hoc

        # 3 Int groups with assumption violation
        result = data.compare_groups('o', ['q'])
        # Kruskal-Wallis
            # jamovi 2.0.0.0 8.37, 0.015
        self.assertTrue('&chi;<sup>2</sup>(2, <i>N</i> = 30) = 8.37, <i>p</i> = .015' in result[8])

        # 2 Ord groups
        data.data_measlevs['o'] = 'ord'
        result = data.compare_groups('o', ['m'])
        # Mann-Whitney
            # jamovi 2.0.0.0 51.0, 0.010 TODO
        self.assertTrue('<i>U</i> = 51.00, <i>p</i> = .011' in result[6])

        # 3 Ord groups
        data.data_measlevs['o'] = 'ord'
        result = data.compare_groups('o', ['q'])
        # Kruskal-Wallis
            # jamovi 2.0.0.0 8.37, 0.015
        self.assertTrue('&chi;<sup>2</sup>(2, <i>N</i> = 30) = 8.37, <i>p</i> = .015' in result[6])
        data.data_measlevs['o'] = 'int'

        # 2 Nom groups
        result = data.compare_groups('i', ['j'])
        # Cramer's V
            # jamovi 2.0.0.0 0.222 # This is without continuity correction
        self.assertTrue('&phi;<i><sub>c</sub></i> = 0.154' in result[3])
            # jamovi 2.0.0.0 continuity corection 0.710, 0.399
        self.assertTrue('&chi;<sup>2</sup></i>(1, <i>N</i> = 30) = 0.71, <i>p</i> = .399' in result[5])  # &chi;<sup>2</sup></i>(1, <i>N</i> = 30) = 0.710

        # 3 Nom groups
        result = data.compare_groups('i', ['c'])
            # jamovi 2.0.0.0 0.00899
        self.assertTrue('&phi;<i><sub>c</sub></i> = 0.009' in result[3])
        # Chi-squared test
            # jamovi 2.0.0.0 0.00242, 0.999
        self.assertTrue('&chi;<sup>2</sup></i>(2, <i>N</i> = 30) = 0.00, <i>p</i> = .999' in result[5])  # TODO validate  # &chi;<sup>2</sup></i>(2, <i>N</i> = 30) = 0.002

        # 3 × 3 Int groups
        result = data.compare_groups('a', ['c', 'd'])
        self.assertTrue('<td>Mean</td>      <td>1.0695</td>      <td>1.8439</td>      <td>2.3693</td>' in result[3])
        self.assertTrue('<td>Standard deviation</td>      <td>2.7005</td>      <td>2.0891</td>      <td>4.2610</td>' in result[3])
        self.assertTrue('<td>Maximum</td>      <td>4.4130</td>      <td>4.7890</td>      <td>9.1600</td>' in result[3])
        self.assertTrue('<td>Upper quartile</td>      <td>3.0000</td>      <td>3.0213</td>      <td>4.4028</td>' in result[3])
        self.assertTrue('<td>Median</td>      <td>1.3340</td>      <td>2.4590</td>      <td>0.9015</td>' in result[3])
        self.assertTrue('<td>Lower quartile</td>      <td>-0.5965</td>      <td>0.8870</td>      <td>-1.1320</td>' in result[3])
        self.assertTrue('<td>Minimum</td>      <td>-2.8030</td>      <td>-2.2890</td>      <td>-1.4860</td>' in result[3])
            # jamovi 2.0.0.0 0.962, 0.398; 0.536, 0.593; 1.145, 0.363
        self.assertTrue('<i>F</i>(2, 21) = 0.96, <i>p</i> = .398' in result[7])
        self.assertTrue('<i>F</i>(2, 21) = 0.54, <i>p</i> = .593' in result[7])
        self.assertTrue('<i>F</i>(4, 21) = 1.15, <i>p</i> = .363' in result[7])

    def test_single_case(self):

        # TODO validate
        # Test for the slope stat
        data = cs.CogStatData(data='''group	slope	slope_SE
Patient	0.247	0.069
Control	0.492	0.106
Control	0.559	0.108
Control	0.63	0.116
Control	0.627	0.065
Control	0.674	0.105
Control	0.538	0.107''')
        result = data.compare_groups('slope', ['group'], 'slope_SE', 25)
        self.assertTrue('Test d.2: <i>t</i>(42.1) = -4.21, <i>p</i> &lt; .001' in result[8])
        result = data.compare_groups('slope', ['group'])
        self.assertTrue('<i>t</i>(5) = -5.05, <i>p</i> = .004' in result[8])

    def test_data_filtering(self):

        data = cs.CogStatData(data='''data dummy
1
3
3
6
8
10
10
1000''')
        result = data.filter_outlier(var_names=['data'], mode='2.5mad')
        # data of Lays et al. 2013
        # as in Lays et al. 2013: median is 7, MAD is 5.1891
        # median +- 2.5*MAD is −5.97275 and 19.97275
        self.assertTrue('will be excluded: -6.0  --  20.0' in result[1])

if __name__ == '__main__':
    unittest.main()

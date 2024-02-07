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
Validate the statistical calculations in CogStat.  
- All calculated statistical value should be tested at least once.
- All leafs of the decision tree should be tested only once, i.e., no need to test the same leaf in various analyses. 
- Choose data for hypothesis tests so that the tests shouldn't give p<0.001 results because exact p-values cannot be 
tested this way.
- No need to test the details of the statistical methods imported from other modules (e.g., results that are not 
displayed in CogStat, correct data handling) because that is the job of that specific module.
- All numerical data values should be used with 3 digits decimal precision, to ensure that copying 
the data for validation no additional rounding happens.

Validate the calculations of CogStat with other software packages.
- Prefer popular software packages that are applied most often by users, such as SPSS, jamovi, JASP
- Get the data of the calculation tests and run the same analyses in the validation software 
- Validated results should be added as a comment here:
    validating software; version; optionally information about how the analysis can be run; result; any comments
- A single calculation result can be validated in multiple software packages
- If the same variable is used for the same analysis, in the current file, only the first result should be validated.
In the following cases, you'll se a note that no validation is needed. 
- If a statistic is not available in a popular software, a note can be added that it is not available
- When other software and CogStat return different results, then (if the difference is caused by various valid versions 
of the calculation) either add the cause of the difference to the user documentation or (if CogStat  calculated 
the result incorrectly) fix the bug. After a calculation bug is fixed, always add a warning to the release note.
- Software specific notes
    - In JASP, decimals cannot be used as ordinal numbers. To validate related calculations, you may multiply the values 
    by 1000 so that they'll be round numbers but the order information is kept.

See also the JASP verification project: https://jasp-stats.github.io/jasp-verification-project/index.html
"""

# TODO Check the documentation to see if all calculations to be tested are tested and validated

#cs.output_type = 'do not format'

# Generate the dataset on which the tests will be performed
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
# For testing multicollinearity
data_pd['s'] = data_pd['h'] + ((data_pd['r'] * np.round(np.random.normal(loc=3, scale=0.5, size=30), 3)) / 0.15)
data = cs.CogStatData(data=data_pd, measurement_levels=['int', 'int', 'nom', 'nom', 'int', 'int', 'int', 'int', 'nom',
                                                        'nom', 'nom', 'int', 'nom', 'nom', 'int', 'int', 'int', 'int',
                                                        'int'])

# display the data so that it can be copied to validating software packages
#pd.set_option('display.expand_frame_repr', False)
#print (data_pd)

class CogStatTestCase(unittest.TestCase):
    """Unit tests for CogStat."""

    def test_explore_variables(self):
        """Test explore variables"""

        # Int variable
        result = data.explore_variable('a', 1, 2.0)
        #for i, res in enumerate(result): print(i, res)
        self.assertTrue('N of observed cases: 30' in result['raw data info'])
        self.assertTrue('N of missing cases: 0' in result['raw data info'])
        # Descriptives
            # jamovi 2.0.0.0    3.14
            # jasp 0.16.1   3.14
        self.assertTrue('<td>Mean</td>      <td>3.1438</td>' in result['descriptives table'])
            # LibreOffice 7.1.5.2   stdev.p() 3.21518743552465
            # LibreOffice 7.4.3.2   stdev.s() 3.27015188599418
            # jasp 0.16.1   3.270  This is the population estimation (even if it is listed under the descriptives)
        self.assertTrue('<td>Standard deviation</td>      <td>3.2152</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 0.359
            # jasp 0.16.1 0.359
        self.assertTrue('<td>Skewness</td>      <td>0.3586</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 0.0446
            # jasp 0.16.1   0.045
        self.assertTrue('<td>Kurtosis</td>      <td>0.0446</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 12.8
            # jasp 0.16.1   12.784
        self.assertTrue('<td>Range</td>      <td>12.7840</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 9.98
            # JASP 0.15.0.0 9.981
            # jasp 0.16.1   9.981
        self.assertTrue('<td>Maximum</td>      <td>9.9810</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 4.39
            # JASP 0.15.0.0 4.388
            # jasp 0.16.1   4.388
        self.assertTrue('<td>Upper quartile</td>      <td>4.3875</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 2.85
            # JASP 0.15.0.0 2.854
            # jasp 0.16.1   2.854   With more decimals, it is 2.85450
        self.assertTrue('<td>Median</td>      <td>2.8545</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 1.42
            # JASP 0.15.0.0 1.419
            # jasp 0.16.1   1.419
        self.assertTrue('<td>Lower quartile</td>      <td>1.4190</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 -2.80
            # JASP 0.15.0.0 -2.803
            # jasp 0.16.1   -2.803
        self.assertTrue('<td>Minimum</td>      <td>-2.8030</td>' in result['descriptives table'])
        # Shapiro–Wilk normality
            # jamovi 2.0.0.0 0.959
            # jasp 0.16.1   0.959
        self.assertTrue('<i>W</i> = 0.96' in result['normality info'])  # <i>W</i> = 0.959
            # jamovi 2.0.0.0 0.287
            # jasp 0.16.1   0.287
        self.assertTrue('<i>p</i> = .287' in result['normality info'])

        # Population estimation and one sample t-test
            # jamovi 2.3.13.0: 3.14, 1.92, 4.36 based on t-distribution
            # https://www.statskingdom.com/confidence-interval-calculator.html
                # To use the t-value based solution, do not use the population SD (i.e., rely on t-distribution)
            # Mean confidence interval: [1.922672 , 4.364861]
        self.assertTrue('<td>Mean</td>      <td>3.1438</td>      <td>1.9227</td>      <td>4.3649</td>' in result['estimation table'])
            # jamovi 2.0.0.0 3.27 - SD estimates population SD
            # https: // www.statskingdom.com / confidence - interval - calculator.html To use the t-value based solution, do not
            # use the population SD
            # Standard deviation confidence interval: [2.604372 , 4.396115]
        self.assertTrue('<td>Standard deviation</td>      <td>3.2702</td>      <td>2.6044</td>      <td>4.3961</td>' in result['estimation table'])
        # Sensitivity power analysis
            # G*Power 3.1.9.6: 0.6811825
            # jamovi v1.2.19.0, jpower 0.1.2: 0.681
        self.assertTrue('effect size in d: 0.68' in result['hypothesis test'])
            # Note that the test value is 2 here.
            # jamovi 2.0.0.0 1.92, 0.065
        self.assertTrue('t</i>(29) = 1.92' in result['hypothesis test'])
        self.assertTrue('p</i> = .065' in result['hypothesis test'])
        # Bayesian one sample t-test
            # Note that the test value is 2 here.
            # JASP 0.16 BF10: 0.969, BF01: 1.032
        self.assertTrue('BF<sub>10</sub> = 0.97, BF<sub>01</sub> = 1.03' in result['hypothesis test'])

        # Wilcoxon signed-rank test for non-normal interval variable
        result = data.explore_variable('b', 0, 20.0)
            # jamovi 2.0.0.0 W(!) 262, p 0.556
            # before scipy 1.9, the p value was 0.551; with scipy 1.9 the p has the same value as in jamovi
        self.assertTrue('T</i> = 203' in result['hypothesis test'])
        self.assertTrue('p</i> = .556' in result['hypothesis test'])

        # Ord variable
        data.data_measlevs['a'] = 'ord'
        result = data.explore_variable('a', 1, 2.0)
        self.assertTrue('N of observed cases: 30' in result['raw data info'])
        self.assertTrue('N of missing cases: 0' in result['raw data info'])
        # Descriptives - Other software validation is not needed here
        self.assertTrue('<td>Maximum</td>      <td>9.9810</td>' in result['descriptives table'])
        self.assertTrue('<td>Upper quartile</td>      <td>4.3875</td>' in result['descriptives table'])
        self.assertTrue('<td>Median</td>      <td>2.8545</td>' in result['descriptives table'])
        self.assertTrue('<td>Lower quartile</td>      <td>1.4190</td>' in result['descriptives table'])
        self.assertTrue('<td>Minimum</td>      <td>-2.8030</td>' in result['descriptives table'])

        # TODO median CI
        # Wilcoxon signed-rank test
            # Note that the test value is 2 here.
            # jamovi 2.0.0.0 W(!) 320, p 0.073
            # JASP 0.15.0.0 W(!) 320, p 0.073
            # before scipy 1.9, the p value was 0.074; with scipy 1.9 the p has the same value as in jamovi
        self.assertTrue('T</i> = 145' in result['hypothesis test'])
        self.assertTrue('p</i> = .073' in result['hypothesis test'])
        data.data_measlevs['a'] = 'int'

        # Nominal variable
        #result = data.explore_variable('c')
        # TODO variation ratio
            # not available in JASP 0.16.1
        # TODO multinomial proportion CI

    def test_regression(self):
        """Test simple and multiple regression, and ordinal and nominal counterparts"""

        # Int variables, one predictor
        result = data.regression(['g'], 'h')
        self.assertTrue('N of observed pairs: 30' in result['raw data info'])
        self.assertTrue('N of missing pairs: 0' in result['raw data info'])
            # Regression model: jamovi 2.2.5.0 g 0.347 Intercept 4.593
            # jasp 0.16.1   g 0.347     Intercept 4.593
        self.assertTrue('h = 0.347 × g + 4.593' in result['descriptives'])
            # Pearson's correlation: jamovi 2.2.5.0 0.168
            # jasp 0.16.1   0.168
        self.assertTrue('<i>r</i> = 0.168' in result['sample effect size table'])
            # Spearman's correlation: jamovi 2.2.5.0 0.103
            # jasp 0.16.1   0.103
        self.assertTrue('<i>r<sub>s</sub></i> = 0.103' in result['sample effect size table'])
            # Henze-Zirkler test: MVN 5.9 using R 4.1.3 statistic 0.3620904 p 0.6555329
        self.assertTrue('<i>W</i> = 0.36, <i>p</i> = .655' in result['assumption info'])
            # Koenker's test: bptest(h~g, data=data, studentize=True) in skedastic 1.0.4 using R 4.1.3, BP 0.11333, df=1, p-value 0.7364
        self.assertTrue('<i>LM</i> = 0.11, <i>p</i> = .736' in result['assumption info'])
            # White's test: white_lm(model, interactions=TRUE, statonly = FALSE) in skedastic 1.0.4 using R 4.1.3, statistic 0.219 p-value 0.896
        self.assertTrue('<i>LM</i> = 0.22, <i>p</i> = .896' in result['assumption info'])
            # Slope CI: jamovi 2.2.5.0 [-0.440, 1.13]
            # jasp 0.16.1   [-0.440, 1.133]
        self.assertTrue('[-0.440, 1.133]' in result['estimation table'])
            # Intercept CI: jamovi 2.2.5.0 [-0.517, 9.70]
            # jasp 0.16.1   [-0.517, 9.703]
        self.assertTrue('[-0.517, 9.703]' in result['estimation table'])
            # Spearman CI: TODO validate
            # Currently validated with JASP, which uses bootstrapping while CogStat uses Fisher transform
            # jasp 0.16.1   [-0.267, 0.447]
        self.assertTrue('[-0.267, 0.447]' in result['population effect size table'])
            # Pearson CI: jamovi 2.2.5.0 [-0.204, 0.498]
            # jasp 0.16.1   [-0.204, 0.498]
        self.assertTrue('[-0.204, 0.498]' in result['population effect size table'])
            # Pearson hypothesis test: jamovi 2.2.5.0 r 0.168 p 0.374
            # jasp 0.16.1   r 0.168     p 0.374
        self.assertTrue("Pearson's correlation: <i>r</i>(28) = 0.17, <i>p</i> = .374" in result['hypothesis test'])
            # Spearman hypothesis test: jamovi 2.2.5.0 r_s 0.103 p 0.585 TODO
            # jasp 0.16.1   r_s 0.103   p 0.585  With more decimals: p 0.58507 TODO
        self.assertTrue("Spearman's rank-order correlation: <i>r<sub>s</sub></i>(28) = 0.10, <i>p</i> = .586" in result['hypothesis test'])
            # Bayes Factor for Pearson: JASP 0.16 BF10: 0.331, BF01: 3.023
        self.assertTrue('BF<sub>10</sub> = 0.33, BF<sub>01</sub> = 3.02' in result['hypothesis test'])

        # Int variables, three predictors
        result = data.regression(['r', 's', 'g'], 'h')
        self.assertTrue('N of observed pairs: 30' in result['raw data info'])
        self.assertTrue('N of missing pairs: 0' in result['raw data info'])

        # Sample effect sizes
        # R2: Jamovi 2.2.5 R2 = 0.156, lm(h ~ g + s + r) in lmtest 0.9-40 using R 4.1.3 R2 = 0.1564
        self.assertTrue('<i>R<sup>2</sup></i></td>      <td>0.156' in result['sample effect size table'])
        # Partial pearson correlations: Jamovi 2.2.5
        self.assertTrue('<td>r</td>      <td><i>r</i> = -0.330</td>' in result['sample effect size table'])  # r = -0.330
        self.assertTrue('<td>s</td>      <td><i>r</i> = 0.357</td>' in result['sample effect size table'])  # r = 0.357
        self.assertTrue('<td>g</td>      <td><i>r</i> = 0.187</td>' in result['sample effect size table'])  # r = 0.187
        # Standardized effect size
        self.assertTrue('<td>Log-likelihood</td>      <td>-95.332</td>' in result['sample effect size table'])  # logLik(model) in stats 4.1.3 in R: -95.33197 (df=5)
        self.assertTrue('<td>AIC</td>      <td>198.664</td>' in result['sample effect size table'])  # Jamovi 2.2.5 AIC = 201
        self.assertTrue('<td>BIC</td>      <td>204.269</td>' in result['sample effect size table'])  # Jamovi 2.2.5 BIC = 208
        # Population properties
        # Assumption tests
        # Multivariate normality: MVN 5.9 using R 4.1.3 statistic 1.079154 p 0.002746894
        self.assertTrue('<i>W</i> = 1.08, <i>p</i> = .003' in result['assumption info'])
        # Homoscedasticity
        # Koenker's test: bptest(h ~ g + s + r, data=data, studentize=TRUE) in skedastic 1.0.4 using R 4.1.3,
        # BP = 2.8965, df = 3, p-value = 0.4079
        self.assertTrue('<i>LM</i> = 2.90, <i>p</i> = .408' in result['assumption info'])
        # White's test: white_lm(model, interactions=TRUE, statonly = FALSE) in skedastic 1.0.4 using R 4.1.3,
        # statistic 7.71 p-value 0.564
        self.assertTrue('<i>LM</i> = 7.71, <i>p</i> = .564' in result['assumption info'])
        # Multicollinearity
        # VIFs: Jamovi 2.2.5 VIF(r) = 16.43, VIF(s) = 16.11, VIF(g) = 1.10
        self.assertTrue('<td>const</td>      <td>5.608</td>' in result['assumption info'])
        self.assertTrue('<td>r</td>      <td>16.434</td>' in result['assumption info'])
        self.assertTrue('<td>s</td>      <td>16.113</td>' in result['assumption info'])
        self.assertTrue('<td>g</td>      <td>1.104</td>' in result['assumption info'])
        # Beta weights when regressing r on all other regressors Jamovi 2.2.5
        self.assertTrue('<td>const</td>      <td>-0.058</td>' in result['assumption info'])  # Intercept: -0.0581
        self.assertTrue('<td>s</td>      <td>0.045</td>' in result['assumption info'])  # s: 0.0453
        self.assertTrue('<td>g</td>      <td>0.051</td>' in result['assumption info'])  # g: 0.0512
        # Beta weights when regressing s on all other regressors Jamovi 2.2.5
        self.assertTrue('<td>const</td>      <td>6.275</td>' in result['assumption info'])  # Intercept: 6.275
        self.assertTrue('<td>r</td>      <td>20.599</td>' in result['assumption info'])  # r: 20.599
        self.assertTrue('<td>g</td>      <td>-0.634</td>' in result['assumption info'])  # g: -0.634
        # Correlation matrix of predictors: Jamovi 2.2.5
        # 1.000     0.968       0.292
        # 0.968     1.000       0.259
        # 0.292     0.259       1.000
        self.assertTrue('<td>r</td>      <td>1.000</td>      <td>0.968</td>      <td>0.292</td>' in result['assumption info'])
        self.assertTrue('<td>s</td>      <td>0.968</td>      <td>1.000</td>      <td>0.259</td>' in result['assumption info'])
        self.assertTrue('<td>g</td>      <td>0.292</td>      <td>0.259</td>      <td>1.000</td>' in result['assumption info'])
        # Regression model parameters: lm(h ~ g + s + r) in lmtest 0.9-40 using R 4.1.3, CIs from gamlj 2.6.1 in Jamovi 2.2.5
        # Intercept = 3.40612 [-2.13583, 8.948], r = -2.32205 [-4.99767, 0.354], s = 0.11901 [-0.00649, 0.245], g = 0.37823 [-0.42342, 1.180]
        self.assertTrue('<td>Intercept</td>      <td>3.406</td>      <td>[-2.136, 8.948]</td>' in result['estimation table'])
        self.assertTrue('<td>Slope for r</td>      <td>-2.322</td>      <td>[-4.998, 0.354]</td>' in result['estimation table'])
        self.assertTrue('<td>Slope for s</td>      <td>0.119</td>      <td>[-0.006, 0.245]</td>' in result['estimation table'])
        self.assertTrue('<td>Slope for g</td>      <td>0.378</td>      <td>[-0.423, 1.180]</td>' in result['estimation table'])
        # Standardized effect size estimates
        # Model metrics: lm(h ~ g + s + r) in lmtest 0.9-40 using R 4.1.3, Adjusted r2 = 0.05906
        self.assertTrue('<td>Adjusted <i>R<sup>2</sup></i></td>      <td>0.059</td>      <td>[-0.083, 0.201]</td>' in result['population effect size table'])
        # Partial correlations TODO validate CIs
        # Point estimates: Jamovi 2.2.5, CIs: cor.test() in stats 4.1.3 in R 4.1.3 run on residuals from respective tests
        # e.g. for partial correlation of r and h: cor.test(resid(lm(h ~ s + g, data)), resid(lm(r ~ s + g, data)))
        # CIs don't exactly match between R stats and python pingouin, but are very close
        # pingouin changed the calculation in 0.4.0, and uses the R ppcor package's method
          # https://pingouin-stats.org/build/html/changelog.html#v0-4-0-august-2021
        # pingouin 0.3.8: <td>-0.330</td>      <td>[-0.620, 0.030]</td>  # r = -0.330 [-0.6171, 0.0341]
        self.assertTrue('<td>r, <i>r</i></td>      <td>-0.330</td>      <td>[-0.630, 0.050]</td>' in result['population effect size table'])
        # pingouin 0.3.8: <td>0.357</td>      <td>[-0.000, 0.640]</td>  # r = 0.357 [-0.00365, 0.63559]
        self.assertTrue('<td>s, <i>r</i></td>      <td>0.357</td>      <td>[-0.020, 0.640]</td>' in result['population effect size table'])
        # pingouin 0.3.8: <td>0.187</td>      <td>[-0.190, 0.510]</td>  # r = 0.187 [-0.186. 0.513]
        self.assertTrue('<td>g, <i>r</i></td>      <td>0.187</td>      <td>[-0.200, 0.520]</td>' in result['population effect size table'])
        # Hypothesis tests
        # lm(h ~ g + s + r) in lmtest 0.9-40 using R 4.1.3, F-statistic: 1.607, DFs: 3 and 26,  p-value: 0.2119
        self.assertTrue('Model F-test: <i>F</i>(3,26) = 1.61, <i>p</i> = .212' in result['hypothesis test'])
        # Jamovi 2.2.5:
        self.assertTrue('r: <i>t</i>(26) = -1.78, <i>p</i> = .086' in result['hypothesis test'])  # t = -1.784 p = 0.086
        self.assertTrue('s: <i>t</i>(26) = 1.95, <i>p</i> = .062' in result['hypothesis test'])  # t = 1.949 p = 0.062
        self.assertTrue('g: <i>t</i>(26) = 0.97, <i>p</i> = .341' in result['hypothesis test'])  # t = 0.970 p = 0.341


        # Ord variables
        data.data_measlevs['a'] = 'ord'
        data.data_measlevs['b'] = 'ord'
        result = data.regression(['a'], 'b')
            # JASP 0.15.0.0 -0.363
        self.assertTrue('-0.363' in result['sample effect size table'])
            # JASP 0.15.0.0 [-0.640, -0.003]
        self.assertTrue('[-0.640, -0.003]' in result['population effect size table'])
            # JASP 0.15.0.0 -0.363 p 0.049 TODO
        self.assertTrue("Spearman's rank-order correlation: <i>r<sub>s</sub></i>(28) = -0.36, <i>p</i> = .048" in result['hypothesis test'])  # <i>r<sub>s</sub></i>(28) = -0.363
        data.data_measlevs['a'] = 'int'
        data.data_measlevs['b'] = 'int'

        # Nom variables
        result = data.regression(['c'], 'd')
        self.assertTrue('N of observed pairs: 30' in result['raw data info'])
        self.assertTrue('N of missing pairs: 0' in result['raw data info'])
        # Cramer's V
            # jamovi 2.0.0.0 0.372
            # jasp 0.16.1   0.372
        self.assertTrue('<sub>c</sub></i> = 0.372' in result['sample effect size table'])
        # Sensitivity power analysis
            # G*Power 3.1.9.6, "Goodness of fit test: Contingency tables", df=4: 0.7868005
        self.assertTrue('effect size in w: 0.79' in result['hypothesis test'])
        # Chi-squared
            # jamovi v1.2.19.0: X2, df, p, N: 8.31, 4, 0.081, 30
            # jasp 0.16.1   x2, df, p, N: 8.312, 4, 0.081, 30
        self.assertTrue('(4, <i>N</i> = 30) = 8.31' in result['hypothesis test'])  # (4, <i>N</i> = 30) = 8.312
        self.assertTrue('<i>p</i> = .081' in result['hypothesis test'])

    def test_diffusion(self):
        """Test diffusion analysis"""
        data_diffusion = cs.CogStatData(data=str(Path('data/diffusion.csv')))
        result = data_diffusion.diffusion(error_name='Error', RT_name='RT_sec', participant_name='Name',
                                          condition_names=['Num1', 'Num2'])
        # Drift rate
        self.assertTrue('<th>zsiraf</th>      <td>0.190</td>      <td>0.276</td>      <td>0.197</td>      <td>0.235</td>      <td>0.213</td>' in result['drift rate'].data.to_html(float_format='{:.3f}'.format).replace('\n', ''))
        # Threshold
        self.assertTrue('<th>zsiraf</th>      <td>0.178</td>      <td>0.096</td>      <td>0.171</td>      <td>0.112</td>      <td>0.088</td>' in result['threshold'].data.to_html(float_format='{:.3f}'.format).replace('\n', ''))
        # Nondecision time
        self.assertTrue('<th>zsiraf</th>      <td>0.481</td>      <td>0.590</td>      <td>0.483</td>      <td>0.561</td>      <td>0.522</td>' in result['nondecision time'].data.to_html(float_format='{:.3f}'.format).replace('\n', ''))

    def test_compare_variables(self):
        """Test compare variables"""

        # 2 Int variables
        result = data.compare_variables(['a', 'e'])
        self.assertTrue('N of observed cases: 30' in result['raw data info'])
        self.assertTrue('N of missing cases: 0' in result['raw data info'])
        # Cohen's d
            # CS formula: https://pingouin-stats.org/generated/pingouin.compute_effsize.html
            # Based on the formula, calculated in LO Calc 6.4: 0.030004573510063
            # jamovi v1.2.19.0: 0.0202; formula: https://github.com/jamovi/jmv/blob/master/R/ttestps.b.R#L54-L66
            # jasp 0.16.1   0.020 TODO With more decimals 0.02017
        self.assertTrue("<td>Cohen's d</td>      <td>0.030</td>" in result['sample effect size'])
        # eta-squared
            # CS formula: https://pingouin-stats.org/generated/pingouin.convert_effsize.html
            # Based on the formula, calculated in LO Calc 6.4: 0.0002250179634
            # jamovi v1.2.19.0: 0.000
        self.assertTrue('<td>Eta-squared</td>      <td>0.000</td>' in result['sample effect size'])
        # Sample means
            # jamovi 2.0.0.0 3.14, 3.05
            # jasp 0.16.1   3.144, 3.050
        self.assertTrue('<td>3.1438</td>      <td>3.0502</td>' in result['descriptives table'])
        # Hedges'g (with CI)
            # CS formula: https://pingouin-stats.org/generated/pingouin.compute_effsize.html
            # https://pingouin-stats.org/generated/pingouin.compute_esci.html
            # Note that the latter (CI) method has changed in v0.3.5 https://pingouin-stats.org/changelog.html
            # Based on the formula, calculated in LO Calc 7.0: 0.029614903724218, -0.34445335392457, 0.403683161373007
            # Note that the last value is 0.404 in LO, not .403 as in pingouin TODO
        self.assertTrue("<td>Hedges' g</td>      <td>0.030</td>      <td>-0.344</td>      <td>0.403</td>" in result['population effect size'])
            # jamovi 2.0.0.0 0.954 0.215
            # jasp 0.16.1   0.954   0.215
        self.assertTrue('<i>W</i> = 0.95, <i>p</i> = .215' in result['hypothesis test'])  # <i>W</i> = 0.954
        # Sensitivity power analysis
            # G*Power 3.1.9.6: 0.6811825
            # jamovi v1.2.19.0, jpower 0.1.2: 0.681
        self.assertTrue('effect size in d: 0.68' in result['hypothesis test'])
        # Paired samples t-test
            # jamovi v1.2.19.0: t, df, p: 0.110, 29.0, 0.913
            # jasp 0.16.1   t:0.110, df:29, p:0.913
        self.assertTrue('<i>t</i>(29) = 0.11, <i>p</i> = .913' in result['hypothesis test'])
        # Bayesian paired samples t-test
            # JASP 0.16 BF10: 0.196, BF01: 5.115
        self.assertTrue('BF<sub>10</sub> = 0.20, BF<sub>01</sub> = 5.11' in result['hypothesis test'])

        # 2 Int variables - non-normal
        result = data.compare_variables(['e', 'f'])
            # jamovi 2.0.0.0 0.915, 0.019
            # jasp 0.16.1   W:0.915 p:0.019     with more decimals W:0.91451
        self.assertTrue('<i>W</i> = 0.91, <i>p</i> = .019' in result['hypothesis test'])  # <i>W</i> = 0.915
        # Wilcoxon signed-rank test
            # jamovi 2.0.0.0 110, 0.011 (0.01060 with more precision)
            # before scipy 1.9, the p value was 0.012; with scipy 1.9 the p shows the same value as jamovi
            # jasp 0.16.1   110, 0.011
        #print(result[7])
        self.assertTrue('<i>T</i> = 110.00, <i>p</i> = .011' in result['hypothesis test'])

        # 3 Int variables
        result = data.compare_variables(['a', 'e', 'g'])
        # jamovi 2.0.0.0 3.14, 3.05, 5.73
            # jasp 0.16.1   3.144, 3.050, 5.729  the last group with more decimals: 5.72950
        self.assertTrue('<td>3.1438</td>      <td>3.0502</td>      <td>5.7295</td>' in result['descriptives table'])
            # Shapiro–Wilk
            # JASP 0.15.0.0 0.959
            # jasp 0.16.1   W:0.959 p:0.287
        self.assertTrue('a: <i>W</i> = 0.96, <i>p</i> = .287' in result['hypothesis test'])  # <i>W</i> = 0.959
            # JASP 0.15.0.0 0.966
            # jasp 0.16.1   W:0.966 p:0.435
        self.assertTrue('e: <i>W</i> = 0.97, <i>p</i> = .435' in result['hypothesis test'])  # <i>W</i> = 0.966
            # JASP 0.15.0.0 0.946
            # jasp 0.16.1   W:0.946 p:0.133
        self.assertTrue('g: <i>W</i> = 0.95, <i>p</i> = .133' in result['hypothesis test'])  # <i>W</i> = 0.946
            # jamovi 2.0.0.0 0.975 0.703
            # jasp 0.16.1   Mauchly's W:0.975 p:0.703
        self.assertTrue('sphericity: <i>W</i> = 0.98, <i>p</i> = .703' in result['hypothesis test'])  # <i>W</i> = 0.975
            # jamovi 2.0.0.0 6.16 0.004
            # jasp 0.16.1   6.174 0.004
        self.assertTrue('<i>F</i>(2, 58) = 6.17, <i>p</i> = .004' in result['hypothesis test'])
            # jamovi 2.0.0.0 (Holm correction) 0.110, 0.913; 3.167, 0.011; 2.883, 0.015
            # jasp 0.16.1   (Holm correction) 0.110, 0.913; -3.167, 0.011; -2.883, 0.015
        self.assertTrue('0.11, <i>p</i> = .913' in result['hypothesis test'])  # TODO keep the order of the variables, and have a fixed sign
        self.assertTrue('3.17, <i>p</i> = .011' in result['hypothesis test'])
        self.assertTrue('2.88, <i>p</i> = .015' in result['hypothesis test'])

        # 3 Int variables, sphericity violated
        result = data.compare_variables(['a', 'e', 'h'])
            # jasp 0.16.1   3.144, 3.050, 6.579
        self.assertTrue('<td>3.1438</td>      <td>3.0502</td>      <td>6.5786</td>' in result['descriptives table'])
            # jasp 0.16.1   0.959 0.287
        self.assertTrue('a: <i>W</i> = 0.96, <i>p</i> = .287' in result['hypothesis test'])  # <i>W</i> = 0.959
            # jasp 0.16.1   0.966 0.435
        self.assertTrue('e: <i>W</i> = 0.97, <i>p</i> = .435' in result['hypothesis test'])  # <i>W</i> = 0.966
            # JASP 0.15.0.0 0.980
            # jasp 0.16.1 0.980 0.824
        self.assertTrue('h: <i>W</i> = 0.98, <i>p</i> = .824' in result['hypothesis test'])
            # jamovi 2.0.0.0  0.793, 0.039
            # jasp 0.16.1   0.793 0.039
        self.assertTrue('sphericity: <i>W</i> = 0.79, <i>p</i> = .039' in result['hypothesis test'])  # <i>W</i> = 0.793
            # jamovi 2.0.0.0 1.66, 48.04, 6.16, 0.007
            # jasp 0.16.1   (Greenhouse-Geisser) 1.657, 48.041, 6.155, 0.007
        self.assertTrue('<i>F</i>(1.66, 48) = 6.16, <i>p</i> = .007' in result['hypothesis test'])  # TODO more precise df values
            # jamovi 2.0.0.0 (Holm correction) 0.110, 0.913; 2.678, 0.026; 2.809, 0.026
            # jasp 0.16.1   (Holm correction) 0.110, 0.913; -2.678, 0.026; -2.809, 0.026
        self.assertTrue('0.11, <i>p</i> = .913' in result['hypothesis test'])  # TODO keep the order of the variables, and have a fixed sign
        self.assertTrue('2.68, <i>p</i> = .026' in result['hypothesis test'])
        self.assertTrue('2.81, <i>p</i> = .026' in result['hypothesis test'])

        # 3 Int variables, non-normal
        result = data.compare_variables(['a', 'e', 'f'])
            # jasp 0.16.1   3.144 3.050 5.368
        self.assertTrue('<td>3.1438</td>      <td>3.0502</td>      <td>5.3681</td>' in result['descriptives table'])
            # jasp 0.16.1   0.959 0.287
        self.assertTrue('a: <i>W</i> = 0.96, <i>p</i> = .287' in result['hypothesis test'])  # <i>W</i> = 0.959
            # jasp 0.16.1   0.966 0.435
        self.assertTrue('e: <i>W</i> = 0.97, <i>p</i> = .435' in result['hypothesis test'])  # <i>W</i> = 0.966
            # jasp 0.16.1   0.818 .001
        self.assertTrue('f: <i>W</i> = 0.82, <i>p</i> &lt; .001' in result['hypothesis test'])  # <i>W</i> = 0.818
            # jamovi 2.0.0.0 df 2, khi2 6.47, p 0.039
        self.assertTrue('&chi;<sup>2</sup>(2, <i>N</i> = 30) = 6.47, <i>p</i> = .039' in result['hypothesis test'])
            # post-hoc Durbin-Conover
            # jamovi 2.2.5 p-values: a-e 0.504, a-f 0.065, e-f: 0.013
        self.assertTrue('<th>a</th>      <td>1.000</td>      <td>0.504</td>      <td>0.065</td>' in result['hypothesis test'])
        self.assertTrue('<th>e</th>      <td>0.504</td>      <td>1.000</td>      <td>0.013</td>' in result['hypothesis test'])

        # 2 × 2 Int variables
        result = data.compare_variables(['a', 'b', 'e', 'f'], factors=[['first', 2], ['second', 2]])
            # jamovi 2.0.0.0 6.06, 0.020; 6.29, 0.018; 6.04, 0.020
        self.assertTrue('Main effect of first: <i>F</i>(1, 29) = 6.06, <i>p</i> = .020' in result['hypothesis test'])
        self.assertTrue('Main effect of second: <i>F</i>(1, 29) = 6.29, <i>p</i> = .018' in result['hypothesis test'])
        self.assertTrue('Interaction of factors first, second: <i>F</i>(1, 29) = 6.04, <i>p</i> = .020' in result['hypothesis test'])

        # 2 Ord variables
        data.data_measlevs['a'] = 'ord'
        data.data_measlevs['e'] = 'ord'
        data.data_measlevs['f'] = 'ord'
        result = data.compare_variables(['e', 'f'])
        # JASP 0.15.0.0 2.3895, 4.2275
        self.assertTrue('<td>2.3895</td>      <td>4.2275</td>' in result['descriptives table'])
        # Wilcoxon signed-rank test
            # jamovi 2.0.0.0 110, 0.011
            # before scipy 1.9, the p value was 0.012; with scipy 1.9 the p shows the same value as jamovi
        self.assertTrue('<i>T</i> = 110.00, <i>p</i> = .011' in result['hypothesis test'])

        # 3 Ord variables
        result = data.compare_variables(['a', 'e', 'f'])
            # JASP 0.15.0.0 2.8545, 2.3895, 4.2275
        self.assertTrue('<td>2.8545</td>      <td>2.3895</td>      <td>4.2275</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 6.47, 0.039
        self.assertTrue('&chi;<sup>2</sup>(2, <i>N</i> = 30) = 6.47, <i>p</i> = .039' in result['hypothesis test'])
        data.data_measlevs['a'] = 'int'
        data.data_measlevs['e'] = 'int'
        data.data_measlevs['f'] = 'int'

        # 2 Nom variables
        result = data.compare_variables(['i', 'j'])
        # TODO on Linux the row labels are 0.0 and 1.0 instead of 0 and 1
             # JASP 0.15.0.0    0, 4, 9, 13
             # JASP 0.15.0.0    1, 9, 8, 17
             # jasp 0.16.1      4, 9, 13; 9, 8, 17
        self.assertTrue('<td>0.0</td>      <td>4</td>      <td>9</td>      <td>13</td>    </tr>    <tr>      <td>1.0</td>      <td>9</td>' in result['descriptives table'])  # TODO more cells
        # McNemar
            # jamovi 2.0.0.0 0.00, 1.000 TODO https://github.com/cogstat/cogstat/issues/55
        self.assertTrue('&chi;<sup>2</sup>(1, <i>N</i> = 30) = 0.06, <i>p</i> = .814' in result['hypothesis test'])  # &chi;<sup>2</sup>(1, <i>N</i> = 30) = 0.0556

        # 3 Nom variables
        result = data.compare_variables(['i', 'j', 'k'])
        # Cochran's Q
            # TODO validate
            # not available in JASP and jamovi?
        self.assertTrue('<i>Q</i>(2, <i>N</i> = 30) = 0.78, <i>p</i> = .676' in result['hypothesis test'])  # <i>Q</i>(2, <i>N</i> = 30) = 0.783

    def test_compare_groups(self):
        """Test compare groups"""

        # 2 Int groups
        result = data.compare_groups('l', ['m'])
            # jamovi 2.0.0.0 2.53,4.58
            # jasp 0.16.1   2.532, 4.576
        self.assertTrue('<td>2.5316</td>      <td>4.5759</td>' in result['descriptives table'])
        # Cohen's d
            # CS formula: https://pingouin-stats.org/generated/pingouin.compute_effsize.html
            # Based on the formula, calculated in LO Calc 6.4: -0.704171924382848
            # jamovi v1.2.19.0: 0.0704
            # jasp 0.16.1   -0.704
        self.assertTrue("<td>Cohen's d</td>      <td>-0.704</td>" in result['sample effect size'])
        # eta-squared
            # jasp 0.16.1   0.117 TODO
            # CS formula: https://pingouin-stats.org/generated/pingouin.convert_effsize.html
            # Based on the formula, calculated in LO Calc 6.4: 0.110292204104377
            # jamovi v1.2.19.0: 0.117 # TODO why the difference? Maybe https://github.com/raphaelvallat/pingouin/releases/tag/v0.5.2
        self.assertTrue('<td>Eta-squared</td>      <td>0.110</td>' in result['sample effect size'])
        # Hedges'g (with CI)
            # CS formula: https://pingouin-stats.org/generated/pingouin.compute_effsize.html
            # https://pingouin-stats.org/generated/pingouin.compute_esci.html
            # Note that the latter (CI) method has changed in v0.3.5 https://pingouin-stats.org/changelog.html
            # Based on the formula, calculated in LO Calc 7.0: -0.685140250750879, -1.45474443187683, 0.084463930375068
            # jasp 0.16.1   -2.044, -4.216, 0.127
        #self.assertTrue('<td>Difference between the two groups:</td>      <td>-2.0443</td>      <td>-4.2157</td>      <td>0.1272</td>' in result[8])
            # TODO CI
            # jasp 0.16.1   -0.685
        self.assertTrue("<td>Hedges' g</td>      <td>-0.685</td>      <td>-1.455</td>      <td>0.084</td>" in result['population effect size'])
            # jasp 0.16.1   W: 0.959 p: 0.683; W: 0.984 p: 0.991
        self.assertTrue('(m: 1.0): <i>W</i> = 0.96, <i>p</i> = .683' in result['hypothesis test'])  # <i>W</i> = 0.959
        self.assertTrue('(m: 2.0): <i>W</i> = 0.98, <i>p</i> = .991' in result['hypothesis test'])  # <i>W</i> = 0.984
        self.assertTrue('<i>W</i> = 0.30, <i>p</i> = .585' in result['hypothesis test'])  # <i>W</i> = 0.305
        # Sensitivity power analysis
            # G*Power 3.1.9.6: 1.3641059
            # jamovi v1.2.19.0, jpower 0.1.2: 1.36
        self.assertTrue('effect size in d: 1.36' in result['hypothesis test'])
        # independent samples t-test
            # jamovi v1.2.19.0: t, df, p: -1.93, 28.0, 0.064
            # jasp 0.16.1   t: -1.928, df: 28, p:0.064
        self.assertTrue('<i>t</i>(28) = -1.93, <i>p</i> = .064' in result['hypothesis test'])
        # Bayesian independent samples t-test
            # JASP 0.16 BF10: 1.348, BF01: 0.742
        self.assertTrue('BF<sub>10</sub> = 1.35, BF<sub>01</sub> = 0.74' in result['hypothesis test'])

        # Non-normal group
        result = data.compare_groups('o', ['m'])
            # jasp 0.16.1   W: 0.808 p: 0.005
        self.assertTrue('(m: 2.0): <i>W</i> = 0.81, <i>p</i> = .005' in result['hypothesis test'])  # <i>W</i> = 0.808
        # Mann-Whitney
            # jamovi 2.0.0.0 51.0, 0.010 TODO
            # jasp 0.16.1   51.000 0.010 TODO with more decimals p 0.00987
        #self.assertTrue('<i>U</i> = 51.00, <i>p</i> = .011' in result['hypothesis test'])
        # Brunner-Munzel
        self.assertTrue('<i>W</i> = 3.13, <i>p</i> = .004' in result['hypothesis test'])

        # Heteroscedastic groups
        result = data.compare_groups('p', ['m'])
        # Welch's t-test
            # jamovi 2.0.0.0 0.119, 0.907
            # jasp 0.16.1   0.119 0.907
        self.assertTrue('<i>t</i>(25.3) = 0.12, <i>p</i> = .907' in result['hypothesis test'])  # <i>t</i>(25.3) = 0.119


        # TODO single case vs. group

        # 3 Int groups
        result = data.compare_groups('r', ['q'])
            # jamovi 2.0.0.0 3.29, 5.04, 7.24
            # jasp 0.16.1   3.287, 5.040, 7.241
        self.assertTrue('<td>3.2869</td>      <td>5.0400</td>      <td>7.2412</td>' in result['descriptives table'])
            # jasp 0.16.1   omega2: 0.167
        self.assertTrue('&omega;<sup>2</sup> = 0.167' in result['population effect size'])
        # Levene's test for homogeneity
            # jamovi 2.0.0.0 0.495, 0.615 TODO https://github.com/cogstat/cogstat/issues/56
            # jasp 0.16.1   F:0.495 p:0.615 TODO
        self.assertTrue('<i>W</i> = 0.68, <i>p</i> = .517' in result['hypothesis test'])  # <i>W</i> = 0.675
        # Sensitivity power analysis
            # TODO eta-square; see also https://github.com/raphaelvallat/pingouin/releases/tag/v0.5.2
            # G*Power (f value) 3.1.9.6: 0.7597473
        self.assertTrue('effect size in eta-square: 0.15' in result['hypothesis test'])
        self.assertTrue('effect size in f: 0.76' in result['hypothesis test'])
            # jamovi 2.0.0.0 4.00, 0.030
            # jasp 0.16.1   4.002, 0.030
        self.assertTrue('<i>F</i>(2, 27) = 4.00, <i>p</i> = .030' in result['hypothesis test'])
        # TODO post-hoc

        # 3 Int groups with assumption violation
        result = data.compare_groups('o', ['q'])
        # Kruskal-Wallis
            # jamovi 2.0.0.0 8.37, 0.015
            # jasp 0.16.1   8.366, 0.015
        self.assertTrue('&chi;<sup>2</sup>(2, <i>N</i> = 30) = 8.37, <i>p</i> = .015' in result['hypothesis test'])
        # TODO post-hoc Dunn's test

        # 2 Ord groups
        data.data_measlevs['o'] = 'ord'
        result = data.compare_groups('o', ['m'])
        # Mann-Whitney
            # jamovi 2.0.0.0 51.0, 0.010 TODO
            # jasp 0.16.1 51.0, 0.010 TODO
        #self.assertTrue('<i>U</i> = 51.00, <i>p</i> = .011' in result['hypothesis test'])
        # Brunner-Munzel
        self.assertTrue('<i>W</i> = 3.13, <i>p</i> = .004' in result['hypothesis test'])


        # 3 Ord groups
        data.data_measlevs['o'] = 'ord'
        result = data.compare_groups('o', ['q'])
        # Kruskal-Wallis
            # jamovi 2.0.0.0 8.37, 0.015
            # jasp 0.16.1   8.366, 0.015
        self.assertTrue('&chi;<sup>2</sup>(2, <i>N</i> = 30) = 8.37, <i>p</i> = .015' in result['hypothesis test'])
        data.data_measlevs['o'] = 'int'

        # 2 Nom groups
        result = data.compare_groups('i', ['j'])
        # Cramer's V
            # jamovi 2.0.0.0 0.222 # This is without continuity correction
            # jasp 0.16.1   0.222 TODO
        self.assertTrue('&phi;<i><sub>c</sub></i> = 0.154' in result['sample effect size'])
            # jamovi 2.0.0.0 continuity corection 0.710, 0.399
            # jasp 0.16.1   0.710, 0.399
        self.assertTrue('&chi;<sup>2</sup></i>(1, <i>N</i> = 30) = 0.71, <i>p</i> = .399' in result['hypothesis test'])  # &chi;<sup>2</sup></i>(1, <i>N</i> = 30) = 0.710

        # 3 Nom groups
        result = data.compare_groups('i', ['c'])
            # jamovi 2.0.0.0 0.00899
            # jasp 0.16.1 0.009
        self.assertTrue('&phi;<i><sub>c</sub></i> = 0.009' in result['sample effect size'])
        # Chi-squared test
            # jamovi 2.0.0.0 0.00242, 0.999
            # jasp 0.16.1   0.002, 0.999
        self.assertTrue('&chi;<sup>2</sup></i>(2, <i>N</i> = 30) = 0.00, <i>p</i> = .999' in result['hypothesis test'])  # TODO validate  # &chi;<sup>2</sup></i>(2, <i>N</i> = 30) = 0.002

        # 3 × 3 Int groups
        result = data.compare_groups('a', ['c', 'd'])
        # JASP does not handle groups with a single participant. Jamovi does.
            # jamovi 2.2.5 1.070, 1.844, 2.369
        self.assertTrue('<td>Mean</td>      <td>1.0695</td>      <td>1.8439</td>      <td>2.3693</td>' in result['descriptives table'])
            # jamovi 2.2.5 3.12, 2.23, 4.92 - jamovi calculates the population estimation
            # LibreOffice 7.3.3.2 pivot table with StDevP (and not StDev) 2.70051934819953, 2.08907368452503, 4.26098869835394

        self.assertTrue('<td>Standard deviation</td>      <td>2.7005</td>      <td>2.0891</td>      <td>4.2610</td>' in result['descriptives table'])
            # jamovi 2.2.5 4.413, 4.789, 9.160
        self.assertTrue('<td>Maximum</td>      <td>4.4130</td>      <td>4.7890</td>      <td>9.1600</td>' in result['descriptives table'])
            # jamovi 2.2.5 3.000, 3.021, 4.403
        self.assertTrue('<td>Upper quartile</td>      <td>3.0000</td>      <td>3.0212</td>      <td>4.4028</td>' in result['descriptives table'])
            # jamovi 2.2.5 1.334, 2.459, 0.902
        self.assertTrue('<td>Median</td>      <td>1.3340</td>      <td>2.4590</td>      <td>0.9015</td>' in result['descriptives table'])
            # jamovi 2.2.5 -0.597, 0.887, -1.132
        self.assertTrue('<td>Lower quartile</td>      <td>-0.5965</td>      <td>0.8870</td>      <td>-1.1320</td>' in result['descriptives table'])
            # jamovi 2.2.5 -2.803, -2.289, -1.486
        self.assertTrue('<td>Minimum</td>      <td>-2.8030</td>      <td>-2.2890</td>      <td>-1.4860</td>' in result['descriptives table'])
            # jamovi 2.0.0.0 0.962, 0.398; 0.536, 0.593; 1.145, 0.363
        self.assertTrue('<i>F</i>(2, 21) = 0.96, <i>p</i> = .398' in result['hypothesis test'])
        self.assertTrue('<i>F</i>(2, 21) = 0.54, <i>p</i> = .593' in result['hypothesis test'])
        self.assertTrue('<i>F</i>(4, 21) = 1.15, <i>p</i> = .363' in result['hypothesis test'])

    def test_single_case(self):

        # TODO validate
        # not available in JASP or in jamovi
        # Test for the slope stat
        data = cs.CogStatData(data='''group	slope	slope_SE
Patient	0.247	0.069
Control	0.492	0.106
Control	0.559	0.108
Control	0.63	0.116
Control	0.627	0.065
Control	0.674	0.105
Control	0.538	0.107''')
        result = data.compare_groups('slope', ['group'], single_case_slope_SE='slope_SE', single_case_slope_trial_n=25)
        self.assertTrue('Test d.2: <i>t</i>(42.1) = -4.21, <i>p</i> &lt; .001' in result['hypothesis test'])
        result = data.compare_groups('slope', ['group'])
        self.assertTrue('<i>t</i>(5) = -5.05, <i>p</i> = .004' in result['hypothesis test'])

    def test_compare_variables_groups(self):
        """Test compare groups"""

        # 2 Int groups
        result = data.compare_variables_groups(var_names=['a', 'e', 'f'], grouping_variables=['i'],
                                               display_factors=[['i', ('Unnamed factor')], [], []])
            # jamovi 2.4.11 2.92, 0.099; 4.546, 0.015; 0.281, 0.756
            # SPSS gives more similar result for the main within subject effect
        self.assertTrue('Main effect of i: <i>F</i>(1, 28) = 2.92, <i>p</i> = .099' in result['hypothesis test'])
        self.assertTrue('Main effect of Unnamed_factor: <i>F</i>(2, 56) = 3.10, <i>p</i> = .053' in
                        result['hypothesis test'])
        self.assertTrue('Interaction of i and Unnamed_factor: <i>F</i>(2, 56) = 0.28, <i>p</i> = .756' in
                        result['hypothesis test'])

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
        self.assertTrue('will be excluded: -6.0  –  20.0' in result['analysis info'])

    def test_reliability_internal(self):
        result = data.reliability_internal(['a', 'e', 'f', 'g'], reverse_items=['e'])
        self.assertTrue('N of observed cases: 30' in result['raw data info'])
        self.assertTrue('N of missing cases: 0' in result['raw data info'])
        self.assertTrue('Reverse coded items: e')

        # Cronbach's alpha Jamovi 2.2.5 alpha=0.429 [-0.023, 0.702]
        # TODO CI value deviation from JASP around zero is acceptable?
        self.assertTrue("Cronbach's alpha = 0.429" in result['descriptives'])  # Sample
        self.assertTrue('<td>0.429</td>      <td>[0.003, 0.702]</td>' in result['estimation table'])  # Population

        # Cronbach's alpha when item removed with CIs, and item-rest correlations with CIs
        # Jamovi 2.2.5 Cronbach's alpha if item dropped top to bottom: 0.517, 0.276, 0.381, 0.235
        # Jamovi 2.2.5 Item-rest correlations top to bottom: 0.076, 0.3303, 0.2245, 0.3617
        # TODO validate CIs
        self.assertTrue('<td>0.517</td>      <td>[0.114, 0.754]</td>      <td>0.076</td>      <td>[-0.292, 0.425]</td>'
                        in result['estimation table'])
        self.assertTrue('<td>0.276</td>      <td>[-0.327, 0.631]</td>      <td>0.330</td>      <td>[-0.034, 0.617]</td>'
                        in result['estimation table'])
        self.assertTrue('<td>0.381</td>      <td>[-0.135, 0.685]</td>      <td>0.224</td>      <td>[-0.148, 0.541]</td>'
                        in result['estimation table'])
        self.assertTrue('<td>0.235</td>      <td>[-0.402, 0.611]</td>      <td>0.362</td>      <td>[0.002, 0.639]</td>'
                        in result['estimation table'])


    def test_reliability_interrater(self):
        # TODO validate! Pingouin, jamovi and JASP all give different ICC values and CIs.
        # The difference is greater at lower values (i.e. around 0), and negligible at medium to higher values.

        # ICC1,1 and CI and assumption tests
        result = data.reliability_interrater(var_names=['a', 'e', 'f', 'g'], ratings_averaged=False)
        self.assertTrue('N of observed cases: 30' in result['raw data info'])
        self.assertTrue('N of missing cases: 0' in result['raw data info'])
        # Assumption tests (same for all ICC types)
        self.assertTrue('Shapiro–Wilk normality test in variable a: <i>W</i> = 0.96, <i>p</i> = .287' in result['assumption'])
        self.assertTrue('Shapiro–Wilk normality test in variable e: <i>W</i> = 0.97, <i>p</i> = .435' in result['assumption'])
        self.assertTrue('Shapiro–Wilk normality test in variable f: <i>W</i> = 0.82, <i>p</i> &lt; .001' in result['assumption'])
        self.assertTrue('Shapiro–Wilk normality test in variable g: <i>W</i> = 0.95, <i>p</i> = .133' in result['assumption'])
        self.assertTrue('Levene test: <i>W</i> = 0.19, <i>p</i> = .904' in result['assumption'])
        # Results
        self.assertTrue('-0.070' in result['descriptives table'])  # Sample
        self.assertTrue('<td>-0.070</td>      <td>[-0.170, 0.090]</td>' in result['estimation table'])  # Population
        self.assertTrue('<i>F</i>(29, 90) = 0.74, <i>p</i> = .820' in result['hypothesis test'])  # Hypothesis test

        # ICC2,1 and CI
        # Results
        self.assertTrue('-0.034' in result['descriptives table'])  # Sample
        self.assertTrue('<td>-0.034</td>      <td>[-0.130, 0.120]</td>' in result['estimation table'])  # Population
        self.assertTrue('<i>F</i>(29, 87) = 0.85, <i>p</i> = .682' in result['hypothesis test'])  # Hypothesis test

        # ICC3,1 and CI
        # Results
        self.assertTrue('-0.039' in result['descriptives table'])  # Sample
        self.assertTrue('<td>-0.039</td>      <td>[-0.150, 0.140]</td>' in result['estimation table'])  # Population
        self.assertTrue('<i>F</i>(29, 87) = 0.85, <i>p</i> = .682' in result['hypothesis test'])  # Hypothesis test

        # ICC1,k and CI
        result = data.reliability_interrater(var_names=['a', 'e', 'f', 'g'], ratings_averaged=True)
        # Results
        self.assertTrue('-0.352' in result['descriptives table'])  # Sample
        self.assertTrue('<td>-0.352</td>      <td>[-1.350, 0.290]</td>' in result['estimation table'])  # Population
        self.assertTrue('<i>F</i>(29, 90) = 0.74, <i>p</i> = .820' in result['hypothesis test'])  # Hypothesis test

        # ICC2,k and CI
        # Results
        self.assertTrue('-0.149' in result['descriptives table'])  # Sample
        self.assertTrue('<td>-0.149</td>      <td>[-0.820, 0.360]</td>' in result['estimation table'])  # Population
        self.assertTrue('<i>F</i>(29, 87) = 0.85, <i>p</i> = .682' in result['hypothesis test'])  # Hypothesis test

        # ICC3,k and CI
        # Results
        self.assertTrue('-0.176' in result['descriptives table'])  # Sample
        self.assertTrue('<td>-0.176</td>      <td>[-1.050, 0.390]</td>' in result['estimation table'])  # Population
        self.assertTrue('<i>F</i>(29, 87) = 0.85, <i>p</i> = .682')


if __name__ == '__main__':
    unittest.main()
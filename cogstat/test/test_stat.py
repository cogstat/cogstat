# -*- coding: utf-8 -*-

import unittest
import cogstat

"""
- All statistical value should be tested at least once.
- All leafs of the decision tree should be tested once.
- No need to test the statistical procedures imported from other modules for 
various numerical situations,since that is the job of that specific module.
"""

# TODO all tests should be checked at least once
# TODO check all branches in cogstat stat compilation

cogstat.output_type = 'do not format'

data = cogstat.CogStatData(data='data/test_data.csv')

class CogStatTestCase(unittest.TestCase):
    """Unit tests for CogStat."""
    
    def test_explore_variables(self):
        """Test explore variables"""
        # TODO these should be different functions

        # Int variable
        result = data.explore_variable('A', [1, 1, 1, 1, 1, 0])
        self.assertTrue('N of valid cases: 30' in result[5])
        self.assertTrue('N of invalid cases: 0' in result[5])
        self.assertTrue('Mean: 5.2019' in result[5])
        self.assertTrue('Standard deviation: 1.7911' in result[5])
        self.assertTrue('Skewness: 0.3455' in result[5])
        self.assertTrue('Kurtosis: -0.7492' in result[5])
        self.assertTrue('Median: 4.8255' in result[5])
        self.assertTrue('Range: 6.552' in result[5])
        # TODO Shapiro-Wilk normality
        # W and p
        # older code for Anderson-Darling normality test
        #self.assertTrue('<i>p</i> = 0.147' in result[4])
        self.assertTrue('Confidence interval [4.5331, 5.8707]' in result[9])
        # One sample t-test
        self.assertTrue(') = 15.9' in result[9])
        self.assertTrue(' &lt; 0.001' in result[9])
        
        # Ord variable
        # TODO
        result = data.explore_variable('D', [1, 1, 1, 1, 1, 0])
        # Median
#        self.assertTrue(result[3][result[3].find('Median: ')+8:][:6], '4.8255')
        # Range
#        self.assertTrue(result[3][result[3].find('Range: ')+7:][:5], '6.552')
        # TODO Wilcoxon test
#        self.assertTrue(result[5][result[5].find(') = ')+4:][:4], '15.9')
#        self.assertTrue(result[5][result[5].find(' < ')+3:][:5], '0.001')

         # print result
         # from IPython import embed; embed()

    def test_explore_variable_pairs(self):
        """Test explore variable pairs"""
        
        # Int variables
        result = data.explore_variable_pair('A', 'B')
        self.assertTrue("Pearson's correlation: <i>r</i>(28) = -0.121, 95% CI [-0.461, 0.250], <i>p</i> = 0.525" in result[1])
        self.assertTrue('y = -0.122x + 3.529' in result[1])
        self.assertTrue("Spearman's rank-order correlation: <i>r</i>(28) = -0.010, 95% CI [-0.369, 0.352], <i>p</i> = 0.960" in result[1])

        # Ord variables
        # TODO

        # Nom variables
        result = data.explore_variable_pair('G', 'H')
        # Cramer's V
        self.assertTrue('<sub>c</sub></i> = 0.348' in result[1])
        # Chi-square
        self.assertTrue('(2, <i>N</i> = 30) = 3.643' in result[1])
        self.assertTrue('<i>p</i> = 0.162' in result[1])

        # print result
        # from IPython import embed; embed()

    def test_compare_variables(self):
        """Test compare variables"""
        
        # 2 Int variables
        result = data.compare_variables(['A', 'B'])
        # print result
        # from IPython import embed; embed()
        # Paired t-test
        self.assertTrue('(29) = 4.69' in result[5])
        self.assertTrue(' &lt; 0.001' in result[5])

        # TODO case vs. group

        # 2 Ord variables
        result = data.compare_variables(['D', 'E'])
        # Wilcoxon signed rank test
        self.assertTrue('W</i> = 118' in result[4])
        self.assertTrue('p</i> = 0.019' in result[4])

        # 3 Ord variables
        result = data.compare_variables(['D', 'E', 'F'])
        # Wilcoxon signed rank test
        # TODO not appropriate value
        self.assertTrue('(2, <i>N</i> = 30) = 2.87' in result[4])
        self.assertTrue('<i>p</i> = 0.239' in result[4])

         # print result
         # from IPython import embed; embed()

    def test_compare_groups(self):
        """Test compare groups"""

        # 2 Int groups
        result = data.compare_groups('A', 'G')
        # TODO Levene is different in SPSS and RopStat and both of them deviates from CS
        # TODO confidence interval
        # independent samples t-test
        self.assertTrue(') = -1.82' in result[5])
        self.assertTrue(' = 0.079' in result[5])

        # 3 Int groups
        result = data.compare_groups('A', 'H')
        self.assertTrue('(2, 27) = 1.61' in result[5])
        self.assertTrue('<i>p</i> = 0.218' in result[5])
        # TODO omega2

        # 2 Ord groups
        result = data.compare_groups('D', 'G')
        # Kruskall-Wallis test
        # TODO CS deviates from others
        self.assertTrue('<i>U</i> = 98' in result[4])
#        self.assertTrue('p</i> = 0.621' in result[4])

        # 3 Ord groups
        result = data.compare_groups('D', 'H')
        # TODO CS deviates from others
#        self.assertTrue(result[5][result[5].find(') = ')+4:][:5], '3.124')
#        self.assertTrue(result[5][result[5].find('p</i> = ')+8:][:5], '0.210')

         # print result
         # from IPython import embed; embed()

if __name__ == '__main__':
    unittest.main()

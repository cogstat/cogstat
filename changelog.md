Upcoming release
===============
## New features
- Repeated measures ANOVA (Ákos Laczkó)
    - Repeated measures one-way ANOVA
    - Mauchly's test for sphericity
    - Greenhouse-Geisser correction
    - Post-hoc pairwise comparison 
- Confidence interval for correlation coefficients
- McNemar test
- Cochran's Q test
- Welch's t-test
- Output improvements
    - More informative and simpler text when checking assumptions in hypothesis tests

1.3.1 (16 January 2016)
===============
## Fixes
- Various bugfixes
- Updated Language files
    - German (László Veller)
	- Romanian (Orsolya Kiss)

1.3.0 (4 September 2015)
===============
## New features
- IPython Notebook integration - see https://github.com/cogstat/cogstat/wiki/IPython-Notebook
    - All CogStat functions are available in IPython Notebook
    - Import pandas data frame (only in IP NB mode)
- Filtering based on single variable outlier (only in IP NB mode at the moment)
- New localization
    - Bulgarian (thanks to Petia Kojouharova)

## Fixes
- Fix memory leak after several analyses (less memory is required now)
- Refactoring
    - Rework of the data import part (uses pandas now), resulting in faster import
- Output improvements
    - Proper html tables are used in most analyses
    - Various minor improvements
- Various minor bugfixes

1.2.0 (22 May 2015)
================
## New features
- Statistics improvements
    - Improved normality test: Shapiro-Wilk test instead of Anderson-Darling test (thanks to Ákos Laczkó)
    - Confidence interval for the difference between two groups
- Output improvements
    - Individual data signs are proportional with the number of cases for a value in  explore variables, compare variables and compare groups functions
    - Titles for all graphs
    - For interval variable mean graphs with confidence interval are shown in variable and group comparisons
    - Wrap labels on graph axis
    - In some cases more concise output with tables. It is also easier now to copy those results to a spreadsheet
    - Confidence interval (numerical result) is in APA format now
- New localization
    - German (László Veller)

## Fixes
- New unit tests are run automatically before each release to ensure that statistics are calculated correctly (thanks to Gábor Lengyel)
- Various bugfixes

1.1.0 (18 August 2014)
================
## New features
- Use modified t-test when comparing a single case and a group (available in group comparison) (thanks to Judit Kárpáti)
- New functions for pivot tables: lower quartile and upper quartile
- Localizations of CogStat
	- All menus and texts (except the dialogs) are localized
	- Initial support for new languages:
		- Hungarian (Attila Krajcsi)
		- Italian (Eszter Temesvári)
		- Romanian (Orsolya Kiss)
	- New command: CogStat > Preferences to set the language

## Fixes
- Correct median in pivot tables for variables including missing data
- Various bugfixes

1.0.0 (16 July 2014)
=====

## New features

### Installation
- Installer for Windows
- Run from source on Linux

### Data import
- From clipbord (copied from a spreadsheet software)
- From csv

### Data
- Display data (optionally only the first few lines)

### Analysis
- Explore variable
	- Frequency
	- Distribution (histogram, raw data points, box plot)
	- Descriptives (N, mean, standard deviation, skewness, kurtosis, median, range)
	- Test normality (Anderson-Darling test, histogram with normality test, Q-Q plot)
	- Test central tendency (one sample t-test, confidence interval of the mean, Wilcoxon sign test)
- Explore variable pairs
	- Pearson and Spearman coefficients
	- Linear regression parameters
	- Scatter plot or mosaic plot
- Pivot
	- Functions: N, sum, mean, median, standard deviation, variance
- Compare variables
	- Descriptives: means, medians or contingency table
	- Diagrams: boxplot with individual data or mosaic plot
	- Hypothesis tests: paired t-test, paired Wilcoxon test, Friedman test
- Compare groups
	- Descriptives: means, medians or contingency tables
	- Diagrams: boxplot with individual data or contingency table
	- Hypothesis tests: independent smaples t-test, Mann-Whitney test, Chi-square test, one sample ANOVA, Kruskal-Wallis test

### Results
- Clear window
- Save results as pdf

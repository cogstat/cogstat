:warning: This symbol means that CogStat will handle data differently compared to previous releases.
Trivial changes when a new feature is added are not denoted. 

Upcoming release
================

## New features
- Behavioral data diffusion analysis
- Sensitivity power analysis for one-sample t-test, two-sample t-test, paired samples t-test, Chi-square test, one-way ANOVA
- Variation ratio for nominal variables
- Various output refinements
- GUI
    - Find text in results

## Fixes
- Various output fixes

1.9.0 (30 January 2020)
================

## New features
- Multi-way repeated measures ANOVA
- Outlier filtering is available from Data > Filter outliers... menu
- Output may be edited (Results > Text is editable menu)
- GUI
    - Add icons for the menus
    - Add a toolbar
- New localizations
    - Norwegian Bokmål (Irmelin Hovland-Hegg)
    - Russian (Nikolay Kuzmenko, Danagul Duisekenova)
    - Kazakh (Danagul Duisekenova)
    - Estonian (Deniss Kovaljov)
    - Updated formerly abandoned Italian version (Roland Kasza, Andrea Bortoletti) and Romanian version (Borbála Tölgyesi)

## Fixes
- Usability refinements
- Dialogs are localized now
- Various bugfixes

1.8.0 (9 Apr 2019)
================

## New features
- Charts
    - Use themes for the charts
    - Set theme in CogStat > Preferences menu
    - Various chart refinements
- New hypothesis tests
    - Dunn's post hoc test after significant Kruskal-Wallis test
    - Single case test for slope index
- Numerical results
    - Display mean estimations numerically in group comparison and in repeated measures comparison 
    - Display standardized effect sizes separately
    - Various smaller refinements
- Output improvements
    - In warning massages add links to pages with more information about fixing the issue
    - Various smaller output refinements
- Sample data
    - Add sample data files
    - Add menu to open sample data files (Data > Open demo data files...)
- Graphical user interface improvements
    - Add text zooming option to Results menu (Results > Increase text size, Results > Decrease text size)
    - Add splash screen
- Simpler installation
    - Simpler Mac installation (thanks to Márton Nagy, Anna Rákóczi and András Csép)
    - Simpler Linux installation
- New localizations
    - Slovak (Katarína Sümegiová)
    - Thai (Jinshana Praemcheun)

# Fixes
- :warning: Fix single-case modified t-test
- Various bugfixes
- Mac specific bug fixes (thanks to Márton Nagy, Anna Rákóczi and András Csép)

1.7.0 (18 June 2018)
================

## New features
- Import SPSS .sav file
- Two-way group comparisons
- Dotted axis for nominal variables and for grouping variables
- More numerical descriptives for explore variables, repeated measures variable comparison and group comparison analyses
- Update checker
- Various output refinements

## Fixes
- Various bugfixes

1.6.0 (11 December 2017)
=================

## New features
- Standard deviation in Explore variable
    - :warning: SD of the sample is not population estimation anymore
    - Add population estimation of the SD
- In Explore variable, for ordinal variables only minimum and maximum is calculated instead of the range
- Population estimations are displayed in tables
- More coherent output for hypothesis tests
- Mac support (thanks to Magdolna Kovács)

## Fixes
- Localization fixes
- Output fixes

1.5.0 (30 September 2017)
===============
## New features
- In an analysis, raw data are diplayed first without any graphical addition
- Display results in groups of (1) raw data, (2) sample properties and (3) population properties 
- Rank information is displayed for ordinal variables (in single variable, in variable pairs (in scatterplot) and in group comparison)
- Aim of the hypothesis tests are displayed.
- Use Wilcoxon signed-rank test when normality is violated in an interval variable

## Fixes
- Spearman rank correlation is denoted as rs (s is subscript) instead of r.
- When quitting CogStat, no confirmation is asked 
- Various smaller fixes

1.4.2 (3 June 2017)
==============
## New features
- :warning: One sample Wilcoxon signed-rank test now shows the T statistic, instead of the W statistic.
- New localization (test versions, they include some bugs)
    - Croatian (Sandra Stojić)
    - Persian (Sara Hosseininezhad)

## Fixes
- Statistics fixes
    - One sample Wilcoxon signed-rank test is available on Windows version, too (Szonja Weigl)
    - Fix test statistic name (show T instead of W) in paired Wilcoxon test
    - :warning: Paired sample t-test assumption check now checks normality of the difference of the variables
- Updated links in the CogStat menu
- Localization related issues
    - Fix some RTL languages related issues
    - Reworked German version (Roman Ricardo Pota)
- Various smaller fixes

1.4.1 (17 April 2016)
===============
## Fixes
- Bugfix: repeated measures ANOVA does not run on Windows 

1.4.0 (16 April 2016)
===============
## New features
- Repeated measures ANOVA (Ákos Laczkó)
    - Repeated measures one-way ANOVA
    - Mauchly's test for sphericity
    - Greenhouse-Geisser correction
    - Post-hoc pairwise comparison with Holm-Bonferroni correction 
- Confidence interval for correlation coefficients
- McNemar test
- Cochran's Q test
- Welch's t-test
- Output improvements
    - More informative and simpler text when checking assumptions in hypothesis tests
- New localization
    - Hebrew (Tzipi Buchman)

## Fixes
- Various bugfixes
- Updated Language files
    - Bulgarian (Petia Kojouharova)
    - Hungarian


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
    - Bulgarian (Petia Kojouharova)

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
    - :warning: Improved normality test: Shapiro-Wilk test instead of Anderson-Darling test (thanks to Ákos Laczkó)
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

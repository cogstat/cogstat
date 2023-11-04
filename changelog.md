:warning: This symbol means that CogStat will handle data differently compared to previous releases.
Trivial changes when a new feature is added are not denoted.

2.4.1 (29 October 2023)
=======================

## Fixes
- Data import fixes 

2.4 (12 September 2023)
=======================
## New features
- Data handling
    - New data view to see the data together with the results (thanks to Belma Bumin)
    - Reload actual data file
    - Multivariate outlier filtering with Mahalanobis distance (Tamás Szűcs)
    - New demo data files https://learningstatisticswithcogstat.com/ (Róbert Fodor)
- Ability to rerun the analyses in the Results pane
- Multiple linear regression analysis (Tamás Szűcs)
    - Scatterplot matrix of raw data
    - Linear regression function
    - Scatterplot with regression line
    - Partial regression plots with regression lines
    - Model fit metrics
    - Partial correlations
    - Residual plot and histogram of residuals
    - Assumptions of inferential statistics
      - Multivariate normality
      - Homoscedasticity
      - Analysis of multicollinearity
    - Population parameter point and interval estimations (including standardized effect sizes)
    - Hypothesis tests
- Reliability analyses (Tamás Szűcs)
    - Internal consistency reliability analysis
        - Item-total scatter plots
        - Cronbach's alpha with and without items and their CIs
        - Item-rest correlation and their CIs
    - Interrater reliability analysis
        - Chart showing scores from different raters
        - ICC values and their CIs
        - Assumption checks for inferential statistics
        - Hypothesis tests whether ICC is 0
- Displaying groups and factors
    - In comparing groups, display groups not only on x-axes but also with colors or in panels
    - In comparing repeated measures variables, display conditions not only on x-axes but also with colors
        - Rearrange the factors flexibly
    - For ordinal repeated measures variables, display the rank of the values
- Comparing variables and groups in mixed design
    - Raw data
    - Descriptives and related charts
    - Parameter estimations and related charts
- Behavioral data diffusion analysis
    - The time unit (sec or msec), error coding (1 or 0), and scaling parameter (0.1 or 1) can be set
    - Slow trials are filtered before the analysis is run
    - Display the number of filtered (missing and slow outlier) trials
    - Number of included trials per conditions are displayed
- Output handling
    - Save results into html file instead of pdf file (Róbert Fodor)
    - Ability to use png or svg image formats for charts (experimental svg support)
- Possibility to print detailed Python error messages to results pane 
- New localization
    - Chinese (Xiaomeng Zhu)
    - Malay (Nur Hidayati Miza binti Junaidi)
    - Arabic (Rahmeh Albursan)
- Python package
    - Pandas DataFrames with MultiIndex columns can be imported
    - Diffusion analysis results are returned as pandas Stylers

## Fixes
- :warning: In outlier filtering, the cases with the limit value will be included and not excluded
- :warning: With the update of the scipy module, the p values of the Wilcoxon tests are fixed
- Extended calculation validations (thanks to Eszter Miklós)
- Most settings in Preferences are applied without the need to restart
- Various GUI, and output fixes

2.3.0 (23 July 2022)
===============

## New features
- Initial support for Bayesian hypothesis tests
    - One sample t-test
    - Pearson correlation
    - Paired two-samples t-test
    - Independent two-samples t-test
- Extended regression analyses (Tamás Szűcs)
    - Residual plot
    - Confidence intervals for regression parameters
    - Population plot with confidence band of regression line
    - :warning: Henze-Zirkler test for assumption of multivariate normality
    - White's test and Koenker's test for assumption of homoscedasticity
- Display the filtered cases when filtering outliers
- Post hoc Durbin-Conover test after significant Friedman test
- New localization
    - Turkish (Belma Feride Bumin)


## Fixes
- :warning: Sensitivity power analysis for Chi-squared test now uses the correct df and makes sure to use w as effect size
- Various GUI, data import, analysis, and output fixes
- Extended calculation validations (thanks to Dóra Hatvani)
- Run from source more simply (thanks to Oliver Lindemann)
- Various usability fixes (thanks to Ádám Szimilkó)
- Mac-specific fixes (Róbert Fodor)

2.2.0 (1 November 2021)
================

## New features
- :warning: For outlier filtering, median +- 2.5 * MAD is used instead of using mean +- 2 * SD
- One-way ANOVA's sensitivity power analysis displays eta-square too
- Missing values are excluded from all analyses
    - :warning: In Explore variable, relative frequency is calculated without missing cases
    - :warning: In Explore variable, population parameter estimations of nominal variable values are calculated without missing cases

## Fixes
- Various UI and output fixes
- Missing cases related fixes
    - :warning: In Explore variable, Wilcoxon signed-rank test p value is fixed when there are missing cases
    - :warning: In Compare repeated measures variables, Hedges'g CI is fixed when there are missing cases
    - :warning: In Compare repeated measures variables, CIs are fixed when there are missing cases for more than two nominal variables
    - :warning: In Compare groups, the mosaic plots of nominal variables do not show value combinations where other variable value is missing
    - :warning: In Compare groups, Cramér's V is fixed when there are missing cases
    - Some statistics were not calculated when there were missing cases

2.1.1 (10 September 2021)
================

## Fixes
- :warning: Multi-way between-subjects ANOVA is fixed – previous releases gave incorrect F and p values for the main effects
- :warning:  Holm-Bonferroni corrected post-hoc tests is performed with the statsmodels module (in some cases, this gives a slightly different result compared with the previous versions)
- Resizable dialogs with shortcuts for buttons and correctly set tab orders
- Various UI, output, and chart fixes
- MacOS app version is now signed (Róbert Fodor)

2.1.0 (21 May 2021)
================

## New features
- New import data file formats
    - Excel spreadsheet .xls and .xlsx files
    - OpenDocument Spreadsheet .ods files
    - SPSS .zsav and .por files
    - JASP .jasp files
    - jamovi .omv files
    - R .rdata, .rds, and .rda files
    - SAS .sas7bdat and .xpt files
    - STATA .dta files
    - Additional text (e.g., .csv) formats
- More demo datasets are provided
- Confidence interval of the standard deviation in Explore variable
- Analysis refinements
    - In behavioral data diffusion analysis, participants and conditions are optional parameters
- New localizations
    - Greek (Zoé Vulgarasz)
    - Spanish (Borbála Zulauf)
    - French (Minka Petánszki)
- :warning: Modified APIs
    - CogStatData() measurement_levels parameter
    - CogStatData.pivot() depend_name parameter
    - CogStatData.compare_groups()  single_case_slope_SE parameter

## Fixes
- :warning: Behavioral data diffusion analysis
    - EZ parameter recovery gave incorrect result when error rate was 50%
    - If any data is missing from a trial, the whole trial is dropped
    - Rows are now ordered to be case-insensitive
    - All tables have the same column order
- Various output fixes
    - :warning: In pivot tables, rows are now ordered to be case-insensitive
- Various data import fixes
- Performance improvements
- On Windows, much smaller installer and much less required space

2.0.0 (13 July 2020)
================

## New features
- Statistical analysis
    - Any number of grouping variables in Compare groups
    - New effect sizes for interval variables for two groups or variables
        - Cohen's d
        - Eta-squared
        - Hedges' g (with 95% CI)
    - Sensitivity power analysis for one-sample t-test, two-sample t-test, paired samples t-test, chi-squared test, one-way ANOVA
    - Confidence intervals
        - Confidence interval for multinomial proportions (in Explore variable, Explore relation of variable pair, Compare repeated measures variables, Compare groups)
        - Confidence interval for medians (in Explore variable, Compare repeated measures variables and Compare groups)
    - Contingency table improvements (in Explore relation of variable pair, Compare repeated measures variables, Compare groups)
        - Display margins
        - Display percentage
        - Confidence interval for multinomial proportions (see confidence interval improvements above)
    - Variation ratio for nominal variables in Explore variable, Compare repeated measures variables and Compare groups
    - Various output refinements
- Data analysis
    - Behavioral data diffusion analysis
- Charts
    - Set minimum and maximum axes values for interval and ordinal variables in variable pair, compare repeated measures and compare groups analyses
    - Various chart refinements
- GUI
    - Find text in results
- New localization
    - Korean (Katalin Wild)
- Installer
    - Mac installer (27 September 2020) (Róbert Fodor)

## Fixes
- Various output fixes
- Various bugfixes

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
    - Dunn's post hoc test after significant Kruskal–Wallis test
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
- Rank information is displayed for ordinal variables (in single variable, in variable pair (in scatterplot) and in group comparison)
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
    - Greenhouse–Geisser correction
    - Post-hoc pairwise comparison with Holm–Bonferroni correction
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
- IPython Notebook integration - see https://doc.cogstat.org/IPython-Notebook
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
    - :warning: Improved normality test: Shapiro–Wilk test instead of Anderson–Darling test (thanks to Ákos Laczkó)
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
    - Test normality (Anderson–Darling test, histogram with normality test, Q-Q plot)
    - Test central tendency (one sample t-test, confidence interval of the mean, Wilcoxon sign test)
- Explore variable pair
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
    - Hypothesis tests: independent samples t-test, Mann–Whitney test, chi-squared test, one sample ANOVA, Kruskal–Wallis test

### Results
- Clear window
- Save results as pdf

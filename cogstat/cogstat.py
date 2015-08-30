# -*- coding: utf-8 -*-

"""
Class for the CogStat data (with import method) and methods to compile the 
appropriate statistics for the main analysis commands.
"""

import csv
import gettext
import logging
from distutils.version import LooseVersion
import os

__version__ = '1.3.0'

import cogstat_config as csc
csc.versions['cogstat'] = __version__
import cogstat_stat as cs_stat
import cogstat_util as cs_util
cs_util.get_versions()

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.figure import Figure

from PyQt4 import QtGui

logging.root.setLevel(logging.INFO)

rcParams['figure.figsize'] = csc.fig_size_x, csc.fig_size_y

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.ugettext

warn_unknown_variable = '<warning>'+_('The properties of the variables are not set. Set them in your data source.')+'\n<default>' # XXX ezt talán elég az importnál nézni, az elemzéseknél lehet már másként.

output_type = 'ipnb'  # if run from GUI, this is switched to 'gui'
                    # any other code will leave the output (e.g., for testing)


class CogStatData:
    """Class to process data."""
    def __init__(self, data='', measurement_level=''):
        """
        In the input data:
        - First line should be the variable name
        --- If there are missing names, Unnamed:0, Unnamed:1, etc. names are given
        --- If there are repeating var names, new available numbers are added, e.g. a.1, a.2, etc.
        - Second line could be the measuring level

        Data structure:
        self.data_frame - pandas DataFrame
        self.data_measlevs - dictionary storing level of measurement of the variables (name:level):
                'nom', 'ord' or 'int'(ratio is included in 'int')
                'unk' - unknown: if no other level is given
        self.orig_data_frame # TODO
        self.filtering_status # TODO

        self.import_source - text info about the import source
        self.import_message - any text warning about the imported data

        data should be
        - filename (one line text)
        - or multiline string (usually clipboard data from spreadsheet)
        - or pandas DataFrame

        measurement_level:
        - if measurement level is set,then use it
        - otherwise look for it in the file/multiline string
        - otherwise set the types to nom and unk
        """

        self.orig_data_frame = None
        self.data_frame = None
        self.data_measlevs = None
        self.import_source = ''
        self.import_message = ''  # can't return anything to caller, since we're in an __init__ method, so store the message here
        self.filtering_status = None

        self._import_data(data=data, param_measurement_level=measurement_level)

    ### Import and handle the data ###

    def _import_data(self, data='', param_measurement_level=''):

        delimiter = '\t'
        quotechar = '"'

        def percent2float():
            """ Convert x.x% str format to float in self.data_frame (pandas cannot handle this).
            """
            # TODO A pandas expert could shorten this
            for column in self.data_frame.columns:
                if self.data_frame[column].dtype == 'object':
                    selected_cells = self.data_frame[column].str.endswith('%')
                    if selected_cells.any():
                        # use  selected_cells == True  to overcome the Nan indexes
                        self.data_frame[column][selected_cells == True] = \
                            self.data_frame[column][selected_cells == True].str.replace('%', '').astype('float')/100.0

        def set_measurement_level(measurement_level=''):
            """ Create self.data_measlevs
            measurement_level:
                a single string with measurement levels  (int, ord, nom, unk) in the order of the variables
                or two lines with the names in the first line, and the levels in the second line
            """
            # 1. Set the levels
            if measurement_level:  # If levels were given, set them
                if '\n' in measurement_level:  # Name-level pairs are given in two lines
                    names, levels = measurement_level.splitlines()
                    if len(names.split()) == len(levels.split()):
                        self.data_measlevs = {name: level for name, level in zip(names.split(), levels.split())}
                    else:
                        self.import_message += '\n<warning>' + _('Number of measurement level do not match the number of variables. Measurement level specification is ignored.')
                        measurement_level = ''
                else:  # Only levels are given - in the order of the variables
                    if len(measurement_level.split()) == len(self.data_frame.columns):
                        self.data_measlevs = {name: level for name, level in zip(self.data_frame.columns,
                                                                             measurement_level.split())}
                    else:
                        self.import_message += '\n<warning>' + _('Number of measurement level do not match the number of variables. Measurement level specification is ignored.')
                        measurement_level = ''
            if not measurement_level:  # Otherwise (or if the given measurement level is incorrect) set them to be nom if type is a str, unk otherwise
                self.data_measlevs = {name: (u'nom' if self.data_frame[name].dtype == 'object' else u'unk') for name in self.data_frame.columns}
                # TODO Does the line above work? Does the line below work ?
                #self.data_measlevs = dict(zip(self.data_frame.columns, [u'nom' if self.data_frame[name].dtype == 'object' else u'unk' for name in self.data_frame.columns]))
                self.import_message += '\n<warning>'+warn_unknown_variable+'<default>'

            # 2. Check for inconsistencies in measurement levels.
            # If str var is set to int or ord set it to nom
            invalid_data = []
            for var_name in self.data_frame.columns:
                if self.data_measlevs[var_name] in ['int', 'ord', 'unk'] and self.data_frame[var_name].dtype == 'object':
                    # 'object' dtype means string variable
                    invalid_data.append(var_name)
            if invalid_data:  # these str variables were set to int or ord
                for var_name in invalid_data:
                    self.data_measlevs[var_name] = 'nom'
                self.import_message += '\n<warning>'+_(u'String variables cannot be interval or ordinal variables. Those variables are automatically set to nominal: ')+''.join(', %s'%var_name for var_name in invalid_data)[2:]+'. '+_(u'You might consider fixing this in your source table.')

            if set(self.data_measlevs) in ['unk']:
                self.import_message += '\n<warning>'+_('The measurement level was not set for all variables. You might consider fixing this in your data source.')+'<default>'

        file_measurement_level = ''
        # Import from pandas DataFrame
        if isinstance(data, pd.DataFrame):
            self.data_frame = data
            self.import_source = _('pandas dataframe')
        elif isinstance(data, basestring):

            # Import from file
            if not ('\n' in data):  # Single line text, i.e., filename
                filetype = data[data.rfind('.'):]
                if filetype in ['.txt', '.csv', '.log', '.tsv']:
                    # Check if the file exists # TODO
                    # self.import_source = _('Import failed')
                    # return

                    # Check if there is variable type line
                    f = csv.reader(open(data, 'rb'), delimiter=delimiter, quotechar=quotechar)
                    f.next()
                    meas_row = f.next()
                    if set([a.lower() for a in meas_row]) <= set(['unk', 'nom', 'ord', 'int', '']) and set(meas_row) != set(['']):
                        file_measurement_level = ' '.join(meas_row)
                    skiprows = [1] if file_measurement_level else None

                    # Read the file
                    self.data_frame = pd.read_csv(data, delimiter=delimiter, quotechar=quotechar, skiprows=skiprows)
                    self.import_source = _('text file - ')+data  # filename

            # Import from multiline string, clipboard
            else:  # Multi line text, i.e., clipboard data
                # Check if there is variable type line
                import StringIO
                f = StringIO.StringIO(data)
                f.next()
                meas_row = f.next().replace('\n', '').split(delimiter)
                if set([a.lower() for a in meas_row]) <= set(['unk', 'nom', 'ord', 'int', '']) and set(meas_row) != set(['']):
                    file_measurement_level = ' '.join(meas_row)
                skiprows = [1] if file_measurement_level else None

                # Read the clipboard
                clipboard_file = StringIO.StringIO(data)
                self.data_frame = pd.read_csv(clipboard_file, delimiter=delimiter, quotechar=quotechar,
                                              skiprows=skiprows)
                self.import_source = _('clipboard')

        else:  # Invalid data source
            self.import_source = _('Import failed')
            return

        # Set other details for all import sources
        percent2float()
        set_measurement_level(measurement_level=
                              (param_measurement_level if param_measurement_level else file_measurement_level))
                            # param_measurement_level overwrites file_measurement_level

        self.orig_data_frame = self.data_frame.copy()

        # Add keys with pyqt string form, too, because UI returns variable names in this form
        # TODO do we still need this?
        from PyQt4 import QtCore
        for var_name in self.data_frame.columns:
            self.data_measlevs[QtCore.QString(var_name)] = self.data_measlevs[var_name]

    def print_data(self, brief=False):
        """Print data."""
        output = csc.heading_style_begin + _('Data')+csc.heading_style_end
        output += '<default>'+_('Source: ') + self.import_source + '\n'
        output += str(len(self.data_frame.columns))+_(' variables and ') + \
                  str(len(self.data_frame.index))+_(' cases') + '\n'
        output += self._filtering_status()

        dtype_convert = {'int32': 'num', 'int64': 'num', 'float32': 'num', 'float64': 'num', 'object': 'str'}
        data_prop = pd.DataFrame([[dtype_convert[str(self.data_frame[name].dtype)] for name in self.data_frame.columns],
                                  [self.data_measlevs[name] for name in self.data_frame.columns]],
                                 columns=self.data_frame.columns)
        data_comb = pd.concat([data_prop, self.data_frame])
        data_comb.index = [_('Type'), _('Level')]+[' ']*len(self.data_frame)
        output += data_comb[:12 if brief else 1001].to_html(bold_rows=False).replace('\n', '').\
            replace('border="1"', 'style="border:1px solid black;"')
        if brief and (len(self.data_frame.index) > 10):
            output += str(len(self.data_frame.index)-10) + _(' further cases are not displayed...')+'\n'
        if len(self.data_frame.index) > 999:
            output += _('You probably would not want to print the next %s cases...') % \
                      (len(self.data_frame.index)-1000) + '\n'

        return self._convert_output([output+'<default>'])

    def filter_outlier(self, var_names=None, mode='2sd'):  # TODO GUI for this function
        """
        Filter the data_frame based on outliers
        :param var_names: list of name of the variable the exclusion is based on (list of str)
                        or None to include all cases
        :param mode: mode of the exclusion (str)
                only 2sd is available at the moment
        :return:
        """
        title = csc.heading_style_begin + _('Filtering')+csc.heading_style_end
        if var_names is None:  # Switch off outlier filtering
            self.data_frame = self.orig_data_frame.copy()
            self.filtering_status = None
            text_output = _('Filtering is switched off.')
        else:  # Create a filtered dataframe based on the variable
            filtered_data_indexes = []
            text_output = ''
            self.filtering_status = ''
            for var_name in var_names:
                # Currently only this simple method is used: cases with more than 2 SD difference are excluded
                mean = np.mean(self.orig_data_frame[var_name].dropna())
                sd = np.std(self.orig_data_frame[var_name].dropna(), ddof=1)
                filtered_data_indexes.append(self.orig_data_frame[
                    (self.orig_data_frame[var_name] < (mean + 2 * sd)) &
                    (self.orig_data_frame[var_name] > (mean - 2 * sd))].index)
                text_output += _('Filtering based on %s.\n') % (var_name + _(' (2 SD)'))
                prec = cs_util.precision(self.orig_data_frame[var_name])+1
                text_output += _('Cases outside of the range will be excluded: %0.*f  --  %0.*f\n') % \
                               (prec, mean - 2 * sd, prec, mean + 2 * sd)
                excluded_cases = self.orig_data_frame.loc[self.orig_data_frame.index.difference(filtered_data_indexes[-1])]
                #excluded_cases.index = [' '] * len(excluded_cases)  # TODO can we cut the indexes from the html table?
                # TODO uncomment the above line after using pivot indexes in CS data
                if len(excluded_cases):
                    text_output += _('The following cases will be excluded: ')
                    text_output += excluded_cases.to_html(bold_rows=False).replace('\n', '').\
                        replace('border="1"', 'style="border:1px solid black;"')
                else:
                    text_output += _('No cases were excluded.')
            self.data_frame = self.orig_data_frame.copy()
            for filtered_data_index in filtered_data_indexes:
                self.data_frame = self.data_frame.reindex(self.data_frame.index.intersection(filtered_data_index))
            self.filtering_status = ', '.join(var_names) +_(' (2 SD)')
            # TODO Add graph about the excluded cases based on the variable

        return self._convert_output([title, text_output])

    def _filtering_status(self):
        if self.filtering_status:
            return '<b>Filtering is on: %s</b>\n' % self.filtering_status
        else:
            return ''

    ### Handle output ###

    def _convert_output(self, outputs):
        """
        Convert output either to the GUI or to the IPython Notebook
        :param outputs: list of the output items
        :return: converted output, list of items
        """

        def _figure_to_qimage(figure):
            """Convert matplotlib figure to pyqt qImage.
            """
            figure.canvas.draw()
            size_x, size_y = figure.get_size_inches()*rcParams['figure.dpi']
            if LooseVersion(csc.versions['matplotlib']) < LooseVersion('1.2'):
                string_buffer = figure.canvas.buffer_rgba(0, 0)
            else:
                string_buffer = figure.canvas.buffer_rgba()
            return QtGui.QImage(string_buffer, size_x, size_y, QtGui.QImage.Format_ARGB32).copy()
                # this should be a copy, otherwise closing the matplotlib figures would damage the qImages on the GUI

        if output_type in ['ipnb', 'gui']:
            # convert custom notation to html
            new_output = []
            for i, output in enumerate(outputs):
                if isinstance(output, Figure):
                    # For gui convert matplotlib to qImage
                    new_output.append(output if output_type == 'ipnb' else _figure_to_qimage(output))
                elif isinstance(output, basestring):
                    new_output.append(cs_util.reformat_output(output))
                elif isinstance(output, list):  # flat list
                    new_output.extend(self._convert_output(output))
                elif output is None:
                    pass  # drop None-s from outputs
                else:  # No other types are expected
                    logging.error('Output includes wrong type: %s' % type(output))
            return new_output
        else:
            return outputs

    ### Various things ###

    def _meas_lev_vars(self, variables):
        """
        arguments:
        variables: list of variable names
        
        returns:
        meas_lev (string): the lowest measurement level among the listed variables
        unknown_var (boolean):  whether list of variables includes at least one unknown variable
        """
        all_levels = [self.data_measlevs[var_name] for var_name in variables]

        if 'nom' in all_levels:
            meas_lev = 'nom'
        elif 'ord' in all_levels:
            meas_lev = 'ord'
        elif 'int' in all_levels:
            meas_lev = 'int'
        else:  # all variables are unknown
            meas_lev = 'int'

        if 'unk' in all_levels:
            unknown_var = True
        else:
            unknown_var = False

        return meas_lev, unknown_var

    def _test_central_tendency(self, var_name, ttest_value):
        """
        Test central tendency
        """
        meas_level, unknown_type = self._meas_lev_vars([var_name])
        text_result=''
        if meas_level in ['int', 'ord', 'unk']:
            prec = cs_util.precision(self.data_frame[var_name])+1
        if unknown_type:
            text_result += '<decision>'+warn_unknown_variable+'\n<default>'
        if meas_level in ['int', 'unk']:  # TODO check normality?
            text_result += '<decision>'+_('Interval variable.')+' >> '+_('Running one sample t-test.')+'<default>\n'
            text_result += _(u'Mean: %0.*f') % (prec, np.mean(self.data_frame[var_name].dropna()))+'\n'
            text_result2, graph = cs_stat.one_t_test(self.data_frame, self.data_measlevs, var_name, test_value=ttest_value)
        elif meas_level == 'ord':
            text_result += '<decision>'+_('Ordinal variable.')+' >> '+_('Running Wilcoxon signed-rank t-test.')+'<default>\n'
            text_result += _(u'Median: %0.*f') % (prec, np.median(self.data_frame[var_name].dropna()))+'\n'
            text_result2, graph = cs_stat.wilcox_sign_test(self.data_frame, self.data_measlevs, var_name, value=ttest_value)
        else:
            text_result2 = '<decision>'+_('No central tendency can be computed for nominal variables.')+'<default>\n'
            graph = None
        text_result += text_result2
        return text_result, graph

    ### Compile statistics ###

    def explore_variable(self, var_name, frequencies=True, distribution=True, descriptives=True, normality=True,
                         central_test=True, central_value=0.0):
        """Explore variable.

        :param var_name: Name of the variable (str)
        :param frequencies: Run Frequencies (bool)
        :param distribution: Run Distribution (bool)
        :param descriptives:  Run Descriptives (bool)
        :param normality: Run Normality (bool)
        :param central_test: Run Test central tendency (bool)
        :param central_value: Test central tendency value (float)
        :return:
        """
        plt.close('all')
        result_list = [csc.heading_style_begin + _('Explore variable')+csc.heading_style_end]
        result_list.append(_('Exploring variable: '+var_name+'\n'))
        if self._filtering_status():
            result_list[-1] += self._filtering_status()

        if frequencies:
            text_result = '<b>'+_('Frequencies')+'</b>\n'
            text_result += cs_stat.frequencies(self.data_frame, var_name)
            result_list.append(text_result)
        if distribution:
            text_result = '<b>'+_('Distribution')+'</b>\n'
            text_result2, image = cs_stat.histogram(self.data_frame, self.data_measlevs, var_name)
            result_list.append(text_result+text_result2)
            result_list.append(image)
        if descriptives:
            text_result = '<b>'+_('Descriptitve statistics')+'</b>\n'
            text_result += cs_stat.descriptives(self.data_frame, self.data_measlevs, var_name)
            result_list.append(text_result)
            # TODO boxplot also
        if normality:
            text_result = '<b>'+_('Normality')+'</b>\n'
            stat_result, text_result2, image, image2 = cs_stat.normality_test(self.data_frame, self.data_measlevs, var_name)
            result_list.append(text_result+text_result2)
            if image:
                result_list.append(image)
            if image2:
                result_list.append(image2)
        if central_test:
            text_result = '<b>'+_('Test central tendency')+'</b>\n'
            text_result2, image = self._test_central_tendency(var_name, central_value)
            result_list.append(text_result+text_result2)
            if image:
                result_list.append(image)
        return self._convert_output(result_list)

    def explore_variable_pair(self, x, y):
        """Explore variable pairs.

        :param x: name of x variable (str)
        :param y: name of y variable (str)
        :return:
        """
        plt.close('all')
        meas_lev, unknown_var = self._meas_lev_vars([x, y])
        title = csc.heading_style_begin + _('Explore variable pair') + csc.heading_style_end
        text_result = _(u'Exploring variable pair: ') + x + u', ' + y + '\n'
        text_result += self._filtering_status()
        if unknown_var:
            text_result += '<decision>'+warn_unknown_variable+'\n<default>'

        # Prepare data, drop missing data
        # TODO are NaNs interesting in nominal variables?
        data = self.data_frame[[x, y]].dropna()
        valid_n = len(data)
        invalid_n = len(self.data_frame[[x, y]]) - valid_n
        text_result += _('N of valid pairs: %g\n') % valid_n
        text_result += _('N of invalid pairs: %g\n\n') % invalid_n
        
        # 1. Compute and print numeric results
        slope, intercept = None, None
        if meas_lev == 'int':
            text_result += '<decision>'+_('Interval variables.')+' >> '+_("Running Pearson's and Spearman's correlation.")+'\n<default>'
            df = len(data)-2
            r, p = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])  # TODO select variables by name instead of iloc
            text_result += _(u"Pearson's correlation")+': <i>r</i>(%0.3g) = %0.3f, %s\n' %(df, r, cs_util.print_p(p))
            if meas_lev == 'int':
                slope, intercept, r_value, p_value, std_err = stats.linregress(data.iloc[:, 0], data.iloc[:, 1])
                # TODO output with the precision of the data
                text_result += _('Linear regression')+': y = %0.3fx + %0.3f\n' %(slope, intercept)
            r, p = stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])
            text_result += _(u"Spearman's rank-order correlation")+': <i>r</i>(%0.3g) = %0.3f, %s' %(df, r, cs_util.print_p(p))
        elif meas_lev == 'ord':
            text_result += '<decision>'+_('Ordinal variables.')+' >> '+_("Running Spearman's correlation.")+'\n<default>'
            df = len(data)-2
            r, p = stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])
            text_result += _(u"Spearman's rank-order correlation")+': <i>r</i>(%0.3g) = %0.3f, %s' %(df, r, cs_util.print_p(p))
        elif meas_lev == 'nom':
            if not(self.data_measlevs[x] == 'nom' and self.data_measlevs[y] == 'nom'):
                text_result += '<warning>'+_('Not all variables are nominal. Consider comparing groups.')+'<default>\n'
            text_result += '<decision>'+_('Nominal variables.')+' >> '+_(u'Running Cramér\'s V.')+'\n<default>'
            text_result += cs_stat.chi_square_test(self.data_frame, x, y)
        text_result += '\n'
        
        # 2. Make graph
        temp_text_result, graph = cs_stat.var_pair_graph(data, meas_lev, slope, intercept, x, y, self.data_frame)
        if temp_text_result:
            text_result += temp_text_result
        return self._convert_output([title, text_result, graph])
    #correlations(x,y)  # test

    def pivot(self, depend_names=[], row_names=[], col_names=[], page_names=[], function='Mean'):
        """ Computes pivot table
        :param row_names:
        :param col_names:
        :param page_names:
        :param depend_names:
        :param function:
        :return:
        """
        # TODO optionally return pandas DataFrame or Panel
        title = csc.heading_style_begin + _('Pivot table') + csc.heading_style_end
        text_result = cs_stat.pivot(self.data_frame, row_names, col_names, page_names, depend_names, function)
        return self._convert_output([title, text_result])


    def compare_variables(self, var_names):
        """Compare variables

        :param var_names: list of variable names (list of str)
        :return:
        """
        plt.close('all')
        title = csc.heading_style_begin + _('Compare variables') + csc.heading_style_end
        intro_result = '<default>'+_(u'Variables to compare: ') + u', '.join(x for x in var_names) + '\n'
        intro_result += self._filtering_status()

        # Check if the variables have the same measurement levels
        meas_levels = set([self.data_measlevs[var_name] for var_name in var_names])
        if len(meas_levels)>1:
            if 'ord' in meas_levels or 'nom' in meas_levels:  # int and unk can be used together, since unk is taken as int by default
                return intro_result, '<decision>'+_(u"Sorry, you can't compare variables with different measurement levels. You could downgrade higher measurement levels to lowers to have the same measurement level.")+'<default>'
        # level of measurement of the variables
        meas_level, unknown_type = self._meas_lev_vars(var_names)
        if unknown_type:
            intro_result += '\n<decision>'+warn_unknown_variable+'<default>'

        # Prepare data, drop missing data
        # TODO are NaNs interesting in nominal variables?
        data = self.data_frame[var_names].dropna()
        valid_n = len(data)
        invalid_n = len(self.data_frame[var_names])-valid_n
        intro_result += _('N of valid cases: %g\n') % valid_n
        intro_result += _('N of invalid cases: %g\n') % invalid_n

        # 1. Plot the individual data
        temp_intro_result, graph = cs_stat.comp_var_graph(data, var_names, meas_level, self.data_frame)
        if temp_intro_result:
            intro_result += temp_intro_result

        # 2. Descriptives
        descr_result = ''
        if meas_level in ['int', 'unk']:
            descr_result += cs_stat.print_var_stats(self.data_frame, var_names, stat='mean')
        elif meas_level == 'ord':
            descr_result += cs_stat.print_var_stats(self.data_frame, var_names, stat='median')
        elif meas_level == 'nom':
            import itertools
            for var_pair in itertools.combinations(var_names, 2):
                cont_table_data = pd.crosstab(self.data_frame[var_pair[0]], self.data_frame[var_pair[1]])#, rownames = [x], colnames = [y])
                descr_result += cont_table_data.to_html(bold_rows=False).replace('\n', '').replace('border="1"', 'style="border:1px solid black;"')

        # 3. Plot the descriptive data
        graph2 = cs_stat.comp_var_graph_cum(data, var_names, meas_level, self.data_frame)

        # 4. Hypotheses testing
        result = _('Hypothesis testing:')+'\n'
        if len(var_names) < 2:
            result += _('At least two variables required.')
        elif len(var_names) == 2:
            result += '<decision>'+_('Two variables. ')+'<default>'

            if meas_level == 'int':
                # TODO check assumptions
                result += '<decision>'+_('Interval variables.')+' >> '+_('Choosing paired t-test.')+'\n<default>'
                
                result += '<decision>'+_('Checking for normality.')+'\n<default>'
                normal_vars = True
                for var_name in var_names:
                    norm, text_result, graph_dummy, graph2_dummy = cs_stat.normality_test(self.data_frame, self.data_measlevs, var_name, alt_data = data)
                    result += text_result
                    if not norm:
                        result += '<decision>'+_('Normality is violated in variable ')+var_name+'.\n<default>'
                        normal_vars = False
                        
                if normal_vars:
                    result += '<decision>'+_('Normality is not violated. >> Running paired t-test.')+'\n<default>'
                    result += cs_stat.paired_t_test(self.data_frame, var_names)
                else:  # TODO should the descriptive be the mean or the median?
                    result += '<decision>'+_('Normality is violated. >> Running paired Wilcoxon test.')+'\n<default>'
                    result += cs_stat.paired_wilcox_test(self.data_frame, var_names)
            elif meas_level == 'ord':
                result += '<decision>'+_('Ordinal variables.')+' >> '+_('Running paired Wilcoxon test.')+'\n\n<default>'
                result += cs_stat.paired_wilcox_test(self.data_frame, var_names)
            else:
                result += '<decision>'+_('Nominal variables.')+' >> '+_('Sorry, not implemented yet.')+'\n<default>'
                #result += cs_stat.mcnemar_test(self.data_frame, var_names)
        else:
            result += '<decision>'+_('More than two variables. ')+'<default>'
            if meas_level == 'int':
                result += '<decision>'+_('Interval variables.')+' >> '+_('Choosing repeated measures one-way ANOVA.')+'\n<default>'

                result += '<decision>'+_('Checking for normality.')+'\n<default>'
                normal_vars = True
                for var_name in var_names:
                    norm, text_result, graph_dummy, graph2_dummy = cs_stat.normality_test(self.data_frame, self.data_measlevs, var_name, alt_data=data)
                    result += text_result
                    if not norm:
                        result += '<decision>'+_('Normality is violated in variable ')+var_name+'.\n<default>'
                        normal_vars = False
                        
                if normal_vars:
                    result += '<decision>'+_('Normality is not violated. >> Running Repeated measures one-way ANOVA.')+'\n<default>'
                    result += _('Sorry, not implemented yet.')+'\n<default>'
                    #result += cs_stat.repeated_measures_anova(self.data_frame, var_names)
                else:
                    result += '<decision>'+_('Normality is violated. >> Running Friedman test.')+'\n<default>'
                    result += cs_stat.friedman_test(self.data_frame, var_names)
            elif meas_level == 'ord':
                result += '<decision>'+_('Ordinal variables.')+' >> '+_('Running Friedman test.')+'\n<default>'
                result += cs_stat.friedman_test(self.data_frame, var_names)
            else:
                result += '<decision>'+_('Nominal variables.')+' >> '+_('Sorry, not implemented yet.')+'\n<default>'

        return self._convert_output([title, intro_result, graph, descr_result, graph2, result])

    def compare_groups(self, var_name, grouping_variable):
        """Compare groups.

        :param var_name: name of the dependent variables (str)
        :param grouping_variable: name of grouping variable (str)
        :return:
        """
        plt.close('all')
        var_names = [var_name]
        groups = [grouping_variable]
        # TODO check if there is only one dep.var.
        title = csc.heading_style_begin + _('Compare groups') + csc.heading_style_end
        intro_result = '<default>'+_(u'Dependent variable: ') + u', '.join(x for x in var_names) + u'. ' + \
                       _(u'Group(s): ') + u', '.join(x for x in groups) + '\n'
        intro_result += self._filtering_status()
        # level of measurement of the variables
        meas_level, unknown_type = self._meas_lev_vars([var_names[0]])
        if unknown_type:
            intro_result += '<decision>'+warn_unknown_variable+'<default>'
        if len(groups) == 1:
            data = self.data_frame[[groups[0], var_names[0]]].dropna()
            group_levels = list(set(data[groups[0]]))
            # index should be specified to work in pandas 0.11; but this way can't use _() for the labels
            # TODO remove index, and localize row indexes
            pdf_result = pd.DataFrame(columns=group_levels, index=['N of valid cases', 'N of invalid cases'])
            pdf_result.loc['N of valid cases'] = [sum(data[groups[0]] == group) for group in group_levels]
            pdf_result.loc['N of invalid cases'] = [sum(self.data_frame[groups[0]] == group) - sum(data[groups[0]] == group) for group in group_levels]
#            for group in group_levels:
#                valid_n = sum(data[groups[0]]==group)
#                invalid_n = sum(self.data_frame[groups[0]]==group)-valid_n
#                intro_result += _(u'Group: %s, N of valid cases: %g, N of invalid cases: %g\n') %(group, valid_n, invalid_n)
            intro_result += pdf_result.to_html(bold_rows=False).replace('\n', '').replace('border="1"', 'style="border:1px solid black;"') # pyqt doesn't support border styles
            valid_n = len(self.data_frame[groups[0]].dropna())
            invalid_n = len(self.data_frame[groups[0]])-valid_n
            intro_result += '\n\n'+_(u'N of invalid group cases: %g\n') % invalid_n

            # 1. Plot the individual data
            temp_intro_result, graph = cs_stat.comp_group_graph(self.data_frame, meas_level, var_names, groups, group_levels)
            if temp_intro_result:
                intro_result += temp_intro_result

            # 2. Descriptive data
            descr_result = ''
            if meas_level in ['int', 'unk']:
                descr_result += cs_stat.print_var_stats(self.data_frame, [var_names[0]], group_names=[groups[0]], stat='mean')
            elif meas_level == 'ord':
                descr_result += cs_stat.print_var_stats(self.data_frame, [var_names[0]], group_names=[groups[0]], stat='median')
            elif meas_level == 'nom':
                cont_table_data = pd.crosstab(self.data_frame[var_names[0]], self.data_frame[groups[0]])#, rownames = [x], colnames = [y])
                descr_result += cont_table_data.to_html(bold_rows=False).replace('\n', '').replace('border="1"', 'style="border:1px solid black;"')

            # 3. Plot the descriptive data
            graph2 = cs_stat.comp_group_graph_cum(self.data_frame, meas_level, var_names, groups, group_levels)

            # 4. Hypothesis testing
            result=_('Hypothesis testing:')+'\n'
            result += '<decision>'+_('One grouping variable. ')+'<default>'
            if len(group_levels) == 1:
                result += _('There is only one group. At least two groups required.')+'\n<default>'
            elif len(group_levels) == 2:
                result += '<decision>'+_('Two groups. ')+'<default>'
                if meas_level == 'int':
                    group_levels, [var1, var2] = cs_stat._split_into_groups(self.data_frame, var_names[0], groups[0])
                    if len(var1)==1 or len(var2) == 1:
                        result += '<decision>'+_('One group contains only one case. >> Choosing modified t-test.') + '\n<default>'
                        result += '<decision>'+_('Checking for normality.')+'\n<default>'
                        group = group_levels[1] if len(var1) == 1 else group_levels[0]
                        norm, text_result, graph_dummy, graph2_dummy = cs_stat.normality_test(self.data_frame, self.data_measlevs, var_names[0], group_name=groups[0], group_value=group)
                        result += text_result
                        if not norm:
                            result += '<decision>'+_('Normality is violated in variable ')+var_names[0]+', '+_('group ')+unicode(group)+'.\n<default>'
                            result += '<decision>>> '+_('Running Mann-Whitney test.')+'\n<default>'
                            result += cs_stat.mann_whitney_test(self.data_frame, var_names[0], groups[0])
                        else:
                            result += '<decision>'+_('Normality is not violated. >> Running modified t-test.') + '\n<default>'
                            result += cs_stat.modified_t_test(self.data_frame, var_names[0], groups[0])
                    else:
                        result += '<decision>'+_('Interval variable.')+' >> '+_('Choosing two sample t-test.') + '\n<default>'
                        result += '<decision>'+_('Checking for normality.')+'\n<default>'
                        normal_vars = True
                        for group in group_levels:
                            norm, text_result, graph_dummy, graph2_dummy = cs_stat.normality_test(self.data_frame, self.data_measlevs, var_names[0], group_name=groups[0], group_value=group)
                            result += text_result
                            if not norm:
                                result += '<decision>'+_('Normality is violated in variable ')+var_names[0]+', '+_('group ')+unicode(group)+'.\n<default>'
                                normal_vars = False
                        result += '<decision>'+_('Checking for homogeneity of variance across groups.')+'\n<default>'
                        hoemogeneity_vars = True
                        p, text_result = cs_stat.levene_test(self.data_frame, var_names[0], groups[0])
                        result += text_result
                        if p <0.05:
                            result += '<decision>'+_('Homeogeneity of variance violated in variable ')+var_names[0]+'.\n<default>'
                            hoemogeneity_vars = False
                        
                        if normal_vars and hoemogeneity_vars:
                            result += '<decision>'+_('Normality and homeogeneity of variance are not violated. >> Running two sample t-test.')+'\n<default>'
                            result += cs_stat.independent_t_test(self.data_frame, var_names[0], groups[0])
                        if not normal_vars:
                            result += '<decision>'+_('Normality is violated. ')+'<default>'
                        if not hoemogeneity_vars:
                            result += '<decision>'+_('Homeogeneity of variance is violated. ')+'<default>'
                        if (not normal_vars) or (not hoemogeneity_vars):
                            result += '<decision>>> '+_('Running Mann-Whitney test.')+'\n<default>'
                            result += cs_stat.mann_whitney_test(self.data_frame, var_names[0], groups[0])
                            
                if meas_level == 'ord':
                    result += '<decision>'+_('Ordinal variable.')+' >> '+_('Running Mann-Whitney test.')+'<default>\n\n'
                    result += cs_stat.mann_whitney_test(self.data_frame, var_names[0], groups[0])
                if meas_level == 'nom':
                    result += '<decision>'+_('Nominal variable.')+' >> '+_('Running Chi-square test.')+' '+'<default>\n'
                    result += cs_stat.chi_square_test(self.data_frame, var_names[0], groups[0])
            elif len(group_levels) > 2:
                result += '<decision>'+_('More than two groups.')+' >> <default>'
                if meas_level == 'int':
                    result += '<decision>'+_('Interval variable.')+' >> '+_('Choosing one-way ANOVA.')+'<default>'+'\n'

                    result += '<decision>'+_('Checking for normality.')+'\n<default>'
                    normal_vars = True
                    for group in group_levels:
                        norm, text_result, graph_dummy, graph2_dummy = cs_stat.normality_test(self.data_frame, self.data_measlevs, var_names[0], group_name=groups[0], group_value=group)
                        result += text_result
                        if not norm:
                            result += '<decision>'+_('Normality is violated in variable ')+var_names[0]+', '+_('group ')+str(group)+'.\n<default>'
                            normal_vars = False
                    result += '<decision>'+_('Checking for homeogeneity of variance across groups.')+'\n<default>'
                    hoemogeneity_vars = True
                    p, text_result = cs_stat.levene_test(self.data_frame, var_names[0], groups[0])
                    result += text_result
                    if p <0.05:
                        result += '<decision>'+_('Homeogeneity of variance violated in variable ')+var_names[0]+'.\n<default>'
                        hoemogeneity_vars = False

                    if normal_vars and hoemogeneity_vars:
                        result += '<decision>'+_('Normality and homeogeneity of variance are not violated. >> Running one-way ANOVA.')+'\n<default>'
                        result += cs_stat.one_way_anova(self.data_frame, var_names[0], groups[0])
                    if not normal_vars:
                        result += '<decision>'+_('Normality is violated. ')+'<default>'
                    if not hoemogeneity_vars:
                        result += '<decision>'+_('Homeogeneity of variance is violated. ')+'<default>'
                    if (not normal_vars) or (not hoemogeneity_vars):
                        result += '<decision> >>'+_('Running Kruskal-Wallis test.')+'\n<default>'
                        result += cs_stat.kruskal_wallis_test(self.data_frame, var_names[0], groups[0])
                        
                elif meas_level == 'ord':
                    result += '<decision>'+_('Ordinal variable.')+' >> '+_('Running Kruskal-Wallis test.')+'<default>\n\n<default>'
                    result += cs_stat.kruskal_wallis_test(self.data_frame, var_names[0], groups[0])
                elif meas_level == 'nom':
                    result += '<decision>'+_('Nominal variable.')+' >> '+_('Running Chi-square test.')+'<default>\n'
                    result += cs_stat.chi_square_test(self.data_frame, var_names[0], groups[0])

        elif len(groups) > 1:
            intro_result += '<decision>'+_('Several grouping variables.')+' >> '+'<default>\n'+_('Sorry, not implemented yet.')

        return self._convert_output([title, intro_result, graph, descr_result, graph2, result])


def display(results):
    """Display list of output given by CogStat analysis in IPython Notebook

    :param results: list of output
    :return:
    """
    from IPython.display import display
    from IPython.display import HTML
    for result in results:
        if isinstance(result, basestring):
            display(HTML(result))
        else:
            display(result)
    plt.close('all')

if __name__ == '__main__':
    import cogstat_gui
    cogstat_gui.main()

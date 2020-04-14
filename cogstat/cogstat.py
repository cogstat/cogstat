# -*- coding: utf-8 -*-

"""
Class for the CogStat data (with import method) and methods to compile the
appropriate statistics for the main analysis commands.
"""

# if CS is used with GUI, start the splash screen
QString = str

# go on with regular importing, etc.
import csv
import gettext
import logging
import os
import itertools

__version__ = '2.0.0dev'

import matplotlib
matplotlib.use("qt5agg")
#print matplotlib.get_backend()

from . import cogstat_config as csc
csc.versions['cogstat'] = __version__
from . import cogstat_stat as cs_stat
from . import cogstat_stat_num as cs_stat_num
from . import cogstat_hyp_test as cs_hyp_test
from . import cogstat_util as cs_util
from . import cogstat_chart as cs_chart
cs_util.get_versions()

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.figure import Figure

from PyQt5 import QtGui

logging.root.setLevel(logging.INFO)

rcParams['figure.figsize'] = csc.fig_size_x, csc.fig_size_y

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext

warn_unknown_variable = '<warning>'+_('The measurement levels of the variables are not set. Set them in your data source.') \
                        + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                        % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                        + '\n<default>'  # TODO it might not be necessary to repeat this warning in the analyses, use only at import?

app_devicePixelRatio = 1.0 # this will be overwritten from cogstat_gui; this is needed for high dpi screens

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
        self.import_message = ''  # can't return anything to caller,
                                #  since we're in an __init__ method, so store the message here
        self.filtering_status = None

        self._import_data(data=data, param_measurement_level=measurement_level.lower())

    ### Import and handle the data ###

    def _import_data(self, data='', param_measurement_level=''):

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
                        try:
                            self.data_frame[column] = self.data_frame[column].astype('float')
                        except:
                            pass

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
                        self.import_message += '\n<warning>' + \
                                               _('Number of measurement level do not match the number of variables. Measurement level specification is ignored.')
                        measurement_level = ''
                else:  # Only levels are given - in the order of the variables
                    if len(measurement_level.split()) == len(self.data_frame.columns):
                        self.data_measlevs = {name: level for name, level in zip(self.data_frame.columns,
                                                                                 measurement_level.split())}
                    else:
                        self.import_message += '\n<warning>' + \
                                               _('Number of measurement level do not match the number of variables. Measurement level specification is ignored.')
                        measurement_level = ''
            if not measurement_level:  # Otherwise (or if the given measurement level is incorrect)
                # set them to be nom if type is a str, unk otherwise
                self.data_measlevs = {name: ('nom' if self.data_frame[name].dtype == 'object' else 'unk')
                                      for name in self.data_frame.columns}
                # TODO Does the line above work? Does the line below work ?
                #self.data_measlevs =
                # dict(zip(self.data_frame.columns, [u'nom' if self.data_frame[name].dtype == 'object'
                # else u'unk' for name in self.data_frame.columns]))
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
                self.import_message += '\n<warning>' + \
                                       _('String variables cannot be interval or ordinal variables in CogStat. Those variables are automatically set to nominal: ')\
                                       + ''.join(', %s' % var_name for var_name in invalid_data)[2:]+'. ' + \
                                       _('You can fix this in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '<default>'

            if set(self.data_measlevs) in ['unk']:
                self.import_message += '\n<warning>' + \
                                       _('The measurement level was not set for all variables.') + ' '\
                                       +_('You can fix this in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '<default>'

        file_measurement_level = ''
        # Import from pandas DataFrame
        if isinstance(data, pd.DataFrame):
            self.data_frame = data
            self.import_source = _('pandas dataframe')
        elif isinstance(data, str):

            delimiter = '\t'
            #print csv.Sniffer().sniff(data).delimiter  # In some tests, it didn't find the delimiter reliably and correctly

            # Import from file
            if not ('\n' in data):  # Single line text, i.e., filename
                filetype = data[data.rfind('.'):]
                # Import csv file
                if filetype in ['.txt', '.csv', '.log', '.tsv']:
                    # Check if the file exists # TODO
                    # self.import_source = _('Import failed')
                    # return

                    # Check if there is variable type line
                    f = csv.reader(open(data, 'r'), delimiter=delimiter, quotechar=quotechar)
                    next(f)
                    meas_row = next(f)
                    if {a.lower() for a in meas_row} <= {'unk', 'nom', 'ord', 'int', ''} \
                            and set(meas_row) != {''}:
                        file_measurement_level = ' '.join(meas_row).lower()
                    skiprows = [1] if file_measurement_level else None

                    # Read the file
                    self.data_frame = pd.read_csv(data, delimiter=delimiter, quotechar=quotechar, skiprows=skiprows)
                    self.import_source = _('text file - ')+data  # filename
                # Import SPSS .sav file
                elif filetype == '.sav':
                    import savReaderWriter
                    # Get the values
                    with savReaderWriter.SavReader(data, ioUtf8=True) as reader:
                        spss_data = [line for line in reader]
                    # Get the variable names and measurement levels
                    with savReaderWriter.SavHeaderReader(data, ioUtf8=True) as header:
                        metadata = header.all()
                    # Create the CogStat dataframe
                    self.data_frame = pd.DataFrame.from_records(spss_data, columns=metadata.varNames)
                    # Convert SPSS measurement levels to CogStat
                    spss_to_cogstat_measurement_levels = {'unknown': 'unk', 'nominal': 'nom', 'ordinal': 'ord', 'scale': 'int', 'ratio': 'int', 'flag': 'nom', 'typeless': 'unk'}
                    file_measurement_level = ' '.join([spss_to_cogstat_measurement_levels[metadata.measureLevels[spss_var]] for spss_var in metadata.varNames])
                    self.import_source = _('SPSS file - ') + data  # filename

            # Import from multiline string, clipboard
            else:  # Multi line text, i.e., clipboard data
                # Check if there is variable type line
                import io
                f = io.StringIO(data)
                next(f)
                meas_row = next(f).replace('\n', '').replace('\r', '').split(delimiter)
                # \r was used in Mac after importing from Excel clipboard
                if {a.lower() for a in meas_row} <= {'unk', 'nom', 'ord', 'int', ''} \
                        and set(meas_row) != {''}:
                    meas_row = ['unk' if item == '' else item for item in meas_row]  # missing level ('') means 'unk'
                    file_measurement_level = ' '.join(meas_row).lower()
                skiprows = [1] if file_measurement_level else None

                # Read the clipboard
                clipboard_file = io.StringIO(data)
                self.data_frame = pd.read_csv(clipboard_file, delimiter=delimiter, quotechar=quotechar,
                                              skiprows=skiprows)
                self.import_source = _('clipboard')

        else:  # Invalid data source
            self.import_source = _('Import failed')
            return

        # Set other details for all import sources
        percent2float()
        # Convert boolean variables to string
        # True and False values should be imported as string, not as boolean - CogStat does not know boolean variables
        # Although this solution changes upper and lower cases: independent of the text, it will be 'True' and 'False'
        self.data_frame[self.data_frame.select_dtypes(include=['bool']).columns] = \
            self.data_frame.select_dtypes(include=['bool']).astype('object')
        set_measurement_level(measurement_level=
                              (param_measurement_level if param_measurement_level else file_measurement_level))
                            # param_measurement_level overwrites file_measurement_level

        # Check for unicode chars in the data to warn user not to use it
        # TODO this might be removed with Python3 and with unicode encoding
        non_ascii_var_names = []
        non_ascii_vars = []
        for variable_name in self.data_frame:
            if not all(ord(char) < 128 for char in variable_name):  # includes non ascii char
                non_ascii_var_names.append(variable_name)
            if self.data_frame[variable_name].dtype == 'object':  # check only string variables
                for ind_data in self.data_frame[variable_name]:
                    if not(ind_data != ind_data) and not (isinstance(ind_data, bool)):
                            # if not NaN, otherwise the next condition is invalid and if not boolean
                        if not all(ord(char) < 128 for char in ind_data):
                            non_ascii_vars.append(variable_name)
                            break  # after finding the first non-ascii data, we can leave the variable
        if non_ascii_var_names:
            self.import_message += '\n<warning>' + \
                                   _('Some variable name(s) include non-English characters, which will cause problems in some analyses: %s.') \
                                   % ''.join(' %s' % non_ascii_var_name for non_ascii_var_name in non_ascii_var_names) \
                                   + ' ' + _('You can fix this in your data source.') \
                                   + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                   % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                   + '<default>'
        if non_ascii_vars:
            self.import_message += '\n<warning>' + \
                                   _('Some variable(s) include non-English characters, which will cause problems in some analyses: %s.') \
                                   % ''.join(' %s' % non_ascii_var for non_ascii_var in non_ascii_vars) \
                                   + ' ' + _('You can fix this in your data source.') \
                                   + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                   % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                   + '<default>'

        self.orig_data_frame = self.data_frame.copy()

        # Add keys with pyqt string form, too, because UI returns variable names in this form
        # TODO do we still need this?
        from PyQt5 import QtCore
        for var_name in self.data_frame.columns:
            self.data_measlevs[QString(var_name)] = self.data_measlevs[var_name]

    def print_data(self, brief=False):
        """Print data."""
        output = f"{csc.heading_style_begin}{_('Data')}{csc.heading_style_end}"
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
        output += cs_stat._format_html_table(data_comb[:12 if brief else 1002].to_html(bold_rows=False,
                                                                                       classes="table_cs_pd"))
        if brief and (len(self.data_frame.index) > 10):
            output += str(len(self.data_frame.index)-10) + _(' further cases are not displayed...')+'\n'
        elif len(self.data_frame.index) > 999:
            output += _('The next %s cases will not be printed. You can check all cases in the original data source.') % \
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
        title = f"{csc.heading_style_begin}{_('Filtering')}{csc.heading_style_end}"
        if var_names is None or var_names == []:  # Switch off outlier filtering
            self.data_frame = self.orig_data_frame.copy()
            self.filtering_status = None
            text_output = _('Filtering is switched off.')
        else:  # Create a filtered dataframe based on the variable
            filtered_data_indexes = []
            text_output = ''
            self.filtering_status = ''
            for var_name in var_names:
                if self.data_measlevs[var_name] in ['ord', 'nom']:
                    text_output += _('Only interval variables can be used for filtering. Ignoring variable %s.') % var_name + '\n'
                    continue
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
                excluded_cases = \
                    self.orig_data_frame.loc[self.orig_data_frame.index.difference(filtered_data_indexes[-1])]
                #excluded_cases.index = [' '] * len(excluded_cases)  # TODO can we cut the indexes from the html table?
                # TODO uncomment the above line after using pivot indexes in CS data
                if len(excluded_cases):
                    text_output += _('The following cases will be excluded: ')
                    text_output += cs_stat._format_html_table(excluded_cases.to_html(bold_rows=False,
                                                                                     classes="table_cs_pd"))
                else:
                    text_output += _('No cases were excluded.') + '\n'
            self.data_frame = self.orig_data_frame.copy()
            for filtered_data_index in filtered_data_indexes:
                self.data_frame = self.data_frame.reindex(self.data_frame.index.intersection(filtered_data_index))
            self.filtering_status = ', '.join(var_names) + _(' (2 SD)')
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
            # TODO is it better to use figure.canvas.width(), figure.canvas.height()
            string_buffer = figure.canvas.buffer_rgba()
            qimage = QtGui.QImage(string_buffer, size_x*app_devicePixelRatio, size_y*app_devicePixelRatio, QtGui.QImage.Format_ARGB32).rgbSwapped().copy()
            QtGui.QImage.setDevicePixelRatio(qimage, app_devicePixelRatio)
            return qimage
                # I couldn't see it documented, but seemingly the figure uses BGR, not RGB coding
                # this should be a copy, otherwise closing the matplotlib figures would damage the qImages on the GUI

        if csc.output_type in ['ipnb', 'gui']:
            # convert custom notation to html
            new_output = []
            for i, output in enumerate(outputs):
                if isinstance(output, Figure):
                    # For gui convert matplotlib to qImage
                    new_output.append(output if csc.output_type == 'ipnb' else _figure_to_qimage(output))
                elif isinstance(output, str):
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


    ### Compile statistics ###

    def explore_variable(self, var_name, frequencies=True, central_value=0.0):
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
        meas_level, unknown_type = self._meas_lev_vars([var_name])
        result_list = [f"{csc.heading_style_begin}{_('Explore variable')}{csc.heading_style_end}"]
        result_list.append(_('Exploring variable: ') + var_name + ' (%s)\n'%meas_level)
        if self._filtering_status():
            result_list[-1] += self._filtering_status()

        # 1. Raw data
        text_result = f"{csc.subheading_style_begin}{_('Raw data')}{csc.subheading_style_end}"
        text_result2, image = cs_stat.display_variable_raw_data(self.data_frame, self.data_measlevs, var_name)
        result_list.append(text_result+text_result2)
        result_list.append(image)

        # 2. Sample properties
        text_result = f"{csc.subheading_style_begin}{_('Sample properties')}{csc.subheading_style_end}"

        # Frequencies
        if frequencies:
            text_result += '<b>'+_('Frequencies')+'</b>'
            text_result += cs_stat.frequencies(self.data_frame, var_name, meas_level) + '\n\n'

        # Descriptives
        if self.data_measlevs[var_name] in ['int', 'unk']:
            text_result += cs_stat.print_var_stats(self.data_frame, [var_name], self.data_measlevs,
                                                   statistics=['mean', 'std', 'skewness', 'kurtosis', 'range', 'max',
                                                               'upper quartile', 'median', 'lower quartile', 'min'])
        elif self.data_measlevs[var_name] == 'ord':
            text_result += cs_stat.print_var_stats(self.data_frame, [var_name], self.data_measlevs,
                                                   statistics=['max', 'upper quartile', 'median', 'lower quartile',
                                                               'min'])
            # TODO boxplot also
        elif self.data_measlevs[var_name] == 'nom':
            text_result += cs_stat.print_var_stats(self.data_frame, [var_name], self.data_measlevs,
                                                   statistics=['variation ratio'])
        result_list.append(text_result)

        # Distribution
        if self.data_measlevs[var_name] != 'nom': # histogram for nominal variable has already been shown in raw data
            image = cs_chart.create_histogram_chart(self.data_frame, self.data_measlevs, var_name)
            result_list.append(image)

        # 3. Population properties
        text_result = f"{csc.subheading_style_begin}{_('Population properties')}{csc.subheading_style_end}"

        # Normality
        if meas_level in ['int', 'unk']:
            text_result += '<b>'+_('Normality')+'</b>\n'
            stat_result, text_result2, image, image2 = cs_stat.normality_test(self.data_frame, self.data_measlevs,
                                                                              var_name)
            text_result += text_result2
            result_list.append(text_result)
            if image:
                result_list.append(image)
            if image2:
                result_list.append(image2)
        else:
            result_list.append(text_result[:-2])

        # Population estimations
        if meas_level in ['int', 'ord', 'unk']:
            prec = cs_util.precision(self.data_frame[var_name]) + 1

        population_param_text = '\n<b>' + _('Population parameter estimations') + '</b>\n'
        if meas_level in ['int', 'unk']:
            population_param_text += cs_stat.variable_estimation(self.data_frame[var_name], ['mean', 'std'])
        elif meas_level == 'ord':
            population_param_text += cs_stat.variable_estimation(self.data_frame[var_name], ['median'])
        elif meas_level == 'nom':
            population_param_text += cs_stat.proportions_ci(self.data_frame, var_name)
        text_result = '\n'

        # Hypothesis tests
        text_result += f"<b>{_('Hypothesis tests')}</b>\n" + '<decision>'
        if self.data_measlevs[var_name] in ['int', 'unk']:
            text_result += _('Testing if mean deviates from the value %s.') % central_value + '<default>\n'
        elif self.data_measlevs[var_name] == 'ord':
            text_result += _('Testing if median deviates from the value %s.') % central_value + '<default>\n'

        if unknown_type:
            text_result += '<decision>' + warn_unknown_variable + '\n<default>'
        if meas_level in ['int', 'unk']:
            text_result += '<decision>' + _('Interval variable.') + ' >> ' + \
                           _(
                               'Choosing one-sample t-test or Wilcoxon signed-rank test depending on the assumption.') + \
                           '<default>\n'
            text_result += '<decision>' + _('Checking for normality.') + '\n<default>'
            norm, text_result_norm, graph_dummy, graph2_dummy = cs_stat.normality_test(self.data_frame,
                                                                                       self.data_measlevs, var_name)
            text_result += text_result_norm
            if norm:
                text_result += '<decision>' + _('Normality is not violated.') + ' >> ' + \
                               _('Running one-sample t-test.') + '<default>\n'
                text_result2, graph = cs_stat.one_t_test(self.data_frame, self.data_measlevs, var_name,
                                                         test_value=central_value)

            else:
                text_result += '<decision>' + _('Normality is violated.') + ' >> ' + \
                               _('Running Wilcoxon signed-rank test.') + '<default>\n'
                text_result += _('Median: %0.*f') % (prec, np.median(self.data_frame[var_name].dropna())) + '\n'
                text_result2, graph = cs_stat.wilcox_sign_test(self.data_frame, self.data_measlevs, var_name,
                                                               value=central_value)

        elif meas_level == 'ord':
            text_result += '<decision>' + _('Ordinal variable.') + ' >> ' + _(
                'Running Wilcoxon signed-rank test.') + \
                           '<default>\n'
            text_result2, graph = cs_stat.wilcox_sign_test(self.data_frame, self.data_measlevs, var_name,
                                                           value=central_value)
        else:
            text_result2 = '<decision>' + _('Sorry, not implemented yet.') + '<default>\n'
            graph = None
        text_result += text_result2

        result_list.append(population_param_text)
        if graph:
            result_list.append(graph)
        result_list.append(text_result)
        return self._convert_output(result_list)

    def explore_variable_pair(self, x, y, xlims=[None, None], ylims=[None, None]):
        """Explore variable pairs.

        :param x: name of x variable (str)
        :param y: name of y variable (str)
        :param xlims: List of values that may overwrite the automatic xlim values for interval and ordinal variables
        :param ylims: List of values that may overwrite the automatic ylim values for interval and ordinal variables
        :return:
        """
        plt.close('all')
        meas_lev, unknown_var = self._meas_lev_vars([x, y])
        title = f"{csc.heading_style_begin}{_('Explore relation of variable pair')}{csc.heading_style_end}"
        raw_result = _('Exploring variable pair: ') + x + ' (%s), '%self.data_measlevs[x] + y + ' (%s)\n'%self.data_measlevs[y]
        raw_result += self._filtering_status()
        if unknown_var:
            raw_result += '<decision>'+warn_unknown_variable+'\n<default>'

        # 1. Raw data
        raw_result = f"{csc.subheading_style_begin}{_('Raw data')}{csc.subheading_style_end}"
        # Prepare data, drop missing data
        # TODO are NaNs interesting in nominal variables?
        data = self.data_frame[[x, y]].dropna()
        valid_n = len(data)
        missing_n = len(self.data_frame[[x, y]]) - valid_n
        raw_result += _('N of valid pairs') + ': %g' % valid_n + '\n'
        raw_result += _('N of missing pairs') + ': %g' % missing_n + '\n'

        # Raw data chart
        raw_graph = cs_chart.create_variable_pair_chart(data, meas_lev, 0, 0, x, y, self.data_frame, raw_data=True,
                                                        xlims=xlims, ylims=ylims)
                                                        # slope and intercept are set to 0, but they
                                                        # are not used with raw_data

        # 2-3. Sample and population properties
        sample_result = f"{csc.subheading_style_begin}{_('Sample properties')}{csc.subheading_style_end}"
        if meas_lev == 'nom':
            sample_result += cs_stat.contingency_table(self.data_frame, [x], [y], count=True, percent=True, margins=True)
        standardized_effect_size_result = _('Standardized effect size:') + '\n'
        estimation_result = f"{csc.subheading_style_begin}{_('Population properties')}{csc.subheading_style_end}" + \
                            f"<b>{_('Population parameter estimations')}</b>\n"
        pdf_result = pd.DataFrame(columns=[_('Point estimation'), _('95% confidence interval')])
        population_result = '\n'

        # Compute and print numeric results
        slope, intercept = None, None
        if meas_lev == 'int':
            population_result += f"<b>{_('Hypothesis tests')}</b>\n" + '<decision>' + \
                                 _('Testing if correlation differs from 0.') + '<default>\n'
            population_result += '<decision>'+_('Interval variables.')+' >> '+\
                           _("Running Pearson's and Spearman's correlation.")+'\n<default>'
            df = len(data)-2
            r, p = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])  # TODO select variables by name instead of iloc
            r_ci_low, r_ci_high = cs_stat_num.corr_ci(r, df + 2)
            standardized_effect_size_result += _("Pearson's correlation") + ': <i>r</i> = %0.3f\n' % r
            pdf_result.loc[_("Pearson's correlation") + ', <i>r</i>'] = ['%0.3f' % (r), '[%0.3f, %0.3f]' % (r_ci_low, r_ci_high)]
            population_result += _("Pearson's correlation") + \
                           ': <i>r</i>(%d) = %0.3f, %s\n' % (df, r, cs_util.print_p(p))

            slope, intercept, r_value, p_value, std_err = stats.linregress(data.iloc[:, 0], data.iloc[:, 1])
            # TODO output with the precision of the data
            sample_result += _('Linear regression')+': y = %0.3fx + %0.3f' % (slope, intercept)

            r, p = stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])
            r_ci_low, r_ci_high = cs_stat_num.corr_ci(r, df + 2)
            standardized_effect_size_result += _("Spearman's rank-order correlation") + ': <i>r<sub>s</sub></i> = %0.3f\n' % r
            pdf_result.loc[_("Spearman's rank-order correlation") + ', <i>r<sub>s</sub></i>'] = ['%0.3f' % (r), '[%0.3f, %0.3f]' % (r_ci_low, r_ci_high)]
            estimation_result += _('Standardized effect size:') \
                                 + cs_stat._format_html_table(pdf_result.to_html(bold_rows=False, classes="table_cs_pd",
                                                                                 escape=False))

            population_result += _("Spearman's rank-order correlation") + \
                           ': <i>r<sub>s</sub></i>(%d) = %0.3f, %s' % \
                           (df, r, cs_util.print_p(p))
        elif meas_lev == 'ord':
            population_result += f"<b>{_('Hypothesis tests')}</b>\n" + '<decision>' + \
                                 _('Testing if correlation differs from 0.') + '<default>\n'
            population_result += '<decision>'+_('Ordinal variables.')+' >> '+_("Running Spearman's correlation.") + \
                           '\n<default>'
            df = len(data)-2
            r, p = stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])
            r_ci_low, r_ci_high = cs_stat_num.corr_ci(r, df + 2)
            standardized_effect_size_result += _("Spearman's rank-order correlation") + ': <i>r<sub>s</sub></i> = %0.3f\n' % r
            pdf_result.loc[_("Spearman's rank-order correlation") + ', <i>r<sub>s</sub></i>'] = ['%0.3f' % (r), '[%0.3f, %0.3f]' % (r_ci_low, r_ci_high)]
            estimation_result += _('Standardized effect size:') \
                                 + cs_stat._format_html_table(pdf_result.to_html(bold_rows=False, classes="table_cs_pd",
                                                                                 escape=False))
            population_result += _("Spearman's rank-order correlation") + \
                           ': <i>r<sub>s</sub></i>(%d) = %0.3f, %s' % \
                           (df, r, cs_util.print_p(p))
        elif meas_lev == 'nom':
            estimation_result += cs_stat.contingency_table(self.data_frame, [x], [y], ci=True)
            population_result += f"<b>{_('Hypothesis tests')}</b>\n" + '<decision>' + \
                                 _('Testing if variables are independent.') + '<default>\n'
            if not(self.data_measlevs[x] == 'nom' and self.data_measlevs[y] == 'nom'):
                population_result += '<warning>'+_('Not all variables are nominal. Consider comparing groups.')+'<default>\n'
            population_result += '<decision>'+_('Nominal variables.')+' >> '+_('Running Cramér\'s V.')+'\n<default>'
            cramer_result, chi_result = cs_stat.chi_square_test(self.data_frame, x, y)
            standardized_effect_size_result += cramer_result
            population_result += chi_result
        sample_result += '\n'
        population_result += '\n'

        # Make graph
        # extra chart is needed only for int variables, otherwise the chart would just repeat the raw data
        if meas_lev in ['int', 'unk']:
            sample_graph = cs_chart.create_variable_pair_chart(data, meas_lev, slope, intercept, x, y, self.data_frame,
                                                               xlims=xlims, ylims=ylims)
        else:
            sample_graph = None
        return self._convert_output([title, raw_result, raw_graph, sample_result, sample_graph,
                                     standardized_effect_size_result, estimation_result, population_result])

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
        title = f"{csc.heading_style_begin}{_('Pivot table')}{csc.heading_style_end}"
        pivot_result = cs_stat.pivot(self.data_frame, row_names, col_names, page_names, depend_names, function)
        return self._convert_output([title, pivot_result])

    def diffusion(self, error_name=[], RT_name=[], participant_name=[], condition_names=[]):
        """Runs diffusion analysis on behavioral data

        :param error_name:
        :param RT_name:
        :param participant_name:
        :param condition_names:

        :return:
        """
        # TODO return pandas DataFrame
        title = f"{csc.heading_style_begin}{_('Behavioral data diffusion analysis')}{csc.heading_style_end}"
        pivot_result = cs_stat.diffusion(self.data_frame, error_name, RT_name, participant_name, condition_names)
        return self._convert_output([title, pivot_result])

    def compare_variables(self, var_names, factors=[], ylims=[None, None]):
        """Compare variables

        :param var_names: list of variable names (list of str)
        :param factors: list of lists, [['name of the factor', number_of_the_levels], ['name of the factor 2', number_of_the_levels]]
        :param ylims: List of values that may overwrite the automatic ylim values for interval and ordinal variables
        :return:
        """
        plt.close('all')
        title = f"{csc.heading_style_begin}{_('Compare repeated measures variables')}{csc.heading_style_end}"
        meas_levels = [self.data_measlevs[var_name] for var_name in var_names]
        raw_result = '<default>'+_('Variables to compare: ') + ', '.join('%s (%s)' % (var, meas) for var, meas in zip(var_names, meas_levels)) + '\n'
        if factors:
            raw_result += _('Factors (number of levels): ') + ', '.join('%s (%d)' % (factor[0], factor[1]) for factor in factors) + '\n'
            factor_combinations = ['']
            for factor in factors:
                factor_combinations = ['%s - %s %s' % (factor_combination, factor[0], level_i+1) for factor_combination in factor_combinations for level_i in range(factor[1])]
            factor_combinations = [factor_combination[3:] for factor_combination in factor_combinations]
            for factor_combination, var_name in zip(factor_combinations, var_names):
                raw_result += '%s: %s\n' % (factor_combination, var_name)

        raw_result += self._filtering_status()

        # Check if the variables have the same measurement levels
        meas_levels = {self.data_measlevs[var_name] for var_name in var_names}
        if len(meas_levels) > 1:
            if 'ord' in meas_levels or 'nom' in meas_levels:  # int and unk can be used together,
                                                                # since unk is taken as int by default
                return self._convert_output([title, raw_result, '<decision>'+_("Sorry, you can't compare variables with different measurement levels. You could downgrade higher measurement levels to lowers to have the same measurement level.")+'<default>'])
        # level of measurement of the variables
        meas_level, unknown_type = self._meas_lev_vars(var_names)
        if unknown_type:
            raw_result += '\n<decision>'+warn_unknown_variable+'<default>'

        # 1. Raw data
        raw_result += f"{csc.subheading_style_begin}{_('Raw data')}{csc.subheading_style_end}"
        # Prepare data, drop missing data
        # TODO are NaNs interesting in nominal variables?
        data = self.data_frame[var_names].dropna()
        valid_n = len(data)
        missing_n = len(self.data_frame[var_names])-valid_n
        raw_result += _('N of valid cases') +': %g\n' % valid_n
        raw_result += _('N of missing cases') + ': %g\n' % missing_n

        # Plot the raw data
        raw_graph = cs_chart.create_repeated_measures_sample_chart(data, var_names, meas_level, self.data_frame,
                                                                   raw_data=True, ylims=ylims)

        # Plot the individual data with box plot
        # There's no need to repeat the mosaic plot for nominal variables
        if meas_level in ['int', 'unk', 'ord']:
            sample_graph = cs_chart.create_repeated_measures_sample_chart(data, var_names, meas_level, self.data_frame,
                                                                          ylims=ylims)
        else:
            sample_graph = None

        # 2. Sample properties
        sample_result = f"{csc.subheading_style_begin}{_('Sample properties')}{csc.subheading_style_end}"

        if meas_level in ['int', 'unk']:
            sample_result += cs_stat.print_var_stats(self.data_frame, var_names, self.data_measlevs,
                             statistics=['mean', 'std', 'max', 'upper quartile', 'median', 'lower quartile', 'min'])
        elif meas_level == 'ord':
            sample_result += cs_stat.print_var_stats(self.data_frame, var_names, self.data_measlevs,
                             statistics=['max', 'upper quartile', 'median', 'lower quartile', 'min'])
        elif meas_level == 'nom':
            sample_result += cs_stat.print_var_stats(self.data_frame, var_names, self.data_measlevs,
                             statistics=['variation ratio'])
            import itertools
            for var_pair in itertools.combinations(var_names, 2):
                sample_result += cs_stat.contingency_table(self.data_frame, [var_pair[1]], [var_pair[0]],
                                                           count=True, percent=True, margins=True)
            sample_result += '\n'

        # 3. Population properties
        population_result = f"{csc.subheading_style_begin}{_('Population properties')}{csc.subheading_style_end}"

        # 3a. Population estimations
        population_result += f"<b>{_('Population parameter estimations')}</b>\n"
        if meas_level in ['int', 'unk']:
            population_result += f"{_('Means')}\n{_('Present confidence interval values suppose normality.')}"
            mean_estimations = cs_stat.repeated_measures_estimations(data, meas_level)
            prec = cs_util.precision(self.data_frame[var_names[0]]) + 1
            population_result += \
                cs_stat._format_html_table(mean_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                    float_format=lambda x: '%0.*f' % (prec, x)))
        elif meas_level == 'ord':
            population_result += f"{_('Median')}"
            median_estimations = cs_stat.repeated_measures_estimations(data, meas_level)
            prec = cs_util.precision(self.data_frame[var_names[0]]) + 1
            population_result += \
                cs_stat._format_html_table(median_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                      float_format=lambda x: '%0.*f' % (prec, x)))
        elif meas_level == 'nom':
            for var_pair in itertools.combinations(var_names, 2):
                population_result += cs_stat.contingency_table(self.data_frame, [var_pair[1]], [var_pair[0]], ci=True)
        population_result += '\n'

        population_graph = cs_chart.create_repeated_measures_population_chart(data, var_names, meas_level,
                                                                              self.data_frame, ylims=ylims)

        # 3b. Hypothesis tests
        result_ht = f"<b>{_('Hypothesis tests')}</b>\n" '<decision>'
        if meas_level in ['int', 'unk']:
            result_ht += _('Testing if the means are the same.') + '<default>\n'
        elif meas_level == 'ord':
            result_ht += _('Testing if the medians are the same.') + '<default>\n'
        elif meas_level == 'nom':
            result_ht += _('Testing if the distributions are the same.') + '<default>\n'
        if not(factors):  # one-way comparison
            if len(var_names) < 2:
                result_ht += _('At least two variables required.')
            elif len(var_names) == 2:
                result_ht += '<decision>'+_('Two variables. ')+'<default>'

                if meas_level == 'int':
                    result_ht += '<decision>'+_('Interval variables.')+' >> '+_('Choosing paired t-test or paired Wilcoxon test depending on the assumptions.')+'\n<default>'

                    result_ht += '<decision>'+_('Checking for normality.')+'\n<default>'
                    non_normal_vars = []
                    temp_diff_var_name = 'Difference of %s and %s' %tuple(var_names)
                    data[temp_diff_var_name] = data[var_names[0]] - data[var_names[1]]
                    norm, text_result, graph_dummy, graph2_dummy = \
                        cs_stat.normality_test(self.data_frame, {temp_diff_var_name:'int'}, temp_diff_var_name, alt_data=data)
                    result_ht += text_result
                    if not norm:
                        non_normal_vars.append(temp_diff_var_name)

                    if not non_normal_vars:
                        result_ht += '<decision>'+_('Normality is not violated. >> Running paired t-test.')+'\n<default>'
                        result_ht += cs_hyp_test.paired_t_test(self.data_frame, var_names)
                    else:  # TODO should the descriptive be the mean or the median?
                        result_ht += '<decision>'+_('Normality is violated in variable(s): %s.') % ', '.\
                            join(non_normal_vars) + ' >> ' + _('Running paired Wilcoxon test.')+'\n<default>'
                        result_ht += cs_hyp_test.paired_wilcox_test(self.data_frame, var_names)
                elif meas_level == 'ord':
                    result_ht += '<decision>'+_('Ordinal variables.')+' >> '+_('Running paired Wilcoxon test.')+'\n<default>'
                    result_ht += cs_hyp_test.paired_wilcox_test(self.data_frame, var_names)
                else:  # nominal variables
                    if len(set(data.values.ravel())) == 2:
                        result_ht += '<decision>'+_('Nominal dichotomous variables.')+' >> ' + _('Running McNemar test.') \
                                  + '\n<default>'
                        result_ht += cs_hyp_test.mcnemar_test(self.data_frame, var_names)
                    else:
                        result_ht += '<decision>'+_('Nominal non dichotomous variables.')+' >> ' + \
                                  _('Sorry, not implemented yet.')+'\n<default>'
            else:
                result_ht += '<decision>'+_('More than two variables. ')+'<default>'
                if meas_level in ['int', 'unk']:
                    result_ht += '<decision>'+_('Interval variables.')+' >> ' + \
                              _('Choosing repeated measures ANOVA or Friedman test depending on the assumptions.') + \
                              '\n<default>'

                    result_ht += '<decision>'+_('Checking for normality.')+'\n<default>'
                    non_normal_vars = []
                    for var_name in var_names:
                        norm, text_result, graph_dummy, graph2_dummy = cs_stat.normality_test(self.data_frame,
                                                                                              self.data_measlevs, var_name,
                                                                                              alt_data=data)
                        result_ht += text_result
                        if not norm:
                            non_normal_vars.append(var_name)

                    if not non_normal_vars:
                        result_ht += '<decision>'+_('Normality is not violated.') + ' >> ' + \
                                  _('Running repeated measures one-way ANOVA.')+'\n<default>'
                        result_ht += cs_hyp_test.repeated_measures_anova(self.data_frame, var_names)
                    else:
                        result_ht += '<decision>'+_('Normality is violated in variable(s): %s.') % ', '.\
                            join(non_normal_vars) + ' >> ' + _('Running Friedman test.')+'\n<default>'
                        result_ht += cs_hyp_test.friedman_test(self.data_frame, var_names)
                elif meas_level == 'ord':
                    result_ht += '<decision>'+_('Ordinal variables.')+' >> '+_('Running Friedman test.')+'\n<default>'
                    result_ht += cs_hyp_test.friedman_test(self.data_frame, var_names)
                else:
                    if len(set(data.values.ravel())) == 2:
                        result_ht += '<decision>'+_('Nominal dichotomous variables.')+' >> '+_("Running Cochran's Q test.") + \
                                  '\n<default>'
                        result_ht += cs_hyp_test.cochran_q_test(self.data_frame, var_names)
                    else:
                        result_ht += '<decision>'+_('Nominal non dichotomous variables.')+' >> ' \
                                  + _('Sorry, not implemented yet.')+'\n<default>'
        else: # two- or more-ways comparison
            if meas_level in ['int', 'unk']:
                result_ht += '<decision>' + _('Interval variables with several factors.') + ' >> ' + \
                             _('Choosing repeated measures ANOVA.') + \
                             '\n<default>'
                result_ht += cs_hyp_test.repeated_measures_anova(self.data_frame, var_names, factors)
            elif meas_level == 'ord':
                result_ht += '<decision>' + _('Ordinal variables with several factors.') + ' >> ' \
                             + _('Sorry, not implemented yet.') + '\n<default>'
            elif meas_level == 'nom':
                result_ht += '<decision>' + _('Nominal variables with several factors.') + ' >> ' \
                             + _('Sorry, not implemented yet.') + '\n<default>'

        return self._convert_output([title, raw_result, raw_graph, sample_result, sample_graph, population_result, population_graph, result_ht])

    def compare_groups(self, var_name, grouping_variables,  single_case_slope_SEs=[], single_case_slope_trial_n=None,
                       ylims=[None, None]):
        """Compare groups.

        :param var_name: name of the dependent variables (str)
        :param grouping_variables: list of names of grouping variables (list of str)
        :param single_case_slope_SEs: list of a single string with the name of the slope SEs for singla case control group
        :param single_case_slope_trial: number of trials in slope calculation for single case
        :param ylims: List of values that may overwrite the automatic ylim values for interval and ordinal variables
        :return:
        """
        plt.close('all')
        var_names = [var_name]
        groups = grouping_variables
        # TODO check if there is only one dep.var.
        title = f"{csc.heading_style_begin}{_('Compare groups')}{csc.heading_style_end}"
        meas_levels = [self.data_measlevs[var_name] for var_name in var_names]
        group_meas_levels = [self.data_measlevs[group] for group in groups]
        raw_result = '<default>'+_('Dependent variable: ') + ', '.join('%s (%s)'%(var, meas) for var, meas in zip(var_names, meas_levels)) + '. ' + \
                       _('Group(s): ') + ', '.join('%s (%s)'%(var, meas) for var, meas in zip(groups, group_meas_levels)) + '\n'
        raw_result += self._filtering_status()

        # level of measurement of the variables
        meas_level, unknown_type = self._meas_lev_vars([var_names[0]])
        if unknown_type:
            raw_result += '<decision>'+warn_unknown_variable+'<default>'

        # One grouping variable
        if len(groups) == 1:
            # 1. Raw data
            raw_result += f"{csc.subheading_style_begin}{_('Raw data')}{csc.subheading_style_end}"

            data = self.data_frame[[groups[0], var_names[0]]].dropna()
            group_levels = sorted(set(data[groups[0]]))
            # index should be specified to work in pandas 0.11; but this way can't use _() for the labels
            pdf_result = pd.DataFrame(columns=group_levels)
            pdf_result.loc[_('N of valid cases')] = [sum(data[groups[0]] == group) for group in group_levels]
            pdf_result.loc[_('N of missing cases')] = [sum(self.data_frame[groups[0]] == group) -
                                                    sum(data[groups[0]] == group) for group in group_levels]
#            for group in group_levels:
#                valid_n = sum(data[groups[0]]==group)
#                missing_n = sum(self.data_frame[groups[0]]==group)-valid_n
#                raw_result += _(u'Group: %s, N of valid cases: %g, N of missing cases: %g\n') %(group, valid_n, missing_n)
            raw_result += cs_stat._format_html_table(pdf_result.to_html(bold_rows=False, classes="table_cs_pd"))
            valid_n = len(self.data_frame[groups[0]].dropna())
            missing_n = len(self.data_frame[groups[0]])-valid_n
            raw_result += '\n\n'+_('N of missing group cases') + ': %g' % missing_n +'\n'

            # Plot individual data
            raw_graph = cs_chart.create_compare_groups_sample_chart(self.data_frame, meas_level, var_names, groups,
                                                                    group_levels, raw_data_only=True, ylims=ylims)

            # Plot the individual data with boxplots
            # There's no need to repeat the mosaic plot for the nominal variables
            if meas_level in ['int', 'unk', 'ord']:
                sample_graph = cs_chart.create_compare_groups_sample_chart(self.data_frame, meas_level, var_names,
                                                                           groups, group_levels, ylims=ylims)
            else:
                sample_graph = None

            # 2. Descriptive data
            sample_result = f"{csc.subheading_style_begin}{_('Sample properties')}{csc.subheading_style_end}"

            if meas_level in ['int', 'unk']:
                sample_result += cs_stat.print_var_stats(self.data_frame, [var_names[0]], self.data_measlevs,
                                                         groups=groups,
                                                         statistics=['mean', 'std', 'max', 'upper quartile', 'median',
                                                                     'lower quartile', 'min'])
            elif meas_level == 'ord':
                sample_result += cs_stat.print_var_stats(self.data_frame, [var_names[0]], self.data_measlevs,
                                                         groups=groups,
                                                         statistics=['max', 'upper quartile', 'median',
                                                                     'lower quartile', 'min'])
            elif meas_level == 'nom':
                sample_result += cs_stat.print_var_stats(self.data_frame, [var_names[0]], self.data_measlevs,
                                                         groups=groups,
                                                         statistics=['variation ratio'])
                sample_result += '\n' + cs_stat.contingency_table(self.data_frame, groups, var_names,
                                                                  count=True, percent=True, margins=True)

            # 3. Population properties
            # Plot population estimations
            group_estimations = cs_stat.comp_group_estimations(self.data_frame, meas_level, var_names, groups)
            population_graph = cs_chart.create_compare_groups_population_chart(self.data_frame, meas_level, var_names,
                                                                               groups, group_levels, ylims=ylims)

            # Population estimations
            population_result = f"{csc.subheading_style_begin}{_('Population properties')}{csc.subheading_style_end}" + \
                                f"<b>{_('Population parameter estimations')}</b>\n"
            if meas_level in ['int', 'unk']:
                population_result += _('Means') + '\n' + _('Present confidence interval values suppose normality.')
                prec = cs_util.precision(self.data_frame[var_names[0]]) + 1
                population_result += \
                    cs_stat._format_html_table(group_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                         float_format=lambda x: '%0.*f' % (prec, x)))
            elif meas_level == 'ord':
                population_result += f"{_('Medians')}\n"
                prec = cs_util.precision(self.data_frame[var_names[0]]) + 1
                population_result += \
                    cs_stat._format_html_table(group_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                         float_format=lambda x: '%0.*f' % (prec, x)))
                population_result += '\n'
            elif meas_level == 'nom':
                population_result += cs_stat.contingency_table(self.data_frame, groups, var_names, ci=True)
                population_result += '\n'

            standardized_effect_size_result = None

            # Hypothesis testing
            standardized_effect_size_result, sample_result_temp, result_ht = cs_hyp_test.decision_one_grouping_variable(self.data_frame, meas_level, self.data_measlevs, var_names, groups, group_levels, single_case_slope_SEs, single_case_slope_trial_n)
            sample_result += sample_result_temp

        # Two grouping variables
        elif len(groups) == 2:
            # 1. Raw data
            raw_result += f"{csc.subheading_style_begin}{_('Raw data')}{csc.subheading_style_end}"

            standardized_effect_size_result = None

            data = self.data_frame[groups + [var_names[0]]].dropna()
            # create a list of sets with the levels of all grouping variables
            levels = [list(set(data[group])) for group in groups]
            for i in range(len(levels)):
                levels[i].sort()
            # TODO sort the levels in other parts of the output, too
            # create all level combinations for the grouping variables
            level_combinations = list(itertools.product(*levels))

            # index should be specified to work in pandas 0.11; but this way can't use _() for the labels
            columns = pd.MultiIndex.from_tuples(level_combinations, names=groups)
            pdf_result = pd.DataFrame(columns=columns)

            pdf_result.loc[_('N of valid cases')] = [sum(
                (data[groups] == pd.Series({group: level for group, level in zip(groups, group_level)})).all(axis=1))
                                                     for group_level in level_combinations]
            pdf_result.loc[_('N of missing cases')] = [sum(
                (self.data_frame[groups] == pd.Series({group: level for group, level in zip(groups, group_level)})).all(
                    axis=1)) -
                                                       sum((data[groups] == pd.Series({group: level for group, level in
                                                                                       zip(groups, group_level)})).all(
                                                           axis=1)) for group_level in level_combinations]
            #            for group in group_levels:
            #                valid_n = sum(data[groups[0]]==group)
            #                missing_n = sum(self.data_frame[groups[0]]==group)-valid_n
            #                raw_result += _(u'Group: %s, N of valid cases: %g, N of missing cases: %g\n') %(group, valid_n, missing_n)
            raw_result += cs_stat._format_html_table(pdf_result.to_html(bold_rows=False, classes="table_cs_pd"))
            raw_result += '\n\n'
            for group in groups:
                valid_n = len(self.data_frame[group].dropna())
                missing_n = len(self.data_frame[group]) - valid_n
                raw_result += _('N of missing grouping variable in %s') % group + ': %g\n' % missing_n

            # Plot individual data

            raw_graph = cs_chart.create_compare_groups_sample_chart(self.data_frame, meas_level, var_names, groups,
                                                                  level_combinations, raw_data_only=True, ylims=ylims)

            # Plot the individual data with boxplots
            # There's no need to repeat the mosaic plot for the nominal variables
            if meas_level in ['int', 'unk', 'ord']:
                sample_graph = cs_chart.create_compare_groups_sample_chart(self.data_frame, meas_level, var_names, groups,
                                                                         level_combinations, ylims=ylims)
            else:
                sample_graph = None

            # 2. Sample properties
            sample_result = f"{csc.subheading_style_begin}{_('Sample properties')}{csc.subheading_style_end}"

            if meas_level in ['int', 'unk']:
                sample_result += cs_stat.print_var_stats(self.data_frame, [var_names[0]], self.data_measlevs,
                                                         groups=groups,
                                                         statistics=['mean', 'std', 'max', 'upper quartile', 'median',
                                                                     'lower quartile', 'min'])
            elif meas_level == 'ord':
                sample_result += cs_stat.print_var_stats(self.data_frame, [var_names[0]], self.data_measlevs,
                                                         groups=groups,
                                                         statistics=['max', 'upper quartile', 'median',
                                                                     'lower quartile', 'min'])
            elif meas_level == 'nom':
                sample_result += cs_stat.print_var_stats(self.data_frame, [var_names[0]], self.data_measlevs,
                                                         groups=groups,
                                                         statistics=['variation ratio'])
                sample_result += '\n' + cs_stat.contingency_table(self.data_frame, groups, var_names,
                                                                  count=True, percent=True, margins=True)

            # 3. Population properties
            # Plot population estimations
            group_estimations = cs_stat.comp_group_estimations(self.data_frame, meas_level, var_names, groups)
            population_graph = cs_chart.create_compare_groups_population_chart(self.data_frame, meas_level, var_names,
                                                                               groups, level_combinations, ylims=ylims)

            # Population estimation
            population_result = f"{csc.subheading_style_begin}{_('Population properties')}{csc.subheading_style_end}" + \
                                f"<b>{_('Population parameter estimations')}</b>\n"
            if meas_level in ['int', 'unk']:
                population_result += _('Means') + '\n' + _('Present confidence interval values suppose normality.')
            elif meas_level == 'ord':
                population_result += f"{_('Medians')}"
            if meas_level in ['int', 'unk', 'ord']:
                prec = cs_util.precision(self.data_frame[var_names[0]]) + 1
                population_result += \
                    cs_stat._format_html_table(group_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                         float_format=lambda x: '%0.*f' % (prec, x)))
                population_result += '\n'
            if meas_level == 'nom':
                population_result += '\n' + cs_stat.contingency_table(self.data_frame, groups, var_names, ci=True)

            # Hypothesis testing
            result_ht = cs_hyp_test.decision_two_grouping_variables(self.data_frame, meas_level, var_names, groups)

        elif len(groups) > 2:
            raw_result += '<decision>'+_('Several grouping variables.')+' >> '+'<default>\n'
            return self._convert_output([title, raw_result])

        return self._convert_output([title, raw_result, raw_graph, sample_result, sample_graph, population_result,
                                     standardized_effect_size_result, population_graph, result_ht])


def display(results):
    """Display list of output given by CogStat analysis in IPython Notebook

    :param results: list of output
    :return:
    """
    from IPython.display import display
    from IPython.display import HTML
    for result in results:
        if isinstance(result, str):
            display(HTML(result))
        else:
            display(result)
    plt.close('all')

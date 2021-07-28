# -*- coding: utf-8 -*-

"""This module is the main engine for CogStat. It includes the class for the CogStat data; initialization handles data
import; methods implement some data handling and they compile the appropriate statistics for the main analysis commands.
"""

# if CS is used with GUI, start the splash screen
QString = str

# go on with regular importing, etc.
import gettext
import itertools
import logging
import os
import datetime
import string

__version__ = '2.1.1dev'

import matplotlib
matplotlib.use("qt5agg")
#print matplotlib.get_backend()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import cogstat_config as csc
csc.versions['cogstat'] = __version__
from . import cogstat_stat as cs_stat
from . import cogstat_hyp_test as cs_hyp_test
from . import cogstat_util as cs_util
from . import cogstat_chart as cs_chart
cs_util.get_versions()

logging.root.setLevel(logging.INFO)
t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext

warn_unknown_variable = '<warning>'+_('The measurement levels of the variables are not set. '
                                      'Set them in your data source.') \
                        + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                        % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                        + '\n</warning>'
                        # TODO it might not be necessary to repeat this warning in the analyses, use only at import?


class CogStatData:
    """
    Import data and create CogStat data object.

    The measurement levels are set in the following order:

    - All variables are 'unknown'.
    - Then, if import data includes measurement level, then that information will be used.
    - Then, if measurement_levels parameter is set, then that will be used.
    - Finally, constraints (e.g., string variables can be nominal variables) will overwrite measurement level.

    Parameters
    ----------
    data : pandas.DataFrame or str
        Data to be imported. This can be:

        - Pandas DataFrame
        - Clipboard data from a spreadsheet (identified as multiline string)
        - Filename (identified as one line text)

    measurement_levels : None, list of {'nom', 'ord', 'int'} or dict of {str: {'nom', 'ord', 'int'}}
        Optional measurement levels of the variables

        - None: measurement level of the import file or the clipboard information will be used
        - List of strings ('nom', 'ord', 'int'): measurement levels will be assigned to variables in that order. It
        overwrites the import data information. Additional constraints (e.g., string variables can be nominal variables)
        will overwrite this.
        - Dictionary, items are variable name and measurement level pairs: measurement levels will be assigned to the
        appropriate variables. It overwrites the import data information. Additional constraints (e.g., string variables
        can be nominal variables) will overwrite this.
    """

    def __init__(self, data, measurement_levels=None):
        pass
        """In the input data:
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
        """

        self.orig_data_frame = None
        self.data_frame = None
        self.data_measlevs = None
        self.import_source = ''
        self.import_message = ''  # can't return anything to caller,
                                  # since we're in an __init__ method, so store the message here
        self.filtering_status = None

        self._import_data(data=data, measurement_levels=measurement_levels)

    ### Import and handle the data ###

    def _import_data(self, data='', measurement_levels=None):

        def _percent2float():
            """ Convert x.x% str format to float in self.data_frame (pandas cannot handle this).
            """
            for column in self.data_frame.select_dtypes('object'):  # check only string variables (or boolean with NaN)
                selected_cells = self.data_frame[column].astype(str).str.endswith('%')
                # .str can be used only with strings, but not booleans; so use astype(str) to convert boolean cells
                # in an object type variable to str when boolean variable with missing value was imported (therefore
                # the dtype is object)
                if selected_cells.any():
                    # use  selected_cells == True  to overcome the Nan indexes
                    self.data_frame[column][selected_cells == True] = \
                        self.data_frame[column][selected_cells == True].str.replace('%', '').astype('float') / 100.0
                    try:
                        self.data_frame[column] = self.data_frame[column].astype('float')
                    except ValueError:  # there may be other non xx% strings in the variable
                        pass

        def _set_measurement_level(measurement_levels=None):
            """ Create self.data_measlevs.

            Parameters
            ----------
            measurement_levels: None or list of str or dict
                None: measurement level of the import file or the clipboard information will be used
                List of {'nom', 'ord', 'int', 'unk', '', 'nan'}:
                    measurement levels will be assigned to variables in that order
                Dictionary, items are variable name and measurement level pairs:
                    measurement levels will be assigned to the appropriate variables
            '' and 'nan' is converted to 'unk'
            List and dict will overwrite the import data information. Additional constraints (e.g., string variables
            can be nominal variables) will overwrite this.
            """

            # By default, all variables have 'unknown' measurement levels
            self.data_measlevs = {name: 'unk' for name in self.data_frame.columns}

            # 0. Measurement levels are not set
            if not measurement_levels:
                self.import_message += '\n<warning>'+warn_unknown_variable+'</warning>'

            # 1. Set the levels (coming either from the import data information or from measurement_levels object
            # parameter)
            elif measurement_levels:  # If levels were given, set them

                # Only levels are given - in the order of the variables
                if type(measurement_levels) is list:
                    # Check if only valid measurement levels were given
                    if not (set(measurement_levels) <= {'unk', 'nom', 'ord', 'int', '', 'nan'}):
                        raise ValueError('Invalid measurement level')
                    # make levels lowercase and replace '' or 'nan' with 'unk'
                    measurement_levels = ['unk' if level.lower in ['', 'nan'] else level.lower()
                                          for level in measurement_levels]
                    self.data_measlevs = {name: level for name, level in
                                          zip(self.data_frame.columns, measurement_levels)}

                # Name-level pairs are given in a dictionary
                elif type(measurement_levels) is dict:
                    # Check if only valid measurement levels were given
                    if not (set(measurement_levels.values()) <= {'unk', 'nom', 'ord', 'int', '', 'nan'}):
                        raise ValueError('Invalid measurement level')
                    # make levels lowercase and replace '' or 'nan' with 'unk'
                    measurement_levels = {name: ('unk' if measurement_levels[name].lower() in ['', 'nan']
                                                 else measurement_levels[name].lower())
                                          for name in measurement_levels.keys()}
                    self.data_measlevs = {name: measurement_levels[name] for name in measurement_levels.keys()}

                if len(self.data_frame.columns) != len(measurement_levels):
                    self.import_message += '\n<warning>' + \
                                           _('Number of measurement levels do not match the number of variables. '
                                             'You may want to correct the number of measurement levels.')

            # 2. Apply constraints to measurement levels.
            # String variables cannot be interval or nominal variables in CogStat, so change them to nominal
            invalid_var_names = [var_name for var_name in self.data_frame.columns if
                                 (self.data_measlevs[var_name] in ['int', 'ord', 'unk'] and
                                  self.data_frame[var_name].dtype == 'object')]
                # 'object' dtype means string variable
            if invalid_var_names:  # these str variables were set to int or ord
                for var_name in invalid_var_names:
                    self.data_measlevs[var_name] = 'nom'
                self.import_message += '\n<warning>' + \
                                       _('String variables cannot be interval or ordinal variables in CogStat. '
                                         'Those variables are automatically set to nominal: ')\
                                       + ''.join(', %s' % var_name for var_name in invalid_var_names)[2:] + '. ' + \
                                       _('You can fix this issue in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</warning>'

            # Warn when not all measurement levels are set
            if set(self.data_measlevs) in ['unk']:
                self.import_message += '\n<warning>' + \
                                       _('The measurement level was not set for all variables.') + ' '\
                                       + _('You can fix this issue in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</warning>'
        # end of set_measurement_level()

        def _convert_dtypes():
            # Convert dtypes
            # CogStat does not know boolean variables, it is converted to string
            #   Although this solution changes upper and lower cases: independent of the text,
            #   it will be 'True' and 'False'
            # Some analyses do not handle Int types, but int types
            # Some analyses do not handle category types
            convert_dtypes = [['bool', 'string'],
                              ['Int32', 'int32'],
                              ['Int64', 'int64'], ['Int64', 'float64'],
                              ['category', 'object']]
            for old_dtype, new_dtype in convert_dtypes:
                try:
                    self.data_frame[self.data_frame.select_dtypes(include=[old_dtype]).columns] = \
                        self.data_frame.select_dtypes(include=[old_dtype]).astype(new_dtype)
                except ValueError:
                    pass
                    # next convert_dtype pair will be used in a next loop if convert_dtypes includes alternatives

        def _check_valid_chars():
            # Check if only valid chars are used in the data, and warn the user if invalid chars are used
            # TODO this might be removed with Python3 and with unicode encoding
            non_ascii_var_names = []
            non_ascii_vars = []
            valid_chars = string.ascii_letters + string.digits + '_'
            for variable_name in self.data_frame:
                # check the variable name
                if not all(char in valid_chars for char in variable_name):  # includes non-valid char
                    non_ascii_var_names.append(variable_name)
                # check the values
                if self.data_frame[variable_name].dtype == 'object':  # check only string variables
                    for ind_data in self.data_frame[variable_name]:
                        if not (ind_data != ind_data) and not (isinstance(ind_data, (bool, int, float, datetime.date))):
                            # if not NaN, otherwise the next condition is invalid
                            # and if not boolean, etc. (int and float can occur in object dtype)
                            if not all(char in valid_chars for char in ind_data):
                                non_ascii_vars.append(variable_name)
                                break  # after finding the first non-ascii data, we can skip the rest of the variable data
            if non_ascii_var_names:
                self.import_message += '\n<warning>' + \
                                       _('Some variable name(s) include other than English characters, numbers, or '
                                         'underscore which can cause problems in some analyses: %s.') \
                                       % ''.join(
                    ' %s' % non_ascii_var_name for non_ascii_var_name in non_ascii_var_names) \
                                       + ' ' + _('If some analyses cannot be run, fix this in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</warning>'
            if non_ascii_vars:
                self.import_message += '\n<warning>' + \
                                       _('Some string variable(s) include other than English characters, numbers, or '
                                         'underscore which can cause problems in some analyses: %s.') \
                                       % ''.join(' %s' % non_ascii_var for non_ascii_var in non_ascii_vars) \
                                       + ' ' + _('If some analyses cannot be run, fix this in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</warning>'


        import_measurement_levels = None

        # 1. Import from pandas DataFrame
        if isinstance(data, pd.DataFrame):
            self.data_frame = data
            self.import_source = _('pandas dataframe')

        # 2. Import from file
        elif isinstance(data, str) and not ('\n' in data):  # Single line text, i.e., filename
            # Check if the file exists # TODO
            # self.import_source = _('Import failed')
            # return
            filetype = data[data.rfind('.'):]

            # Import csv file
            if filetype in ['.txt', '.csv', '.log', '.dat', '.tsv']:
                # Check if there is a measurement level line
                meas_row = list(pd.read_csv(data, sep=None, engine='python').iloc[0])
                meas_row = list(map(str, meas_row))
                if {a.lower() for a in meas_row} <= {'unk', 'nom', 'ord', 'int', '', 'nan'} and set(meas_row) != {''}:
                    import_measurement_levels = meas_row
                skiprows = [1] if import_measurement_levels else None

                # Read the file
                self.data_frame = pd.read_csv(data, sep=None, engine='python', skiprows=skiprows,
                                              skip_blank_lines=False)
                self.import_source = _('Text file') + ' - ' + data  # filename

            # Import from spreadsheet files
            elif filetype in ['.ods', '.xls', '.xlsx']:
                # engine should be set manually in pandas 1.0.5, later pandas version may handle this automatically
                engine = {'.ods': 'odf', '.xls': 'xlrd', '.xlsx': 'openpyxl'}
                self.data_frame = pd.read_excel(data, engine=engine[filetype])
                # if there is a measurement level line, use it, and reread the spreadsheet
                meas_row = list(map(str, list(self.data_frame.iloc[0])))
                if {a.lower() for a in meas_row} <= {'unk', 'nom', 'ord', 'int', '', 'nan'} and set(meas_row) != {''}:
                    import_measurement_levels = meas_row
                    self.data_frame = pd.read_excel(data, engine=engine[filetype], skiprows=[1])
                self.import_source = _('Spreadsheet file') + ' - ' + data  # filename

            # Import SPSS, SAS and STATA files
            elif filetype in ['.sav', '.zsav', '.por', '.sas7bdat', '.xpt', '.dta']:
                import pyreadstat

                # Read import file
                if filetype in ['.sav', '.zsav']:
                    import_data, import_metadata = pyreadstat.read_sav(data)
                    # pandas (as of v1.2) uses pyreadstat, but ignores measurement level information
                elif filetype == '.por':
                    import_data, import_metadata = pyreadstat.read_por(data)
                    # pandas (as of v1.2) uses pyreadstat, but ignores measurement level information
                elif filetype == '.sas7bdat':
                    import_data, import_metadata = pyreadstat.read_sas7bdat(data)
                    # alternative solution in pandas:
                    # https://pandas.pydata.org/pandas-docs/stable/reference/io.html#sas
                elif filetype == '.xpt':
                    import_data, import_metadata = pyreadstat.read_xport(data)
                    # alternative solution in pandas:
                    # https://pandas.pydata.org/pandas-docs/stable/reference/io.html#sas
                elif filetype == '.dta':
                    import_data, import_metadata = pyreadstat.read_dta(data)
                    # alternative solution in pandas:
                    # https://pandas.pydata.org/pandas-docs/stable/reference/io.html#stata
                self.data_frame = pd.DataFrame.from_records(import_data, columns=import_metadata.column_names)

                # Convert measurement levels from import format to CogStat
                # We use pyreadstat variable_measure
                # https://ofajardo.github.io/pyreadstat_documentation/_build/html/index.html#metadata-object-description
                if filetype in ['.sav', '.zsav', '.por']:
                    import_to_cs_meas_lev = {'unknown': 'unk', 'nominal': 'nom', 'ordinal': 'ord', 'scale': 'int',
                                             'ratio': 'int', 'flag': 'nom', 'typeless': 'unk'}
                elif filetype in ['.sas7bdat', '.xpt']:
                    # TODO this should be checked; I couldn't find relevant information or test file
                    import_to_cs_meas_lev = {'unknown': 'unk', 'nominal': 'nom', 'ordinal': 'ord',
                                             'interval': 'int','ratio': 'int'}
                elif filetype == '.dta':  # filetype does not include measurement level information
                    import_to_cs_meas_lev = {'unknown': 'unk'}
                import_measurement_levels = [import_to_cs_meas_lev[import_metadata.variable_measure[var_name]]
                                             for var_name in import_metadata.column_names]

                self.import_source = _('SPSS/SAS/STATA file') + ' - ' + data  # filename

            # Import from R files
            elif filetype.lower() in ['.rdata', '.rds', '.rda']:
                import pyreadr
                import_data = pyreadr.read_r(data)
                self.data_frame = import_data[list(import_data.keys())[0]]
                self.data_frame= self.data_frame.convert_dtypes()
                self.import_source = _('R file') + ' - ' + data  # filename

            # Import JASP files
            elif filetype == '.jasp':
                from . import cogstat_stat_num as cs_stat_num
                import_pdf, import_measurement_levels = cs_stat_num.read_jasp_file(data)
                self.data_frame = import_pdf.convert_dtypes()
                self.import_source = _('JASP file') + ' - ' + data  # filename

            # Import jamovi files
            elif filetype == '.omv':
                from . import cogstat_stat_num as cs_stat_num
                import_pdf, import_measurement_levels = cs_stat_num.read_jamovi_file(data)
                self.data_frame = import_pdf.convert_dtypes()
                self.import_source = _('jamovi file') + ' - ' + data  # filename

        # 3. Import from clipboard
        elif isinstance(data, str) and ('\n' in data):  # Multi line text, i.e., clipboard data
            # Check if there is variable type line
            import io
            clipboard_file = io.StringIO(data)
            """ # old version; if everything goes smoothly, this can be removed
            f = io.StringIO(data)
            next(f)
            meas_row = next(f).replace('\n', '').replace('\r', '').split(delimiter)
            # \r was used in Mac after importing from Excel clipboard
            """
            meas_row = list(pd.read_csv(clipboard_file, sep=None, engine='python').iloc[0])
            meas_row = list(map(str, meas_row))
            if {a.lower() for a in meas_row} <= {'unk', 'nom', 'ord', 'int', '', 'nan'} and set(meas_row) != {''}:
                import_measurement_levels = meas_row
            skiprows = [1] if import_measurement_levels else None

            # Read the clipboard
            clipboard_file = io.StringIO(data)
            self.data_frame = pd.read_csv(clipboard_file, sep=None, engine='python',
                                          skiprows=skiprows, skip_blank_lines=False)
            self.import_source = _('clipboard')

        # 4. Invalid data source
        else:
            self.import_source = _('Import failed')
            return

        # Set additional details for all import sources
        _percent2float()
        _convert_dtypes()
        _set_measurement_level(measurement_levels=(measurement_levels if measurement_levels else
                                                  import_measurement_levels))
                               # measurement_levels overwrites import_measurement_levels
        _check_valid_chars()
        self.orig_data_frame = self.data_frame.copy()

        # Add keys with pyqt string form, too, because UI returns variable names in this form
        # TODO do we still need this?
        from PyQt5 import QtCore
        for var_name in self.data_frame.columns:
            self.data_measlevs[QString(var_name)] = self.data_measlevs[var_name]

    def print_data(self, brief=False):
        """
        Display the data.

        Parameters
        ----------
        brief : bool
            Should only the first few cases or the whole data frame be displayed?

        Returns
        -------
        str
            HTML string showing the data.
        """
        output = '<cs_h1>' + _('Data') + '</cs_h1>'
        output += _('Source: ') + self.import_source + '\n'
        output += str(len(self.data_frame.columns)) + _(' variables and ') + \
                  str(len(self.data_frame.index)) + _(' cases') + '\n'
        output += self._filtering_status()

        dtype_convert = {'int32': 'num', 'int64': 'num', 'float32': 'num', 'float64': 'num',
                         'object': 'str', 'string': 'str', 'category': 'str', 'datetime64[ns]': 'str'}
        data_prop = pd.DataFrame([[dtype_convert[str(self.data_frame[name].dtype).lower()] for name in self.data_frame.columns],
                                  [self.data_measlevs[name] for name in self.data_frame.columns]],
                                 columns=self.data_frame.columns)
        data_comb = pd.concat([data_prop, self.data_frame])
        data_comb.index = [_('Type'), _('Level')]+[' ']*len(self.data_frame)
        output += cs_stat._format_html_table(data_comb[:12 if brief else 1002].to_html(bold_rows=False,
                                                                                       classes="table_cs_pd"))
        if brief and (len(self.data_frame.index) > 10):
            output += str(len(self.data_frame.index)-10) + _(' further cases are not displayed...')+'\n'
        elif len(self.data_frame.index) > 999:
            output += _('The next %s cases will not be printed. You can check all cases in the original data source.') \
                      % (len(self.data_frame.index)-1000) + '\n'

        return cs_util.convert_output([output])

    def filter_outlier(self, var_names=None, mode='2sd'):
        """
        Filter the data_frame based on outliers.

        All variables are investigated independently and cases are excluded if any variables shows they are outliers.
        If var_names is None, then all cases are used.

        Parameters
        ----------
        var_names : None or list of str
            Names of the variables the exclusion is based on or None to include all cases.
        mode : {'2sd'}
            Mode of the exclusion - only 2sd is available at the moment

        Returns
        -------
        list of str
            List of HTML strings showing the filtered cases. The method modifies the dataframe in place.
        """
        title = '<cs_h1>' + _('Filtering') + '</cs_h1>'
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
                    text_output += _('Only interval variables can be used for filtering. Ignoring variable %s.') % \
                                   var_name + '\n'
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

        return cs_util.convert_output([title, text_output])

    def _filtering_status(self):
        if self.filtering_status:
            return '<b>Filtering is on: %s</b>\n' % self.filtering_status
        else:
            return ''

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
        """
        Explore a single variable.

        Parameters
        ----------
        var_name : str
            Name of the variable
        frequencies : bool
            Should the frequencies be shown?
        central_value : float
            Test value for testing central tendency.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """
        plt.close('all')
        meas_level, unknown_type = self._meas_lev_vars([var_name])
        result_list = ['<cs_h1>' + _('Explore variable') + '</cs_h1>']
        result_list.append(_('Exploring variable: ') + var_name + ' (%s)\n' % meas_level)
        if self._filtering_status():
            result_list[-1] += self._filtering_status()

        # 1. Raw data
        text_result = '<cs_h2>' + _('Raw data') + '</cs_h2>'
        text_result2 = cs_stat.display_variable_raw_data(self.data_frame, var_name)
        image = cs_chart.create_variable_raw_chart(self.data_frame, self.data_measlevs, var_name,
                                                   self.data_frame[var_name].dropna())
        result_list.append(text_result+text_result2)
        result_list.append(image)

        # 2. Sample properties
        text_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'

        # Frequencies
        if frequencies:
            text_result += '<cs_h3>'+_('Frequencies')+'</cs_h3>'
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
        if self.data_measlevs[var_name] != 'nom':  # histogram for nominal variable has already been shown in raw data
            image = cs_chart.create_histogram_chart(self.data_frame, self.data_measlevs, var_name)
            result_list.append(image)

        # 3. Population properties
        text_result = '<cs_h2>' + _('Population properties') + '</cs_h2>'

        # Normality
        if meas_level in ['int', 'unk']:
            text_result += '<cs_h3>'+_('Normality')+'</cs_h3>\n'
            stat_result, text_result2 = cs_hyp_test.normality_test(self.data_frame, self.data_measlevs, var_name)
            image, image2 = cs_chart.create_normality_chart(self.data_frame[var_name].dropna(), var_name)
                # histogram with normality and qq plot
            text_result += text_result2
            result_list.append(text_result)
            if image:
                result_list.append(image)
            if image2:
                result_list.append(image2)
        else:
            result_list.append(text_result)

        # Population estimations
        if meas_level in ['int', 'ord', 'unk']:
            prec = cs_util.precision(self.data_frame[var_name]) + 1

        population_param_text = '\n<cs_h3>' + _('Population parameter estimations') + '</cs_h3>\n'
        if meas_level in ['int', 'unk']:
            population_param_text += cs_stat.variable_estimation(self.data_frame[var_name], ['mean', 'std'])
        elif meas_level == 'ord':
            population_param_text += cs_stat.variable_estimation(self.data_frame[var_name], ['median'])
        elif meas_level == 'nom':
            population_param_text += cs_stat.proportions_ci(self.data_frame, var_name)
        text_result = '\n'

        # Hypothesis tests
        text_result += '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>\n'
        if self.data_measlevs[var_name] in ['int', 'unk']:
            text_result += '<decision>' + _('Testing if mean deviates from the value %s.') % central_value +\
                           '</decision>\n'
        elif self.data_measlevs[var_name] == 'ord':
            text_result += '<decision>' + _('Testing if median deviates from the value %s.') % central_value +\
                           '</decision>\n'

        if unknown_type:
            text_result += '<decision>' + warn_unknown_variable + '\n</decision>'
        if meas_level in ['int', 'unk']:
            text_result += '<decision>' + _('Interval variable.') + ' >> ' + \
                           _('Choosing one-sample t-test or Wilcoxon signed-rank test depending on the assumption.') + \
                           '</decision>\n'
            text_result += '<decision>' + _('Checking for normality.') + '\n</decision>'
            norm, text_result_norm = cs_hyp_test.normality_test(self.data_frame, self.data_measlevs, var_name)

            text_result += text_result_norm
            if norm:
                text_result += '<decision>' + _('Normality is not violated.') + ' >> ' + \
                               _('Running one-sample t-test.') + '</decision>\n'
                text_result2, ci = cs_hyp_test.one_t_test(self.data_frame, self.data_measlevs, var_name,
                                                          test_value=central_value)
                graph = cs_chart.create_variable_population_chart(self.data_frame[var_name].dropna(), var_name, ci)

            else:
                text_result += '<decision>' + _('Normality is violated.') + ' >> ' + \
                               _('Running Wilcoxon signed-rank test.') + '</decision>\n'
                text_result += _('Median: %0.*f') % (prec, np.median(self.data_frame[var_name].dropna())) + '\n'
                text_result2 = cs_hyp_test.wilcox_sign_test(self.data_frame, self.data_measlevs, var_name,
                                                            value=central_value)
                graph = cs_chart.create_variable_population_chart_2(self.data_frame[var_name].dropna(), var_name)

        elif meas_level == 'ord':
            text_result += '<decision>' + _('Ordinal variable.') + ' >> ' + _('Running Wilcoxon signed-rank test.') + \
                           '</decision>\n'
            text_result2 = cs_hyp_test.wilcox_sign_test(self.data_frame, self.data_measlevs, var_name,
                                                        value=central_value)
            graph = cs_chart.create_variable_population_chart_2(self.data_frame[var_name].dropna(), var_name)
        else:
            text_result2 = '<decision>' + _('Sorry, not implemented yet.') + '</decision>\n'
            graph = None
        text_result += text_result2

        result_list.append(population_param_text)
        if graph:
            result_list.append(graph)
        result_list.append(text_result)
        return cs_util.convert_output(result_list)

    def explore_variable_pair(self, x, y, xlims=[None, None], ylims=[None, None]):
        """
        Explore a variable pair.

        Parameters
        ----------
        x : str
            Name of the x variable.
        y : str
            Name of the y variable.
        xlims : list of {int or float}
            Limit of the x axis for interval and ordinal variables instead of using automatic values.
        ylims : list of {int or float}
            Limit of the y axis for interval and ordinal variables instead of using automatic values.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """
        plt.close('all')
        meas_lev, unknown_var = self._meas_lev_vars([x, y])
        title = '<cs_h1>' + _('Explore relation of variable pair') + '</cs_h1>'
        raw_result = _('Exploring variable pair: ') + x + ' (%s), ' % self.data_measlevs[x] + y + \
                     ' (%s)\n' % self.data_measlevs[y]
        raw_result += self._filtering_status()
        if unknown_var:
            raw_result += '<decision>'+warn_unknown_variable+'\n</decision>'

        # 1. Raw data
        raw_result += '<cs_h2>' + _('Raw data') + '</cs_h2>'
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
        sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'
        if meas_lev == 'nom':
            sample_result += cs_stat.contingency_table(self.data_frame, [x], [y], count=True, percent=True,
                                                       margins=True)
        estimation_result = '<cs_h2>' + _('Population properties') + '</cs_h2>' + \
                            '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>\n'
        pdf_result = pd.DataFrame(columns=[_('Point estimation'), _('95% confidence interval')])
        population_result = '\n'

        # Compute and print numeric results
        slope, intercept = None, None
        if meas_lev == 'int':
            population_result += '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>\n' + '<decision>' + \
                                 _('Testing if correlation differs from 0.') + '</decision>\n'
            population_result += '<decision>'+_('Interval variables.')+' >> ' + \
                                  _("Running Pearson's and Spearman's correlation.") + '\n</decision>'
            df = len(data)-2
            r, p = stats.pearsonr(data.iloc[:, 0], data.iloc[:, 1])  # TODO select variables by name instead of iloc
            population_result += _("Pearson's correlation") + \
                                 ': <i>r</i>(%d) = %0.*f, %s\n' % \
                                 (df, cs_hyp_test.non_data_dim_precision, r, cs_hyp_test.print_p(p))

            slope, intercept, r_value, p_value, std_err = stats.linregress(data.iloc[:, 0], data.iloc[:, 1])
            # TODO output with the precision of the data
            sample_result += _('Linear regression')+': y = %0.3fx + %0.3f' % (slope, intercept)

            r, p = stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])
            population_result += _("Spearman's rank-order correlation") + \
                                 ': <i>r<sub>s</sub></i>(%d) = %0.*f, %s' % \
                                 (df, cs_hyp_test.non_data_dim_precision, r, cs_hyp_test.print_p(p))
        elif meas_lev == 'ord':
            population_result += '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>\n' + '<decision>' + \
                                 _('Testing if correlation differs from 0.') + '</decision>\n'
            population_result += '<decision>'+_('Ordinal variables.')+' >> '+_("Running Spearman's correlation.") + \
                                 '\n</decision>'
            df = len(data)-2
            r, p = stats.spearmanr(data.iloc[:, 0], data.iloc[:, 1])
            population_result += _("Spearman's rank-order correlation") + \
                                 ': <i>r<sub>s</sub></i>(%d) = %0.*f, %s' % \
                                 (df, cs_hyp_test.non_data_dim_precision, r, cs_hyp_test.print_p(p))
        elif meas_lev == 'nom':
            estimation_result += cs_stat.contingency_table(self.data_frame, [x], [y], ci=True)
            population_result += '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>\n' + '<decision>' + \
                                 _('Testing if variables are independent.') + '</decision>\n'
            if not(self.data_measlevs[x] == 'nom' and self.data_measlevs[y] == 'nom'):
                population_result += '<warning>' + _('Not all variables are nominal. Consider comparing groups.') + \
                                     '</warning>\n'
            population_result += '<decision>' + _('Nominal variables.') + ' >> ' + _('Running Cramér\'s V.') + \
                                 '\n</decision>'
            chi_result = cs_hyp_test.chi_squared_test(self.data_frame, x, y)
            population_result += chi_result
        standardized_effect_size_result = cs_stat.variable_pair_standard_effect_size(data, meas_lev, sample=True)
        estimation_result += cs_stat.variable_pair_standard_effect_size(data, meas_lev, sample=False)
        sample_result += '\n'
        population_result += '\n'

        # Make graph
        # extra chart is needed only for int variables, otherwise the chart would just repeat the raw data
        if meas_lev in ['int', 'unk']:
            sample_graph = cs_chart.create_variable_pair_chart(data, meas_lev, slope, intercept, x, y, self.data_frame,
                                                               xlims=xlims, ylims=ylims)
        else:
            sample_graph = None
        return cs_util.convert_output([title, raw_result, raw_graph, sample_result, standardized_effect_size_result,
                                       sample_graph, estimation_result, population_result])

    #correlations(x,y)  # test

    def pivot(self, depend_name='', row_names=[], col_names=[], page_names=[], function='Mean'):
        """
        Compute pivot table.

        Parameters
        ----------
        depend_name : str
            Variable serving as dependent variables.
        row_names : list of str
            Variable names serving as row grouping variables.
        col_names : list of str
            Variable names serving as column grouping variables.
        page_names : list of str
            Variable names serving as page  grouping variables.
        function : {'N', 'Sum', 'Mean', 'Median', 'Lower quartile', 'Upper quartile', 'Standard deviation', 'Variance'}
            Functions applied to pivot cells. Use localized version if CogStat is used in a non-English language.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """
        # TODO optionally return pandas DataFrame or Panel
        title = '<cs_h1>' + _('Pivot table') + '</cs_h1>'
        pivot_result = cs_stat.pivot(self.data_frame, row_names, col_names, page_names, depend_name, function)
        return cs_util.convert_output([title, pivot_result])

    def diffusion(self, error_name=[], RT_name=[], participant_name=[], condition_names=[]):
        """
        Run diffusion analysis on behavioral data.

        Dataframe should include a single trial in a case (row).

        Parameters
        ----------
        error_name : list of str
            Name of the variable storing the errors.
            Error should be coded as 1, correct response as 0.
        RT_name : list of str
            Name of the variable storing response times.
            Time should be stored in sec.
        participant_name : list of str
            Name of the variable storing participant IDs.
        condition_names : list of str
            Name(s) of the variable(s) storing conditions.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """
        # TODO return pandas DataFrame
        title = '<cs_h1>' + _('Behavioral data diffusion analysis') + '</cs_h1>'
        pivot_result = cs_stat.diffusion(self.data_frame, error_name, RT_name, participant_name, condition_names)
        return cs_util.convert_output([title, pivot_result])

    def compare_variables(self, var_names, factors=[], ylims=[None, None]):
        """
        Compare repeated measures variables.

        Parameters
        ----------
        var_names: list of str
            The variable to be compared.
        factors : list of list of [str, int]
            The factors and their levels, e.g.,

                [['name of the factor', number_of_the_levels],
                ['name of the factor 2', number_of_the_levels]]

            Factorial combination of the factors will be generated, and variables will be assigned respectively
        ylims : list of {int or float}
            Limit of the y axis for interval and ordinal variables instead of using automatic values.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """
        plt.close('all')
        title = '<cs_h1>' + _('Compare repeated measures variables') + '</cs_h1>'
        meas_levels = [self.data_measlevs[var_name] for var_name in var_names]
        raw_result = _('Variables to compare: ') + ', '.\
            join('%s (%s)' % (var, meas) for var, meas in zip(var_names, meas_levels)) + '\n'
        if factors:
            raw_result += _('Factors (number of levels): ') + ', '.\
                join('%s (%d)' % (factor[0], factor[1]) for factor in factors) + '\n'
            factor_combinations = ['']
            for factor in factors:
                factor_combinations = ['%s - %s %s' % (factor_combination, factor[0], level_i+1) for factor_combination
                                       in factor_combinations for level_i in range(factor[1])]
            factor_combinations = [factor_combination[3:] for factor_combination in factor_combinations]
            for factor_combination, var_name in zip(factor_combinations, var_names):
                raw_result += '%s: %s\n' % (factor_combination, var_name)

        raw_result += self._filtering_status()

        # Check if the variables have the same measurement levels
        meas_levels = {self.data_measlevs[var_name] for var_name in var_names}
        if len(meas_levels) > 1:
            if 'ord' in meas_levels or 'nom' in meas_levels:  # int and unk can be used together,
                                                              # since unk is taken as int by default
                return cs_util.convert_output([title, raw_result, '<decision>' +
                                               _("Sorry, you can't compare variables with different measurement levels."
                                                 " You could downgrade higher measurement levels to lowers to have the "
                                                 "same measurement level.") + '</decision>'])
        # level of measurement of the variables
        meas_level, unknown_type = self._meas_lev_vars(var_names)
        if unknown_type:
            raw_result += '\n<decision>'+warn_unknown_variable+'</decision>'

        # 1. Raw data
        raw_result += '<cs_h2>' + _('Raw data') + '</cs_h2>'
        # Prepare data, drop missing data
        # TODO are NaNs interesting in nominal variables?
        data = self.data_frame[var_names].dropna()
        valid_n = len(data)
        missing_n = len(self.data_frame[var_names])-valid_n
        raw_result += _('N of valid cases') + ': %g\n' % valid_n
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
        sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'

        if meas_level in ['int', 'unk']:
            sample_result += cs_stat.print_var_stats(self.data_frame, var_names, self.data_measlevs,
                                                     statistics=['mean', 'std', 'max', 'upper quartile',
                                                                 'median', 'lower quartile', 'min'])
        elif meas_level == 'ord':
            sample_result += cs_stat.print_var_stats(self.data_frame, var_names, self.data_measlevs,
                                                     statistics=['max', 'upper quartile', 'median',
                                                                 'lower quartile', 'min'])
        elif meas_level == 'nom':
            sample_result += cs_stat.print_var_stats(self.data_frame, var_names, self.data_measlevs,
                                                     statistics=['variation ratio'])
            import itertools
            for var_pair in itertools.combinations(var_names, 2):
                sample_result += cs_stat.contingency_table(self.data_frame, [var_pair[1]], [var_pair[0]],
                                                           count=True, percent=True, margins=True)
            sample_result += '\n'

        # 2b. Effect size
        effect_size_result = cs_stat.repeated_measures_effect_size(self.data_frame, var_names, factors,
                                                                   meas_level, sample=True)
        if effect_size_result:
            sample_result += '\n\n' + effect_size_result

        # 3. Population properties
        population_result = '<cs_h2>' + _('Population properties') + '</cs_h2>'

        # 3a. Population estimations
        population_result += '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>\n'
        if meas_level in ['int', 'unk']:
            population_result += _('Means') + '\n' + _('Present confidence interval values suppose normality.')
            mean_estimations = cs_stat.repeated_measures_estimations(data, meas_level)
            prec = cs_util.precision(self.data_frame[var_names[0]]) + 1
            population_result += \
                cs_stat._format_html_table(mean_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                    float_format=lambda x: '%0.*f' % (prec, x)))
        elif meas_level == 'ord':
            population_result += _('Median')
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

        # 3b. Effect size
        effect_size_result = cs_stat.repeated_measures_effect_size(self.data_frame, var_names, factors, meas_level,
                                                                   sample=False)
        if effect_size_result:
            population_result += '\n' + effect_size_result

        # 3c. Hypothesis tests
        result_ht = cs_hyp_test.decision_repeated_measures(self.data_frame, meas_level, factors, var_names, data,
                                                           self.data_measlevs)

        return cs_util.convert_output([title, raw_result, raw_graph, sample_result, sample_graph, population_result,
                                       population_graph, result_ht])

    def compare_groups(self, var_name, grouping_variables,  single_case_slope_SE=None, single_case_slope_trial_n=None,
                       ylims=[None, None]):
        """
        Compare groups.

        Parameters
        ----------
        var_name : str
            Name of the dependent variable
        grouping_variables : list of str
            List of name(s) of grouping variable(s).
        single_case_slope_SE : str
            When comparing the slope between a single case and a group, variable name storing the slope SEs
        single_case_slope_trial : int
            When comparing the slope between a single case and a group, number of trials.
        ylims : list of {int or float}
            Limit of the y axis for interval and ordinal variables instead of using automatic values.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """
        plt.close('all')
        var_names = [var_name]
        groups = grouping_variables
        # TODO check if there is only one dep.var.
        title = '<cs_h1>' + _('Compare groups') + '</cs_h1>'
        meas_levels = [self.data_measlevs[var_name] for var_name in var_names]
        group_meas_levels = [self.data_measlevs[group] for group in groups]
        raw_result = _('Dependent variable: ') + ', '.join('%s (%s)' % (var, meas) for var, meas in
                                                           zip(var_names, meas_levels)) + '. ' + _('Group(s): ') + \
                     ', '.join('%s (%s)' % (var, meas) for var, meas in zip(groups, group_meas_levels)) + '\n'
        raw_result += self._filtering_status()

        # level of measurement of the variables
        meas_level, unknown_type = self._meas_lev_vars([var_names[0]])
        if unknown_type:
            raw_result += '<decision>'+warn_unknown_variable+'</decision>'

        # 1. Raw data
        raw_result += '<cs_h2>' + _('Raw data') + '</cs_h2>'

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
        #                raw_result += _(u'Group: %s, N of valid cases: %g, N of missing cases: %g\n') %
        #                              (group, valid_n, missing_n)
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
        sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'

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

        # Effect size
        sample_effect_size = cs_stat.compare_groups_effect_size(self.data_frame, var_names, groups, meas_level,
                                                                sample=True)
        if sample_effect_size:
            sample_result += '\n\n' + sample_effect_size

        # 3. Population properties
        # Plot population estimations
        group_estimations = cs_stat.comp_group_estimations(self.data_frame, meas_level, var_names, groups)
        population_graph = cs_chart.create_compare_groups_population_chart(self.data_frame, meas_level, var_names,
                                                                           groups, level_combinations, ylims=ylims)

        # Population estimation
        population_result = '<cs_h2>' + _('Population properties') + '</cs_h2>' + \
                            '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>\n'
        if meas_level in ['int', 'unk']:
            population_result += _('Means') + '\n' + _('Present confidence interval values suppose normality.')
        elif meas_level == 'ord':
            population_result += _('Medians')
        if meas_level in ['int', 'unk', 'ord']:
            prec = cs_util.precision(self.data_frame[var_names[0]]) + 1
            population_result += \
                cs_stat._format_html_table(group_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                     float_format=lambda x: '%0.*f' % (prec, x)))
            population_result += '\n'
        if meas_level == 'nom':
            population_result += '\n' + cs_stat.contingency_table(self.data_frame, groups, var_names, ci=True) + '\n'

        # effect size
        standardized_effect_size_result = cs_stat.compare_groups_effect_size(self.data_frame, var_names, groups,
                                                                             meas_level, sample=False)

        # Hypothesis testing
        if len(groups) == 1:
            group_levels = sorted(set(data[groups[0]]))
            result_ht = cs_hyp_test.decision_one_grouping_variable(self.data_frame, meas_level, self.data_measlevs,
                                                                   var_names, groups, group_levels,
                                                                   single_case_slope_SE, single_case_slope_trial_n)
        else:
            result_ht = cs_hyp_test.decision_several_grouping_variables(self.data_frame, meas_level, var_names, groups)

        return cs_util.convert_output([title, raw_result, raw_graph, sample_result, sample_graph, population_result,
                                       standardized_effect_size_result, population_graph, result_ht])


def display(results):
    """
    Display list of output given by CogStat analysis in IPython Notebook.

    Parameters
    ----------
    results : list of {str, image}
        HTML results.
    """
    from IPython.display import display
    from IPython.display import HTML
    for result in results:
        if isinstance(result, str):
            display(HTML(result))
        else:
            display(result)
    plt.close('all')

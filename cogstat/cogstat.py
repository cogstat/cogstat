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

__version__ = '2.4dev'

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

warn_unknown_variable = '<warning><b>' + _('Measurement level warning') + '</b> ' + \
                        _('The measurement levels of the variables are not set. Set them in your data source.') \
                        + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                        % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                        + '</warning>'
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
        """Initialize the cogstat data object.

        In the input data:
        - First line should be the variable name
        --- If there are missing names, Unnamed:0, Unnamed:1, etc. names are given
        --- If there are repeating var names, new available numbers are added, e.g. a.1, a.2, etc.
        - Second line could be the measuring level

        Data structure that is created:
        self.data_frame - pandas DataFrame
        self.data_measlevs - dictionary storing level of measurement of the variables (name:level):
                'nom', 'ord', or 'int' (ratio is included in 'int')
                'unk' - unknown: if no other level is given
        self.orig_data_frame # TODO
        self.filtering_status # TODO

        self.import_source - list of 2 strings:
                             [0]: import data type
                             [1]: path to the data file or '' if the data source is not a file
        self.import_message - text output of the imported process
                              can't return anything to caller, since we're in an __init__ method, so store the message
                              here

        Parameters
        ----------
        See the class docstring.

        """

        self.orig_data_frame = None
        self.data_frame = None
        self.data_measlevs = None
        self.import_source = ['', '']
        self.import_message = ''
        self.filtering_status = None

        self._import_data(data=data, measurement_levels=measurement_levels)

    ### Import and handle the data ###

    def _import_data(self, data='', measurement_levels=None, show_heading=True):
        """Import the data to initialize the object.

        See __init__ for more information

        Parameters
        ----------
        See the class docstring
        show_heading : bool
            Should we show a heading?

        Returns
        -------
        It creates the data related properties in place. See __init__ for more information.
        """

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
                    self.data_frame[column][selected_cells] = \
                        self.data_frame[column][selected_cells == True].str.replace('%', '').astype('float') / 100.0
                    try:
                        self.data_frame[column] = self.data_frame[column].astype('float')
                    except ValueError:  # there may be other non xx% strings in the variable
                        pass

        def _special_values_to_nan():
            """Some additional values are converted to NaNs."""
            self.data_frame.replace('', np.nan, inplace=True)
            self.data_frame.replace(r'^#.*!$', np.nan, regex=True, inplace=True)
                # spreadsheet errors, such as #DIV/0!, #VALUE!
            # spreadsheet errors make the variable object dtype, although they may be numeric variables
            try:
                self.data_frame[self.data_frame.select_dtypes(include=['object']).columns] = \
                    self.data_frame.select_dtypes(include=['object']).astype(float)
            except (ValueError, TypeError):
                pass

        def _convert_dtypes():
            """Convert dtypes.

            1. CogStat does not know boolean variables, so they are converted to strings.
              This solution changes upper and lower cases: independent of the text, it will be 'True' and 'False'
            2. Some analyses do not handle Int types, but int types
            3. Some analyses do not handle category types

            Returns
            -------
            Changes self.data_frame
            """
            convert_dtypes = [['bool', 'object'],  # although 'string' type is recommended, patsy cannot handle it
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

        def _all_object_data_to_strings():
            """Some object dtype data may include both strings and numbers. This may cause issues in later analyses.
            So we convert all items to string in an object dtype variable."""
            self.data_frame[self.data_frame.select_dtypes(include=['object']).columns] = \
                self.data_frame.select_dtypes(include=['object']).astype('str').astype('object')
                # Finally, we convert back to object because string type may cause issues e.g., for patsy.
            self.data_frame.replace('nan', np.nan, inplace=True)
                # string conversion turns np.nan to 'nan', so we turn it back;
                # a former solution was the skipna=True parameter in the astype() method

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

            Returns
            -------
            It creates self.data_measlevs in place.
            """

            nonlocal warning_text

            # By default, all variables have 'unknown' measurement levels
            self.data_measlevs = {name: 'unk' for name in self.data_frame.columns}

            # 1. Set the levels (coming either from the import data information or from measurement_levels object
            # parameter)
            if measurement_levels:  # If levels were given, set them

                # Only levels are given - in the order of the variables
                if type(measurement_levels) is list:
                    # Check if only valid measurement levels were given
                    if not (set(measurement_levels) <= {'unk', 'nom', 'ord', 'int', '', 'nan'}):
                        raise ValueError('Invalid measurement level')
                    # make levels lowercase and replace '' or 'nan' with 'unk'
                    measurement_levels = ['unk' if level.lower() in ['', 'nan'] else level.lower()
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
                    warning_text += '\n<warning>' + \
                                    _('Number of measurement levels do not match the number of variables. '
                                    'You may want to correct the number of measurement levels.')

            # 2. Apply constraints to measurement levels.
            # String variables cannot be interval or nominal variables in CogStat, so change them to nominal
            invalid_var_names = [var_name for var_name in self.data_frame.columns if
                                 (self.data_measlevs[var_name] in ['int', 'ord', 'unk'] and
                                  str(self.data_frame[var_name].dtype) in ['object', 'string'])]
                # 'object' dtype means string variable
            if invalid_var_names:  # these str variables were set to int or ord
                for var_name in invalid_var_names:
                    self.data_measlevs[var_name] = 'nom'
                warning_text += '\n<warning><b>' + _('String variable conversion warning') + '</b> ' + \
                                _('String variables cannot be interval or ordinal variables in CogStat. '
                                'Those variables are automatically set to nominal: ')\
                                + '<i>' + ', '.join('%s' % var_name for var_name in invalid_var_names) + \
                                '</i>. ' + _('You can fix this issue in your data source.') \
                                + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                + '</warning>'

            # Warn when any measurement levels are not set
            if 'unk' in set(self.data_measlevs.values()):
                warning_text += '\n<warning><b>' + _('Measurement level warning') + '</b> ' + \
                                       _('The measurement level was not set for all variables.') + ' '\
                                       + _('You can fix this issue in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</warning>'
        # end of set_measurement_level()

        def _check_valid_chars():
            # Check if only valid chars are used in the data, and warn the user if invalid chars are used
            # TODO this might be removed with Python3 and with unicode encoding

            nonlocal warning_text

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
                                break  #after finding the first non-ascii data, we can skip the rest variable data
            if non_ascii_var_names:
                warning_text += '\n<warning><b>' + _('Recommended characters in variable names warning') + \
                                       '</b> ' + \
                                       _('Some variable name(s) include characters other than English letters, '
                                         'numbers, or underscore which can cause problems in some analyses: %s.') \
                                       % ('<i>' + ', '.join(
                    '%s' % non_ascii_var_name for non_ascii_var_name in non_ascii_var_names) + '</i>')\
                                       + ' ' + _('If some analyses cannot be run, fix this in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</warning>'
            if non_ascii_vars:
                warning_text += '\n<warning><b>' + _('Recommended characters in data values warning') + \
                                       '</b> ' + \
                                       _('Some string variable(s) include characters other than English letters, '
                                         'numbers, or underscore which can cause problems in some analyses: %s.') \
                                       % ('<i>' + ', '.join('%s' % non_ascii_var for non_ascii_var in non_ascii_vars) +
                                          '</i>')\
                                       + ' ' + _('If some analyses cannot be run, fix this in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</warning>'

        self.import_message = ''
        import_measurement_levels = None
        warning_text = ''

        # I. Import the DataFrame/file/clipboard

        # 1. Import from pandas DataFrame
        if isinstance(data, pd.DataFrame):
            self.data_frame = data
            self.import_source[0] = _('Pandas dataframe')

        # 2. Import from file
        elif isinstance(data, str) and not ('\n' in data):  # Single line text, i.e., filename
            # Check if the file exists # TODO
            # self.import_source[0] = _('Import failed')
            # self.import_message += '<cs_h1>' + _('Data') + '</cs_h1>' + _('Import failed. File does not exist.')
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
                self.import_source = [_('Text file'), data]  # filename

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
                self.import_source = [_('Spreadsheet file'), data]  # filename

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

                self.import_source = [_('SPSS/SAS/STATA file'), data]  # filename

            # Import from R files
            elif filetype.lower() in ['.rdata', '.rds', '.rda']:
                import pyreadr
                import_data = pyreadr.read_r(data)
                self.data_frame = import_data[list(import_data.keys())[0]]
                self.data_frame= self.data_frame.convert_dtypes()
                self.import_source = [_('R file'), data]  # filename

            # Import JASP files
            elif filetype == '.jasp':
                from . import cogstat_stat_num as cs_stat_num
                import_pdf, import_measurement_levels = cs_stat_num.read_jasp_file(data)
                self.data_frame = import_pdf.convert_dtypes()
                self.import_source = [_('JASP file'), data]  # filename

            # Import jamovi files
            elif filetype == '.omv':
                from . import cogstat_stat_num as cs_stat_num
                import_pdf, import_measurement_levels = cs_stat_num.read_jamovi_file(data)
                self.data_frame = import_pdf.convert_dtypes()
                self.import_source = [_('jamovi file'), data]  # filename

            # File type is not supported
            else:
                self.import_source[0] = _('Import failed')
                self.import_message += '<cs_h1>' + _('Data') + '</cs_h1>' + \
                                       _('Import failed. File type is not supported.')
                return

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
            self.import_source[0] = _('Clipboard')

        # 4. Invalid data source
        else:
            self.import_source[0] = _('Import failed')
            self.import_message += '<cs_h1>' + _('Data') + '</cs_h1>' + _('Import failed. Invalid data source.')
            return

        # II. Set additional details for all import sources

        # Convert some values and data types
        self.data_frame.columns = self.data_frame.columns.astype('str')  # variable names can only be strings
        _percent2float()
        _special_values_to_nan()
        _convert_dtypes()
        _all_object_data_to_strings()

        # Set and check data properties
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

        self.import_message += self.print_data(show_heading=show_heading, brief=True)[0]
        self.import_message += cs_util.convert_output([warning_text])[0]

    def reload_data(self):
        """Reload actual data from the path it has been read previously.

        Returns
        -------
        list of a single str
            Report in HTML format
        """

        output = '<cs_h1>' + _('Reload actual data file') + '</cs_h1>'

        if self.import_source[1]:  # if the actual dataset was imported from a file, then reload it
            self._import_data(data=self.import_source[1], show_heading=False)  # measurement level should be reimported too
            output += _('The file was successfully reloaded.') + '\n'
            output += cs_util.reformat_output(self.import_message)
        else:
            output += _('The data was not imported from a file. It cannot be reloaded.') + '\n'
            # or do we assume that this method is not called when the actual file was not imported from a file?

        return cs_util.convert_output([output])

    def print_data(self, show_heading=True, brief=False):
        """
        Display the data.

        Parameters
        ----------
        show_heading : bool
            Add heading to the output string?
        brief : bool
            Should only the first few cases or the whole data frame be displayed?

        Returns
        -------
        str
            HTML string showing the data.
        """
        output = ''
        if show_heading:
            output += '<cs_h1>' + _('Data') + '</cs_h1>'
        output += _('Source: ') + self.import_source[0] + (self.import_source[1] if self.import_source[1] else '')\
                  + '\n'
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

    def filter_outlier(self, var_names=None, mode='mahalanobis'):
        """
        Filter self.data_frame based on outliers.

        All variables are investigated independently and cases are excluded if any variables shows they are outliers.
        If mode is 'mahalanobis', then variables are jointly investigated for multivariate outliers.
        If var_names is None, then all cases are used (i.e., filtering is switched off).

        Parameters
        ----------
        var_names : None or list of str
            Names of the variables the exclusion is based on or None to include all cases.
        mode : {'2.5mad', '2sd', 'mahalanobis'}
            Mode of the exclusion:
                2.5mad: median +- 2.5 * MAD
                2sd: mean +- 2 * SD
                mahalanobis: MCCD Mahalanobis distance with .05 chi squared cut-off
            CogStat uses only a single method (MAD), but for possible future code change, the previous (2sd) and
            multivariate (mahalanobis) version is also included.

        Returns
        -------
        list of str
            List of HTML strings showing the filtered cases.
            The method modifies the self.data_frame in place.
        list of charts
            If cases were filtered, then filtered and remaining cases are shown.
        """
        mode_names = {'2sd': _('Mean ± 2 SD'),  # Used in the output
                      '2.5mad': _('Median ± 2.5 MAD'),
                      'mahalanobis': _('MCCD Mahalanobis distance with .05 chi squared cut-off')}

        title = '<cs_h1>' + _('Filter outliers') + '</cs_h1>'

        chart_results = []

        if var_names is None or var_names == []:  # Switch off outlier filtering
            self.data_frame = self.orig_data_frame.copy()
            self.filtering_status = None
            text_output = _('Filtering is switched off.')
        else:  # Create a filtered dataframe based on the variable(s)
            remaining_cases_indexes = []
            text_output = ''
            self.filtering_status = ''
            if mode == 'mahalanobis':
                # Based on the robust mahalanobis distance in Leys et al, 2017 and Rousseeuw, 1999
                ignored_variables = []
                for var_name in var_names:
                    if self.data_measlevs[var_name] in ['ord', 'nom']:
                        var_names.remove(var_name)
                        ignored_variables.append(var_name)
                text_output += _('Only interval variables can be used for filtering. Ignoring variable(s) %s.') % \
                               ignored_variables + '\n'

                # Calculating the robust mahalanobis distances
                from sklearn import covariance
                cov = covariance.EllipticEnvelope(contamination=0.25).fit(self.data_frame[var_names])

                # Custom filtering criteria based on Leys et al. (2017)
                limit = np.sqrt(
                    stats.chi2.ppf(0.95, len(self.data_frame[var_names].columns)))  # Appropriate cut-off point based on chi2
                distances = cov.mahalanobis(self.data_frame[var_names])  # Get robust mahalanobis distances from model object
                filtering_data_frame = self.orig_data_frame.copy()
                filtering_data_frame['mahalanobis'] = distances

                # Find the cases to be kept
                remaining_cases_indexes.append(filtering_data_frame[
                                                   (filtering_data_frame['mahalanobis'] < limit)].index)

                # Display filtering information
                text_output += _('Filtering based on the variables: %s.\n') % (var_names)
                prec = cs_util.precision(filtering_data_frame['mahalanobis']) + 1
                text_output += _('Cases above the cutoff mahalanobis distance will be excluded:') + \
                               ' %0.*f\n' % (prec, limit)

                # Display the excluded cases
                excluded_cases = \
                    self.orig_data_frame.drop(remaining_cases_indexes[-1])
                # excluded_cases.index = [' '] * len(excluded_cases)  # TODO can we cut the indexes from the html table?
                # TODO uncomment the above line after using pivot indexes in CS data
                if len(excluded_cases):
                    text_output += _('The following cases will be excluded: ')
                    text_output += cs_stat._format_html_table(excluded_cases.to_html(bold_rows=False,
                                                                                     classes="table_cs_pd"))
                    for var_name in var_names:
                        chart_results.append(cs_chart.create_filtered_cases_chart(
                            self.orig_data_frame.loc[remaining_cases_indexes[-1]][var_name],
                            excluded_cases[var_name], var_name,
                            lower_limit=None, upper_limit=None))

                else:
                    text_output += _('No cases were excluded.')


            else:
                for var_name in var_names:
                    if self.data_measlevs[var_name] in ['ord', 'nom']:
                        text_output += _('Only interval variables can be used for filtering. Ignoring variable %s.') % \
                                       var_name + '\n'
                        continue
                    # Find the lower and upper limit
                    if mode == '2sd':
                        mean = np.mean(self.orig_data_frame[var_name].dropna())
                        sd = np.std(self.orig_data_frame[var_name].dropna(), ddof=1)
                        lower_limit = mean - 2 * sd
                        upper_limit = mean + 2 * sd
                    elif mode == '2.5mad':
                        # Python implementations:
                        # https://www.statsmodels.org/stable/generated/statsmodels.robust.scale.mad.html
                        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_absolute_deviation.html
                        from statsmodels.robust.scale import mad as mad_function
                        median = np.median(self.orig_data_frame[var_name].dropna())
                        mad_value = mad_function(self.orig_data_frame[var_name].dropna())
                        lower_limit = median - 2.5 * mad_value
                        upper_limit = median + 2.5 * mad_value
                    else:
                        raise ValueError('Invalid mode parameter was given')

                    # Display filtering information
                    text_output += _('Filtering based on %s.\n') % (var_name + ' (%s)' % mode_names[mode])
                    prec = cs_util.precision(self.orig_data_frame[var_name]) + 1
                    text_output += _('Cases outside of the range will be excluded:') + \
                                   ' %0.*f  –  %0.*f\n' % (prec, lower_limit, prec, upper_limit)
                    # Find the cases to be kept
                    remaining_cases_indexes.append(self.orig_data_frame[
                                                     (self.orig_data_frame[var_name] > lower_limit) &
                                                     (self.orig_data_frame[var_name] < upper_limit)].index)

                    # Display the excluded cases
                    excluded_cases = \
                        self.orig_data_frame.drop(remaining_cases_indexes[-1])
                    #excluded_cases.index = [' '] * len(excluded_cases)  # TODO can we cut the indexes from the html table?
                    # TODO uncomment the above line after using pivot indexes in CS data
                    if len(excluded_cases):
                        text_output += _('The following cases will be excluded: ')
                        text_output += cs_stat._format_html_table(excluded_cases.to_html(bold_rows=False,
                                                                                         classes="table_cs_pd"))
                        if mode != 'mahalanobis':
                            chart_results.append(cs_chart.create_filtered_cases_chart(self.orig_data_frame.loc[remaining_cases_indexes[-1]][var_name],
                                                                                      excluded_cases[var_name], var_name,
                                                                                      lower_limit, upper_limit))
                        elif mode == 'mahalanobis':
                            chart_results.append(cs_chart.create_filtered_cases_chart(self.orig_data_frame.loc[remaining_cases_indexes[-1]][var_name],
                                                                                      excluded_cases[var_name], var_name,
                                                                                      lower_limit, upper_limit))

                    else:
                        text_output += _('No cases were excluded.')
                    if var_name != var_names[-1]:
                        text_output += '\n\n'

            # Do the filtering (remove outliers), modify self.data_frame in place
            self.data_frame = self.orig_data_frame.copy()
            for remaining_cases_index in remaining_cases_indexes:
                self.data_frame = self.data_frame.loc[self.data_frame.index.intersection(remaining_cases_index)]
            self.filtering_status = ', '.join(var_names) + ' (%s)' % mode_names[mode]

        return cs_util.convert_output([title, text_output, chart_results])

    def _filtering_status(self):
        if self.filtering_status:
            return '<b>' + _('Filtering is on:') + ' %s</b>\n' % self.filtering_status
        else:
            return ''

    ### Various things ###

    def _meas_lev_vars(self, variables):
        """
        For the set of variables, find the lowest measurement level and if there is unknown variable.

        Parameters
        ----------
        variables : list of str
            Variable names

        Returns:
        {'nom', 'ord', 'int'}
            the lowest measurement level among the listed variables
        bool
            does the list of variables include at least one unknown variable?
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
            unknown_meas_lev = True
        else:
            unknown_meas_lev = False

        return meas_lev, unknown_meas_lev

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

        data = pd.DataFrame(self.data_frame[var_name].dropna())

        text_result2 = _('N of valid cases: %g') % len(data) + '\n'
        missing_cases = len(self.data_frame[var_name])-len(data)
        text_result2 += _('N of missing cases: %g') % missing_cases + '\n'

        image = cs_chart.create_variable_raw_chart(data, self.data_measlevs, var_name)

        result_list.append(text_result+text_result2)
        result_list.append(image)

        # 2. Sample properties
        text_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'

        # Frequencies
        if frequencies:
            text_result += '<cs_h3>'+_('Frequencies')+'</cs_h3>'
            text_result += cs_stat.frequencies(data, var_name, meas_level) + '\n\n'

        # Descriptives
        if self.data_measlevs[var_name] in ['int', 'unk']:
            text_result += cs_stat.print_var_stats(data, [var_name], self.data_measlevs,
                                                   statistics=['mean', 'std', 'skewness', 'kurtosis', 'range', 'max',
                                                               'upper quartile', 'median', 'lower quartile', 'min'])
        elif self.data_measlevs[var_name] == 'ord':
            text_result += cs_stat.print_var_stats(data, [var_name], self.data_measlevs,
                                                   statistics=['max', 'upper quartile', 'median', 'lower quartile',
                                                               'min'])
            # TODO boxplot also
        elif self.data_measlevs[var_name] == 'nom':
            text_result += cs_stat.print_var_stats(data, [var_name], self.data_measlevs,
                                                   statistics=['variation ratio'])
        result_list.append(text_result)

        # Distribution
        if self.data_measlevs[var_name] != 'nom':  # histogram for nominal variable has already been shown in raw data
            image = cs_chart.create_histogram_chart(data, self.data_measlevs, var_name)
            result_list.append(image)

        # 3. Population properties
        text_result = '<cs_h2>' + _('Population properties') + '</cs_h2>'

        # Normality
        if meas_level in ['int', 'unk']:
            text_result += '<cs_h3>'+_('Normality')+'</cs_h3>\n'
            stat_result, text_result2 = cs_hyp_test.normality_test(data, self.data_measlevs, var_name)
            image = cs_chart.create_normality_chart(data, var_name)
                # histogram with normality and qq plot
            text_result += text_result2
            result_list.append(text_result)
            if image:
                result_list.append(image)
        else:
            result_list.append(text_result)

        # Population estimations
        if meas_level in ['int', 'ord', 'unk']:
            prec = cs_util.precision(data[var_name]) + 1

        population_param_text = '\n<cs_h3>' + _('Population parameter estimations') + '</cs_h3>\n'
        if meas_level in ['int', 'unk']:
            population_param_text += cs_stat.variable_estimation(data[var_name], ['mean', 'std'])
        elif meas_level == 'ord':
            population_param_text += cs_stat.variable_estimation(data[var_name], ['median'])
        elif meas_level == 'nom':
            population_param_text += cs_stat.proportions_ci(data, var_name)
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
            norm, text_result_norm = cs_hyp_test.normality_test(data, self.data_measlevs, var_name)

            text_result += text_result_norm
            if norm:
                text_result += '<decision>' + _('Normality is not violated.') + ' >> ' + \
                               _('Running one-sample t-test.') + '</decision>\n'
                text_result2, ci = cs_hyp_test.one_t_test(data, self.data_measlevs, var_name,
                                                          test_value=central_value)
                graph = cs_chart.create_variable_population_chart(data[var_name], var_name, 'mean', ci)

            else:
                text_result += '<decision>' + _('Normality is violated.') + ' >> ' + \
                               _('Running Wilcoxon signed-rank test.') + '</decision>\n'
                text_result += _('Median: %0.*f') % (prec, np.median(data[var_name])) + '\n'
                text_result2 = cs_hyp_test.wilcox_sign_test(data, self.data_measlevs, var_name,
                                                            value=central_value)
                graph = cs_chart.create_variable_population_chart(data[var_name], var_name, 'median')

        elif meas_level == 'ord':
            text_result += '<decision>' + _('Ordinal variable.') + ' >> ' + _('Running Wilcoxon signed-rank test.') + \
                           '</decision>\n'
            text_result2 = cs_hyp_test.wilcox_sign_test(data, self.data_measlevs, var_name,
                                                        value=central_value)
            graph = cs_chart.create_variable_population_chart(data[var_name], var_name, 'median')
        else:
            text_result2 = '<decision>' + _('Sorry, not implemented yet.') + '</decision>\n'
            graph = None
        text_result += text_result2

        result_list.append(population_param_text)
        if graph:
            result_list.append(graph)
        result_list.append(text_result)
        return cs_util.convert_output(result_list)

    def regression(self, predictors, predicted, xlims=[None, None], ylims=[None, None]):
        """
        Explore a variable pair.

        Parameters
        ----------
        predictors : list of str
            Name of the predictor variables.
        predicted : str
            Name of the predicted variable.
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

        meas_lev, unknown_var = self._meas_lev_vars(predictors + [predicted])

        # TODO rebuild this whole method to handle various scenarios with a compact code
        if len(predictors) == 1:
            x = predictors[0]
            y = predicted

            # Analysis output header
            # TODO regression - new title
            title = '<cs_h1>' + _('Explore relation of variable pair') + '</cs_h1>'

            # 0. Analysis information
            # TODO regression - new information
            raw_result = _('Exploring variable pair: ') + x + ' (%s), ' % self.data_measlevs[x] + y + \
                         ' (%s)\n' % self.data_measlevs[y]
            raw_result += self._filtering_status()
            if unknown_var:
                raw_result += '<decision>' + warn_unknown_variable + '\n</decision>'

            # 1. Raw data
            raw_result += '<cs_h2>' + _('Raw data') + '</cs_h2>'
            # Prepare data, drop missing data
            # TODO are NaNs interesting in nominal variables?
            # TODO regression - N of valid cases?
            data = self.data_frame[[x, y]].dropna()
            valid_n = len(data)
            missing_n = len(self.data_frame[[x, y]]) - valid_n
            raw_result += _('N of valid pairs') + ': %g' % valid_n + '\n'
            raw_result += _('N of missing pairs') + ': %g' % missing_n + '\n'

            # Raw data chart
            # TODO regression - extend the function to handle multiple regressors
            raw_graph = cs_chart.create_variable_pair_chart(data, meas_lev, x, y, raw_data=True,
                                                            regression=False, CI=False, xlims=xlims, ylims=ylims)

            # 2. Sample properties
            sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'
            residual_title = None
            residual_graph = None
            normality = None  # Do the two variables follow a multivariate normal distribution?
            homoscedasticity = None
            assumptions_result = None
            if meas_lev == 'nom':
                sample_result += cs_stat.contingency_table(data, [x], [y], count=True, percent=True, margins=True)
            elif meas_lev == 'int':

                # Test of multivariate normality
                # TODO regression - several variables
                assumptions_result = '\n' + '<cs_h3>' + _('Checking assumptions of inferential methods') + '</cs_h3>\n'
                assumptions_result += '<decision>' + _('Testing multivariate normality of variables') + '</decision>\n'
                normality, norm_text = cs_hyp_test.multivariate_normality(data, [x, y])
                assumptions_result += norm_text

                # Calculate regression with statsmodels
                # TODO regression - several regressors
                import statsmodels.regression
                import statsmodels.tools

                data_sorted = data.sort_values(by=x)  # Sorting required for subsequent plots to work
                x_var = statsmodels.tools.add_constant(data_sorted[x])
                y_var = data_sorted[y]
                model = statsmodels.regression.linear_model.OLS(y_var, x_var)
                result = model.fit()
                residuals = result.resid

                # Test of homoscedasticity
                # TODO regression
                assumptions_result += '<decision>' + _('Testing homoscedasticity') + '</decision>\n'
                homoscedasticity, het_text = cs_hyp_test.homoscedasticity(data, [x, y],
                                                                          residual=residuals)
                assumptions_result += het_text

                # TODO output with the precision of the data
                sample_result += _('Linear regression')+': y = %0.3fx + %0.3f' % (result.params[1], result.params[0])
            sample_result += '\n'

            # TODO regression
            standardized_effect_size_result = cs_stat.variable_pair_standard_effect_size(data, meas_lev, sample=True,
                                                                                         normality=normality,
                                                                                         homoscedasticity=homoscedasticity)
            standardized_effect_size_result += '\n'

            # Make graphs
            # extra chart is needed only for int variables, otherwise the chart would just repeat the raw data
            if meas_lev == 'int':

                # Residual analysis
                # TODO regression
                residual_title = '<cs_h3>' + _('Residual analysis') + '</cs_h3>\n'
                residual_graph = cs_chart.create_residual_chart(data, meas_lev, x, y)

                # Sample scatter plot with regression line
                # TODO regression
                sample_graph = cs_chart.create_variable_pair_chart(data, meas_lev, x, y, result=result, raw_data=True,
                                                                   regression=True, CI=False, xlims=xlims, ylims=ylims)

            else:
                sample_graph = None

            # 3. Population properties
            population_properties_title = '<cs_h2>' + _('Population properties') + '</cs_h2>'
            estimation_result = '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>\n'
            estimation_parameters, estimation_effect_size, population_graph = None, None, None

            # TODO regression
            if meas_lev == 'nom':
                estimation_result += cs_stat.contingency_table(data, [x], [y], ci=True)
            if meas_lev =='int':
                estimation_parameters = cs_stat.variable_pair_regression_coefficients(result.params[1], result.params[0],
                                                                                      result.bse[1],result.bse[0],
                                                                                      meas_lev, len(data[x]),
                                                                                      normality=normality,
                                                                                      homoscedasticity=homoscedasticity)
                population_graph = cs_chart.create_variable_pair_chart(data, meas_lev, x, y, result=result, raw_data=False,
                                                                       regression=True, CI=True,
                                                                       xlims=[None, None], ylims=[None, None])
            estimation_effect_size = cs_stat.variable_pair_standard_effect_size(data, meas_lev, sample=False,
                                                                                normality=normality,
                                                                                homoscedasticity=homoscedasticity)

            population_result = '\n' + cs_hyp_test.variable_pair_hyp_test(data, x, y, meas_lev, normality,
                                                                          homoscedasticity) + '\n'

            return cs_util.convert_output([title, raw_result, raw_graph, sample_result, sample_graph,
                                           standardized_effect_size_result, residual_title, residual_graph,
                                           population_properties_title, assumptions_result, estimation_result,
                                           estimation_parameters, population_graph, estimation_effect_size,
                                           population_result])
        else:  # several predictors
            return cs_util.convert_output(['<cs_h1>' + _('Explore relation of variable pair') + '</cs_h1>\n' +
                                          _('Sorry, not implemented yet.')])


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
            raw_result += '\n<decision>' + warn_unknown_variable + '</decision>'

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
        raw_graph = cs_chart.create_repeated_measures_sample_chart(data, var_names, meas_level, raw_data_only=True,
                                                                   ylims=ylims)

        # Plot the individual data with box plot
        # There's no need to repeat the mosaic plot for nominal variables
        if meas_level in ['int', 'unk', 'ord']:
            sample_graph = cs_chart.create_repeated_measures_sample_chart(data, var_names, meas_level, ylims=ylims)
        else:
            sample_graph = None

        # 2. Sample properties
        sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'

        if meas_level in ['int', 'unk']:
            sample_result += cs_stat.print_var_stats(data, var_names, self.data_measlevs,
                                                     statistics=['mean', 'std', 'max', 'upper quartile',
                                                                 'median', 'lower quartile', 'min'])
        elif meas_level == 'ord':
            sample_result += cs_stat.print_var_stats(data, var_names, self.data_measlevs,
                                                     statistics=['max', 'upper quartile', 'median',
                                                                 'lower quartile', 'min'])
        elif meas_level == 'nom':
            sample_result += cs_stat.print_var_stats(data, var_names, self.data_measlevs,
                                                     statistics=['variation ratio'])
            import itertools
            for var_pair in itertools.combinations(var_names, 2):
                sample_result += cs_stat.contingency_table(data, [var_pair[1]], [var_pair[0]], count=True,
                                                           percent=True, margins=True)
            sample_result += '\n'

        # 2b. Effect size
        effect_size_result = cs_stat.repeated_measures_effect_size(data, var_names, factors, meas_level, sample=True)
        if effect_size_result:
            sample_result += '\n\n' + effect_size_result

        # 3. Population properties
        population_result = '<cs_h2>' + _('Population properties') + '</cs_h2>'

        # 3a. Population estimations
        population_result += '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>\n'
        if meas_level in ['int', 'unk']:
            population_result += _('Means') + '\n' + _('Present confidence interval values suppose normality.')
            mean_estimations = cs_stat.repeated_measures_estimations(data, meas_level)
            prec = cs_util.precision(data[var_names[0]]) + 1
            population_result += \
                cs_stat._format_html_table(mean_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                    float_format=lambda x: '%0.*f' % (prec, x)))
        elif meas_level == 'ord':
            population_result += _('Median')
            median_estimations = cs_stat.repeated_measures_estimations(data, meas_level)
            prec = cs_util.precision(data[var_names[0]]) + 1
            population_result += \
                cs_stat._format_html_table(median_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                      float_format=lambda x: '%0.*f' % (prec, x)))
        elif meas_level == 'nom':
            for var_pair in itertools.combinations(var_names, 2):
                population_result += cs_stat.contingency_table(data, [var_pair[1]], [var_pair[0]], ci=True)
        population_result += '\n'

        population_graph = cs_chart.create_repeated_measures_population_chart(data, var_names, meas_level, ylims=ylims)

        # 3b. Effect size
        effect_size_result = cs_stat.repeated_measures_effect_size(data, var_names, factors, meas_level, sample=False)
        if effect_size_result:
            population_result += '\n' + effect_size_result

        # 3c. Hypothesis tests
        result_ht = cs_hyp_test.decision_repeated_measures(data, meas_level, factors, var_names, self.data_measlevs)

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
            raw_result += '<decision>' + warn_unknown_variable + '</decision>'

        # 1. Raw data
        raw_result += '<cs_h2>' + _('Raw data') + '</cs_h2>'

        standardized_effect_size_result = None

        data = self.data_frame[groups + [var_names[0]]].dropna()
        if single_case_slope_SE:
            data = self.data_frame[groups + [var_names[0], single_case_slope_SE]].dropna()
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

        raw_graph = cs_chart.create_compare_groups_sample_chart(data, meas_level, var_names, groups,
                                                                level_combinations, raw_data_only=True, ylims=ylims)

        # 2. Sample properties
        sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'

        if meas_level in ['int', 'unk']:
            sample_result += cs_stat.print_var_stats(data, [var_names[0]], self.data_measlevs,
                                                     groups=groups,
                                                     statistics=['mean', 'std', 'max', 'upper quartile', 'median',
                                                                 'lower quartile', 'min'])
        elif meas_level == 'ord':
            sample_result += cs_stat.print_var_stats(data, [var_names[0]], self.data_measlevs,
                                                     groups=groups,
                                                     statistics=['max', 'upper quartile', 'median',
                                                                 'lower quartile', 'min'])
        elif meas_level == 'nom':
            sample_result += cs_stat.print_var_stats(data, [var_names[0]], self.data_measlevs,
                                                     groups=groups,
                                                     statistics=['variation ratio'])
            sample_result += '\n' + cs_stat.contingency_table(data, groups, var_names,
                                                              count=True, percent=True, margins=True)

        # Effect size
        sample_effect_size = cs_stat.compare_groups_effect_size(data, var_names, groups, meas_level,
                                                                sample=True)
        if sample_effect_size:
            sample_result += '\n\n' + sample_effect_size

        # Plot the individual data with boxplots
        # There's no need to repeat the mosaic plot for the nominal variables
        if meas_level in ['int', 'unk', 'ord']:
            sample_graph = cs_chart.create_compare_groups_sample_chart(data, meas_level, var_names, groups,
                                                                       level_combinations, ylims=ylims)
        else:
            sample_graph = None

        # 3. Population properties
        # Plot population estimations
        group_estimations = cs_stat.comp_group_estimations(data, meas_level, var_names, groups)
        population_graph = cs_chart.create_compare_groups_population_chart(data, meas_level, var_names, groups,
                                                                           level_combinations, ylims=ylims)

        # Population estimation
        population_result = '<cs_h2>' + _('Population properties') + '</cs_h2>' + \
                            '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>\n'
        if meas_level in ['int', 'unk']:
            population_result += _('Means') + '\n' + _('Present confidence interval values suppose normality.')
        elif meas_level == 'ord':
            population_result += _('Medians')
        if meas_level in ['int', 'unk', 'ord']:
            prec = cs_util.precision(data[var_names[0]]) + 1
            population_result += \
                cs_stat._format_html_table(group_estimations.to_html(bold_rows=False, classes="table_cs_pd",
                                                                     float_format=lambda x: '%0.*f' % (prec, x)))
        if meas_level == 'nom':
            population_result += '\n' + cs_stat.contingency_table(data, groups, var_names, ci=True)

        # effect size
        standardized_effect_size_result = cs_stat.compare_groups_effect_size(data, var_names, groups,
                                                                             meas_level, sample=False)
        if standardized_effect_size_result is not None:
            standardized_effect_size_result += '\n'

        # Hypothesis testing
        if len(groups) == 1:
            group_levels = sorted(set(data[groups[0]]))
            result_ht = cs_hyp_test.decision_one_grouping_variable(data, meas_level, self.data_measlevs,
                                                                   var_names, groups, group_levels,
                                                                   single_case_slope_SE, single_case_slope_trial_n)
        else:
            result_ht = cs_hyp_test.decision_several_grouping_variables(data, meas_level, var_names, groups)

        return cs_util.convert_output([title, raw_result, raw_graph, sample_result, sample_graph, population_result,
                                       population_graph, standardized_effect_size_result, result_ht])


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

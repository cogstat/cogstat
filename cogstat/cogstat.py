# -*- coding: utf-8 -*-

"""This module is the main engine for CogStat.

It includes the class for the CogStat data; initialization handles data import; methods implement some data handling
and they compile the appropriate statistics for the main analysis pipelines.

For the analyses, headings (<cs_hx>) are included in this module.
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

pd.options.display.html.border = 0

warn_unknown_variable = '<cs_warning><b>' + _('Measurement level warning') + '</b> ' + \
                        _('The measurement levels of the variables are not set. Set them in your data source.') \
                        + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                        % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                        + '</cs_warning>'
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
        self.orig_data_frame - pandas DataFrame, the original data without filtering
        self.data_frame - pandas DataFrame, the actual data with optional filtering
        self.data_measlevs - dictionary storing level of measurement of the variables (name:level):
                'nom', 'ord', or 'int' (ratio is included in 'int')
                'unk' - unknown: if no other level is given
        self.filtering_status - list of two items:
                                [0] list of the variables the filtering is based on (or None)
                                [2] the name of the filtering method (or '')

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
        self.filtering_status = [None, '']

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
                # spreadsheet errors, starting with # and ending with !, such as #DIV/0!, #VALUE!
            self.data_frame.replace(r'^Err:.*$', np.nan, regex=True, inplace=True)
                # spreadsheet errors, starting with Err:, such as Err:502
            # spreadsheet errors make the variable object dtype, although they may be numeric variables
            for column in self.data_frame.select_dtypes(include=['object']).columns:
                try:
                    self.data_frame[column] = self.data_frame[column].astype(float)
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
                    warning_text += '\n<cs_warning>' + \
                                    _('Number of measurement levels do not match the number of variables. '
                                    'You may want to correct the number of measurement levels.') + '</cs_warning>'

            # 2. Apply constraints to measurement levels.
            # String variables cannot be interval or nominal variables in CogStat, so change them to nominal
            invalid_var_names = [var_name for var_name in self.data_frame.columns if
                                 (self.data_measlevs[var_name] in ['int', 'ord', 'unk'] and
                                  str(self.data_frame[var_name].dtype) in ['object', 'string'])]
                # 'object' dtype means string variable
            if invalid_var_names:  # these str variables were set to int or ord
                for var_name in invalid_var_names:
                    self.data_measlevs[var_name] = 'nom'
                warning_text += '\n<cs_warning><b>' + _('String variable conversion warning') + '</b> ' + \
                                _('String variables cannot be interval or ordinal variables in CogStat. '
                                'Those variables are automatically set to nominal: ')\
                                + '<i>' + ', '.join('%s' % var_name for var_name in invalid_var_names) + \
                                '</i>. ' + _('You can fix this issue in your data source.') \
                                + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                + '</cs_warning>'

            # Warn when any measurement levels are not set
            if 'unk' in set(self.data_measlevs.values()):
                warning_text += '\n<cs_warning><b>' + _('Measurement level warning') + '</b> ' + \
                                       _('The measurement level was not set for all variables.') + ' '\
                                       + _('You can fix this issue in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</cs_warning>'
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
                warning_text += '\n<cs_warning><b>' + _('Recommended characters in variable names warning') + \
                                       '</b> ' + \
                                       _('Some variable name(s) include characters other than English letters, '
                                         'numbers, or underscore which can cause problems in some analyses: %s.') \
                                       % ('<i>' + ', '.join(
                    '%s' % non_ascii_var_name for non_ascii_var_name in non_ascii_var_names) + '</i>')\
                                       + ' ' + _('If some analyses cannot be run, fix this in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</cs_warning>'
            if non_ascii_vars:
                warning_text += '\n<cs_warning><b>' + _('Recommended characters in data values warning') + \
                                       '</b> ' + \
                                       _('Some string variable(s) include characters other than English letters, '
                                         'numbers, or underscore which can cause problems in some analyses: %s.') \
                                       % ('<i>' + ', '.join('%s' % non_ascii_var for non_ascii_var in non_ascii_vars) +
                                          '</i>')\
                                       + ' ' + _('If some analyses cannot be run, fix this in your data source.') \
                                       + ' ' + _('Read more about this issue <a href = "%s">here</a>.') \
                                       % 'https://github.com/cogstat/cogstat/wiki/Handling-data' \
                                       + '</cs_warning>'

        self.import_message = ''
        import_measurement_levels = None
        warning_text = ''

        # I. Import the DataFrame/file/clipboard

        # 1. Import from pandas DataFrame
        if isinstance(data, pd.DataFrame):
            self.data_frame = data
            self.import_source[0] = 'Pandas dataframe'  # intentionally, we don't localize this term

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
                                       _('Import failed') + '. ' + _('File type is not supported') + '.'
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
            self.import_message += '<cs_h1>' + _('Data') + '</cs_h1>' + _('Import failed') + '. ' + \
                                   _('Invalid data source') + '.'
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
            output += _('The file was successfully reloaded') + '.\n'
            output += self.import_message
            if self.filtering_status[0]:
                self.filter_outlier(var_names=self.filtering_status[0], mode=self.filtering_status[1])
        else:
            output += _('The data was not imported from a file') + '. ' + _('It cannot be reloaded') + '.\n'
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
        output += data_comb[:12 if brief else 1002].to_html(bold_rows=False).replace('\n', '')
        if brief and (len(self.data_frame.index) > 10):
            output += str(len(self.data_frame.index)-10) + _(' further cases are not displayed...')+'\n'
        elif len(self.data_frame.index) > 999:
            output += _('The next %s cases will not be printed. You can check all cases in the original data source.') \
                      % (len(self.data_frame.index)-1000) + '\n'

        return cs_util.convert_output([output])

    def filter_outlier(self, var_names=None, mode='2.5mad'):
        """
        Filter self.data_frame based on outliers.

        With univariate methods, all variables are investigated independently and cases are excluded if any variables
        shows they are outliers.
        If mode is 'mahalanobis', then variables are jointly investigated for multivariate outliers.
        If var_names is None, then the filtering will be switched off (i.e. all cases will be used).

        If any values in the given variables are missing in a case, the whole case will also be excluded.

        Parameters
        ----------
        var_names : None or list of str
            Names of the variables the exclusion is based on or None to include all cases.
        mode : {'2.5mad', '2sd', 'mahalanobis'}
            Mode of the exclusion:
                2.5mad: median +- 2.5 * MAD
                2sd: mean +- 2 * SD
                mahalanobis: MMCD Mahalanobis distance with .05 chi squared cut-off
            CogStat uses the MAD method for single variable-based outlier, but for possible future code change, the
            previous (2sd) version is also included.

        Returns
        -------
        list of str
            List of HTML strings showing the filtered cases.
            The method modifies the self.data_frame in place.
        list of charts
            If cases were filtered, then filtered and remaining cases are shown.

        Modifies the self.filtering_status.
        """
        mode_names = {'2sd': _('Mean ± 2 SD'),  # Used in the output
                      '2.5mad': _('Median ± 2.5 MAD'),
                      'mahalanobis': _('MMCD Mahalanobis distance with .05 chi squared cut-off')}

        self.filtering_status = [var_names, mode]

        title = '<cs_h1>' + _('Filter outliers') + '</cs_h1>'

        chart_results = []

        # Filtering should be done on the original data, so use self.orig_data_frame

        if var_names is None or var_names == []:  # Switch off outlier filtering
            self.data_frame = self.orig_data_frame.copy()
            text_output = _('Filtering is switched off.')
        else:  # Create a filtered dataframe based on the variable(s)
            remaining_cases_indexes = []
            text_output = ''
            if mode in ['2sd', '2.5mad']:
                for var_name in var_names:
                    # Check if the variable is 'int'
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
                    # Find the cases to be kept
                    remaining_cases_indexes.append(self.orig_data_frame[
                                                       (self.orig_data_frame[var_name] >= lower_limit) &
                                                       (self.orig_data_frame[var_name] <= upper_limit)].index)

                    # Display filtering information
                    text_output += _('Filtering based on %s') % (var_name + ' (%s)' % mode_names[mode]) + '.\n'
                    text_output += _('Cases with missing data will also be excluded') + '.\n'
                    prec = cs_util.precision(self.orig_data_frame[var_name]) + 1
                    text_output += _('Cases outside of the range will be excluded') + \
                                   ': %0.*f  –  %0.*f\n' % (prec, lower_limit, prec, upper_limit)
                    # Display the excluded cases
                    excluded_cases = \
                        self.orig_data_frame.drop(remaining_cases_indexes[-1])
                    # excluded_cases.index = [' '] * len(excluded_cases)  # TODO can we cut the indexes from the html table?
                    # TODO uncomment the above line after using pivot indexes in CS data
                    if len(excluded_cases):
                        text_output += _('Excluded cases (%s cases)') % (len(excluded_cases)) + ':'
                        # Change indexes to be in line with the data view numbering
                        excluded_cases.index = excluded_cases.index + 1
                        text_output += excluded_cases.to_html(bold_rows=False).replace('\n', '')
                        chart_results.append(cs_chart.create_filtered_cases_chart(
                            self.orig_data_frame.loc[remaining_cases_indexes[-1]][var_name],
                            excluded_cases[var_name], var_name, lower_limit=lower_limit, upper_limit=upper_limit))
                    else:
                        text_output += _('No cases were excluded') + '.'
                    if var_name != var_names[-1]:
                        text_output += '\n\n'
            elif mode == 'mahalanobis':
                # Based on the robust Mahalanobis distance in Leys et al., 2017 and Rousseeuw, 1999
                # Removing non-interval variables
                valid_var_names = var_names[:]
                for var_name in valid_var_names:
                    if self.data_measlevs[var_name] in ['ord', 'nom']:
                        valid_var_names.remove(var_name)
                if len(var_names) > len(valid_var_names):
                    text_output += _('Only interval variables can be used for filtering') + '.' + \
                                   _('Ignoring variable(s) %s') % ', '.join(set(var_names) - set(valid_var_names)) + \
                                   '.\n'

                # Calculating the robust Mahalanobis distances
                from sklearn import covariance
                cov = covariance.EllipticEnvelope(contamination=0.25).fit(self.orig_data_frame[valid_var_names].
                                                                          dropna())

                # Custom filtering criteria based on Leys et al. (2017)
                # Appropriate cut-off point based on chi2
                limit = stats.chi2.ppf(0.95, len(self.orig_data_frame[valid_var_names].columns))
                # Get robust Mahalanobis distances from model object
                distances = cov.mahalanobis(self.orig_data_frame[valid_var_names].dropna())
                filtering_data_frame = self.orig_data_frame.dropna(subset=valid_var_names).copy()
                filtering_data_frame['mahalanobis'] = distances

                # Find the cases to be kept
                remaining_cases_indexes.append(filtering_data_frame[(filtering_data_frame['mahalanobis'] <= limit)].
                                               index)

                # Display filtering information
                text_output += _('Multivariate filtering based on the variables: %s (%s)') % \
                               (', '.join(valid_var_names), mode_names[mode]) + '.\n'
                text_output += _('Cases with missing data will also be excluded') + '.\n'
                prec = cs_util.precision(filtering_data_frame['mahalanobis']) + 1  # TODO we should set this to a constant value
                text_output += _('Cases above the cutoff Mahalanobis distance will be excluded') + \
                               ': %0.*f\n' % (prec, limit)

                # Display the excluded cases
                excluded_cases = \
                    self.orig_data_frame.dropna(subset=valid_var_names).drop(remaining_cases_indexes[-1])
                # excluded_cases.index = [' '] * len(excluded_cases)  # TODO can we cut the indexes from the html table?
                # TODO uncomment the above line after using pivot indexes in CS data
                if len(excluded_cases):
                    text_output += _('Excluded cases (%s cases)') % (len(excluded_cases)) + ': '
                    # Change indexes to be in line with the data view numbering
                    excluded_cases.index = excluded_cases.index + 1
                    text_output += excluded_cases.to_html(bold_rows=False).replace('\n', '') + '\n'
                    for var_name in valid_var_names:
                        chart_results.append(cs_chart.create_filtered_cases_chart(
                            self.orig_data_frame.dropna(subset=valid_var_names).loc[remaining_cases_indexes[-1]]
                            [var_name], excluded_cases[var_name], var_name))

                else:
                    text_output += _('No cases were excluded') +'.'
            else:
                raise ValueError('Invalid mode parameter was given')

            # Do the filtering (remove outliers), modify self.data_frame in place
            self.data_frame = self.orig_data_frame.copy()
            for remaining_cases_index in remaining_cases_indexes:
                self.data_frame = self.data_frame.loc[self.data_frame.index.intersection(remaining_cases_index)]

        return cs_util.convert_output([title, text_output, chart_results])

    def _filtering_status(self):
        """Create a message about the filtering status (used variables and the filtering method).

        Returns
        -------
        str
            Filtering status to be printed. If filtering is off, then an empty string.
        """

        mode_names = {'2sd': _('Mean ± 2 SD'),  # Used in the output
                      '2.5mad': _('Median ± 2.5 MAD'),
                      'mahalanobis': _('MMCD Mahalanobis distance with .05 chi squared cut-off')}

        if self.filtering_status[0] is None or self.filtering_status[0] == []:
            return ''
        else:
            filtering_message = ', '.join(self.filtering_status[0]) + ' (%s)' % mode_names[self.filtering_status[1]]
            return '<b>' + _('Filtering is on') + ': %s</b>\n' % filtering_message

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

    def explore_variable(self, var_name='', frequencies=True, central_value=0.0):
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
        if not var_name:
            title = '<cs_h1>' + _('Explore variable') + '</cs_h1>'
            title += _('At least one variable should be set.')
            return cs_util.convert_output([title])

        meas_level, unknown_type = self._meas_lev_vars([var_name])
        result_list = ['<cs_h1>' + _('Explore variable') + '</cs_h1>']
        result_list.append(_('Exploring variable: ') + var_name + ' (%s)\n' % meas_level)
        result_list[-1] += self._filtering_status()

        # 1. Raw data
        text_result = '<cs_h2>' + _('Raw data') + '</cs_h2>'

        data = pd.DataFrame(self.data_frame[var_name].dropna())

        text_result2 = _('N of observed cases') + ': %g' % len(data) + '\n'
        missing_cases = len(self.data_frame[var_name])-len(data)
        text_result2 += _('N of missing cases')  + ': %g' % missing_cases + '\n'

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
        text_result += '<cs_h3>' + _('Descriptives for the variable') + '</cs_h3>'
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
            text_result += '<cs_h3>'+_('Normality')+'</cs_h3>'
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

        population_param_text = '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>'
        if meas_level in ['int', 'unk']:
            population_param_text += cs_stat.variable_estimation(data[var_name], ['mean', 'std'])
        elif meas_level == 'ord':
            population_param_text += cs_stat.variable_estimation(data[var_name], ['median'])
        elif meas_level == 'nom':
            population_param_text += cs_stat.proportions_ci(data, var_name)
        text_result = '\n'

        # Hypothesis tests
        text_result += '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>'
        if self.data_measlevs[var_name] in ['int', 'unk']:
            text_result += '<cs_decision>' + _('Testing if mean deviates from the value %s.') % central_value +\
                           '</cs_decision>\n'
        elif self.data_measlevs[var_name] == 'ord':
            text_result += '<cs_decision>' + _('Testing if median deviates from the value %s.') % central_value +\
                           '</cs_decision>\n'

        if unknown_type:
            text_result += '<cs_decision>' + warn_unknown_variable + '\n</cs_decision>'
        if meas_level in ['int', 'unk']:
            text_result += '<cs_decision>' + _('Interval variable.') + ' >> ' + \
                           _('Choosing one-sample t-test or Wilcoxon signed-rank test depending on the assumption.') + \
                           '</cs_decision>\n'
            text_result += '<cs_decision>' + _('Checking for normality.') + '\n</cs_decision>'
            norm, text_result_norm = cs_hyp_test.normality_test(data, self.data_measlevs, var_name)

            text_result += text_result_norm
            if norm:
                text_result += '<cs_decision>' + _('Normality is not violated.') + ' >> ' + \
                               _('Running one-sample t-test.') + '</cs_decision>\n'
                text_result2, ci = cs_hyp_test.one_t_test(data, self.data_measlevs, var_name,
                                                          test_value=central_value)
                graph = cs_chart.create_variable_population_chart(data[var_name], var_name, 'mean', ci)

            else:
                text_result += '<cs_decision>' + _('Normality is violated.') + ' >> ' + \
                               _('Running Wilcoxon signed-rank test.') + '</cs_decision>\n'
                text_result += _('Median: %0.*f') % (prec, np.median(data[var_name])) + '\n'
                text_result2 = cs_hyp_test.wilcox_sign_test(data, self.data_measlevs, var_name,
                                                            value=central_value)
                graph = cs_chart.create_variable_population_chart(data[var_name], var_name, 'median')

        elif meas_level == 'ord':
            text_result += '<cs_decision>' + _('Ordinal variable.') + ' >> ' + _('Running Wilcoxon signed-rank test.') + \
                           '</cs_decision>\n'
            text_result2 = cs_hyp_test.wilcox_sign_test(data, self.data_measlevs, var_name,
                                                        value=central_value)
            graph = cs_chart.create_variable_population_chart(data[var_name], var_name, 'median')
        else:
            text_result2 = '<cs_decision>' + _('Sorry, not implemented yet.') + '</cs_decision>\n'
            graph = None
        text_result += text_result2

        result_list.append(population_param_text)
        if graph:
            result_list.append(graph)
        result_list.append(text_result)
        return cs_util.convert_output(result_list)


    def reliability_internal(self, var_names=None, reverse_items=None):
        """
        Calculate internal consistency reliability using Cronbach's alpha and it's confidence interval,
        as well as item-rest correlations and their confidence intervals.

        Parameters
        ----------
        var_names : list of str
            Names of the variables or items.
        reverse_items : list of str
            Subset of var_names. Names of reverse coded variables or items.

        Returns
        -------
        list of str and matplotlib image
            Analysis results: str in HTML format
        """

        meas_levels = [self.data_measlevs[var_name] for var_name in var_names]

        title = '<cs_h1>' + _('Internal consistency reliability') + '</cs_h1>'
        title += _('Reliability of items') + ': ' + ', '.join('%s (%s)' % (var, meas)
                                                         for var, meas in zip(var_names, meas_levels))

        data = pd.DataFrame(self.data_frame[var_names].dropna())
        # Items to be reversed will be reversed here, and all functions will get (and expect) the reversed items.
        # This solution assumes that all values in an item are used. Otherwise (e.g., in a 1-5 scale, only 1-4 values
        #  are used), the score will be reversed incorrectly, leading to incorrect total score and other related
        #  statistics.
        if reverse_items:
            for reverse_item in reverse_items:
                data[reverse_item] = np.min(data[reverse_item]) + np.max(data[reverse_item]) - data[reverse_item]
            title += '\n' + _('Reverse coded item(s)') + ': ' + ', '.join('%s' % var for var in reverse_items)

        # Raw data
        raw_title = '<cs_h2>' + _('Raw data') + '</cs_h2>'

        missing_cases = len(self.data_frame[var_names])-len(data)
        raw_title += _('N of observed cases') + ': %g' % len(data) + '\n'
        raw_title += _('N of missing cases') + ': %g' % missing_cases
        raw_graph = cs_chart.create_item_total_matrix(data, regression=False)

        # Sample properties
        sample_title = '<cs_h2>' + _('Sample properties') + '</cs_h2>'
        alpha, item_removed_sample = cs_stat.reliability_internal_calc(data, sample=True)
        sample_graph = cs_chart.create_item_total_matrix(data, regression=True)
        sample_result = '\n' + _("Cronbach's alpha") + ' = %0.3f' % alpha[0] + '\n'

        # Population properties
        population_result = '<cs_h2>' + _('Population properties') + '</cs_h2>'
        alpha, item_removed_pop = cs_stat.reliability_internal_calc(data, sample=False)
        pop_result_df = pd.DataFrame(columns=[_('Point estimation'), _('95% confidence interval')])
        pop_result_df.loc[_("Cronbach's alpha")] = \
            ['%0.3f' % alpha[0], '[%0.3f, %0.3f]' % (alpha[1][0], alpha[1][1])]
        population_result += pop_result_df.to_html(bold_rows=False, escape=False, float_format=lambda x: '%0.3f' % (x))\
                                 .replace('\n', '') + '\n'

        return cs_util.convert_output([title, raw_title, raw_graph, sample_title, sample_graph, sample_result,
                                       item_removed_sample, population_result, item_removed_pop])


    def reliability_interrater(self, var_names=None, ratings_averaged=True, ylims=[None, None]):
        """
        Calculate inter-rater reliability using intraclass correlation. Use the McGraw and Wong, 1996 terms. Follow the
        Liljequist et al. 2019 strategy and display three indexes.

        Parameters
        ----------
        var_names : list of str
            Names of variables containing the ratings of the raters.
        ratings_averaged : bool
            Are the ratings averaged?
        ylims : list of {int or float}
            Limit of the y axis for interval and ordinal variables instead of using automatic values.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """

        meas_levels = [self.data_measlevs[var_name] for var_name in var_names]

        title = '<cs_h1>' + _('Interrater reliability') + '</cs_h1>'
        title += _('Reliability calculated from variables') + ': ' + \
                 ', '.join('%s (%s)' % (var, meas) for var, meas in zip(var_names, meas_levels))

        # Raw data
        raw_title = '<cs_h2>' + _('Raw data') + '</cs_h2>'

        data = pd.DataFrame(self.data_frame[var_names].dropna())
        missing_cases = len(self.data_frame[var_names])-len(data)
        raw_title += _('N of observed cases') + ': %g' % len(data) + '\n'
        raw_title += _('N of missing cases') + ': %g' % missing_cases

        raw_plot = cs_chart.create_repeated_measures_sample_chart(data, var_names, meas_level='int',
                                                                  raw_data_only=True, ylims=ylims)

        # Analysis
        data_copy = data.reset_index()
        data_long = pd.melt(data_copy, id_vars='index')
        sample_result_table, population_result_table, hyp_test_table = \
            cs_stat.reliability_interrater_calc(data_long, targets='index', raters='variable', ratings='value',
                                                ratings_averaged=ratings_averaged)

        # Sample properties
        sample_title = '<cs_h2>' + _('Sample properties') + '</cs_h2>'
        sample_plot = cs_chart.create_repeated_measures_sample_chart(data, var_names, meas_level='int',
                                                                     raw_data_only=False, ylims=ylims)

        # Population properties
        population_result = '<cs_h2>' + _('Population properties') + '</cs_h2>'
        population_result += '<cs_h3>' + _('Checking assumptions of inferential methods') + '</cs_h3>'
        population_result += '<cs_decision>' + _('Testing normality') + '.</cs_decision>'
        non_normal_vars, normality_text, var_hom_p, var_text_result = \
            cs_hyp_test.reliability_interrater_assumptions(data, data_long, var_names, self.data_measlevs)
        population_result += '\n' + normality_text
        warnings = ''
        if not non_normal_vars:
            population_result += '<cs_decision>' + _('Assumption of normality met') + '.</cs_decision>' + '\n'
        else:
            population_result += '<cs_decision>' + _('Assumption of normality violated in variable(s) %s' %
                                                  ', '.join(non_normal_vars)) + '</cs_decision>' + '\n'
            warnings += '<cs_decision>' + _('Assumption of normality violated') + '.</cs_decision>'
        population_result += '\n' + '<cs_decision>' + _('Testing homogeneity of variances') + '.</cs_decision>'
        population_result += '\n' + var_text_result
        if var_hom_p < 0.05:
            population_result += '<cs_decision>' + _('Assumption of homogeneity of variances violated') + \
                                 '.</cs_decision>'
            warnings += '<cs_decision>' + _('Assumption of homogeneity of variances violated') + '.</cs_decision>'
        else:
            population_result += '<cs_decision>' + _('Assumption of homogeneity of variances met') + '.</cs_decision>'

        population_result += '<cs_h3>' + _('Parameter estimates') + '</cs_h3>'
        if non_normal_vars or var_hom_p < 0.05:
            warnings += '<cs_decision>' + _('CIs may be inaccurate') + '.</cs_decision>'
        else:
            warnings += '<cs_decision>' + _('Assumptions met') + '.</cs_decision>'

        hypothesis_tests = cs_hyp_test.reliability_interrater_hyp_test(hyp_test_table, non_normal_vars, var_hom_p)

        return cs_util.convert_output([title, raw_title, raw_plot, sample_title, sample_plot,
                                       sample_result_table, population_result, warnings, population_result_table,
                                       hypothesis_tests])


    def regression(self, predictors=None, predicted=None, xlims=[None, None], ylims=[None, None]):
        """
        Explore a variable pair or multiple predictors and one predicted variable.

        Parameters
        ----------
        predictors : list of str
            Name of the predictor variables.
        predicted : str
            Name of the predicted variable.
        xlims : list of {int or float}
            Limit of the x-axis for interval and ordinal variables instead of using automatic values.
        ylims : list of {int or float}
            Limit of the y-axis for interval and ordinal variables instead of using automatic values.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """
        title = '<cs_h1>' + _('Explore relation of variables') + '</cs_h1>'
        preconditions = True
        if (predictors is None) or (predictors == [None]) or not predictors:
            title += _('At least one predictor variable should be set') + '.\n'
            preconditions = False
        if predicted is None:
            title += _('The predicted variable should be set') + '.'
            preconditions = False
        constant_vars = []
        for var in predictors + [predicted]:
            if len(set(self.data_frame[var])) == 1:
                constant_vars += [var]
        if len(constant_vars) > 0:
            title += _('Analysis cannot be run for constant variable(s): %s') % ', '.join(constant_vars) + '\n'
            preconditions = False

        if not preconditions:
            return cs_util.convert_output([title])

        meas_lev, unknown_var = self._meas_lev_vars(predictors + [predicted])

        # TODO merge the code of 1 and several predictors when it is feasible/makes sense

        if len(predictors) == 1:
            # x and y will be the name of the variables when only a single regressor is used
            x = predictors[0]
            y = predicted

        # Analysis output header
        if len(predictors) == 1:
            title = '<cs_h1>' + _('Explore relation of variable pair') + '</cs_h1>'
        else:
            title = '<cs_h1>' + _('Explore relation of variables') + '</cs_h1>'

        # 0. Analysis information
        if len(predictors) == 1:
            raw_result = _('Exploring variable pair') + ': ' + x + ' (%s), ' % self.data_measlevs[x] \
                         + y + ' (%s)\n' % self.data_measlevs[y]
        else:
            raw_result = _('Predictors') + ': ' + \
                         ', '.join([predictor + ' (%s)' % self.data_measlevs[predictor] for predictor in predictors]) + \
                         '\n' + _('Predicted') + ': ' + predicted + ' (%s)\n' % self.data_measlevs[predicted]
        raw_result += self._filtering_status()
        if unknown_var:
            raw_result += '<cs_decision>' + warn_unknown_variable + '\n</cs_decision>'

        # 1. Raw data
        raw_result += '<cs_h2>' + _('Raw data') + '</cs_h2>'
        # Prepare data, drop missing data
        # TODO are NaNs interesting in nominal variables?
        data = self.data_frame[predictors + [predicted]].dropna()
        observed_n = len(data)
        missing_n = len(self.data_frame[predictors + [predicted]]) - observed_n
        raw_result += _('N of observed pairs') + ': %g' % observed_n + '\n'
        raw_result += _('N of missing pairs') + ': %g' % missing_n + '\n'

        # Raw data chart
        if len(predictors) == 1:
            raw_graph = cs_chart.create_variable_pair_chart(data, meas_lev, x, y, raw_data=True,
                                                            regression=False, CI=False, xlims=xlims, ylims=ylims)
        else:
            # display the predicted variable first
            raw_graph = cs_chart.create_scatter_matrix(data[[predicted] + predictors], meas_lev)

        # 2. Sample properties
        sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'
        residual_title = None
        residual_graph = None
        if meas_lev == 'nom':
            sample_result += cs_stat.contingency_table(data, [x], [y], count=True, percent=True, margins=True)
        elif meas_lev == 'int':

            # Calculate regression with statsmodels
            import statsmodels.regression
            import statsmodels.tools
            from statsmodels.api import add_constant

            if len(predictors) == 1:
                data_sorted = data.sort_values(by=x)  # Sorting required for subsequent plots to work
                model = statsmodels.regression.linear_model.OLS(data_sorted[y], add_constant(data_sorted[x]))
            else:
                model = statsmodels.regression.linear_model.OLS(data[predicted], add_constant(data[predictors]))
            result = model.fit()

            if len(predictors) == 1:
                # TODO output with the right precision of the results
                sample_result += _('Linear regression')+': y = %0.3fx + %0.3f' % (result.params[1], result.params[0])
            else:
                import string
                # Shift constant to the end of the list
                params = [x for x in list(result.params) if x != result.params[0]] + [result.params[0]]
                # Pair up parameter estimates with variable names and unpack the resulting pairs to a non-nested tuple
                content = tuple(c for b in zip(params, predictors+['']) for c in b)
                # Get string to format
                structure = ' + '.join(['%0.3f%s' for i in range(len(params))])
                sample_result += _('Linear regression')+': %s = ' % predicted + structure % content

        sample_result += '\n'

        if len(predictors) == 1:
            standardized_effect_size_result = '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>'
            standardized_effect_size_result += cs_stat.variable_pair_standard_effect_size(data, meas_lev, sample=True) \
                                               + '\n'
        else:
            if meas_lev in ['int', 'unk']:
                standardized_effect_size_result = '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>'
                standardized_effect_size_result += cs_stat.multiple_variables_standard_effect_size(data, predictors,
                                                   predicted, result, sample=True) + '\n'
            else:
                standardized_effect_size_result = None

        # Make graphs
        # extra chart is needed only for int variables, otherwise the chart would just repeat the raw data
        if meas_lev == 'int':

            # Residual analysis
            if len(predictors) == 1:
                residual_title = '<cs_h3>' + _('Residual analysis') + '</cs_h3>'
                residual_graph = cs_chart.create_residual_chart(data, meas_lev, x, y)
            else:
                # TODO multivariate residuals
                residual_title = None
                residual_graph = None

            # Sample scatter plot with regression line
            if len(predictors) == 1:
                sample_graph = cs_chart.create_variable_pair_chart(data, meas_lev, x, y, result=result, raw_data=True,
                                                                   regression=True, CI=False, xlims=xlims, ylims=ylims)
            else:
                # TODO scatter plot with fitted line
                sample_graph = None

            if len(predictors) > 1:
                regression_plot = cs_chart.part_regress_plots(data, predicted, predictors)
            else:
                regression_plot = None
        else:
            sample_graph = None
            regression_plot = None

        # 3. Population properties
        # TODO for the estimations, do not print warning if assumption is not violated
        population_properties_title = '<cs_h2>' + _('Population properties') + '</cs_h2>'
        estimation_result = '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>'
        estimation_parameters, estimation_effect_size, population_graph = None, None, None

        # Initilazing assumptions
        normality = None  # Do the two variables follow a multivariate normal distribution?
        homoscedasticity = None
        assumptions_result = None

        if meas_lev == 'nom':
            estimation_result += cs_stat.contingency_table(data, [x], [y], ci=True)
        elif meas_lev == 'int':

            # Test of multivariate normality
            assumptions_result = '<cs_h3>' + _('Checking assumptions of inferential methods') + '</cs_h3>'
            assumptions_result += '<cs_decision>' + _('Testing multivariate normality of variables') + '</cs_decision>\n'
            normality, norm_text = cs_hyp_test.multivariate_normality(data, predictors + [predicted])
            assumptions_result += norm_text

            # Test of homoscedasticity
            assumptions_result += '<cs_decision>' + _('Testing homoscedasticity') + '</cs_decision>\n'
            homoscedasticity, het_text = cs_hyp_test.homoscedasticity(data, predictors, predicted)
            assumptions_result += het_text

            # Test of multicollinearity
            if len(predictors) > 1:
                vif, multicollinearity = cs_stat.vif_table(data, predictors)
                assumptions_result += '<cs_decision>' + _('Testing multicollinearity') + '</cs_decision>\n'
                assumptions_result += vif
                assumptions_result += "\n" + cs_stat.correlation_matrix(data, predictors)

            estimation_parameters = '<cs_h4>' + _('Regression coefficients') + '</cs_h4>'
            estimation_parameters += cs_stat.variable_pair_regression_coefficients(predictors, meas_lev,
                                                                                   normality=normality,
                                                                                   homoscedasticity=homoscedasticity,
                                                                                   multicollinearity=multicollinearity
                                                                                   if len(predictors) > 1 else None,
                                                                                   result=result)

            if len(predictors) == 1:
                population_graph = cs_chart.create_variable_pair_chart(data, meas_lev, x, y, result=result,
                                                                       raw_data=False, regression=True, CI=True,
                                                                       xlims=[None, None], ylims=[None, None])
            else:
                # TODO multivariate population graph
                pass

        if len(predictors) == 1:
            estimation_effect_size = '<cs_h4>' + _('Standardized effect sizes') + '</cs_h4>'
            estimation_effect_size += cs_stat.variable_pair_standard_effect_size(data, meas_lev, sample=False,
                                                                                 normality=normality,
                                                                                 homoscedasticity=homoscedasticity)
        else:
            if meas_lev in ['int', 'unk']:
                estimation_effect_size = '<cs_h4>' + _('Standardized effect sizes') + '</cs_h4>'
                estimation_effect_size += cs_stat.multiple_variables_standard_effect_size(self.data_frame, predictors,
                                                                                          predicted, result, normality,
                                                                                          homoscedasticity, multicollinearity,
                                                                                          sample=False)

        population_result = '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>'
        if len(predictors) == 1:
            population_result += cs_hyp_test.variable_pair_hyp_test(data, x, y, meas_lev, normality, homoscedasticity) \
                                 + '\n'
        else:
            population_result += cs_hyp_test.multiple_regression_hyp_tests(data=self.data_frame, result=result,
                                                                           predictors=predictors, normality=normality,
                                                                           homoscedasticity=homoscedasticity,
                                                                           multicollinearity=multicollinearity)

        # TODO should we set all optional returned item to None at the beginning of the method? And in all methods
        return cs_util.convert_output([title, raw_result, raw_graph,
                                       sample_result, sample_graph, regression_plot,
                                       standardized_effect_size_result,
                                       residual_title, residual_graph,
                                       population_properties_title, assumptions_result,
                                       estimation_result,
                                       estimation_parameters, population_graph, estimation_effect_size,
                                       population_result])

    def pivot(self, depend_name='', row_names=None, col_names=None, page_names=None, function='Mean'):
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
        if page_names is None:
            page_names = []
        if col_names is None:
            col_names = []
        if row_names is None:
            row_names = []

        title = '<cs_h1>' + _('Pivot table') + '</cs_h1>'
        preconditions = True
        if not depend_name:
            title += _('The dependent variable should be set') + '.\n'
            preconditions = False
        if not (row_names or col_names or page_names):
            title += _('At least one grouping variable should be set') + '\n'
            preconditions = False
        if not preconditions:
            return cs_util.convert_output([title])


        # TODO optionally return pandas DataFrame or Panel
        pivot_result = cs_stat.pivot(self.data_frame, row_names, col_names, page_names, depend_name, function)
        return cs_util.convert_output([title, pivot_result])

    def diffusion(self, error_name='', RT_name='', participant_name='', condition_names=None, correct_coding='0',
                  reaction_time_in='sec', scaling_parameter=0.1):
        """
        Run diffusion analysis on behavioral data.

        Dataframe should include a single trial in a case (row).

        Parameters
        ----------
        error_name : str
            Name of the variable storing the errors.
            Correct and incorrect trials should be coded with 0 and 1. See the correct_coding parameter.
        RT_name : str
            Name of the variable storing response times.
            Time should be stored in sec or msec. See the reaction_time_in parameter.
        participant_name : str
            Name of the variable storing participant IDs.
        condition_names : list of str
            Name(s) of the variable(s) storing conditions.
        correct_coding : {'0', '1'}
            Are correct responses noted with 0 or 1? Incorrect responses are noted with the other value.
        scaling_parameter : float
            Usually either 0.1 or 1
        reaction_time_in : {'sec', 'msec'}
            Unit of reaction time

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """
        if condition_names is None:
            condition_names = []
        # TODO return pandas DataFrame
        title = '<cs_h1>' + _('Behavioral data diffusion analysis') + '</cs_h1>'
        preconditions = True
        if not RT_name:
            title += _('The reaction time should be given') + '.\n'
            preconditions = False
        if not error_name:
            title += _('The error variables should be given') + '.'
            preconditions = False
        if not preconditions:
            return cs_util.convert_output([title])

        pivot_result = cs_stat.diffusion(self.data_frame, error_name, RT_name, participant_name, condition_names,
                                         correct_coding, reaction_time_in, scaling_parameter)
        return cs_util.convert_output([title, pivot_result])

    def compare_variables(self, var_names, factors=None, display_factors=None, ylims=[None, None]):
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
        display_factors: list of two lists of strings
            Factors to be displayed on x-axis, and color (panel cannot be used for repeated measures data).
        ylims : list of {int or float}
            Limit of the y-axis for interval and ordinal variables instead of using automatic values.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """

        # 0. Analysis info
        title = '<cs_h1>' + _('Compare repeated measures variables') + '</cs_h1>'
        meas_levels = [self.data_measlevs[var_name] for var_name in var_names]

        # Check preconditions
        preconditions = True
        if len(var_names) < 2:
            title += _('At least two variables should be set') + '.\n'
            preconditions = False
        if '' in var_names:
            title += _('A variable should be assigned to each level of the factors.') + '\n'
            preconditions = False
        # Check if the variables have the same measurement levels
        # int and unk can be used together, since unk is taken as int by default
        if (len(set(meas_levels)) > 1) and ('ord' in meas_levels or 'nom' in meas_levels):
            title += _('Variables to compare: ') + ', '.\
            join('%s (%s)' % (var, meas) for var, meas in zip(var_names, meas_levels)) + '\n'
            title += _("Sorry, you can't compare variables with different measurement levels."
                       " You could downgrade higher measurement levels to lowers to have the same measurement level.")\
                     + '\n'
        if not preconditions:
            return cs_util.convert_output([title])

        # Prepare missing parameters
        # if factor is not specified, use a single space for factor name, so this can be handled by the rest of the code
        if factors is None or factors == []:
            factors = [[_('Unnamed factor'), len(var_names)]]
        # if display_factors is not specified, then all factors are displayed on the x-axis
        if (display_factors is None) or (display_factors == [[], []]):
            display_factors = [[factor[0] for factor in factors], []]

        # Variables info
        analysis_info = _('Variables to compare') + ': ' + \
                        ', '.join('%s (%s)' % (var, meas) for var, meas in zip(var_names, meas_levels)) + '\n'
        analysis_info += _('Factor(s) (number of levels)') + ': ' + ', '.\
            join('%s (%d)' % (factor[0], factor[1]) for factor in factors) + '\n'
        factor_combinations = ['']
        for factor in factors:
            factor_combinations = ['%s - %s %s' % (factor_combination, factor[0], level_i+1)
                                   for factor_combination in factor_combinations
                                   for level_i in range(factor[1])]
        # remove ' - ' from the beginning of the strings
        factor_combinations = [factor_combination[3:] for factor_combination in factor_combinations]
        analysis_info += _('Factor level combinations and assigned variables') + ':\n'
        for factor_combination, var_name in zip(factor_combinations, var_names):
            analysis_info += '%s: %s\n' % (factor_combination, var_name)

        # Filtering status
        analysis_info += self._filtering_status()

        # level of measurement of the dependent variables
        meas_level, unknown_type = self._meas_lev_vars(var_names)
        if unknown_type:
            analysis_info += '\n<cs_decision>' + warn_unknown_variable + '</cs_decision>'

        # 1. Raw data
        raw_result = '<cs_h2>' + _('Raw data') + '</cs_h2>'

        # Prepare data, drop missing data, display number of observed/missing cases
        # TODO are NaNs interesting in nominal variables?
        data = self.data_frame[var_names].dropna()
        observed_n = len(data)
        missing_n = len(self.data_frame[var_names]) - observed_n
        raw_result += _('N of observed cases') + ': %g\n' % observed_n
        raw_result += _('N of missing cases') + ': %g\n' % missing_n

        # Plot the individual raw data
        raw_graph = cs_chart.create_repeated_measures_sample_chart(data, var_names, meas_level, raw_data_only=True,
                                                                   ylims=ylims)
        factor_info = pd.DataFrame([var_names], columns=pd.MultiIndex.from_product([['%s %s' % (factor[0], i) for i in range(factor[1])] for factor in factors],
                                                                                  names=[factor[0] for factor in factors]))
        raw_graph_new = cs_chart.create_repeated_measures_groups_chart(data=data, dep_meas_level=meas_level,
                                                                       dep_names=var_names,
                                                                       factor_info=factor_info,
                                                                       indep_x=display_factors[0],
                                                                       indep_color=display_factors[1],
                                                                       ylims=ylims, raw_data=True)

        # 2. Sample properties
        sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'

        # 2a. Descriptives
        sample_result += '<cs_h3>' + _('Descriptives for the variables') + '</cs_h3>'
        statistics = {'int': ['mean', 'std', 'max', 'upper quartile', 'median', 'lower quartile', 'min'],
                      'unk': ['mean', 'std', 'max', 'upper quartile', 'median', 'lower quartile', 'min'],
                      'ord': ['max', 'upper quartile', 'median', 'lower quartile', 'min'],
                      'nom': ['variation ratio']}
        sample_result += cs_stat.print_var_stats(data, var_names, self.data_measlevs, statistics=statistics[meas_level])
        if meas_level == 'nom':
            import itertools
            for var_pair in itertools.combinations(var_names, 2):
                sample_result += cs_stat.contingency_table(data, [var_pair[1]], [var_pair[0]], count=True,
                                                           percent=True, margins=True)
            sample_result += '\n'

        # 2b. Effect size
        sample_effect_size = cs_stat.repeated_measures_effect_size(data, var_names, factors, meas_level, sample=True)
        if sample_effect_size:
            sample_result += '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>' + sample_effect_size

        # 2c. Plot the individual data with box plot
        # There's no need to repeat the mosaic plot for nominal variables
        if meas_level in ['int', 'unk', 'ord']:
            sample_graph = cs_chart.create_repeated_measures_sample_chart(data, var_names, meas_level, ylims=ylims)
            sample_graph_new = cs_chart.create_repeated_measures_groups_chart(data=data, dep_meas_level=meas_level,
                                                                              dep_names=var_names,
                                                                              factor_info=factor_info,
                                                                              indep_x=display_factors[0],
                                                                              indep_color=display_factors[1],
                                                                              ylims=ylims, raw_data=True, box_plots=True)
        else:
            sample_graph = None
            sample_graph_new = None

        # 3. Population properties
        population_result = '<cs_h2>' + _('Population properties') + '</cs_h2>'

        # 3a. and 3c. Population estimations and plots
        population_result += '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>'
        if meas_level in ['int', 'unk']:
            population_result += _('Means') + '\n' + _('Present confidence interval values suppose normality.')
            mean_estimations = cs_stat.repeated_measures_estimations(data, meas_level)
            prec = cs_util.precision(data[var_names[0]]) + 1
            population_result += mean_estimations.to_html(bold_rows=False, float_format=lambda x: '%0.*f' % (prec, x))\
                .replace('\n', '')
        elif meas_level == 'ord':
            population_result += _('Median')
            median_estimations = cs_stat.repeated_measures_estimations(data, meas_level)
            prec = cs_util.precision(data[var_names[0]]) + 1
            population_result += median_estimations.to_html(bold_rows=False,float_format=lambda x: '%0.*f' % (prec, x))\
                .replace('\n', '')
        elif meas_level == 'nom':
            for var_pair in itertools.combinations(var_names, 2):
                population_result += cs_stat.contingency_table(data, [var_pair[1]], [var_pair[0]], ci=True)
        population_result += '\n'

        population_graph = cs_chart.create_repeated_measures_population_chart(data, var_names, meas_level, ylims=ylims)
        population_estimation, *population_graph_new = cs_chart.\
            create_repeated_measures_groups_chart(data=data, dep_meas_level=meas_level,
                                                  dep_names=var_names,
                                                  factor_info=factor_info,
                                                  indep_x=display_factors[0],
                                                  indep_color=display_factors[1],
                                                  ylims=ylims, estimations=True,
                                                  estimation_table=True)

        # 3b. Effect size
        population_effect_size = cs_stat.repeated_measures_effect_size(data, var_names, factors, meas_level, sample=False)
        if population_effect_size:
            population_effect_size += '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>' + population_effect_size

        # 3d. Hypothesis tests
        result_ht = '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>' + \
                    cs_hyp_test.decision_repeated_measures(data, meas_level, factors, var_names, self.data_measlevs)

        return cs_util.convert_output([title, analysis_info, raw_result, raw_graph, raw_graph_new, sample_result,
                                       sample_graph, sample_graph_new, population_result, population_estimation,
                                       population_effect_size, population_graph, population_graph_new, result_ht])

    def compare_groups(self, var_name,
                       grouping_variables=None, display_groups=None,
                       single_case_slope_SE=None, single_case_slope_trial_n=None,
                       ylims=[None, None]):
        """
        Compare groups.

        Parameters
        ----------
        var_name : str
            Name of the dependent variable
        grouping_variables : list of str
            List of name(s) of grouping variable(s).
        display_groups : list of three list of strings
            List of name(s) of grouping variable(s) displayed on x-axis, with colors, and on panels.
        single_case_slope_SE : str
            When comparing the slope between a single case and a group, variable name storing the slope SEs
        single_case_slope_trial : int
            When comparing the slope between a single case and a group, number of trials.
        ylims : list of {int or float}
            Limit of the y-axis for interval and ordinal variables instead of using automatic values.

        Returns
        -------
        list of str and image
            Analysis results in HTML format
        """

        # 0. Analysis info
        title = '<cs_h1>' + _('Compare groups') + '</cs_h1>'

        # Check preconditions
        # TODO check if there is only one dep.var.
        preconditions = True
        if not var_name or (var_name is None):
            title += _('The dependent variable should be set') + '.\n'
            preconditions = False
        if (grouping_variables is None) or grouping_variables == []:
            title += _('At least one grouping variable should be set') + '.\n'
            preconditions = False
        if not preconditions:
            return cs_util.convert_output([title])

        var_names = [var_name]
        if grouping_variables is None:
            grouping_variables = []

        # Prepare missing parameters
        # if display_groups are not specified, then all group will be displayed on x-axis
        if (display_groups is None) or (display_groups == [[], [], []]):
            display_groups = [grouping_variables, [], []]

        # Variables info
        analysis_info = _('Dependent variable: ') + '%s (%s)' % (var_name, self.data_measlevs[var_name]) + '\n' + \
                     _('Grouping variable(s)') + ': ' + \
                     ', '.join('%s (%s)' % (var, meas) for var, meas
                               in zip(grouping_variables, [self.data_measlevs[group] for group in grouping_variables]))\
                     + '\n'

        # Filtering status
        analysis_info += self._filtering_status()

        # level of measurement of the dependent variables
        meas_level, unknown_type = self._meas_lev_vars([var_names[0]])
        if unknown_type:
            analysis_info += '<cs_decision>' + warn_unknown_variable + '</cs_decision>'

        # 1. Raw data
        raw_result = '<cs_h2>' + _('Raw data') + '</cs_h2>'

        # Prepare data, drop missing data, display number of observed/missing cases
        single_case_slope_SE_list = [single_case_slope_SE] if single_case_slope_SE else []
        data = self.data_frame[grouping_variables + [var_names[0]] + single_case_slope_SE_list].dropna()

        # display the number of observed/missing cases for (a) grouping variable level combinations and (b) missing
        #  level information
        # create a list of sets with the levels of all grouping variables
        levels = [list(set(data[grouping_variable])) for grouping_variable in grouping_variables]
        for i in range(len(levels)):
            levels[i].sort()
        # TODO sort the levels in other parts of the output, too
        # create all level combinations for the grouping variables
        level_combinations = list(itertools.product(*levels))
        # index should be specified to work in pandas 0.11; but this way can't use _() for the labels
        columns = pd.MultiIndex.from_tuples(level_combinations, names=grouping_variables)
        pdf_result = pd.DataFrame(columns=columns)

        pdf_result.loc[_('N of observed cases')] = [sum((data[grouping_variables] == pd.Series(
            {grouping_variable: level for grouping_variable, level in zip(grouping_variables, level_combination)})).all(
            axis=1)) for level_combination in level_combinations]
        pdf_result.loc[_('N of missing cases')] = [sum((self.data_frame[grouping_variables] == pd.Series(
            {grouping_variable: level for grouping_variable, level in zip(grouping_variables, level_combination)})).all(
            axis=1)) - sum((data[grouping_variables] == pd.Series(
            {grouping_variable: level for grouping_variable, level in zip(grouping_variables, level_combination)})).all(
            axis=1)) for level_combination in level_combinations]
        raw_result += pdf_result.to_html(bold_rows=False).replace('\n', '')
        raw_result += '\n\n'
        # display missing grouping level information
        for grouping_variable in grouping_variables:
            observed_n = len(self.data_frame[grouping_variable].dropna())
            missing_n = len(self.data_frame[grouping_variable]) - observed_n
            raw_result += _('N of missing grouping variable in %s') % grouping_variable + ': %g\n' % missing_n

        # Plot individual raw data

        raw_graph = cs_chart.create_compare_groups_sample_chart(data, meas_level, var_names, grouping_variables,
                                                                level_combinations, raw_data_only=True, ylims=ylims)
        raw_graph_new = cs_chart.create_repeated_measures_groups_chart(data, meas_level,
                                                                       dep_names=[var_name],
                                                                       indep_x=display_groups[0],
                                                                       indep_color=display_groups[1],
                                                                       indep_panel=display_groups[2],
                                                                       ylims=ylims, raw_data=True)

        # 2. Sample properties
        sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'

        # 2a. Descriptives
        sample_result += '<cs_h3>' + _('Descriptives for the groups') + '</cs_h3>'
        statistics = {'int': ['mean', 'std', 'max', 'upper quartile', 'median', 'lower quartile', 'min'],
                      'unk': ['mean', 'std', 'max', 'upper quartile', 'median', 'lower quartile', 'min'],
                      'ord': ['max', 'upper quartile', 'median', 'lower quartile', 'min'],
                      'nom': ['variation ratio']}
        sample_result += cs_stat.print_var_stats(data, [var_names[0]], self.data_measlevs,
                                                 grouping_variables=grouping_variables,
                                                 statistics=statistics[meas_level])
        if meas_level == 'nom':
            sample_result += '\n' + cs_stat.contingency_table(data, grouping_variables, var_names,
                                                              count=True, percent=True, margins=True)

        # 2b. Effect size
        sample_effect_size = cs_stat.compare_groups_effect_size(data, var_names, grouping_variables, meas_level,
                                                                sample=True)
        if sample_effect_size:
            sample_result += '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>' + sample_effect_size

        # Plot the individual data with boxplots
        # There's no need to repeat the mosaic plot for the nominal variables
        if meas_level in ['int', 'unk', 'ord']:
            sample_graph = cs_chart.create_compare_groups_sample_chart(data, meas_level, var_names, grouping_variables,
                                                                       level_combinations, ylims=ylims)
            sample_graph_new = cs_chart.create_repeated_measures_groups_chart(data, meas_level,
                                                                              dep_names=[var_name],
                                                                              indep_x=display_groups[0],
                                                                              indep_color=display_groups[1],
                                                                              indep_panel=display_groups[2],
                                                                              ylims=ylims,
                                                                              raw_data=True,
                                                                              box_plots=True)
        else:
            sample_graph = None
            sample_graph_new = None

        # 3. Population properties
        population_result = '<cs_h2>' + _('Population properties') + '</cs_h2>'

        # 3a. and c. Population estimation and plots
        population_result += '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>'

        group_estimations = cs_stat.comp_group_estimations(data, meas_level, var_names, grouping_variables)
        population_graph = cs_chart.create_compare_groups_population_chart(data, meas_level, var_names, grouping_variables,
                                                                           level_combinations, ylims=ylims)
        population_estimation, *population_graph_new = cs_chart.create_repeated_measures_groups_chart(data, meas_level,
                                                                              dep_names=[var_name],
                                                                              indep_x=display_groups[0],
                                                                              indep_color=display_groups[1],
                                                                              indep_panel=display_groups[2],
                                                                              estimations=True, ylims=ylims,
                                                                              estimation_table=True)

        if meas_level in ['int', 'unk', 'ord']:
            if meas_level in ['int', 'unk']:
                population_result += _('Means') + '\n' + _('Present confidence interval values suppose normality.')
            elif meas_level == 'ord':
                population_result += _('Medians')
            prec = cs_util.precision(data[var_names[0]]) + 1
            population_result += group_estimations.to_html(bold_rows=False, float_format=lambda x: '%0.*f' % (prec, x))\
                .replace('\n', '')
        if meas_level == 'nom':
            population_result += '\n' + cs_stat.contingency_table(data, grouping_variables, var_names, ci=True)

        # 3b. Effect size
        population_effect_size = cs_stat.compare_groups_effect_size(data, var_names, grouping_variables,
                                                                             meas_level, sample=False)
        if population_effect_size is not None:
            population__effect_size = '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>' + \
                                              population_effect_size + '\n'

        # 3d. Hypothesis testing
        if len(grouping_variables) == 1:
            group_levels = sorted(set(data[grouping_variables[0]]))
            result_ht = '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>' + \
                        cs_hyp_test.decision_one_grouping_variable(data, meas_level, self.data_measlevs,
                                                                   var_names, grouping_variables, group_levels,
                                                                   single_case_slope_SE, single_case_slope_trial_n)
        else:
            result_ht = '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>' + \
                        cs_hyp_test.decision_several_grouping_variables(data, meas_level, var_names, grouping_variables)

        return cs_util.convert_output([title, analysis_info, raw_result, raw_graph, raw_graph_new, sample_result,
                                       sample_graph, sample_graph_new, population_result, population_estimation,
                                       population_effect_size, population_graph, population_graph_new, result_ht])

    def compare_variables_groups(self, var_names=None, factors=None, grouping_variables=None, display_factors=None,
                          single_case_slope_SE=None, single_case_slope_trial_n=None, ylims=[None, None]):
        """ Compare mixed-design (repeated measures and groups) data.

        Parameters
        ----------
        var_names: list of str
            The variable to be compared.
        factors : list of list of [str, int]
            The factors and their levels, e.g.,
                [['name of the factor', number_of_the_levels],
                ['name of the factor 2', number_of_the_levels]]
            Factorial combination of the factors will be generated, and variables will be assigned respectively
        grouping_variables : list of str
            List of name(s) of grouping variable(s).
        display_factors: list of two lists of strings
            Factors to be displayed on x-axis, and color (panel cannot be used for repeated measures data).
        single_case_slope_SE : str
            When comparing the slope between a single case and a group, variable name storing the slope SEs
        single_case_slope_trial : int
            When comparing the slope between a single case and a group, number of trials.
        ylims : list of {int or float}
            Limit of the y-axis for interval and ordinal variables instead of using automatic values.

        Returns
        -------
        list of str and image
            Analysis results in HTML format

        """

        if var_names is None:
            var_names = []
        if grouping_variables is None:
            grouping_variables = []

        # 0. Analysis info
        if len(var_names) == 1 and grouping_variables:
            title = '<cs_h1>' + _('Compare groups') + '</cs_h1>'
        elif not grouping_variables:
            title = '<cs_h1>' + _('Compare repeated measures variables') + '</cs_h1>'
        else:
            title = '<cs_h1>' + _('Compare repeated measures variables and groups') + '</cs_h1>'
        meas_levels = [self.data_measlevs[var_name] for var_name in var_names]

        # Check preconditions
        preconditions = True
        if len(var_names) < 1:
            title += _('At least one dependent variable should be set') + '.\n'
            preconditions = False
        if '' in var_names:
            title = _('A variable should be assigned to each level of the factors') + '.\n'
            preconditions = False
        # Check if the repeated measures variables have the same measurement levels
        # int and unk can be used together, since unk is taken as int by default
        if (len(set(meas_levels)) > 1) and ('ord' in meas_levels or 'nom' in meas_levels):
            title += _('Variables to compare: ') + ', '.\
            join('%s (%s)' % (var, meas) for var, meas in zip(var_names, meas_levels)) + '\n'
            title += _("Sorry, you can't compare variables with different measurement levels."
                       " You could downgrade higher measurement levels to lowers to have the same measurement level.")\
                     + '\n'
        if not preconditions:
            return cs_util.convert_output([title])

        # Prepare missing parameters
        # if factor is not specified, use a single space for factor name, so this can be handled by the rest of the code
        if (factors is None or factors == []) and len(var_names) > 1:
            factors = [[_('Unnamed factor'), len(var_names)]]
        # handle if display_factors is not specified
        if (display_factors is None) or (display_factors == [[], []]) or (display_factors == [[], [], []]):  # TODO check what is possible here
            if grouping_variables and not factors:  # only between-subject: all group will be displayed on x-axis
                display_factors = [grouping_variables, [], []]
            elif factors and not grouping_variables:  # only within-subject:  all factors are displayed on the x-axis
                display_factors = [[factor[0] for factor in factors], [], []]
            else:  # mixed design
                display_factors = [grouping_variables + [factor[0] for factor in factors], [], []]

        # Variables info
        if len(var_names) == 1:
            analysis_info = _('Dependent variable: ') + '%s (%s)' % (var_names[0], self.data_measlevs[var_names[0]]) + '\n'
        else:
            analysis_info = _('Variables to compare: ') + ', '. \
                join('%s (%s)' % (var, meas) for var, meas in zip(var_names, meas_levels)) + '\n'
            analysis_info += _('Factor(s) (number of levels)') + ': ' + ', '. \
                join('%s (%d)' % (factor[0], factor[1]) for factor in factors) + '\n'
            factor_combinations = ['']
            for factor in factors:
                factor_combinations = ['%s - %s %s' % (factor_combination, factor[0], level_i + 1)
                                       for factor_combination in factor_combinations
                                       for level_i in range(factor[1])]
            # remove ' - ' from the beginning of the strings
            factor_combinations = [factor_combination[3:] for factor_combination in factor_combinations]
            analysis_info += _('Factor level combinations and assigned variables') + ':\n'
            for factor_combination, var_name in zip(factor_combinations, var_names):
                analysis_info += '%s: %s\n' % (factor_combination, var_name)
        if grouping_variables:
            analysis_info += _('Grouping variable(s)') + ': ' + \
                          ', '.join('%s (%s)' % (var, meas) for var, meas
                                    in zip(grouping_variables,
                                           [self.data_measlevs[group] for group in grouping_variables])) + '\n'

        # Filtering status
        analysis_info += self._filtering_status()

        # level of measurement of the dependent variables
        meas_level, unknown_type = self._meas_lev_vars(var_names)
        if unknown_type:
            analysis_info += '\n<cs_decision>' + warn_unknown_variable + '</cs_decision>'

        # 1. Raw data
        raw_result = '<cs_h2>' + _('Raw data') + '</cs_h2>'

        # Prepare data, drop missing data, display number of observed/missing cases
        # TODO are NaNs interesting in nominal variables?
        single_case_slope_SE_list = [single_case_slope_SE] if single_case_slope_SE else []
        data = self.data_frame[grouping_variables + var_names + single_case_slope_SE_list].dropna()

        if not grouping_variables:
            observed_n = len(data)
            missing_n = len(self.data_frame[var_names]) - observed_n
            raw_result += _('N of observed cases') + ': %g\n' % observed_n
            raw_result += _('N of missing cases') + ': %g\n' % missing_n
        else:  # there are grouping variables
            # display the number of observed/missing cases for (a) grouping variable level combinations and (b) missing
            #  level information
            # create a list of sets with the levels of all grouping variables
            levels = [list(set(data[grouping_variable])) for grouping_variable in grouping_variables]
            for i in range(len(levels)):
                levels[i].sort()
            # TODO sort the levels in other parts of the output, too
            # create all level combinations for the grouping variables
            level_combinations = list(itertools.product(*levels))
            # index should be specified to work in pandas 0.11; but this way can't use _() for the labels
            columns = pd.MultiIndex.from_tuples(level_combinations, names=grouping_variables)
            pdf_result = pd.DataFrame(columns=columns)

            pdf_result.loc[_('N of observed cases')] = [sum((data[grouping_variables] == pd.Series(
                {grouping_variable: level for grouping_variable, level in zip(grouping_variables, level_combination)}))
                                                            .all(axis=1)) for level_combination in level_combinations]
            pdf_result.loc[_('N of missing cases')] = [sum((self.data_frame[grouping_variables] == pd.Series(
                {grouping_variable: level for grouping_variable, level in zip(grouping_variables, level_combination)}))
                                                           .all(axis=1)) - sum((data[grouping_variables] == pd.Series(
                {grouping_variable: level for grouping_variable, level in zip(grouping_variables, level_combination)}))
                                                           .all(axis=1)) for level_combination in level_combinations]
            raw_result += pdf_result.to_html(bold_rows=False).replace('\n', '')
            raw_result += '\n\n'

            # display missing grouping level information
            for grouping_variable in grouping_variables:
                observed_n = len(self.data_frame[grouping_variable].dropna())
                missing_n = len(self.data_frame[grouping_variable]) - observed_n
                raw_result += _('N of missing grouping variable in %s') % grouping_variable + ': %g\n' % missing_n

        factor_info = pd.DataFrame([var_names],
                                   columns=pd.MultiIndex.from_product(
                                       [['%s %s' % (factor[0], i) for i in range(factor[1])] for factor in factors],
                                       names=[factor[0] for factor in factors]))

        #print('cs.py first call:', var_names, factors, grouping_variables, display_factors)
        #print(factor_info)

        # Plot the individual raw data
        raw_graph_new = cs_chart.create_repeated_measures_groups_chart(data=data, dep_meas_level=meas_level,
                                                                       dep_names=var_names,
                                                                       factor_info=factor_info,
                                                                       indep_x=display_factors[0],
                                                                       indep_color=display_factors[1],
                                                                       indep_panel=display_factors[2],
                                                                       ylims=ylims, raw_data=True)

        # 2. Sample properties
        sample_result = '<cs_h2>' + _('Sample properties') + '</cs_h2>'

        sample_result += '<cs_h3>' + _('Descriptives for the variables') + '</cs_h3>'

        statistics = {'int': ['mean', 'std', 'max', 'upper quartile', 'median', 'lower quartile', 'min'],
                      'unk': ['mean', 'std', 'max', 'upper quartile', 'median', 'lower quartile', 'min'],
                      'ord': ['max', 'upper quartile', 'median', 'lower quartile', 'min'],
                      'nom': ['variation ratio']}

        descriptive_table, *sample_graph_new = cs_chart.\
            create_repeated_measures_groups_chart(data=data, dep_meas_level=meas_level,
                                                  dep_names=var_names,
                                                  factor_info=factor_info,
                                                  indep_x=display_factors[0],
                                                  indep_color=display_factors[1],
                                                  indep_panel=display_factors[2],
                                                  ylims=ylims, raw_data=True, box_plots=True,
                                                  descriptives_table=True, statistics=statistics[meas_level])
        #sample_graph_new = cs_chart.create_repeated_measures_groups_chart(dep_name=var_name)
        # TODO for nominal dependent variable include the contingency table
        #  See the variable and the group comparison solutions

        # 2b. Effect size
        if not grouping_variables:  # no grouping variables
            sample_effect_size = cs_stat.repeated_measures_effect_size(data, var_names, factors, meas_level,
                                                                       sample=True)
        elif len(var_names) == 1:  # grouping variables with one dependent variable
            sample_effect_size = cs_stat.compare_groups_effect_size(data, var_names, grouping_variables, meas_level,
                                                                    sample=True)
        else:  # mixed design
            sample_effect_size = None
            # TODO
        if sample_effect_size:
            sample_result += '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>' + sample_effect_size

        # 3. Population properties
        population_result = '<cs_h2>' + _('Population properties') + '</cs_h2>'

        # 3a. and 3c. Population estimations and plots
        population_result += '<cs_h3>' + _('Population parameter estimations') + '</cs_h3>'
        population_estimation, *population_graph_new = cs_chart.\
            create_repeated_measures_groups_chart(data=data, dep_meas_level=meas_level,
                                                  dep_names=var_names,
                                                  factor_info=factor_info,
                                                  indep_x=display_factors[0],
                                                  indep_color=display_factors[1],
                                                  indep_panel=display_factors[2],
                                                  ylims=ylims, estimations=True,
                                                  estimation_table=True)
        prec = cs_util.precision(data[var_names[0]]) + 1  # TODO which variables should be used here?
        # 3b. Effect size
        if not grouping_variables:  # no grouping variables
            population_effect_size = cs_stat.repeated_measures_effect_size(data, var_names, factors, meas_level,
                                                                           sample=False)
        elif len(var_names) == 1:  # grouping variables with one dependent variable
            population_effect_size = cs_stat.compare_groups_effect_size(data, var_names, grouping_variables, meas_level,
                                                                        sample=False)
        else:  # mixed design
            population_effect_size = None
            # TODO
        if population_effect_size:
            population_effect_size += '<cs_h3>' + _('Standardized effect sizes') + '</cs_h3>' + population_effect_size

        # 3d. Hypothesis tests
        result_ht = '<cs_h3>' + _('Hypothesis tests') + '</cs_h3>'
        if not grouping_variables:  # no grouping variables
            result_ht += cs_hyp_test.decision_repeated_measures(data, meas_level, factors, var_names,
                                                                self.data_measlevs)
        elif len(var_names) == 1:  # grouping variables with one dependent variable
            if len(grouping_variables) == 1:
                group_levels = sorted(set(data[grouping_variables[0]]))
                result_ht += cs_hyp_test.decision_one_grouping_variable(data, meas_level, self.data_measlevs,
                                                                        var_names, grouping_variables, group_levels,
                                                                        single_case_slope_SE, single_case_slope_trial_n)
            else:
                result_ht += cs_hyp_test.decision_several_grouping_variables(data, meas_level, var_names,
                                                                             grouping_variables)
        else:  # mixed design
            result_ht += _('Sorry, not implemented yet.')

        return cs_util.convert_output([title, analysis_info, raw_result, raw_graph_new,
                                       sample_result, descriptive_table, sample_graph_new,
                                       population_result, population_estimation, population_effect_size,
                                       population_graph_new, result_ht])


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
    plt.close('all')  # free memory after everything is displayed

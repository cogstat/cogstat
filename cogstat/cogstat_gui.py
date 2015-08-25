# -*- coding: utf-8 -*-
"""
GUI for CogStat.
"""

import sys
import os
import webbrowser
import gettext
import logging

import cogstat
import cogstat_dialogs
import cogstat_config as csc
csc.versions['cogstat'] = cogstat.__version__
import cogstat_util as cs_util

from PyQt4 import QtGui
from PyQt4 import QtCore

cs_util.get_versions()

logging.root.setLevel(logging.INFO)

reload(sys)
sys.setdefaultencoding("utf-8")  # TODO Not sure if this will work correctly for most systems.

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.ugettext


class StatMainWindow(QtGui.QMainWindow):
    """
    CogStat GUI.
    """
    def __init__(self):
        super(StatMainWindow, self).__init__()  # TOD do we need super()?
        self._init_UI()

        # Check if all required components are installed
        # TODO Maybe all these checking can be removed
        missing_required_components, missing_recommended_components = self._check_installed_components()
        if missing_required_components or missing_recommended_components:
            QtGui.QMessageBox.critical(self, 'Incomplete installation', u'Install missing component(s): ' + ''.join([x+u', ' for x in missing_required_components+missing_recommended_components])[:-2]+u'.<br><br>'+u'<a href = "https://sites.google.com/site/cogstatprogram/home/telepites">Visit the installation help page</a> to see how to complete the installation.', QtGui.QMessageBox.Ok)
            if missing_required_components:
                sys.exit()
        
        self.analysis_results = []
        # analysis_result stores list of GuiResultPackages.
        # It will be useful when we can rerun all the previous analysis in the GUI output
        # At the moment no former results can be manipulated later

        cogstat.output_type = 'gui'  # For some GUI specific formatting

        # Only for testing
        # self.open_file('dev_data/example2.csv'); #self.compare_groups()
#        self.open_clipboard()
#        self.print_data()
#        self.explore_variable('X')
#        self.explore_variable(u'a', freq=False)
#         self.explore_variable_pair(['X', 'Y'])
#         self.pivot([u'X'], row_names=[], col_names=[], page_names=[u'CONDITION', u'TIME3'], function='N')
#         self.compare_variables(['X', 'Y'])
#        self.compare_variables([u'CONDITION', u'CONDITION2', u'TIME'])
#        self.compare_groups([u'dep_var'], [u'group_var'])
#         self.compare_groups([u'A'], [u'H'])
#         self.compare_groups(['X'], ['TIME'])
#        self.save_result_as()
#        self.save_result_as(filename='pdf_test.pdf')

    def _init_UI(self):
        self.resize(800, 600)
        self.setWindowTitle('CogStat')
        self.setWindowIcon(QtGui.QIcon('resources/CogStat.ico'))
        # FIXME Icon not showing up in Win7 64 and Win8.1 64, but works on WinXP 32 and Linux Mint 13 64

        # Menus and commands
        menu_commands = [  # This list will be used to construct the menus
                            [_('&Data'),
                                ['', _('&Open data file'), _('Ctrl+O'), _('Open data file (csv text file)'), 'self.open_file'],
                                ['', _('&Paste data'), _('Ctrl+V'), _('Paste data from clipboard'), 'self.open_clipboard'],
                                ['separator'],
                                # ['', _('&Filter outliers'), _('Ctrl+L'), _('Filter cases based on outliers'), 'self.xxx'],
                                # ['separator'],
                                ['', _('&Display data'), _('Ctrl+D'), _('Print data to the output'), 'self.print_data'],
                                ['', _('Display data &briefly'), '', _('Print beginning of the data to the output'), 'self._print_data_brief'],
                            ],
                            [_('&Analysis'),
                                ['', _('&Explore variable'), _('Ctrl+1'), _('Main properties of variables'), 'self.explore_variable'],
                                ['', _('Explore variable &pair'), _('Ctrl+2'), _('Properties of variable pairs'), 'self.explore_variable_pair'],
                                ['separator'],
                                ['', _('Pivot &table'), 'Ctrl+T', _('Build a pivot table'), 'self.pivot'],
                                ['separator'],
                                ['', _('Compare va&riables'), 'Ctrl+R', _('Compare variables'), 'self.compare_variables'],
                                ['', _('Compare &groups'), 'Ctrl+G', _('Compare groups'), 'self.compare_groups'],
                            ],
                            [_('&Results'),
                                ['', _('&Clear results'), _('Del'), _('Delete the output window'), 'self.delete_output'],
                                ['separator'],
                                ['', _('Save results'), _('Ctrl+P'), _('Save the output to .pdf format'), 'self.save_result'],
                                ['', _('Save results as'), _('Shift+Ctrl+P'), _('Save the results'), 'self.save_result_as']
                            ],
                            [_('&CogStat'),
                                ['', _('&Help'), _('F1'), _('Read online documentation'), 'self._open_help_webpage'],
                                ['', _('&Preferences'), '', _('Set the preferences'), 'self._show_preferences'],
                                ['', _('Request a &feature'), '', _("Can't find a feature? Ask for it!"), 'self._open_reqfeat_webpage'],
                                ['separator'],
                                ['', _('&Report a problem'), '', _('Fill online form to report a problem'), 'self._open_reportbug_webpage'],
                                ['', _('&Diagnosis information'), '', _('List the version of the components on your system'), 'self.print_versions'],
                                ['separator'],
                                ['', _('&About'), '', _('About CogStat'), 'self._show_about'],
                                ['separator'],
                                ['', _('&Exit'), _('Ctrl+Q'), _('Exit CogStat'), 'self.closeEvent']
                            ]
                        ]
        # Enable these commands only when active_data is available
        self.analysis_commands = [_('&Save data'), _('Save data &as'), _('&Display data'), _('Display data &briefly'),
                                  _('&Set variable properties'), _('Pivot &table'), _('&Explore variable'),
                                  _('Explore variable &pair'), _('Compare va&riables'), _('Compare &groups'),
                                  _('&Compare groups and variables')]

        # Create menus and commands
        self.menubar = self.menuBar()
        self.menus = []
        self.menu_commands = {}
        for menu in menu_commands:
            self.menus.append(self.menubar.addMenu(menu[0]))
            for i in range(1, len(menu)):
                if menu[i][0] == 'separator':
                    self.menus[-1].addSeparator()
                else:
                    self.menu_commands[menu[i][1]] = QtGui.QAction(QtGui.QIcon(menu[i][0]), menu[i][1], self)
                    self.menu_commands[menu[i][1]].setShortcut(menu[i][2])
                    self.menu_commands[menu[i][1]].setStatusTip(menu[i][3])
                    self.menu_commands[menu[i][1]].triggered.connect(eval(menu[i][4]))
                    self.menus[-1].addAction(self.menu_commands[menu[i][1]])
        for menu in self.analysis_commands:
            try:
                self.menu_commands[menu].setEnabled(False)
            except KeyError:
                pass
        
        # Prepare Output pane
        self.output_pane = QtGui.QTextBrowser()  # QTextBrowser can handle links, QTextEdit cannot
        self.output_pane.setLineWrapMode (QtGui.QTextEdit.NoWrap)
        self.output_pane.setText(_('<br><b>Welcome to CogStat!</b><br>CogStat makes statistical analysis more simple and efficient.<br>To start working open a data file or paste your data from a spreadsheet.<br>Find more information about CogStat on its <a href = "http://sites.google.com/site/cogstatprogram/">webpage</a>.<br>'))
        self.welcome_text_on = True  # Used for deleting the welcome text at the first analysis
        self.output_pane.setReadOnly(True)
        self.output_pane.setOpenExternalLinks(True)
        self.output_pane.setStyleSheet("QWidget { background-color: white; }")
            # Some styles use non-white background (e.g. Linux Mint 17 Mate uses gray)
        # Set default font
        #print self.output_pane.currentFont().toString()
        # http://stackoverflow.com/questions/2475750/using-qt-css-to-set-own-q-propertyqfont
        font = QtGui.QFont()
        font.setFamily(csc.default_font)
        font.setPointSizeF(csc.default_font_size)
        self.output_pane.setFont(font)
        #print self.output_pane.currentFont().toString()

        self.setCentralWidget(self.output_pane)
        self.setAcceptDrops(True)
        self.statusBar().showMessage(_('Ready'))

        self.unsaved_output = True
        self.output_filename = ''
        
        self.show()

    def _show_data_menus(self, on=True):
        """
        Enable or disable data handling menus depending on whether data is loaded.
        
        parameters:
        on: True to enable menus
            False to disable
            default is True
        """
        for menu in self.analysis_commands:
            try:
                self.menu_commands[menu].setEnabled(on)
            except:
                pass
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("text/uri-list"):
            event.accept()
        elif event.mimeData().hasFormat("text/plain"):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasFormat("text/uri-list"):
            # print 'Dropped URL: ', event.mimeData().urls()[0].toString()[7:]
            self.open_file(filename=event.mimeData().urls()[0].toString()[7:])
        elif event.mimeData().hasFormat("text/plain"):
            # print 'Dropped Text: ', event.mimeData().text()
            self.open_clipboard_data(data=event.mimeData().text())
        
    def _check_installed_components(self):
        """
        Check if all required and recommended components are installed.
        Return the list of missing components as strings.
        """
        missing_required_components = []
        missing_recommended_components = []

        # Required components
        for module in ['pyqt', 'numpy', 'pandas', 'scipy', 'statsmodels']:
            if csc.versions[module] is None:
                missing_required_components.append(module)

        # Recommended components
        for module in []:  # At the moment it's empty
            if csc.versions[module] is None:
                missing_recommended_components.append(module)
        # Check R only on Linux, since Win doesn't have a working rpy at the moment
        if sys.platform in ['linux2', 'linux']:
            for module in ['r', 'rpy2', 'car']:
                if csc.versions[module] is None:
                    missing_recommended_components.append(module)
        
        if missing_required_components:
            logging.error('Missing required components: %s' % missing_required_components)
        if missing_recommended_components:
            logging.error('Missing recommended components: %s' % missing_recommended_components)
        
        return missing_required_components, missing_recommended_components

    def _busy_signal(self, on):
        """
        Changes the mouse, signalling that the system is busy
        """
        # http://qt-project.org/doc/qt-4.7/qt.html see CursorShape
        # http://qt-project.org/doc/qt-4.7/qapplication.html#id-19f00dae-ec43-493e-824c-ef07ce96d4c6
        if on:
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.BusyCursor))
        else:
            while QtGui.QApplication.overrideCursor() is not None:
                # TODO if for some reason (unhandled exception) the cursor was not set back formerly,
                # then next time set it back
                # FIXME exception handling should solve this problem on the long term
                QtGui.QApplication.restoreOverrideCursor()
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        
    def _print_to_output_pane(self, index=-1):
        """Print a GuiResultPackage to GUI output pane
        :param index: index of the item in self.analysis_results to be printed
                      If no index is given, the last item is printed.
        """
        if self.welcome_text_on:
            self.output_pane.clear()
            self.welcome_text_on = False
        for output in self.analysis_results[index].output:
            if isinstance(output, basestring):
                self.output_pane.append(output)
            elif isinstance(output, QtGui.QImage):
                self.output_pane.moveCursor(11, 0)  # Moves cursor to the end
                self.output_pane.textCursor().insertImage(output)
            elif output is None:
                pass  # We simply don't do anything with None-s
            else:
                logging.error('Unknown output type: %s' % type(output))
        self.unsaved_output = True
                        
    ### Data menu methods ###
    def open_file(self, filename=''):
        """Open file.
        :param filename: filename with path
        """
        if filename in ['', False]:
            filename = cogstat_dialogs.open_data_file()
        if filename:
            self._open_data(unicode(filename))

    def open_clipboard(self):
        """Open data copied to clipboard."""
        clipboard = QtGui.QApplication.clipboard()
        if clipboard.mimeData().hasFormat("text/plain"):
            self._open_data(unicode(clipboard.text("plain", QtGui.QClipboard.Clipboard)))
    
    def _open_data(self, data):
        """ Core of the import process.
        """
        self._busy_signal(True)
        self.active_data = cogstat.CogStatData(data=data)
        if self.active_data.import_source == _('Import failed'):
            QtGui.QMessageBox.warning(self, _('Import error'), _('Data could not be loaded.'), QtGui.QMessageBox.Ok)
            self._show_data_menus(False)
        else:
            self._show_data_menus()
            self.statusBar().showMessage((_('Data loaded from clipboard: ') if data else _('Data loaded from file: '))
                                        + _('%s variables and %s cases.') % (len(self.active_data.data_frame.columns),
                                                                             len(self.active_data.data_frame.index)))
            self.print_data(brief=True, display_import_message=True)
        self._busy_signal(False)
            
    def print_data(self, brief=False, display_import_message=False):
        """Print the current data to the output.
        
        :param brief (bool): print only the first 10 rows
        :param display_import_message (bool):
        """
        self.analysis_results.append(GuiResultPackage())
        self.analysis_results[-1].add_command('self.print_data')  # TODO commands will be used to rerun the analysis
        self.analysis_results[-1].add_output(self.active_data.print_data(brief=brief))
        if self.active_data.import_message and display_import_message:
            self.analysis_results[-1].add_output(cs_util.reformat_output(self.active_data.import_message))
        self._print_to_output_pane()

    def _print_data_brief(self):
        self.print_data(brief=True)

    ### Statistics menu methods ###
        
    def explore_variable(self, var_names=None, freq=True, dist=True, descr=True, norm=True, loc_test=True,
                         loc_test_value=0):
        """Computes various properties of variables.

        Arguments:
        var_names (list): variable names
        freq (bool): compute frequencies (default True)
        dist (bool): compute distribution (default True)
        descr (bool): compute descriptive statistics (default True)
        norm (bool): check normality (default True)
        loc_test (bool): test location (e.g. t-test) (default True)
        loc_test_value (numeric): test location against this value (default 0.0)
        """
        if not var_names:
            try:
                self.dial_var_prop
            except:
                self.dial_var_prop = cogstat_dialogs.explore_var_dialog(names = self.active_data.data_frame.columns)
            else:  # TODO is it not necessary anymore? For all dialogs
                self.dial_var_prop.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_var_prop.exec_():
                var_names, freq, dist, descr, norm, loc_test, loc_test_value = self.dial_var_prop.read_parameters()
            else:
                return
        self._busy_signal(True)
        for var_name in var_names:
            self.analysis_results.append(GuiResultPackage())
            self.analysis_results[-1].add_command('self.explore_variable()')  # TODO
            result = self.active_data.explore_variable(var_name, frequencies=freq, distribution=dist,
                                                            descriptives=descr, normality=norm, central_test=loc_test,
                                                            central_value=loc_test_value)
            self.analysis_results[-1].add_output(result)
            self._print_to_output_pane()
        self._busy_signal(False)

    def explore_variable_pair(self, var_names=None):
        """Explore variable pairs.
        
        Arguments:
        var_names (list): variable names
        """
        if not var_names:
            try:
                self.dial_var_pair
            except:
                self.dial_var_pair = cogstat_dialogs.explore_var_pairs_dialog(names=self.active_data.data_frame.columns)
            else:
                self.dial_var_pair.init_vars(names = self.active_data.data_frame.columns)
            if self.dial_var_pair.exec_():
                var_names = self.dial_var_pair.read_parameters()
            else:
                return
        self._busy_signal(True)
        if len(var_names) < 2:  # TODO this check should go to the appropriate dialog
            self.analysis_results.append(GuiResultPackage())
            text_result = '<default>'+_(u'At least two variables should be set.')
            self.analysis_results[-1].add_output(text_result)
        else:
            for x in var_names:
                pass_diag = False
                for y in var_names:
                    if pass_diag:
                        self.analysis_results.append(GuiResultPackage())
                        self.analysis_results[-1].add_command('self.explore_variable_pair')  # TODO
                        result_list = self.active_data.explore_variable_pair(x, y)
                        self.analysis_results[-1].add_output(result_list)
                    if x == y:
                        pass_diag = True
        self._print_to_output_pane()
        self._busy_signal(False)
            
    def pivot(self, depend_names=None, row_names=[], col_names=[], page_names=[], function='Mean'):
        """Build a pivot table.
        
        Arguments:
        depend_names (str): name of the dependent variable
        row_names, col_names, page_names (lists of str): name of the independent variables
        function (str): available functions: N,Sum, Mean, Median, Standard Deviation, Variance (default Mean)
        """
        if not depend_names:
            try:
                self.dial_pivot
            except:
                self.dial_pivot = cogstat_dialogs.pivot_dialog(names=self.active_data.data_frame.columns)
            else:
                self.dial_pivot.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_pivot.exec_():
                row_names, col_names, page_names, depend_names, function = self.dial_pivot.read_parameters()
            else:
                return
        self._busy_signal(True)
        self.analysis_results.append(GuiResultPackage())
        if not depend_names or not (row_names or col_names or page_names):  # TODO this check should go to the dialog
            text_result = '<default>'+_('The dependent variable and at least one grouping variable should be given.')
        else:
            text_result = self.active_data.pivot(depend_names, row_names, col_names, page_names, function)
        self.analysis_results[-1].add_output(text_result)
        self._print_to_output_pane()
        self._busy_signal(False)

    def compare_variables(self, var_names=None):
        """Compare variables.
        
        Arguments:
        var_names (list): variable names
        """
        if not var_names:
            try:
                self.dial_comp_var
            except:
                self.dial_comp_var = cogstat_dialogs.compare_vars_dialog(names=self.active_data.data_frame.columns)
            else:
                self.dial_comp_var.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_comp_var.exec_():
                var_names = self.dial_comp_var.read_parameters()  # TODO check if settings are appropriate
            else:
                return
        self._busy_signal(True)
        self.analysis_results.append(GuiResultPackage())
        self.analysis_results[-1].add_command('self.compare_variables()')  # TODO
        if len(var_names) < 2:
            text_result = '<default>'+_(u'At least two variables should be set.')
            self.analysis_results[-1].add_output(text_result)
        else:
            result_list = self.active_data.compare_variables(var_names)
            for result in result_list:  # TODO is this a list of lists? Can we remove the loop?
                self.analysis_results[-1].add_output(result)
        self._print_to_output_pane()
        self._busy_signal(False)
        
    def compare_groups(self, var_names=None, groups=None):
        """Compare groups.
        
        Arguments:
        var_names (list): dependent variable names
        groups (list): grouping variable names
        """
        if not var_names:
            try:
                self.dial_comp_grp
            except:
                self.dial_comp_grp = cogstat_dialogs.compare_groups_dialog(names=self.active_data.data_frame.columns)
            else:
                self.dial_comp_grp.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_comp_grp.exec_():
                var_names, groups = self.dial_comp_grp.read_parameters()  # TODO check if settings are appropriate
            else:
                return
        self._busy_signal(True)
        self.analysis_results.append(GuiResultPackage())
        self.analysis_results[-1].add_command('self.compare_groups()')  # TODO
        if not var_names or not groups:
            text_result = '<default>'+_(u'Both the dependent and the grouping variables should be set.')
            self.analysis_results[-1].add_output(text_result)
        else:
            result_list = self.active_data.compare_groups(var_names[0], groups[0])
            self.analysis_results[-1].add_output(result_list)
        self._print_to_output_pane()
        self._busy_signal(False)

    ### Result menu methods ###
    def delete_output(self):
        reply = QtGui.QMessageBox.question(self, _('Clear output'),
            _('Are you sure you want to delete the output?'), QtGui.QMessageBox.Yes | 
            QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            self.output_pane.clear()
            self.analysis_results = []

    def save_result(self):
        """Save the output pane to pdf file."""
        if self.output_filename == '':
            self.save_result_as()
        else:
            pdf_printer = QtGui.QPrinter()
            pdf_printer.setOutputFormat(QtGui.QPrinter.PdfFormat)
            pdf_printer.setColorMode(QtGui.QPrinter.Color)
            pdf_printer.setOutputFileName(self.output_filename)
            self.output_pane.print_(pdf_printer)
            self.unsaved_output = False
            
    def save_result_as(self, filename=None):
        """Save the output pane to pdf file.
        
        Arguments:
        filename (str): name of the file to save to
        """
        if not filename:
            filename = cogstat_dialogs.save_output()
        self.output_filename = filename
        if filename:
            # self.output_pane.setLineWrapMode (QtGui.QTextEdit.FixedPixelWidth)  # TODO
            pdf_printer = QtGui.QPrinter()
            pdf_printer.setOutputFormat(QtGui.QPrinter.PdfFormat)
            pdf_printer.setOutputFileName(self.output_filename)
            self.output_pane.print_(pdf_printer)
            # self.output_pane.setLineWrapMode (QtGui.QTextEdit.NoWrap)
            self.unsaved_output = False

    ### Cogstat menu  methods ###
    def _open_help_webpage(self):
        webbrowser.open(_('http://sites.google.com/site/cogstatprogram/'))
        
    def _show_preferences(self):
        try:
            self.dial_pref
        except:
            self.dial_pref = cogstat_dialogs.preferences_dialog()
        self.dial_pref.exec_()
    
    def _open_reqfeat_webpage(self):
        webbrowser.open(_('https://docs.google.com/spreadsheet/viewform?formkey=dEZ2aU0teXdxVEltOHNsLTBmTk9QM2c6MQ#gid=0'))
        
    def _open_reportbug_webpage(self):
        webbrowser.open(_('https://docs.google.com/spreadsheet/viewform?formkey=dHBNOWJZMTEtNXBYaFlndTZnTEwwX0E6MQ#gid=0'))
        
    def _show_about(self):
        QtGui.QMessageBox.about(self, _('About CogStat ')+csc.versions['cogstat'], u'CogStat '+csc.versions['cogstat']+_(u'<br>Simple statistical solutions for cognitive scientists<br><br>Copyright © %s-%s Attila Krajcsi<br><br><a href = "http://sites.google.com/site/cogstatprogram/">Visit CogStat website</a>'%(2012, 2015)))

    def print_versions(self):
        """Print the versions of the software components CogStat uses."""
        # Intentionally not localized.
        self._busy_signal(True)
        
        text_output = cs_util.print_versions()
        
        self.analysis_results.append(GuiResultPackage())
        self.analysis_results[-1].add_output(csc.heading_style_begin + _('System components') + csc.heading_style_end)
        self.analysis_results[-1].add_output(text_output)
        self._print_to_output_pane()
        self._busy_signal(False)

    def closeEvent(self, event):
        # Override the close behavior, otherwise alt+F4 quits unconditionally.
        # http://stackoverflow.com/questions/1414781/prompt-on-exit-in-pyqt-application
        
        # Check if everything is saved
        tosave = True
        while self.unsaved_output and tosave:
            reply = QtGui.QMessageBox.question(self, _('Save output'),
                _('Output has unsaved results. Do you want to save it?'), QtGui.QMessageBox.Yes | 
                QtGui.QMessageBox.No, QtGui.QMessageBox.Yes)
            if reply == QtGui.QMessageBox.Yes:
                self.save_result()
            else:
                tosave=False

        reply = QtGui.QMessageBox.question(self, _('Confirm exit'), 
            _('Are you sure you want to exit the program?'), QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            QtGui.qApp.quit()
        else:
            event.ignore()

# -*- coding: utf-8 -*-

class GuiResultPackage():
    """ A class for storing a package of results.

    Result object includes:
    - self.command: Command to run (python code) - not used yet
    - self.output:
        - list of strings (html) or figures (QImages)
        - the first item is recommended to be the title line
    """

    def __init__(self):
        self.command = []
        self.output = []

    def add_command(self, command):
        self.command.append(command)

    def add_output(self, output):
        """Add output to the self.output

        :param output: item or list of items to add
        """
        if isinstance(output, list):
            for outp in output:
                self.output.append(outp)
        else:
            self.output.append(output)

def main():
    app = QtGui.QApplication(sys.argv)
    ex = StatMainWindow()
    sys.exit(app.exec_())
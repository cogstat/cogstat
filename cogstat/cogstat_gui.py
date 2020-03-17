# -*- coding: utf-8 -*-
"""
GUI for CogStat.
"""

# Splash screen
import os
import sys

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
import imp

app = QtWidgets.QApplication(sys.argv)
pixmap = QtGui.QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'CogStat splash screen.png'), 'PNG')
splash_screen = QtWidgets.QSplashScreen(pixmap)
splash_screen.show()
splash_screen.showMessage('', Qt.AlignBottom, Qt.white)  # TODO find something else to make the splash visible

# go on with regular imports, etc.
import sys
import os
import webbrowser
import gettext
import logging
import traceback
from urllib.request import urlopen
from distutils.version import LooseVersion

from . import cogstat
from . import cogstat_dialogs
from . import cogstat_config as csc
csc.versions['cogstat'] = cogstat.__version__
from . import cogstat_util as cs_util

from PyQt5 import QtCore, QtGui, QtWidgets, QtPrintSupport

cogstat.app_devicePixelRatio = app.devicePixelRatio()

cs_util.get_versions()

logging.root.setLevel(logging.INFO)

imp.reload(sys)

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext

rtl_lang = True if csc.language in ['he', 'fa', 'ar'] else False

broken_analysis = '<default>'+_('%s Oops, something went wrong, CogStat could not run the analysis. You may want to report it.') \
                  + ' ' + _('Read more about how to report an issue <a href = "%s">here</a>.') \
                  % 'https://github.com/cogstat/cogstat/wiki/Report-a-bug'

class StatMainWindow(QtWidgets.QMainWindow):
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
            QtWidgets.QMessageBox.critical(self, 'Incomplete installation', 'Install missing component(s): ' + ''.join([x+', ' for x in missing_required_components+missing_recommended_components])[:-2]+'.<br><br>'+'<a href = "https://github.com/cogstat/cogstat/wiki/Installation">Visit the installation help page</a> to see how to complete the installation.', QtWidgets.QMessageBox.Ok)
            if missing_required_components:
                sys.exit()
        
        self.analysis_results = []
        # analysis_result stores list of GuiResultPackages.
        # It will be useful when we can rerun all the previous analysis in the GUI output
        # At the moment no former results can be manipulated later

        cogstat.output_type = 'gui'  # For some GUI specific formatting

        self.check_for_update()

        # Only for testing
#        self.open_file('cogstat/sample_data/example_data.csv'); #self.compare_groups()
#        self.open_file('cogstat/test/data/diffusion.csv')
#        self.open_clipboard()
#        self.print_data()
#        self.explore_variable(['X'])
#        self.explore_variable(['a'], freq=False)
#        self.explore_variable_pair(['X', 'Y'])
#        self.pivot([u'X'], row_names=[], col_names=[], page_names=[u'CONDITION', u'TIME3'], function='N')
#        self.diffusion(error_name=['Error'], RT_name=['RT_sec'], participant_name=['Name'], condition_names=['Num1', 'Num2'])
#        self.compare_variables(['X', 'Y'])
#        self.compare_variables(['A', 'B', 'C1'])
#        self.compare_variables(['D', 'E', 'F'])
#        self.compare_variables()
#        self.compare_variables(['a', 'b', 'c1', 'd', 'e', 'f', 'g', 'h'],
#                               factors=[['factor1', 2], ['factor2', 2], ['factor3', 2]])
#        self.compare_variables([u'CONDITION', u'CONDITION2', u'CONDITION3'])
#        self.compare_groups(['slope'], ['group'],  ['slope_SE'], 25)
#        self.compare_groups(['A'], ['G', 'H'])
#        self.compare_groups(['X'], ['TIME', 'CONDITION'])
#        self.save_result_as()
#        self.save_result_as(filename='CogStat analysis result.pdf')

    def check_for_update(self):
        """Check for update, and if update is available, display a message box with the download link.

        The version number is available in a plain text file, at the appropriate web address."""
        try:
            latest_version = urlopen('http://kognitiv.elte.hu/cogstat/version', timeout=3).read().decode('utf-8')
            if LooseVersion(cogstat.__version__) < LooseVersion(latest_version):
                QtWidgets.QMessageBox.about(self, _('Update available'),
                                        _('New version is available.') + '<br><br>' +
                                        _('You can download the new version<br>from the <a href = "%s">CogStat download page</a>.')%'http://www.cogstat.org/download.html')
        except:
            print("Couldn't check for update")

    def _init_UI(self):
        self.resize(830, 600)
        self.setWindowTitle('CogStat')
        # FIXME there could be issues if the __file__ path includes unicode chars
        # e.g., see pixmap = QtGui.QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)).decode('utf-8'), u'resources', u'CogStat splash screen.png'), 'PNG')
        self.setWindowIcon(QtGui.QIcon(os.path.dirname(os.path.abspath(__file__)) + '/resources/CogStat.ico'))

        if rtl_lang:
            self.setLayoutDirection(QtCore.Qt.RightToLeft)

        # Menus and commands
        # The list will be used to construct the menus
        # Items include the icon name, the menu name, the shortcuts and the function to call, add it to the toolbar
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'icons')
        menu_commands = [
                            [_('&Data'),
                                ['/icons8-folder-48.png', _('&Open data file')+'...', _('Ctrl+O'), 'self.open_file', True],
                                ['/icons8-folder-eye-48.png', _('Open d&emo data file')+'...', _('Ctrl+E'), 'self.open_demo_file',True],
                                ['/icons8-paste-48.png', _('&Paste data'), _('Ctrl+V'), 'self.open_clipboard', True],
                                ['separator'],
                                ['/icons8-filter-48.png', _('&Filter outliers')+'...', _('Ctrl+L'), 'self.filter_outlier', True],
                                ['separator'],
                                ['/icons8-data-sheet-48.png', _('&Display data'), _('Ctrl+D'), 'self.print_data', False],
                                ['/icons8-data-sheet-check-48.png', _('Display data &briefly'), _('Ctrl+B'), 'self._print_data_brief', True],
                                ['toolbar separator']
                            ],
                            [_('&Analysis'),
                                ['/icons8-normal-distribution-histogram-48.png', _('&Explore variable')+'...', _('Ctrl+1'), 'self.explore_variable', True],
                                ['/icons8-scatter-plot-48.png', _('Explore relation of variable &pair')+'...', _('Ctrl+2'), 'self.explore_variable_pair', True],
                                ['separator'],
                                ['/icons8-pivot-table-48.png', _('Pivot &table')+'...', 'Ctrl+T', 'self.pivot', True],
                                ['/icons8-electrical-threshold-48.png', _('Behavioral data di&ffusion analysis') + '...', 'Ctrl+F', 'self.diffusion', True],
                                ['separator'],
                                ['/icons8-combo-chart-48.png', _('Compare repeated measures va&riables')+'...', 'Ctrl+R', 'self.compare_variables', True],
                                ['/icons8-bar-chart-48.png', _('Compare &groups')+'...', 'Ctrl+G', 'self.compare_groups', True],
                                ['toolbar separator']
                             ],
                            [_('&Results'),
                                ['/icons8-file-48.png', _('&Clear results'), _('Ctrl+Del'), 'self.delete_output', True],
                                ['separator'],
                                ['/icons8-zoom-in-48.png', _('&Increase text size'), _('Ctrl++'), 'self.zoom_in', True],
                                ['/icons8-zoom-out-48.png', _('&Decrease text size'), _('Ctrl+-'), 'self.zoom_out', True],
                                #['', _('Reset &zoom'), _('Ctrl+0'), _(''), 'self.zoom_reset'],
                                # TODO how can we reset to 100%?
                                ['/icons8-edit-file-48.png', _('Text is &editable'), _('Ctrl+Shift+E'), 'self.text_editable', False],
                                ['separator'],
                                ['/icons8-pdf-48.png', _('&Save results'), _('Ctrl+P'), 'self.save_result', False],
                                ['/icons8-pdf-edit-48.png', _('Save results &as')+'...', _('Ctrl+Shift+P'), 'self.save_result_as', False],
                                ['toolbar separator']
                            ],
                            [_('&CogStat'),
                                ['/icons8-help-48.png', _('&Help'), _('F1'), 'self._open_help_webpage', True],
                                ['/icons8-settings-48.png', _('&Preferences')+'...', _('Ctrl+Shift+R'), 'self._show_preferences', True],
                                ['/icons8-file-add-48.png', _('Request a &feature'), '', 'self._open_reqfeat_webpage', False],
                                ['separator'],
                                #['/icons8-toolbar-50.png', _('Show the &toolbar'), '', 'self.toolbar.toggleViewAction().trigger', False],
                                #['separator'],
                                ['/icons8-bug-48.png', _('&Report a problem'), '', 'self._open_reportbug_webpage', False],
                                ['/icons8-system-report-48.png', _('&Diagnosis information'), '', 'self.print_versions', False],
                                ['separator'],
                                ['/icons8-info-48.png', _('&About'), '', 'self._show_about', False],
                                ['separator'],
                                ['/icons8-exit-48.png', _('&Exit'), _('Ctrl+Q'), 'self.close', False]
                            ]
                        ]
        # Enable these commands only when active_data is available
        self.analysis_commands = [_('&Save data'), _('Save data &as')+'...', _('&Display data'), _('Display data &briefly'), _('&Filter outliers')+'...',
                                  _('Pivot &table')+'...', _('&Explore variable')+'...',
                                  _('Behavioral data di&ffusion analysis') + '...',
                                  _('Explore relation of variable &pair')+'...', _('Compare repeated measures va&riables')+'...', _('Compare &groups')+'...',
                                  _('&Compare groups and variables')+'...']

        # Create menus and commands, create toolbar
        self.menubar = self.menuBar()
        self.menus = []
        self.menu_commands = {}
        self.toolbar_actions = {}
        self.toolbar = self.addToolBar('General')
        for menu in menu_commands:
            self.menus.append(self.menubar.addMenu(menu[0]))
            for menu_item in menu:
                if isinstance(menu_item, str):  # Skip the name of the main menus
                    continue
                if menu_item[0] == 'separator':
                    self.menus[-1].addSeparator()
                elif menu_item[0] == 'toolbar separator':
                    self.toolbar.addSeparator()
                else:
                    self.menu_commands[menu_item[1]] = QtWidgets.QAction(QtGui.QIcon(icon_path + menu_item[0]), menu_item[1], self)
                    self.menu_commands[menu_item[1]].setShortcut(menu_item[2])
                    #self.menu_commands[menu_item[1]].setStatusTip(menu_item[3])
                    self.menu_commands[menu_item[1]].triggered.connect(eval(menu_item[3]))
                    self.menus[-1].addAction(self.menu_commands[menu_item[1]])
                    if menu_item[4]:  # if the menu item should be added to the toolbar
                        self.toolbar_actions[menu_item[1]] = QtWidgets.QAction(QtGui.QIcon(icon_path + menu_item[0]), menu_item[1] + ' (' + menu_item[2] + ')', self)
                        self.toolbar_actions[menu_item[1]].triggered.connect(eval(menu_item[3]))
                        self.toolbar.addAction(self.toolbar_actions[menu_item[1]])


        self.menus[2].actions()[4].setCheckable(True)  # _('&Text is editable') menu is a checkbox
        #self.toolbar.actions()[15].setCheckable(True)  # TODO rewrite Text is editable switches, because the menu and the toolbar works independently
        #self.menus[3].actions()[4].setCheckable(True)  # Show the toolbar menu is a checkbox
        #self.menus[3].actions()[4].setChecked(True)  # Set the default value On
            # TODO if the position of these menus are changed, then this setting will not work
        self._show_data_menus(on=False)

        # Prepare Output pane
        self.output_pane = QtWidgets.QTextBrowser()  # QTextBrowser can handle links, QTextEdit cannot
        #self.output_pane.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.output_pane.setText('<br><b>%s</b><br>%s<br>%s<br>%s<br>' %
                                 (_('Welcome to CogStat!'), _('CogStat makes statistical analysis more simple and efficient.'),
                                  _('To start working open a data file or paste your data from a spreadsheet.'),
                                  _('Find more information about CogStat on its <a href = "https://www.cogstat.org">webpage</a> or read the <a href="https://github.com/cogstat/cogstat/wiki/Quick-Start-Tutorial">quick start tutorial.</a>')))
        self.welcome_text_on = True  # Used for deleting the welcome text at the first analysis
        self.output_pane.setReadOnly(True)
        self.output_pane.setOpenExternalLinks(True)
        self.output_pane.setStyleSheet("QTextBrowser { background-color: white; }")
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
        #self.statusBar().showMessage(_('Ready'))

        self.unsaved_output = False  # Do not want to save the output with the welcome message
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
                self.toolbar_actions[menu].setEnabled(on)
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
            self._open_data(data=str(event.mimeData().text()))
        
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
        '''
        # Check R only on Linux, since Win doesn't have a working rpy at the moment
        if sys.platform in ['linux2', 'linux']:
            for module in ['r', 'rpy2', 'car']:
                if csc.versions[module] is None:
                    missing_recommended_components.append(module)
        '''

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
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.BusyCursor))
        else:
            while QtWidgets.QApplication.overrideCursor() is not None:
                # TODO if for some reason (unhandled exception) the cursor was not set back formerly,
                # then next time set it back
                # FIXME exception handling should solve this problem on the long term
                QtWidgets.QApplication.restoreOverrideCursor()
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
            if isinstance(output, str):
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
        #print(filename)
        if filename:
            self._open_data(str(filename))

    ### Data menu methods ###
    def open_demo_file(self, filename=''):
        """Open file.
        :param filename: filename with path
        """
        if filename in ['', False]:
            filename = cogstat_dialogs.open_demo_data_file()
        #print(filename)
        if filename:
            self._open_data(str(filename))

    def open_clipboard(self):
        """Open data copied to clipboard."""
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard.mimeData().hasFormat("text/plain"):
            self._open_data(str(clipboard.text()))
    
    def _open_data(self, data):
        """ Core of the import process.
        """
        self._busy_signal(True)
        try:
            self.active_data = cogstat.CogStatData(data=data)
            if self.active_data.import_source == _('Import failed'):
                QtWidgets.QMessageBox.warning(self, _('Import error'), _('Data could not be loaded.'), QtWidgets.QMessageBox.Ok)
                self._show_data_menus(False)
            else:
                self._show_data_menus()
                '''
                self.statusBar().showMessage((_('Data loaded from file: ') if self.active_data.import_source[:9] in ['text file', 'SPSS file'] else _('Data loaded from clipboard: '))
                                            + _('%s variables and %s cases.') % (len(self.active_data.data_frame.columns),
                                                                                 len(self.active_data.data_frame.index)))
                '''
                self.print_data(brief=True, display_import_message=True)
        except:
            self.analysis_results.append(GuiResultPackage())
            self.analysis_results[-1].add_command('self._open_data()')  # TODO
            self.analysis_results[-1].\
                add_output(cs_util.reformat_output('<default>' +
                                                   _('Open data. Oops, something went wrong, CogStat could not open the data. You may want to report the issue.') \
                                                   + ' ' + _('Read more about how to report an issue <a href = "%s">here</a>.') \
                                                   % 'https://github.com/cogstat/cogstat/wiki/Report-a-bug'))
            traceback.print_exc()
            self._print_to_output_pane()
        self._busy_signal(False)

    def filter_outlier(self, var_names=None):
        """Filter outliers.

        Arguments:
        var_names (list): variable names
        """
        if not var_names:
            try:
                self.dial_filter
            except:
                # Only interval variables can be used for filtering
                names = [name for name in self.active_data.data_frame.columns if (self.active_data.data_measlevs[name] in ['int', 'unk'])]
                self.dial_filter = cogstat_dialogs.filter_outlier(names=names)
            else:  # TODO is it not necessary anymore? For all dialogs
                # Only interval variables can be used for filtering
                names = [name for name in self.active_data.data_frame.columns if (self.active_data.data_measlevs[name] in ['int', 'unk'])]
                self.dial_filter.init_vars(names=names)
            if self.dial_filter.exec_():
                var_names = self.dial_filter.read_parameters()
            else:
                return
        self._busy_signal(True)
        try:
            self.analysis_results.append(GuiResultPackage())
            self.analysis_results[-1].add_command('self.filter_outlier()')  # TODO
            result = self.active_data.filter_outlier(var_names)
            self.analysis_results[-1].add_output(result)
            self._print_to_output_pane()
        except:
            self.analysis_results[-1].add_output(cs_util.reformat_output(broken_analysis % _('Filter outlier.')))
            traceback.print_exc()
            self._print_to_output_pane()
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

    ### Analysis menu methods ###

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
                self.dial_var_prop = cogstat_dialogs.explore_var_dialog(names=self.active_data.data_frame.columns)
            else:  # TODO is it not necessary anymore? For all dialogs
                self.dial_var_prop.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_var_prop.exec_():
                var_names, freq, loc_test_value = self.dial_var_prop.read_parameters()
            else:
                return
        self._busy_signal(True)
        try:
            for var_name in var_names:
                self.analysis_results.append(GuiResultPackage())
                self.analysis_results[-1].add_command('self.explore_variable()')  # TODO
                result = self.active_data.explore_variable(var_name, frequencies=freq, central_value=loc_test_value)
                self.analysis_results[-1].add_output(result)
                self._print_to_output_pane()
        except:
            self.analysis_results[-1].add_output(cs_util.reformat_output(broken_analysis % _('Explore variable.')))
            traceback.print_exc()
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
                self.dial_var_pair.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_var_pair.exec_():
                var_names = self.dial_var_pair.read_parameters()
            else:
                return
        self._busy_signal(True)
        if len(var_names) < 2:  # TODO this check should go to the appropriate dialog
            self.analysis_results.append(GuiResultPackage())
            text_result = cs_util.reformat_output('<default> %s %s'%(_('Explore variable pair.'), _('At least two variables should be set.')))
            self.analysis_results[-1].add_output(text_result)
        else:
            try:
                for x in var_names:
                    pass_diag = False
                    for y in var_names:
                        if pass_diag:
                            self.analysis_results.append(GuiResultPackage())
                            self.analysis_results[-1].add_command('self.explore_variable_pair')  # TODO
                            result_list = self.active_data.explore_variable_pair(x, y)
                            self.analysis_results[-1].add_output(result_list)
                            self._print_to_output_pane()
                        if x == y:
                            pass_diag = True
            except:
                self.analysis_results[-1].add_output(cs_util.reformat_output(broken_analysis % _('Explore variable pair.')))
                traceback.print_exc()
                self._print_to_output_pane()
        self._busy_signal(False)
            
    def pivot(self, depend_names=None, row_names=[], col_names=[], page_names=[], function='Mean'):
        """Build a pivot table.
        
        Arguments:
        depend_names (list of str): name of the dependent variable
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
            text_result = cs_util.reformat_output('<default>%s %s'%(_('Pivot table.'), _('The dependent variable and at least one grouping variable should be given.')))
        else:
            try:
                text_result = self.active_data.pivot(depend_names, row_names, col_names, page_names, function)
            except:
                text_result = cs_util.reformat_output(broken_analysis % _('Pivot table.'))
                traceback.print_exc()
        self.analysis_results[-1].add_output(text_result)
        self._print_to_output_pane()
        self._busy_signal(False)

    def diffusion(self, error_name=[], RT_name=[], participant_name=[], condition_names=[]):
        """Run a diffusion analysis on behavioral data.

        Arguments:
        RT_name, error name, participant_name (lists of str): name of the variables
        condition_names (lists of str): name of the condition(s) variables
        """
        if not RT_name:
            try:
                self.dial_diffusion
            except:
                self.dial_diffusion = cogstat_dialogs.diffusion_dialog(names=self.active_data.data_frame.columns)
            else:
                self.dial_diffusion.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_diffusion.exec_():
                error_name, RT_name, participant_name, condition_names = self.dial_diffusion.read_parameters()
            else:
                return
        self._busy_signal(True)
        self.analysis_results.append(GuiResultPackage())
        if (not RT_name) or (not error_name):  # TODO this check should go to the dialog
            text_result = cs_util.reformat_output('<default>%s %s' % (
            _('Diffusion analysis.'), _('At least the RT and the error variables should be given.')))
        else:
            try:
                text_result = self.active_data.diffusion(error_name, RT_name, participant_name, condition_names)
            except:
                text_result = cs_util.reformat_output(broken_analysis % _('Diffusion analysis.'))
                traceback.print_exc()
        self.analysis_results[-1].add_output(text_result)
        self._print_to_output_pane()
        self._busy_signal(False)

    def compare_variables(self, var_names=None, factors=[]):
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
                var_names, factors = self.dial_comp_var.read_parameters()  # TODO check if settings are appropriate
            else:
                return
        self._busy_signal(True)
        self.analysis_results.append(GuiResultPackage())
        self.analysis_results[-1].add_command('self.compare_variables()')  # TODO
        if len(factors) == 1:
            factors = []  # ignore single factor
        if len(var_names) < 2:
            text_result = cs_util.reformat_output('<default>%s %s' %
                                                  (_('Compare variables.'), _('At least two variables should be set.')))
            self.analysis_results[-1].add_output(text_result)
        else:
            try:
                result_list = self.active_data.compare_variables(var_names, factors)
                for result in result_list:  # TODO is this a list of lists? Can we remove the loop?
                    self.analysis_results[-1].add_output(result)
            except:
                self.analysis_results[-1].add_output(cs_util.reformat_output(broken_analysis % _('Compare variables.')))
                traceback.print_exc()
        self._print_to_output_pane()
        self._busy_signal(False)
        
    def compare_groups(self, var_names=None, groups=None, single_case_slope_SEs=None, single_case_slope_trial_n=None):
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
                var_names, groups, single_case_slope_SEs, single_case_slope_trial_n = self.dial_comp_grp.read_parameters()  # TODO check if settings are appropriate
            else:
                return
        self._busy_signal(True)
        if not var_names or not groups:
            self.analysis_results.append(GuiResultPackage())
            self.analysis_results[-1].add_command('self.compare_groups()')  # TODO
            text_result = cs_util.reformat_output('<default>%s %s' % (_('Compare groups.'), _('Both the dependent and the grouping variables should be set.')))
            self.analysis_results[-1].add_output(text_result)
        else:
            for var_name in var_names:
                try:
                    self.analysis_results.append(GuiResultPackage())
                    self.analysis_results[-1].add_command('self.compare_groups()')  # TODO
                    result_list = self.active_data.compare_groups(var_name, groups,  single_case_slope_SEs, single_case_slope_trial_n)
                    self.analysis_results[-1].add_output(result_list)
                    self._print_to_output_pane()
                except:
                    self.analysis_results[-1].add_output(cs_util.reformat_output(broken_analysis % _('Compare groups.')))
                    traceback.print_exc()
                    self._print_to_output_pane()
        self._busy_signal(False)

    ### Result menu methods ###
    def delete_output(self):
        reply = QtWidgets.QMessageBox.question(self, _('Clear output'),
            _('Are you sure you want to delete the output?'), QtWidgets.QMessageBox.Yes | 
            QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.output_pane.clear()
            self.analysis_results = []
            self.unsaved_output = False  # Not necessary to save the empty output

    def zoom_in(self):
        self.output_pane.zoomIn(1)

    def zoom_out(self):
        self.output_pane.zoomOut(1)

    def text_editable(self):
        self.output_pane.setReadOnly(not(self.menus[2].actions()[4].isChecked()))
        #self.output_pane.setReadOnly(not(self.toolbar.actions()[15].isChecked()))
        # TODO if the position of this menu is changed, then this function will not work
        # TODO rewrite Text is editable switches, because the menu and the toolbar works independently


    def save_result(self):
        """Save the output pane to pdf file."""
        if self.output_filename == '':
            self.save_result_as()
        else:
            pdf_printer = QtPrintSupport.QPrinter()
            pdf_printer.setOutputFormat(QtPrintSupport.QPrinter.PdfFormat)
            pdf_printer.setColorMode(QtPrintSupport.QPrinter.Color)
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
            pdf_printer = QtPrintSupport.QPrinter()
            pdf_printer.setOutputFormat(QtPrintSupport.QPrinter.PdfFormat)
            pdf_printer.setOutputFileName(self.output_filename)
            self.output_pane.print_(pdf_printer)
            self.unsaved_output = False

    ### Cogstat menu  methods ###
    def _open_help_webpage(self):
        webbrowser.open('https://github.com/cogstat/cogstat/wiki')
        
    def _show_preferences(self):
        try:
            self.dial_pref
        except:
            self.dial_pref = cogstat_dialogs.preferences_dialog()
        self.dial_pref.exec_()
    
    def _open_reqfeat_webpage(self):
        webbrowser.open('https://github.com/cogstat/cogstat/wiki/Suggest-a-new-feature')
        
    def _open_reportbug_webpage(self):
        webbrowser.open('https://github.com/cogstat/cogstat/wiki/Report-a-bug')
        
    def _show_about(self):
        QtWidgets.QMessageBox.about(self, _('About CogStat ')+csc.versions['cogstat'], 'CogStat '+csc.versions['cogstat']+('<br>%s<br><br>Copyright © %s-%s Attila Krajcsi<br><br><a href = "http://www.cogstat.org">%s</a>'%(_('Simple automatic data analysis software'), 2012, 2020, _('Visit CogStat website'))))

    def print_versions(self):
        """Print the versions of the software components CogStat uses."""
        # Intentionally not localized.
        self._busy_signal(True)
        
        text_output = cs_util.reformat_output(cs_util.print_versions())
        
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
            reply = QtWidgets.QMessageBox.question(self, _('Save output'),
                _('Output has unsaved results. Do you want to save it?'), QtWidgets.QMessageBox.Yes | 
                QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
            if reply == QtWidgets.QMessageBox.Yes:
                self.save_result()
            else:
                tosave=False

        """
        reply = QtGui.QMessageBox.question(self, _('Confirm exit'), 
            _('Are you sure you want to exit the program?'), QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            QtGui.qApp.quit()
        else:
            event.ignore()
        """

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
    splash_screen.close()
    ex = StatMainWindow()
    sys.exit(app.exec_())

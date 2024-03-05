# -*- coding: utf-8 -*-
"""
GUI for CogStat.

The GUI includes
- a menu bar
- a toolbar (selected items from menu bar)
- the data in a QTableView
- the result pane

"""

# Splash screen
import os
import sys

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt

app = QtWidgets.QApplication(sys.argv)
#app.setAttribute(Qt.AA_UseHighDpiPixmaps)  # TODO we don't need this anymore?
pixmap = QtGui.QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources',
                                    'CogStat splash screen.png'), 'PNG')
splash_screen = QtWidgets.QSplashScreen(pixmap)
splash_screen.show()
splash_screen.showMessage('', Qt.AlignmentFlag.AlignBottom, Qt.GlobalColor.white)  # TODO find something else to make the splash visible

# go on with regular imports, etc.
import random
import importlib
import base64
from distutils.version import LooseVersion
import gettext
import io
import logging
import matplotlib.figure
import matplotlib.pyplot as plt
import os
import sys
import time
import traceback
from urllib.request import urlopen
import webbrowser
import pandas as pd
from operator import attrgetter

from PyQt6 import QtCore, QtGui, QtWidgets

from . import cogstat
from . import cogstat_dialogs
from . import cogstat_config as csc
from . import cogstat_util as cs_util

csc.app_devicePixelRatio = app.devicePixelRatio()

logging.root.setLevel(logging.INFO)

importlib.reload(sys)  # TODO why do we need this?

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext

rtl_lang = True if csc.language in ['he', 'fa', 'ar'] else False

broken_analysis = '<cs_h1>%s</cs_h1>' + \
                  _('Oops, something went wrong, CogStat could not run the analysis. You may want to report it.') \
                  + ' ' + _('Read more about how to report an issue <a href = "%s">here</a>.') \
                  % 'https://doc.cogstat.org/Report-a-bug'


class StatMainWindow(QtWidgets.QMainWindow):
    """
    CogStat GUI.
    """
    def __init__(self):
        super(StatMainWindow, self).__init__()  # TOD do we need super()?
        self._init_UI()

        self.unsaved_output = False  # Do not want to save the output with the welcome message
        self.output_filename = ''
        self.last_file_dir = os.path.dirname(csc.__file__)
        self.last_demo_file_dir = os.path.join(os.path.dirname(csc.__file__), 'demo_data')

        self.watched_file = QtCore.QFileSystemWatcher()
        self.watched_file.fileChanged.connect(self.reload_file)

        # Check if all required components are installed
        # TODO Maybe all these checking can be removed
        missing_required_components, missing_recommended_components = self._check_installed_components()
        if missing_required_components or missing_recommended_components:
            QtWidgets.QMessageBox.critical(self, 'Incomplete installation', 'Install missing component(s): ' +
                                           ''.join([x+', ' for x in
                                                    missing_required_components+missing_recommended_components])[:-2] +
                                           '.<br><br>' + '<a href = "https://doc.cogstat.org/'
                                                         'Installation">Visit the installation help page</a> to see how '
                                                         'to complete the installation.', QtWidgets.QMessageBox.StandardButton.Ok)
            if missing_required_components:
                sys.exit()
        
        self.analysis_results = []  # analysis_result stores list of GuiResultPackages objects

        csc.output_type = 'gui'  # For some GUI specific formatting

        self.check_for_update()

        # Only for testing
#        self.open_file('cogstat/test/dummy data/example_data.csv'); #self.compare_groups()
#        self.open_file('cogstat/test/data/VA_test.csv')
#        self.open_file('cogstat/test/data/test2.csv')
#        self.open_file('cogstat/test/data/diffusion.csv')
#        self.open_clipboard()
#        self.print_data()
#        self.filter_outlier(['before', 'after'], True)
#        self.explore_variable(['X'])
#        self.explore_variable(['a'], freq=False)
#        self.explore_variable_pair(['X', 'Y'])
#        self.regression(['a'], 'b')
#        self.regression(['b', 'f', 'g'], 'a')
#        self.pivot('X', row_names=[], col_names=[], page_names=['CONDITION', 'TIME3'], function='N')
#        self.diffusion(error_name='Error', RT_name='RT_sec', participant_name='Name', condition_names=['Num1', 'Num2'])
#        self.compare_variables(['X', 'Y'])
#        self.compare_variables(['a', 'e', 'g'])
#        self.compare_variables(['D', 'E', 'F'])
#        self.compare_variables()
#        self.compare_variables(['a', 'b'], factors=[['factor', 2]], display_factors=[['factor'], []])
#        self.compare_variables(['a', 'g', 'b', 'h', 'e', 'f'],
#                               factors=[['factor1', 2], ['factor2', 3]],
#                               display_factors=[['factor1'], ['factor2']])
#        self.compare_variables(['CONDITION', 'CONDITION2', 'CONDITION3'])
#        self.compare_groups(['slope'], ['group'],  ['slope_SE'], 25)
#        self.compare_groups(['b'], groups=['i', 'j']),
#        self.compare_groups(['b'], groups=['i', 'j', 'k'], display_groups=[['k', 'i'], ['j'], []]),
#        self.compare_groups(['b'], groups=['i', 'j'], display_groups=[['i'], ['j'], []])
#        self.compare_groups(['X'], ['TIME', 'CONDITION'])
#        self.compare_groups(['dep_nom'], ['g0', 'g1', 'g2', 'g3'])
#        self.compare_variables_groups(var_names=['a', 'e', 'f'], groups=['i'], display_factors=[['i', _('Unnamed factor')], [], []])
#        self.reliability_internal(var_names=['a', 'e', 'f', 'g'])
#        self.reliability_interrater(var_names=['a', 'e', 'f', 'g'])
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
                                            _('You can download the new version<br>from the <a href = "%s">CogStat '
                                              'download page</a>.') % 'http://www.cogstat.org/download.html')
        except:
            print("Couldn't check for update")

    def _init_UI(self):
        self.resize(1250, 1000)  # for the height we assume that a full HD screen is available; if the value is larger
                                # than the available screen size, the window will be the max height
        #print(QtWidgets.QDesktopWidget().screenGeometry(-1).height())  # height of the actual screen
        self.setWindowTitle('CogStat')
        # FIXME there could be issues if the __file__ path includes unicode chars
        # e.g., see pixmap = QtGui.QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)).decode('utf-8'),
        # u'resources', u'CogStat splash screen.png'), 'PNG')
        self.setWindowIcon(QtGui.QIcon(os.path.dirname(os.path.abspath(__file__)) + '/resources/CogStat.ico'))

        if rtl_lang:
            self.setLayoutDirection(QtCore.Qt.LayoutDirection.RightToLeft)

        # Menus and commands
        # The list will be used to construct the menus
        # Items include the icon name, the menu name, the shortcuts, the function to call, whether to add it to the
        # toolbar, whether it is active only when data is loaded
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'icons')
        menu_commands = [
                            [_('&Data'),
                                ['/icons8-folder.svg', _('&Open data file')+'...', _('Ctrl+O'), 'self.open_file',
                                 True, False],
                                ['/icons8-folder-eye.svg', _('Open d&emo data file')+'...', _('Ctrl+E'),
                                 'self.open_demo_file', True, False],
                                ['/icons8-folder-reload.svg', _('Re&load actual data file'), _('Ctrl+Shift+L'),
                                 'self.reload_file', True, True],
                                ['/icons8-folder-link.svg', _('Reload data file when &changed'), _('Ctrl+Shift+Alt+L'),
                                 'self.watch_file', False, True],
                                ['/icons8-paste.svg', _('&Paste data'), _('Ctrl+V'), 'self.open_clipboard', True,
                                 False],
                                ['separator'],
                                ['/icons8-filter.svg', _('&Filter outliers')+'...', _('Ctrl+L'), 'self.filter_outlier',
                                 True, True],
                                ['separator'],
                                ['/icons8-data-sheet-check.svg', _('Display &data briefly'), _('Ctrl+D'),
                                 'self._print_data_brief', False, True],
                                ['toolbar separator']
                            ],
                            [_('&Analysis'),
                                ['/icons8-normal-distribution-histogram.svg', _('Explore &variable')+'...',
                                 _('Ctrl+1'), 'self.explore_variable', True, True],
                                ['/icons8-scatter-plot.svg', _('Explore relation of variable &pair')+'...',
                                 _('Ctrl+2'), 'self.explore_variable_pair', True, True],
                                ['/icons8-heat-map-100.png', _('Explore &relation of variables')+'...',
                                 _('Ctrl+R'), 'self.regression', True, True],
                                ['/icons8-combo-chart.svg', _('Compare repeated &measures variables')+'...',
                                 'Ctrl+P', 'self.compare_variables', True, True],
                                ['/icons8-bar-chart.svg', _('Compare &groups')+'...', 'Ctrl+G',
                                 'self.compare_groups', True, True],
                                ['/icons8-combo-chart-100.png', _('&Compare repeated measures variables and groups')+'...',
                                 'Ctrl+M', 'self.compare_variables_groups', True, True],
                                ['separator'],
                                ['toolbar separator'],
                                ['/icons8-goal-100.png', _('Internal consistenc&y reliability analysis')+'...',
                                 'Ctrl+Shift+C', 'self.reliability_internal', True, True],
                                ['/icons8-collect-100.png', _('&Interrater reliability analysis')+'...',
                                 'Ctrl+Shift+I', 'self.reliability_interrater', True, True],
                                ['separator'],
                                ['toolbar separator'],
                                ['/icons8-pivot-table.svg', _('Pivot &table')+'...', 'Ctrl+T', 'self.pivot', True,
                                 True],
                                ['/icons8-electrical-threshold.svg', _('Behavioral data &diffusion analysis') +
                                 '...', 'Ctrl+Shift+D', 'self.diffusion', True, True],
                                ['separator'],
                                ['toolbar separator'],
                                ['/icons8-reboot-100.png', _('Rerun all &analyses'), 'Ctrl+Shift+R',
                                 'self.rerun_analyses', True, True],
                                ['/icons8-reboot-link-100.png', _('Rerun all analyses when &file reloaded'),
                                 'Ctrl+Shift+Alt+R', 'self._pass', False, True],
                                ['toolbar separator']
                             ],
                            [_('&Results'),
                                ['/icons8-file.svg', _('&Clear results'), _('Ctrl+Del'), 'self.delete_output', True,
                                 False],
                                ['/icons8-search.svg', _('&Find text...'), _('Ctrl+F'), 'self.find_text', True, False],
                                ['separator'],
                                ['/icons8-zoom-in.svg', _('&Increase text size'), _('Ctrl++'), 'self.zoom_in', True,
                                 False],
                                ['/icons8-zoom-out.svg', _('&Decrease text size'), _('Ctrl+-'), 'self.zoom_out',
                                 True, False],
                                #['', _('Reset &zoom'), _('Ctrl+0'), _(''), 'self.zoom_reset'],
                                # TODO how can we reset to 100%?
                                ['/icons8-edit-file.svg', _('Text is &editable'), _('Ctrl+Shift+E'),
                                 'self.text_editable', False, False],
                                ['separator'],
                                ['/icons8-document.svg', _('&Save results'), _('Ctrl+S'), 'self.save_result', False, False],
                                ['/icons8-document-plus.svg', _('Save results &as')+'...', _('Ctrl+Shift+S'),
                                 'self.save_result_as', False, False],
                                ['toolbar separator']
                            ],
                            [_('&CogStat'),
                                ['/icons8-help.svg', _('&Help'), _('F1'), 'self._open_help_webpage', True, False],
                                ['/icons8-settings.svg', _('&Preferences')+'...', _('Ctrl+Shift+P'),
                                 'self._show_preferences', True, False],
                                ['/icons8-file-add.svg', _('Request a &feature'), '', 'self._open_reqfeat_webpage',
                                 False, False],
                                ['separator'],
                                #['/icons8-toolbar.svg', _('Show the &toolbar'), '',
                                # 'self.toolbar.toggleViewAction().trigger', False],
                                #['separator'],
                                ['/icons8-bug.svg', _('&Report a problem'), '', 'self._open_reportbug_webpage',
                                 False, False],
                                ['/icons8-system-report.svg', _('&Diagnosis information'), '', 'self.print_versions',
                                 False, False],
                                ['separator'],
                                ['/icons8-info.svg', _('&About'), '', 'self._show_about', False, False],
                                ['separator'],
                                ['/icons8-exit.svg', _('&Exit'), _('Ctrl+Q'), 'self.close', False, False]
                            ]
                        ]

        # Create menus and commands, create toolbar
        self.menubar = self.menuBar()
        self.menus = []
        self.menu_commands = {}
        self.toolbar_actions = {}
        self.toolbar = self.addToolBar('General')
        self.active_menu_with_data = []  # Enable these commands only when active_data is available
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
                    self.menu_commands[menu_item[1]] = QtGui.QAction(QtGui.QIcon(icon_path + menu_item[0]),
                                                                         menu_item[1], self)
                    self.menu_commands[menu_item[1]].setShortcut(menu_item[2])
                    self.menu_commands[menu_item[1]].triggered.connect(eval(menu_item[3]))
                    self.menus[-1].addAction(self.menu_commands[menu_item[1]])
                    if menu_item[4]:  # if the menu item should be added to the toolbar
                        self.toolbar_actions[menu_item[1]] = QtGui.QAction(QtGui.QIcon(icon_path + menu_item[0]),
                                                                               menu_item[1] + ' (' + menu_item[2] + ')',
                                                                               self)
                        self.toolbar_actions[menu_item[1]].triggered.connect(eval(menu_item[3]))
                        self.toolbar.addAction(self.toolbar_actions[menu_item[1]])
                    if menu_item[5]:  # the menu should be enabled only when data are loaded
                        self.active_menu_with_data.append(menu_item[1])

        self.menus[0].actions()[3].setCheckable(True)  # _('Reload data file when &changed') menu is
                                                       # a checkbox
        self.menus[1].actions()[14].setCheckable(True)  # _('Rerun all analyses when &file reloaded') menu is a checkbox
        self.menus[2].actions()[5].setCheckable(True)  # _('&Text is editable') menu is a checkbox
                                                       # see also text_editable()
        #self.toolbar.actions()[15].setCheckable(True)  # TODO rewrite Text is editable switches, because the menu and
                                                        # the toolbar works independently
        #self.menus[3].actions()[4].setCheckable(True)  # Show the toolbar menu is a checkbox
        #self.menus[3].actions()[4].setChecked(True)  # Set the default value On
            # TODO if the position of these menus are changed, then this setting will not work
        self._show_data_menus(on=False)

        # Initialize dialogs
        self.dial_var_prop = cogstat_dialogs.explore_var_dialog()
        self.dial_filter = cogstat_dialogs.filter_outlier()
        self.dial_var_pair = cogstat_dialogs.explore_var_pairs_dialog()
        self.dial_regression = cogstat_dialogs.regression_dialog()
        self.dial_pivot = cogstat_dialogs.pivot_dialog()
        self.dial_diffusion = cogstat_dialogs.diffusion_dialog()
        self.dial_comp_var = cogstat_dialogs.compare_vars_dialog()
        self.dial_comp_grp = cogstat_dialogs.compare_groups_dialog()
        self.dial_comp_var_groups = cogstat_dialogs.compare_vars_groups_dialog()
        self.dial_rel_internal = cogstat_dialogs.reliability_internal_dialog()
        self.dial_rel_interrater = cogstat_dialogs.reliability_interrater_dialog()
        self.dial_pref = cogstat_dialogs.preferences_dialog()

        # Prepare result and data panes
        self.centralwidget = QtWidgets.QWidget()
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.table_view = QtWidgets.QTableView(self.splitter)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.table_view.horizontalHeader().setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.result_pane = QtWidgets.QTextBrowser(self.splitter)  # QTextBrowser can handle links, QTextEdit cannot
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 4)
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.addWidget(self.splitter)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        #self.output_pane.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)

        # Currently, it doesn't make sense to use a loop here, but we keep it, until we decide how to implement the ToC
        for pane in [self.result_pane]:
            # some html styles are modified for the GUI version (but not for the Jupyter Notebook version)
            # Because qt does not support table borders, use padding to have a more reviewable table
            pane.document().setDefaultStyleSheet('body {color:black;} '
                                                 'h2 {color:%s;} h3 {color:%s;} '
                                                 'h4 {color:%s;} h5 {color:%s; font-size: medium;} '
                                                 'th {font-weight:normal; white-space:nowrap; '
                                                 'padding-right: 5px; padding-left: 5px} '
                                                 'td {white-space:nowrap; padding-right: 5px; padding-left: 5px}' %
                                                 (cs_util.change_color(csc.mpl_theme_color, brightness=1.1),
                                                  cs_util.change_color(csc.mpl_theme_color, brightness=1.0),
                                                  cs_util.change_color(csc.mpl_theme_color, brightness=0.8),
                                                  cs_util.change_color(csc.mpl_theme_color, brightness=0.4)))
            pane.setReadOnly(True)
            pane.setOpenExternalLinks(True)
            pane.setStyleSheet("QTextBrowser { background-color: white; }")
                # Some styles use non-white background (e.g. Linux Mint 17 Mate uses gray)
            # Set default font
            #print pane.currentFont().toString()
            # http://stackoverflow.com/questions/2475750/using-qt-css-to-set-own-q-propertyqfont
            font = QtGui.QFont()
            font.setFamily(csc.default_font)
            font.setPointSizeF(csc.default_font_size)
            pane.setFont(font)
            #print pane.currentFont().toString()

        output_welcome_message = '%s%s%s%s<br>%s<br>%s<br>' % \
                                 ('<cs_h1>', _('Welcome to CogStat!'), '</cs_h1>',
                                 _('CogStat makes statistical analysis more simple and efficient.'),
                                 _('To start working open a data file or paste your data from a spreadsheet.'),
                                 _('Find more information about CogStat on its <a href = "https://www.cogstat.org">webpage</a> or read the <a href="https://doc.cogstat.org/Quick-Start-Tutorial">quick start tutorial.</a>'))
        data_welcome_message = '%s%s%s%s<br>' % \
                               ('<cs_h1>', _('Data view'), '</cs_h1>',
                               _('To start working open a data file or paste your data from a spreadsheet.'))
        self.result_pane.setText(cs_util.convert_output({'info': output_welcome_message})['info'])
        # We add these extra properties to track if the welcome message  is still on
        self.result_pane.welcome_message_on = True

        self.setCentralWidget(self.centralwidget)
        self.setAcceptDrops(True)

        self.show()

    def _show_data_menus(self, on=True):
        """Enable or disable data handling menus depending on whether data is loaded.

        Parameters
        ----------
        on : bool
            True to enable menus
            False to disable
        Returns
        -------

        """
        for menu in self.active_menu_with_data:
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
            self.open_file(path=event.mimeData().urls()[0].toString(options=QtCore.QUrl.UrlFormattingOption.PreferLocalFile))
        elif event.mimeData().hasFormat("text/plain"):
            # print 'Dropped Text: ', event.mimeData().text()
            self._open_data(data=str(event.mimeData().text()))
        

    def _pass(self):
        pass

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
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.BusyCursor))
        else:
            while QtWidgets.QApplication.overrideCursor() is not None:
                # TODO if for some reason (unhandled exception) the cursor was not set back formerly,
                # then next time set it back
                # FIXME exception handling should solve this problem on the long term
                QtWidgets.QApplication.restoreOverrideCursor()
            #QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        
    def _print_to_pane(self, pane=None, output_dict=None, scroll_to_analysis=True):
        """Print a GuiResultPackage to the output or data pane.

        The pane should have a pane.welcome_message_on property.

        Parameters
        ----------
        pane : QtWidgets.QTextBrowser object
            The pane the message should be printed to.
        output_dict : dict including str (html) or matplotlib figure or pandas dataframe styler or list of these in the
                      values
        scroll_to_analysis : bool
            Should the pane scroll to the beginning of the analysis?

        Returns
        -------

        """

        if pane.welcome_message_on:
            pane.clear()
            #pane.setHtml(cs_util.convert_output(['<cs_h1>&nbsp;</cs_h1>'])[0])
            pane.welcome_message_on = False
        #pane.append('<h2>test2</h2>testt<h3>test3</h3>testt<br>testpbr')
        #pane.output_pane.append('<h2>test2</h2>testt<h3>test3</h3>testt<br>testpbr')
        #print(pane.toHtml())

        # anchor and scrolling is relevant only in the result pane
        # TODO we may remove it for other panes
        anchor = str(random.random())
        pane.append('<a id="%s">&nbsp;</a>' % anchor)  # nbsp is needed otherwise qt will ignore the string

        # convert dict values to list
        output_list = []
        for value in output_dict.values():
            if isinstance(value, list):
                output_list.extend([item for item in value])
            else:
                output_list.extend([value])

        for output in output_list:
            if isinstance(output, str):
                pane.append(output)  # insertHtml() messes up the html doc,
                                                 # check it with value.toHtml()
            elif isinstance(output, matplotlib.figure.Figure):
                if csc.image_format == 'png':
                    chart_buffer = io.BytesIO()
                    output.savefig(chart_buffer, format='png')  # TODO dpi= and modify html width to keep the original image size
                    chart_buffer.seek(0)
                    html_img = '<img src="data:image/png;base64,{0}">'.\
                        format(base64.b64encode(chart_buffer.read()).decode())  # TODO width=...gui.physicaldpi * 6.4
                    chart_buffer.close()
                    # print('PNG size', sys.getsizeof(html_img))
                elif csc.image_format == 'svg':
                    """
                    # matplotlib-based method
                    chart_buffer = io.BytesIO()
                    # in savefig(), for 'svg' format, the dpi parameter is ignored, 72 is used instead
                    output.savefig(chart_buffer, format='svg')
                    chart_buffer.seek(0)
                    # width="800" height="600" won't help because the image is blurry
                    html_img = '<img src="data:image/svg-xml;base64,{0}">'.\
                        format(base64.b64encode(chart_buffer.read()).decode())
                    chart_buffer.close()
                    #print('SVG matplotlib size', sys.getsizeof(html_img))
                    #"""
                    # svgutils-based method
                    import svgutils.transform
                    svg_fig = svgutils.transform.from_mpl(output)
                    svg_size = output.get_size_inches() * output.dpi
                    svg_fig.set_size((str(svg_size[0]), str(svg_size[1])))
                    html_img = '<img src="data:image/svg-xml;base64,{0}">'.\
                        format(base64.b64encode(svg_fig.to_str()).decode())
                    #print('SVG svgutils size', sys.getsizeof(html_img))
                pane.append(html_img)
            elif isinstance(output, pd.io.formats.style.Styler):
                # Styler may have pipe_func attribute, which should be a function, which will be run right before
                # converting the Styler to html
                if hasattr(output, 'pipe_func'):
                    pipe_func = output.pipe_func
                else:
                    pipe_func = lambda x: x
                # 1. make row headers left aligned
                # 2. headers use None formatter resulting in a format used in DataFrame.to_html() for floats
                # 3. call pipe_func if availabe
                # 4. convert to html, and remove \n-s
                pane.append(output.set_table_styles([{'selector': 'th.row_heading', 'props': 'text-align: left;'}]).
                            format_index(formatter='{}', axis=0).format_index(formatter='{}', axis=1).
                            pipe(pipe_func).
                            to_html().replace('\n', ''))
            else:
                logging.error('Unknown output type: %s' % type(output))
        self.unsaved_output = True
        if scroll_to_analysis:
            pane.scrollToAnchor(anchor)
        plt.close('all')  # free memory after everything is displayed
        #pane.moveCursor(QtGui.QTextCursor.End)

    def _display_data(self, reset=False):
        """ Display the actual data in the tableview.
        Show the variable names, the measurement levels and the data.
        Show the filtered cases.

        Parameters
        ----------
        reset : bool
            Should the tableview be cleared?

        """
        # When reset is True, reset the view; when reset is False, initialize
        self.table_view.setModel(None)
        if not reset:
            # Make a copy of the original data so that both filtered and included cases can be displayed.
            data_to_display = self.active_data.orig_data_frame.copy()
            # This new column is used for formatting the rows in the tableview.
            # We use a column name that is not likely to be used by the users.
            # By default, all cases are excluded.
            data_to_display['cogstat_filtered_cases'] = 1
            # Modify the included cases.
            data_to_display.loc[self.active_data.data_frame.index, 'cogstat_filtered_cases'] = 0
            # Start row numbers from 1, instead of 0, if it starts with 0 (as in default index).
            # Otherwise, keep the original index.
            if data_to_display.index[0] == 0:
                data_to_display.index = data_to_display.index + 1
            # Add the variable type and measurement level to the dataframe.
            dtype_convert = {'int32': 'num', 'int64': 'num', 'float32': 'num', 'float64': 'num',
                             'object': 'str', 'string': 'str', 'category': 'str', 'datetime64[ns]': 'str'}
            data_to_display = pd.concat(
               [pd.DataFrame([[dtype_convert[str(self.active_data.data_frame[name].dtype).lower()] for name in
                               self.active_data.data_frame.columns],
                              [self.active_data.data_measlevs[name] for name in self.active_data.data_frame.columns]],
                             columns=self.active_data.data_frame.columns,
                             index=[_('Type'), _('Level')]), data_to_display])
            # Prepare table view
            model = PandasModel(data_to_display)
            self.table_view.setModel(model)
            # Hide the filtering column
            self.table_view.setColumnHidden(model.columnCount() - 1, True)
            # For automatic data file reload, provide visual feedback
            if self.menus[0].actions()[3].isChecked():
                self.table_view.setGridStyle(QtCore.Qt.PenStyle.NoPen)
                self.table_view.repaint()
                time.sleep(0.2)
                self.table_view.setGridStyle(QtCore.Qt.PenStyle.SolidLine)
            self.table_view.show()  # TODO do we need this?

    def _run_analysis(self, title, function_name, parameters=None, scroll_to_analysis=True, store_in_results=True):
        """Run an analysis by calling the function with the parameters.
        If it fails, provide an error message with the title as heading.

        Parameters
        ----------
        title : str
            Used for the heading of the error message
        function_name : str
            Name of the function or method that implements the analysis
        parameters : dict
            Optional parameters
        scroll_to_analysis : bool
            Should the pane scroll to the beginning of the analysis?
        store_in_results : bool
            Should the command and the output stored in analysis_results (for later reanalysis)?

        Returns
        -------
        bool
            Whether the analysis could run without an exception

        """
        from . import cogstat_util as cs_util  # import cs_util so that it is available in locals()

        #print(title, function_name, parameters)

        self._busy_signal(True)
        if store_in_results:
            self.analysis_results.append(GuiResultPackage())
            self.analysis_results[-1].add_command([title, function_name, parameters])
        result = None
        successful_run = True
        try:
            # split the function name into a first part and the rest of it
            function_highest_level, function_rest_levels = function_name.split('.', 1)
            if parameters is None:  # no parameters are stored
                result = attrgetter(function_rest_levels)(locals()[function_highest_level])()
            else:  # there are parameters stored in a dict
                result = attrgetter(function_rest_levels)(locals()[function_highest_level])(**parameters)
            if store_in_results:
                self.analysis_results[-1].add_output(result)
                self._print_to_pane(pane=self.result_pane, output_dict=self.analysis_results[-1].output,
                                    scroll_to_analysis=scroll_to_analysis)
        except Exception as e:
            if not store_in_results:  # if analyses was not intended to be stored, but exception happened, and the
                                      # error message should be displayed
                self.analysis_results.append(GuiResultPackage())
                self.analysis_results[-1].add_command([title, function_name, parameters])
            if csc.detailed_error_message:
                error_message = '\n' + '<cs_warning>' + _('Detailed error message') + \
                                ' (%s):</cs_warning>\n' % 'you can turn this off in Preferences' + traceback.format_exc()
            else:
                error_message = ''
            self.analysis_results[-1].add_output(cs_util.convert_output({'warning': (broken_analysis % title) +
                                                                                    error_message}))
            if title == _('Data'):  # Data import-specific error message
                data = parameters['data']
                try:
                    file_content = _('Data file content') + ':<br>' + open(data, 'r').read()[:1000].replace(
                        '\n', '<br>') if os.path.exists(data) else ''
                except:
                    file_content = ''
                self.analysis_results[-1].add_output(cs_util.convert_output(
                    {'warning 2': '<cs_warning>' + _('Data to be imported') +
                                ':</cs_warning><br>%s<br>%s' % (data, file_content)}))
                self._display_data(reset=True)
            traceback.print_exc()
            successful_run = False
            self._print_to_pane(pane=self.result_pane, output_dict=self.analysis_results[-1].output,
                                scroll_to_analysis=scroll_to_analysis)
        self._busy_signal(False)
        return successful_run


    ### Data menu methods ###
    def open_file(self, path=''):
        """Open data file.

        Parameters
        ----------
        path : str
            Path of the file.
        """
        if path in ['', False]:
            path = cogstat_dialogs.open_data_file(self.last_file_dir)
        if path:
            self.last_file_dir = os.path.dirname(path)
            self._open_data(path)

    def open_demo_file(self, path=''):
        """Open demo data file.

        Parameters
        ----------
        path : str
            Path of the demo file.
        """
        if path in ['', False]:
            # If the last directory was outside the demo directory, offer the demo root directory again
            if self.last_demo_file_dir.find(os.path.join(os.path.dirname(csc.__file__), 'demo_data')) != 0:
                    self.last_demo_file_dir = os.path.join(os.path.dirname(csc.__file__), 'demo_data')
            path = cogstat_dialogs.open_demo_data_file(self.last_demo_file_dir)
        if path:
            self.last_demo_file_dir = os.path.normpath(os.path.dirname(path))
            self._open_data(path)

    def watch_file(self):
        """Add and remove files to/from the file watcher.

        Action depends on whether the current data comes from a file and whether automatic file reload is on.

        Call this, when (a) a data is loaded or reloaded (since this may mean that the file is changed and in some
        systems/settings, the file is created; therefore, the file is removed from the watcher) or (b) automatic data
        reload setting is changed.

        Returns
        -------
        Changes the files in the file watcher
        """

        # If the current data comes from a file and if the automatic data file reload is on, watch the file.
        # Otherwise, don't watch anything.
        if self.active_data.import_source[1] and self.menus[0].actions()[3].isChecked():
            self.watched_file.addPath(self.active_data.import_source[1])
        else:
            # unless the list of the watched files is already empty, remove the files
            if self.watched_file.files():
                self.watched_file.removePaths(self.watched_file.files())


    def reload_file(self):
        """Reload data file."""

        store_in_results = not self.menus[0].actions()[3].isChecked()
        successful = self._run_analysis(title=_('Reload data'), function_name='self.active_data.reload_data',
                                        store_in_results=store_in_results)
        if successful:
            self._display_data()
        else:
            self._display_data(reset=True)
        self.watch_file()

        if self.menus[1].actions()[14].isChecked():  # rerun all analyses when file is reloaded
            self.rerun_analyses()

    def open_clipboard(self):
        """Open data copied to clipboard."""
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard.mimeData().hasFormat("text/plain"):
            self._open_data(str(clipboard.text()))
    
    def _open_data_core(self, data):
        """ Core of the import process.
        """
        self.active_data = cogstat.CogStatData(data=data)
        if self.active_data.import_source[0] == _('Import failed'):
            self._show_data_menus(False)
        else:
            self._show_data_menus()

            # Make Reload menu available, if the imported data is coming from a file
            if self.active_data.import_source[1]:
                self.menu_commands[_('Re&load actual data file')].setEnabled(True)
                self.toolbar_actions[_('Re&load actual data file')].setEnabled(True)
                self.menu_commands[_('Reload data file when &changed')].setEnabled(True)
            else:
                self.menu_commands[_('Re&load actual data file')].setEnabled(False)
                self.toolbar_actions[_('Re&load actual data file')].setEnabled(False)
                self.menu_commands[_('Reload data file when &changed')].setEnabled(False)
        self._display_data()
        self.watch_file()
        return cs_util.convert_output({'imported data': self.active_data.import_message})

    def _open_data(self, data):
        """Open all kind of data. It calls _open_data_core() that opens the data and returns the output message.
        With the returned method, this can be used with the _run_analysis() method.

        Parameters
        ----------
        data :

        Returns
        -------

        """
        self._run_analysis(title=_('Data'), function_name='self._open_data_core', parameters={'data': data})

    def filter_outlier(self, var_names=None, multivariate_outliers=False):
        """Filter outliers.

        Parameters
        ----------
        var_names : list of str
            variable names
        multivariate_outliers : bool

        Returns
        -------

        """
        if not var_names:
            # Only interval variables can be used for filtering
            names = [name for name in self.active_data.data_frame.columns if (self.active_data.data_measlevs[name]
                                                                              in ['int', 'unk'])]
            self.dial_filter.init_vars(names=names)
            if self.dial_filter.exec():
                var_names, multivariate_outliers = self.dial_filter.read_parameters()
            else:
                return
        if self._run_analysis(title=_('Filter outliers'), function_name='self.active_data.filter_outlier',
                              parameters={'var_names': var_names,
                                          'mode': 'mahalanobis' if multivariate_outliers else '2.5mad'}):
            self._display_data()

    def print_data(self, brief=False):
        """Print the current data to the output.

        Parameters
        ----------
        brief : bool
            print only the first 10 rows
        Returns
        -------

        """
        self._run_analysis(title=_('Data'), function_name='self.active_data.print_data', parameters={'brief': brief})

    def _print_data_brief(self):
        """Print the data briefly to GUI output pane
        """
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
            self.dial_var_prop.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_var_prop.exec():
                var_names, freq, loc_test_value = self.dial_var_prop.read_parameters()
                if not var_names:
                    var_names = ['']  # error message for missing variable come from the explore_variable() method
            else:
                return
        for i, var_name in enumerate(var_names):
            self._run_analysis(title=_('Explore variable'), function_name='self.active_data.explore_variable',
                               parameters={'var_name': var_name, 'frequencies': freq, 'central_value': loc_test_value},
                               scroll_to_analysis=not i)

    def explore_variable_pair(self, var_names=None, xlims=[None, None], ylims=[None, None]):
        """Explore variable pairs.

        Parameters
        ----------
        var_names : list of str
            Names of the variables
        xlims : list of floats
            Minimum and maximum value of the x-axis
        ylims : list of floats
            Minimum and maximum value of the y-axis

        Returns
        -------

        """
        if not var_names:
            self.dial_var_pair.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_var_pair.exec():
                var_names, xlims, ylims = self.dial_var_pair.read_parameters()
            else:
                return
        if len(var_names) < 2:
            var_names = (var_names + [None, None])[:2]  # regression() method handles the missing variables
        scroll_to_analysis = True
        for x in var_names:
            pass_diag = False
            for y in var_names:
                if pass_diag:
                    self._run_analysis(title=_('Explore relation of variable pair'),
                                       function_name='self.active_data.regression',
                                       parameters={'predictors': [x], 'predicted': y, 'xlims': xlims, 'ylims': ylims},
                                       scroll_to_analysis=scroll_to_analysis)
                    scroll_to_analysis = False
                if x == y:
                    pass_diag = True
            if x is None:  # with [None, None] var_names regression() is called only once
                break

    def regression(self, predictors=[], predicted=None, xlims=[None, None], ylims=[None, None]):
        """Regression analysis.

        Parameters
        ----------
        predicted : str
            Name of the outcome variable
        predictors : list of str
            Name of the regressors
        xlims : list of floats
            Minimum and maximum value of the x-axis
        ylims : list of floats
            Minimum and maximum value of the y-axis
        Returns
        -------

        """
        if not predicted:
            self.dial_regression.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_regression.exec():
                predicted, predictors, xlims, ylims = self.dial_regression.read_parameters()
                if predicted == []:  # regression() method handles missing parameters
                    predicted = [None]
                predicted = predicted[0]  # currently, GUI predicted is a list, but it should be a string
            else:
                return
        self._run_analysis(title=_('Explore relation of variables'), function_name='self.active_data.regression',
                           parameters={'predictors': predictors, 'predicted': predicted,
                                       'xlims': xlims, 'ylims': ylims})


    def pivot(self, depend_name=None, row_names=None, col_names=None, page_names=None, function='Mean'):
        """Build a pivot table.
        
        Arguments:
        depend_name (str): name of the dependent variable
        row_names, col_names, page_names (lists of str): name of the independent variables
        function (str): available functions: N,Sum, Mean, Median, Standard Deviation, Variance (default Mean)
        """
        if page_names is None:
            page_names = []
        if col_names is None:
            col_names = []
        if row_names is None:
            row_names = []
        if not depend_name:
            self.dial_pivot.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_pivot.exec():
                row_names, col_names, page_names, depend_name, function = self.dial_pivot.read_parameters()
            else:
                return
        self._run_analysis(title=_('Pivot table'), function_name='self.active_data.pivot',
                           parameters={'depend_name': depend_name, 'row_names': row_names, 'col_names': col_names,
                                       'page_names': page_names, 'function': function})

    def diffusion(self, error_name='', RT_name='', participant_name='', condition_names=None, correct_coding='0',
                  reaction_time_in='sec', scaling_parameter=0.1):
        """Run a diffusion analysis on behavioral data.

        Arguments:
        RT_name, error name, participant_name (lists of str): name of the variables
        condition_names (lists of str): name of the condition(s) variables
        """
        if condition_names is None:
            condition_names = []
        if not RT_name:
            self.dial_diffusion.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_diffusion.exec():
                error_name, RT_name, participant_name, condition_names, correct_coding, reaction_time_in, \
                scaling_parameter = self.dial_diffusion.read_parameters()
            else:
                return
            # use the original term in the function call, not the translated one
            reaction_time_in = 'sec' if reaction_time_in == _('s') else 'msec'
        self._run_analysis(title=_('Behavioral data diffusion analysis'), function_name='self.active_data.diffusion',
                           parameters={'error_name': error_name, 'RT_name': RT_name,
                                       'participant_name': participant_name, 'condition_names': condition_names,
                                       'correct_coding': correct_coding[0], 'reaction_time_in': reaction_time_in,
                                       'scaling_parameter': scaling_parameter})

    def compare_variables(self, var_names=None, factors=None, display_factors=None, ylims=[None, None]):
        """Compare variables.
        
        Arguments:
        var_names (list): variable names
        """
        if factors is None:
            factors = []
        if display_factors is None:
            display_factors = [[factor[0] for factor in factors] if factors else [], []]
        if not var_names:
            self.dial_comp_var.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_comp_var.exec():
                var_names, factors, display_factors, ylims = self.dial_comp_var.read_parameters()  # TODO check if settings are
                                                                                  # appropriate
            else:
                return
        self._run_analysis(title=_('Compare repeated measures variables'),
                           function_name='self.active_data.compare_variables',
                           parameters={'var_names': var_names, 'factors': factors, 'display_factors': display_factors,
                                       'ylims': ylims})

    def compare_groups(self, var_names=None, groups=None, display_groups=None,
                       single_case_slope_SE=None, single_case_slope_trial_n=None,
                       ylims=[None, None]):
        """Compare groups.
        
        Arguments:
        var_names (list): dependent variable names
        groups (list): grouping variable names
        """
        if not var_names:
            self.dial_comp_grp.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_comp_grp.exec():
                var_names, groups, display_groups, single_case_slope_SE, single_case_slope_trial_n, ylims = \
                    self.dial_comp_grp.read_parameters()  # TODO check if settings are appropriate
                if var_names == []:
                    var_names = [None]  # compare_groups() method handles the missing parameters
            else:
                return
        for i, var_name in enumerate(var_names):
            self._run_analysis(title=_('Compare groups'), function_name='self.active_data.compare_groups',
                               parameters={'var_name': var_name, 'grouping_variables': groups,
                                           'display_groups': display_groups,
                                           'single_case_slope_SE': single_case_slope_SE,
                                           'single_case_slope_trial_n': single_case_slope_trial_n, 'ylims': ylims},
                               scroll_to_analysis=not i)

    def compare_variables_groups(self, var_names=None, groups=None, factors=None, display_factors=None,
                                 single_case_slope_SE=None, single_case_slope_trial_n=None, ylims=[None, None]):
        """Compare variables and groups.
        """
        if groups is None:
            groups = []
        if factors is None:
            factors = []
        if display_factors is None:
            display_factors = [[factor[0] for factor in factors] if factors else [], []]
        if not var_names:
            self.dial_comp_var_groups.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_comp_var_groups.exec():
                var_names, groups, factors, display_factors, single_case_slope_SE, single_case_slope_trial_n, ylims = \
                    self.dial_comp_var_groups.read_parameters()
            else:
                return
        self._run_analysis(title=_('Compare repeated measures variables and groups'),
                           function_name='self.active_data.compare_variables_groups',
                           parameters={'var_names': var_names, 'factors': factors, 'grouping_variables': groups,
                                       'display_factors': display_factors, 'single_case_slope_SE': single_case_slope_SE,
                                       'single_case_slope_trial_n': single_case_slope_trial_n, 'ylims': ylims})

    def reliability_internal(self, var_names=None, reversed_names=None):
        if not var_names:
            self.dial_rel_internal.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_rel_internal.exec():
                var_names, reversed_names = self.dial_rel_internal.read_parameters()
                if not var_names:
                    var_names = ['']  # error message for missing variable come from the explore_variable() method
            else:
                return
        self._run_analysis(title=_('Internal consistency reliability analysis'),
                           function_name='self.active_data.reliability_internal',
                           parameters={'var_names': var_names, 'reverse_items': reversed_names})

    def reliability_interrater(self, var_names=None, ratings_averaged=True):
        if not var_names:
            self.dial_rel_interrater.init_vars(names=self.active_data.data_frame.columns)
            if self.dial_rel_interrater.exec():
                var_names, ratings_averaged = self.dial_rel_interrater.read_parameters()
                if not var_names:
                    var_names = ['']  # error message for missing variable come from the explore_variable() method
            else:
                return
            print(var_names)
        self._run_analysis(title=_('Interrater reliability analysis'),
                           function_name='self.active_data.reliability_interrater',
                           parameters={'var_names': var_names, 'ratings_averaged': ratings_averaged})

    def rerun_analyses(self):
        """Rerun the analyses that are currently visible in the results pane.

        """
        from . import cogstat_util as cs_util  # import cs_util so that it is available in locals()

        # Collect the analyses to be run from the current result pane
        analyses_to_run = [analysis_result.command for analysis_result in self.analysis_results]
        # Clear the results pane and the related list
        self.result_pane.clear()
        self.analysis_results = []
        self.unsaved_output = False  # Not necessary to save the empty output
        # Rerun the collected analyses
        for analysis_to_run in analyses_to_run:
            # analysis_to_run is a list of three items: 0. the title of teh analysis, 1. function/method to be run,
            #  2. the optional parameters in a dict
            #print('Analysis:', analysis_to_run )
            # TODO refactor the data import
            self._run_analysis(*analysis_to_run)


    ### Result menu methods ###
    def delete_output(self):
        reply = QtWidgets.QMessageBox.question(self, _('Clear output'),
                                               _('Are you sure you want to delete the output?'),
                                               QtWidgets.QMessageBox.StandardButton.Yes |
                                               QtWidgets.QMessageBox.StandardButton.No,
                                               QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.result_pane.clear()
            self.analysis_results = []
            self.unsaved_output = False  # Not necessary to save the empty output

    def find_text(self):
        self.dial_find_text = cogstat_dialogs.find_text_dialog(output_pane=self.result_pane)
        self.dial_find_text.exec()

    def zoom_in(self):
        self.result_pane.zoomIn(1)

    def zoom_out(self):
        self.result_pane.zoomOut(1)

    def text_editable(self):
        self.result_pane.setReadOnly(not (self.menus[2].actions()[5].isChecked()))  # see also _init_UI
        #self.output_pane.setReadOnly(not(self.toolbar.actions()[15].isChecked()))
        # TODO if the position of this menu is changed, then this function will not work
        # TODO rewrite Text is editable switches, because the menu and the toolbar works independently

    def save_result(self):
        """Save the results pane to an html file."""
        if self.output_filename == '':
            self.save_result_as()
        else:
            try:
                html_file = self.result_pane.toHtml()
                # replace non-breaking spaces with html code for nbsp
                html_file = html_file.replace(' ', '&nbsp;')
                with open(self.output_filename, 'w') as f:
                    f.write(html_file)
                self.unsaved_output = False
            except:
                if csc.detailed_error_message:
                    error_message = '\n' + '<cs_warning>' + _('Detailed error message') + \
                                    ' (%s):</cs_warning>\n' % 'you can turn this off in Preferences' + \
                                     traceback.format_exc()
                else:
                    error_message = ''
                self.result_pane.append(cs_util.convert_output({'warning': ((broken_analysis % _('Saving the results'))
                                                                            + error_message)})['warning'])

    def save_result_as(self, filename=None):
        """Save the results pane to an html file.

        Parameters
        ----------
        filename : str
            name of the file to save to

        Returns
        -------

        """
        if not filename:
            filename = cogstat_dialogs.save_output()
            self.output_filename = filename
        if filename:
            if filename[-5:] != ".html":
                filename = filename + '.html'
            self.output_filename = filename
            self.save_result()

    ### Cogstat menu methods ###
    def _open_help_webpage(self):
        webbrowser.open('https://doc.cogstat.org/')
        
    def _show_preferences(self):
        self.dial_pref.exec()
    
    def _open_reqfeat_webpage(self):
        webbrowser.open('https://doc.cogstat.org/Suggest-a-new-feature')
        
    def _open_reportbug_webpage(self):
        webbrowser.open('https://doc.cogstat.org/Report-a-bug')
        
    def _show_about(self):
        QtWidgets.QMessageBox.about(self, _('About CogStat ') + csc.versions['cogstat'], 'CogStat ' +
                                    csc.versions['cogstat'] + ('<br>%s<br><br>Copyright © %s-%s Attila Krajcsi and CogStat contributors<br><br>'
                                                               '<a href = "http://www.cogstat.org">%s</a>' %
                                                               (_('Simple automatic data analysis software'),
                                                                2012, 2023, _('Visit CogStat website'))))

    def print_versions(self):
        """Print the versions of the software components CogStat uses."""
        # Intentionally not localized.
        self._run_analysis(title=_('System components'), function_name='cs_util.print_versions',
                           parameters={'main_window': self})

    def closeEvent(self, event):
        # Override the close behavior, otherwise alt+F4 quits unconditionally.
        # http://stackoverflow.com/questions/1414781/prompt-on-exit-in-pyqt-application
        
        # Check if everything is saved
        tosave = True
        while self.unsaved_output and tosave:
            reply = QtWidgets.QMessageBox.question(self, _('Save output'),
                    _('Output has unsaved results. Do you want to save it?'), QtWidgets.QMessageBox.StandardButton.Yes |
                                                   QtWidgets.QMessageBox.StandardButton.No,
                                                   QtWidgets.QMessageBox.StandardButton.Yes)
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self.save_result()
            else:
                tosave = False

        """
        reply = QtGui.QMessageBox.question(self, _('Confirm exit'), 
            _('Are you sure you want to exit the program?'), QtGui.QMessageBox.StandardButton.Yes, QtGui.QMessageBox.StandardButton.No)
        if reply == QtGui.QMessageBox.StandardButton.Yes:
            QtGui.qApp.quit()
        else:
            event.ignore()
        """


class PandasModel(QtCore.QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe """

    # Based on https://doc.qt.io/qtforpython/examples/example_external__pandas.html

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        """ Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QtCore.QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QtCore.QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QtCore.QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        # Filtered data have different background
        if role == Qt.ItemDataRole.ForegroundRole and not(index.row() in [0, 1]):  # don't change the measurement level row (row 0)
            if self._dataframe['cogstat_filtered_cases'].iloc[index.row()]:
                return QtGui.QColor('lightGray')

        # Use different background for the data type and measurement level
        if role == Qt.ItemDataRole.BackgroundRole:
            if index.row() in [0, 1]:
                return QtGui.QColor('lightGray')

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Orientation.Vertical:
                return str(self._dataframe.index[section])

        return None

class GuiResultPackage():
    """ A class for storing a package of results.

    Result object includes:
    - self.command: Command to run. List of str (command) and optional dict (parameters)
        e.g., ['self.active_data.explore_variable',
               {'var_name': var_name, 'frequencies': freq, 'central_value': loc_test_value}]
    - self.output:
        - dict of strings (html) or matplotlib figures or pandas Stylers
    """

    def __init__(self):
        self.command = []
        self.output = {}

    def add_command(self, command):
        self.command.extend(command)

    def add_output(self, output):
        """Add new output dict to the self.output dict

        :param output: result dictionary to add
        """
        self.output.update(output)

def main():
    splash_screen.close()
    ex = StatMainWindow()
    sys.exit(app.exec())

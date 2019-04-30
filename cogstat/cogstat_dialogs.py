# -*- coding: utf-8 -*-

import gettext
import os
from . import cogstat_config as csc
from PyQt5 import QtWidgets

QString = str

t = gettext.translation('cogstat', 'locale/', [csc.language], fallback=True)
_ = t.gettext

### File menu commands ###

# http://developer.qt.nokia.com/doc/qt-4.8/QFileDialog.html


def open_data_file():
    return str(QtWidgets.QFileDialog.getOpenFileName(None, _('Open data file'), '', '*.csv *.sav')[0])  #*.txt *.log *.tsv


def open_demo_data_file():
    return str(QtWidgets.QFileDialog.getOpenFileName(None, _('Open data file'), os.path.dirname(csc.__file__)+'/sample_data', '*.csv *.sav')[0])  #*.txt *.log *.tsv


def save_output():
    return str(QtWidgets.QFileDialog.getSaveFileName(None, _('Save result file'), 'CogStat analysis result.pdf', '*.pdf')[0])

# XXX self.buttonBox.Ok.setEnabled(False) # TODO how can we disable the OK button without the other?
# TODO Some variables are CamelCase - change them

### Various functions ###

# TODO functions should be private


def init_source_vars(list_widget, names):
    list_widget.clear() # clear source list in case new data is loaded
    for var_name in names:
        list_widget.addItem(QString(var_name))


def remove_ceased_vars(list_widget, names):
    """
    If list_widget includes items that are not in the names list,
    then remove those items.
    """
    for item_i in range(list_widget.count()-1, -1, -1):
        if not str(list_widget.item(item_i).text()) in names:
            list_widget.takeItem(item_i)


def add_to_list_widget(source_list_widget, target_list_widget):
    """
    Add the selected items of the source_list_widget to the target_list_widget.
    """
    for item in source_list_widget.selectedItems():
        item_in_the_list = False
        for i in range(target_list_widget.count()):
            if item.text() == target_list_widget.item(i).text():
                item_in_the_list = True
                break
        if not item_in_the_list:
            target_list_widget.addItem(QString(item.text()))


def remove_item_from_list_widget(list_widget):
    """
    Remove selected item from list_widget.
    """
    for item in list_widget.selectedItems():
        list_widget.takeItem(list_widget.row(item))

### Data dialogs ###

from .ui import pivot
class pivot_dialog(QtWidgets.QDialog, pivot.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.addRows.clicked.connect(self.add_rows)
        self.removeRows.clicked.connect(self.remove_rows)
        self.rowsListWidget.doubleClicked.connect(self.remove_rows)
        self.addColumns.clicked.connect(self.add_columns)
        self.removeColumns.clicked.connect(self.remove_columns)
        self.columnsListWidget.doubleClicked.connect(self.remove_columns)
        self.addPages.clicked.connect(self.add_pages)
        self.removePages.clicked.connect(self.remove_pages)
        self.pagesListWidget.doubleClicked.connect(self.remove_pages)
        self.addDependent.clicked.connect(self.add_dependent)
        self.removeDependent.clicked.connect(self.remove_dependent)
        self.dependentListWidget.doubleClicked.connect(self.remove_dependent)

        self.init_vars(names)                
        self.show()

    def init_vars(self, names):
        init_source_vars(self.sourceListWidget, names)
        remove_ceased_vars(self.pagesListWidget, names)
        remove_ceased_vars(self.columnsListWidget, names)
        remove_ceased_vars(self.rowsListWidget, names)
        remove_ceased_vars(self.dependentListWidget, names)

    def add_rows(self):
        add_to_list_widget(self.sourceListWidget, self.rowsListWidget)
    def remove_rows(self):
        remove_item_from_list_widget(self.rowsListWidget)
    def add_columns(self):
        add_to_list_widget(self.sourceListWidget, self.columnsListWidget)
    def remove_columns(self):
        remove_item_from_list_widget(self.columnsListWidget)
    def add_pages(self):
        add_to_list_widget(self.sourceListWidget, self.pagesListWidget)
    def remove_pages(self):
        remove_item_from_list_widget(self.pagesListWidget)
    def add_dependent(self):
        if self.dependentListWidget.count() == 0:  # do this only if the list is empty
            self.dependentListWidget.addItem(QString(self.sourceListWidget.currentItem().text()))
    def remove_dependent(self):
        self.dependentListWidget.takeItem(self.dependentListWidget.currentRow())
    
    def read_parameters(self):
        return ([str(self.rowsListWidget.item(i).text()) for i in range(self.rowsListWidget.count())],
                [str(self.columnsListWidget.item(i).text()) for i in range(self.columnsListWidget.count())],
                [str(self.pagesListWidget.item(i).text()) for i in range(self.pagesListWidget.count())], 
                [str(self.dependentListWidget.item(i).text()) for i in range(self.dependentListWidget.count())], 
                str(self.function.currentText()))


from .ui import var_properties
class explore_var_dialog(QtWidgets.QDialog, var_properties.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)

        self.init_vars(names)
        self.show()

    def init_vars(self, names):
        init_source_vars(self.source_listWidget, names)
        remove_ceased_vars(self.selected_listWidget, names)
    def add_var(self):
        add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        remove_item_from_list_widget(self.selected_listWidget)
    
    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                 self.freq_checkbox.isChecked(),
                 str(self.ttest_value.text()))


from .ui import explore_var_pairs
class explore_var_pairs_dialog(QtWidgets.QDialog, explore_var_pairs.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)

        self.init_vars(names)        
        self.show()
        
    def init_vars(self, names):
        init_source_vars(self.source_listWidget, names)
        remove_ceased_vars(self.selected_listWidget, names)
    def add_var(self):
        add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        remove_item_from_list_widget(self.selected_listWidget)
    
    def read_parameters(self):
        return [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())]


from .ui import compare_vars
class compare_vars_dialog(QtWidgets.QDialog, compare_vars.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)

        self.init_vars(names)
        self.show()
        
    def init_vars(self, names):
        init_source_vars(self.source_listWidget, names)
        remove_ceased_vars(self.selected_listWidget, names)
    def add_var(self):
        add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        remove_item_from_list_widget(self.selected_listWidget)
    
    def read_parameters(self):
        return [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())]

from .ui import compare_groups_single_case_slope
class compare_groups_single_case_slope_dialog(QtWidgets.QDialog, compare_groups_single_case_slope.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)

        self.init_vars(names)
        #self.show()

    def init_vars(self, names):
        init_source_vars(self.source_listWidget, names)
        remove_ceased_vars(self.selected_listWidget, names)

    def add_var(self):
        if self.selected_listWidget.count() == 0:  # allow only if the list is empty
            add_to_list_widget(self.source_listWidget, self.selected_listWidget)

    def remove_var(self):
        remove_item_from_list_widget(self.selected_listWidget)

    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                str(self.spinBox.text()))


from .ui import compare_groups
class compare_groups_dialog(QtWidgets.QDialog, compare_groups.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.group_listWidget.doubleClicked.connect(self.remove_group)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)
        self.add_group_button.clicked.connect(self.add_group)
        self.remove_group_button.clicked.connect(self.remove_group)
        self.pushButton.clicked.connect(self.on_slopeButton_clicked)

        self.slope_dialog = compare_groups_single_case_slope_dialog(self, names=names)
        self.single_case_slope_SEs, self.single_case_slope_trial_n = [], 0

        self.init_vars(names)
        self.show()

    def init_vars(self, names):
        init_source_vars(self.source_listWidget, names)
        remove_ceased_vars(self.selected_listWidget, names)
        remove_ceased_vars(self.group_listWidget, names)
        self.slope_dialog.init_vars(names)

    def add_var(self):
        add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        remove_item_from_list_widget(self.selected_listWidget)

    def add_group(self):
        if self.group_listWidget.count() < 2:  # allow maximum two grouping variables
            add_to_list_widget(self.source_listWidget, self.group_listWidget)
    def remove_group(self):
        remove_item_from_list_widget(self.group_listWidget)

    def on_slopeButton_clicked(self):
        self.slope_dialog.exec_()
        self.single_case_slope_SEs, self.single_case_slope_trial_n = self.slope_dialog.read_parameters()

    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                [str(self.group_listWidget.item(i).text()) for i in range(self.group_listWidget.count())],
                self.single_case_slope_SEs, int(self.single_case_slope_trial_n))


from .ui import preferences
class preferences_dialog(QtWidgets.QDialog, preferences.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.write_settings)
        self.buttonBox.rejected.connect(self.reject)
    
        self.init_langs()
        self.init_themes()
        self.show()

    def init_langs(self):
        """Set the available languages.
        """
        import glob
        import os

        def available_langs(domain=None, localedir=None):
            """Look for available languages"""
            if domain is None:
                domain = gettext._current_domain
            if localedir is None:
                localedir = gettext._default_localedir
            files = glob.glob(os.path.join(localedir, '*', 'LC_MESSAGES', '%s.mo' % domain))
            langs = [file_name.split(os.path.sep)[-3] for file_name in files]
            return langs

        langs = sorted(['en']+available_langs(domain='cogstat', localedir=os.path.dirname(os.path.abspath(__file__))+'/locale'))
        # local language names based on https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
        lang_names = {'bg':'Български (Bulgarian)', 'de':'Deutsch (German)', 'en': 'English', 'fa':'فارسی (Persian)',
                      'he':'עברית (Hebrew)', 'hr':'Hrvatski (Croatian)',
                      'hu':'Magyar (Hungarian)', 'it':'Italiano (Italian)', 'nb':'Norsk Bokmål (Norvegian Bokmål)',
                      'ro':'Română (Romanian)', 'sk':'Slovenčina (Slovak)', 'th':'ไทย (Thai)'}
        lang_names_sorted = sorted([lang_names[lang] for lang in langs])
        self.lang_codes = {lang_name:lang_code for lang_code, lang_name in zip(lang_names.keys(), lang_names.values())}

        self.langComboBox.clear()
        for lang_name in lang_names_sorted:
            self.langComboBox.addItem(lang_name)
        self.langComboBox.setCurrentIndex(lang_names_sorted.index(lang_names[csc.language]))

    def init_themes(self):
        """Set the available themes.
        """
        import matplotlib.pyplot as plt

        themes = sorted(plt.style.available)
        self.themeComboBox.clear()
        for theme in themes:
            self.themeComboBox.addItem(theme)
        self.themeComboBox.setCurrentIndex(themes.index(csc.theme))

    def write_settings(self):
        """Save the settings when OK is pressed.
        """
        csc.save(['language'], self.lang_codes[str(self.langComboBox.currentText())])
        csc.save(['graph', 'theme'], str(self.themeComboBox.currentText()))
        self.accept()

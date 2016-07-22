# -*- coding: utf-8 -*-

import gettext
import cogstat_config as csc
from PyQt4 import QtCore, QtGui

t = gettext.translation('cogstat', 'locale/', [csc.language], fallback=True)
_ = t.ugettext

### File menu commands ###

# http://developer.qt.nokia.com/doc/qt-4.8/QFileDialog.html


def open_data_file():
    return unicode(QtGui.QFileDialog.getOpenFileName(None, _('Open data file'), '',  '*.txt *.log *.csv *.tsv'))


def save_output():
    return unicode(QtGui.QFileDialog.getSaveFileName(None, _('Save result file'), 'result.pdf', '*.pdf'))

# XXX self.buttonBox.Ok.setEnabled(False) # TODO how can we disable the OK button without the other?
# TODO Some variables are CamelCase - change them

### Various functions ###

# TODO functions should be private


def init_source_vars(list_widget, names):
    list_widget.clear() # clear source list in case new data is loaded
    for var_name in names:
        list_widget.addItem(QtCore.QString(var_name))


def remove_ceased_vars(list_widget, names):
    """
    If list_widget includes items that are not in the names list,
    then remove those items.
    """
    for item_i in range(list_widget.count()-1, -1,-1):
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
            target_list_widget.addItem(QtCore.QString(item.text()))


def remove_item_from_list_widget(list_widget):
    """
    Remove selected item from list_widget.
    """
    for item in list_widget.selectedItems():
        list_widget.takeItem(list_widget.row(item))

### Data dialogs ###

import ui.pivot
class pivot_dialog(QtGui.QDialog, ui.pivot.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
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
            self.dependentListWidget.addItem(QtCore.QString(self.sourceListWidget.currentItem().text()))
    def remove_dependent(self):
        self.dependentListWidget.takeItem(self.dependentListWidget.currentRow())
    
    def read_parameters(self):
        return ([unicode(self.rowsListWidget.item(i).text()) for i in range(self.rowsListWidget.count())],
                [unicode(self.columnsListWidget.item(i).text()) for i in range(self.columnsListWidget.count())],
                [unicode(self.pagesListWidget.item(i).text()) for i in range(self.pagesListWidget.count())], 
                [unicode(self.dependentListWidget.item(i).text()) for i in range(self.dependentListWidget.count())], 
                unicode(self.function.currentText()))


import ui.var_properties
class explore_var_dialog(QtGui.QDialog, ui.var_properties.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
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
        return ([unicode(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                 self.freq_checkbox.isChecked(),
                 self.dist_checkbox.isChecked(), 
                 self.descr_checkbox.isChecked(),
                 self.norm_checkbox.isChecked(), 
                 self.ttest_checkbox.isChecked(),
                 unicode(self.ttest_value.text()))


import ui.explore_var_pairs
class explore_var_pairs_dialog(QtGui.QDialog, ui.explore_var_pairs.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
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
        return [unicode(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())]


import ui.compare_vars
class compare_vars_dialog(QtGui.QDialog, ui.compare_vars.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
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
        return [unicode(self.selected_listWidget.item(i).text(), 'utf-8') for i in range(self.selected_listWidget.count())]


import ui.compare_groups
class compare_groups_dialog(QtGui.QDialog, ui.compare_groups.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.group_listWidget.doubleClicked.connect(self.remove_group)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)
        self.add_group_button.clicked.connect(self.add_group)
        self.remove_group_button.clicked.connect(self.remove_group)

        self.init_vars(names)
        self.show()

    def init_vars(self, names):
        init_source_vars(self.source_listWidget, names)
        remove_ceased_vars(self.selected_listWidget, names)
        remove_ceased_vars(self.group_listWidget, names)
    def add_var(self):
        add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        remove_item_from_list_widget(self.selected_listWidget)

    def add_group(self):
        if self.group_listWidget.count() == 0:  # do this only if the list is empty
            add_to_list_widget(self.source_listWidget, self.group_listWidget)
    def remove_group(self):
        remove_item_from_list_widget(self.group_listWidget)
    
    def read_parameters(self):
        return ([unicode(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                [unicode(self.group_listWidget.item(i).text()) for i in range(self.group_listWidget.count())])


import ui.preferences
class preferences_dialog(QtGui.QDialog, ui.preferences.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtGui.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.buttonBox.accepted.connect(self.write_settings)
        self.buttonBox.rejected.connect(self.reject)
    
        self.init_langs()
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

        langs = ['en']+available_langs(domain='cogstat', localedir=os.path.dirname(os.path.abspath(__file__))+'/locale')

        # TODO is there any automatic method to show the name and not the code 
        # of the languages? Or should we use our own solution (e.g., dictionary)?
        self.langComboBox.clear()
        for lang in sorted(langs):
            self.langComboBox.addItem(lang)
        self.langComboBox.setCurrentIndex(langs.index(csc.language))

    def write_settings(self):
        """Save the settings when OK is pressed.
        """
        csc.save('language', unicode(self.langComboBox.currentText()))
        self.accept()

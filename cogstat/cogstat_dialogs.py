# -*- coding: utf-8 -*-

import gettext
import os

from PyQt5 import QtWidgets, QtCore, QtGui

from . import cogstat_config as csc

QString = str

t = gettext.translation('cogstat', os.path.dirname(os.path.abspath(__file__))+'/locale/', [csc.language], fallback=True)
_ = t.gettext
# Overwrite the qt _translate function, use gettext
def _gui(a, b):
    return(_(b))
QtCore.QCoreApplication.translate = _gui

### File menu commands ###

# http://developer.qt.nokia.com/doc/qt-4.8/QFileDialog.html


def open_data_file():
    #dialog = QtWidgets.QFileDialog()
    #dialog.setFilter(QtCore.QDir.CaseSensitive)
    # TODO how to make the filter case insensitive?
    return str(QtWidgets.QFileDialog.getOpenFileName(None, _('Open data file'), '',
                                                     '%s (*.ods *.xls *.xlsx *.csv *.txt *.tsv *.dat *.log '
                                                     '*.sav *.zsav *.por *.jasp *.omv *.sas7bdat *.xpt *.dta '
                                                     '*.rdata *.Rdata *.rds *.rda);;'
                                                     '%s *.ods *.xls *xlsx;;%s *.csv *.txt *.tsv *.dat *.log;;'
                                                     '%s *.sav *.zsav *.por;;%s *.omv;;%s *.jasp;;'
                                                     '%s *.sas7bdat *.xpt;; %s *.dta;;'
                                                     '%s *.rdata *.Rdata *.rds *.rda' %
                                                     (_('All importable data files'),
                                                      _('Spreadsheet files'), _('Text files'),
                                                      _('SPSS data files'), _('jamovi data files'),
                                                      _('JASP data files'), _('SAS data files'),
                                                      _('STATA data files'), _('R data files'))
                                                     )[0])


def open_demo_data_file():
    return str(QtWidgets.QFileDialog.getOpenFileName(None, _('Open data file'), os.path.dirname(csc.__file__) +
                                                     '/demo_data',
                                                     '%s (*.ods *.xls *.xlsx *.csv *.txt *.tsv *.dat *.log '
                                                     '*.sav *.zsav *.por *.jasp *.omv *.sas7bdat *.xpt *.dta '
                                                     '*.rdata *.Rdata *.rds *.rda);;'
                                                     '%s *.ods *.xls *xlsx;;%s *.csv *.txt *.tsv *.dat *.log;;'
                                                     '%s *.sav *.zsav *.por;;%s *.omv;;%s *.jasp;;'
                                                     '%s *.sas7bdat *.xpt;; %s *.dta;;'
                                                     '%s *.rdata *.Rdata *.rds *.rda' %
                                                     (_('All importable data files'),
                                                      _('Spreadsheet files'), _('Text files'),
                                                      _('SPSS data files'), _('jamovi data files'),
                                                      _('JASP data files'), _('SAS data files'),
                                                      _('STATA data files'), _('R data files'))
                                                     )[0])


def save_output():
    return str(QtWidgets.QFileDialog.getSaveFileName(None, _('Save result file'), 'CogStat analysis result.pdf',
                                                     '*.pdf')[0])

# XXX self.buttonBox.Ok.setEnabled(False) # TODO how can we disable the OK button without the other?
# TODO Some variables are CamelCase - change them

### Various functions ###

# TODO functions should be private


def init_source_vars(list_widget, names, already_in_use):
    """
    :param list_widget: source variables
    :param names: names of the variables
    :param already_in_use: list of listWidgets that may contain variables already in use
    :return:
    """
    already_in_use_vars = []
    for in_use_list_widget in already_in_use:
        already_in_use_vars.extend([in_use_list_widget.item(i).text() for i in range(in_use_list_widget.count())])
    list_widget.clear()  # clear source list in case new data is loaded
    for var_name in names:
        if not(var_name in already_in_use_vars):
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
    Add the selected item(s) of the source_list_widget to the target_list_widget, and remove the item(s) from the
    source_list_widget.
    """
    # TODO add a maximum parameter, for the maximum number of items that can be added
    for item in source_list_widget.selectedItems():
        target_list_widget.addItem(QString(item.text()))
        source_list_widget.takeItem(source_list_widget.row(item))


def remove_item_from_list_widget(source_list_widget, target_list_widget, names):
    """
    Remove selected item(s) from target_list_widget, and add it to the source_list_widget.
    """
    for item in target_list_widget.selectedItems():
        target_list_widget.takeItem(target_list_widget.row(item))
        source_list_widget.insertItem(find_previous_item_position(source_list_widget, names, item.text()), item.text())


def find_previous_item_position(list_widget, names, text_item):
    """
    TODO
    """
    names = list(names)
    if list(reversed(names[:names.index(text_item)])):  # check if the text_item is not the first in the variable list,
                                                        # otherwise return zero
        for item in reversed(names[:names.index(text_item)]):
            try:  # if the item is in the list_widget, then return its position
                return list_widget.row(list_widget.findItems(item, QtCore.Qt.MatchExactly)[0])+1
            except:  # otherwise look further for next variable names
                pass
    return 0  # if no earlier variables were found on list_widget (or the text_item is the first in the variable list)
              # insert the item at the beginning of the list_widget


def add_to_list_widget_with_factors(source_list_widget, target_list_widget, names=[]):
    """
    Add the selected items of the source_list_widget to the target_list_widget,
    """

    if target_list_widget.selectedItems():  # there are selected items in the target list
        start_target_row = target_list_widget.row(target_list_widget.selectedItems()[0])
    else:
        start_target_row = 0

    for item_source in source_list_widget.selectedItems():
        for item_target_i in range(start_target_row, target_list_widget.count()):
            item_text = target_list_widget.item(item_target_i).text()
            if item_text.endswith(' :: '):
                target_list_widget.item(item_target_i).setText(item_text + item_source.text())
                source_list_widget.takeItem(source_list_widget.row(item_source))
                break


def remove_from_list_widget_with_factors(source_list_widget, target_list_widget, names=[]):
    """
    Remove selected items from target_list_widget.
    """
    for item in target_list_widget.selectedItems():
        if item.text().split(' :: ')[1]:
            source_list_widget.insertItem(find_previous_item_position(source_list_widget, names, item.text().
                                                                      split(' :: ')[1]), item.text().split(' :: ')[1])
            item.setText(item.text().split(' :: ')[0]+' :: ')


def _float_or_none(x):
    try:
        return float(x)
    except ValueError:
        return None

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
        self.names = names
        remove_ceased_vars(self.pagesListWidget, names)
        remove_ceased_vars(self.columnsListWidget, names)
        remove_ceased_vars(self.rowsListWidget, names)
        remove_ceased_vars(self.dependentListWidget, names)
        init_source_vars(self.sourceListWidget, names, [self.pagesListWidget, self.columnsListWidget,
                                                        self.rowsListWidget, self.dependentListWidget])

    def add_rows(self):
        add_to_list_widget(self.sourceListWidget, self.rowsListWidget)
    def remove_rows(self):
        remove_item_from_list_widget(self.sourceListWidget, self.rowsListWidget, self.names)
    def add_columns(self):
        add_to_list_widget(self.sourceListWidget, self.columnsListWidget)
    def remove_columns(self):
        remove_item_from_list_widget(self.sourceListWidget, self.columnsListWidget, self.names)
    def add_pages(self):
        add_to_list_widget(self.sourceListWidget, self.pagesListWidget)
    def remove_pages(self):
        remove_item_from_list_widget(self.sourceListWidget, self.pagesListWidget, self.names)
    def add_dependent(self):
        if self.dependentListWidget.count() == 0:  # do this only if the list is empty
            add_to_list_widget(self.sourceListWidget, self.dependentListWidget)
    def remove_dependent(self):
        remove_item_from_list_widget(self.sourceListWidget, self.dependentListWidget, self.names)
    
    def read_parameters(self):
        return ([str(self.rowsListWidget.item(i).text()) for i in range(self.rowsListWidget.count())],
                [str(self.columnsListWidget.item(i).text()) for i in range(self.columnsListWidget.count())],
                [str(self.pagesListWidget.item(i).text()) for i in range(self.pagesListWidget.count())], 
                [str(self.dependentListWidget.item(i).text()) for i in range(self.dependentListWidget.count())][0],
                str(self.function.currentText()))


from .ui import diffusion


class diffusion_dialog(QtWidgets.QDialog, diffusion.Ui_Dialog):
    def __init__(self, parent=None, names=[]):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.addRT.clicked.connect(self.add_RT)
        self.removeRT.clicked.connect(self.remove_RT)
        self.RTListWidget.doubleClicked.connect(self.remove_RT)
        self.addError.clicked.connect(self.add_error)
        self.removeError.clicked.connect(self.remove_error)
        self.errorListWidget.doubleClicked.connect(self.remove_error)
        self.addParticipant.clicked.connect(self.add_participant)
        self.removeParticipant.clicked.connect(self.remove_participant)
        self.participantListWidget.doubleClicked.connect(self.remove_participant)
        self.addCondition.clicked.connect(self.add_condition)
        self.removeCondition.clicked.connect(self.remove_condition)
        self.conditionListWidget.doubleClicked.connect(self.remove_condition)

        self.init_vars(names)
        self.show()

    def init_vars(self, names):
        self.names = names
        remove_ceased_vars(self.RTListWidget, names)
        remove_ceased_vars(self.errorListWidget, names)
        remove_ceased_vars(self.participantListWidget, names)
        remove_ceased_vars(self.conditionListWidget, names)
        init_source_vars(self.sourceListWidget, names,
                         [self.RTListWidget, self.errorListWidget, self.participantListWidget,
                          self.conditionListWidget])

    def add_RT(self):
        if self.RTListWidget.count() == 0:  # do this only if the list is empty
            add_to_list_widget(self.sourceListWidget, self.RTListWidget)

    def remove_RT(self):
        remove_item_from_list_widget(self.sourceListWidget, self.RTListWidget, self.names)

    def add_error(self):
        if self.errorListWidget.count() == 0:  # do this only if the list is empty
            add_to_list_widget(self.sourceListWidget, self.errorListWidget)

    def remove_error(self):
        remove_item_from_list_widget(self.sourceListWidget, self.errorListWidget, self.names)

    def add_participant(self):
        if self.participantListWidget.count() == 0:  # do this only if the list is empty
            add_to_list_widget(self.sourceListWidget, self.participantListWidget)

    def remove_participant(self):
        remove_item_from_list_widget(self.sourceListWidget, self.participantListWidget, self.names)

    def add_condition(self):
        add_to_list_widget(self.sourceListWidget, self.conditionListWidget)

    def remove_condition(self):
        remove_item_from_list_widget(self.sourceListWidget, self.conditionListWidget, self.names)

    def read_parameters(self):
        return ([str(self.errorListWidget.item(i).text()) for i in range(self.errorListWidget.count())],
                [str(self.RTListWidget.item(i).text()) for i in range(self.RTListWidget.count())],
                [str(self.participantListWidget.item(i).text()) for i in range(self.participantListWidget.count())],
                [str(self.conditionListWidget.item(i).text()) for i in range(self.conditionListWidget.count())])


from .ui import filter_outlier
class filter_outlier(QtWidgets.QDialog, filter_outlier.Ui_Dialog):
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
        self.names = names
        remove_ceased_vars(self.selected_listWidget, names)
        init_source_vars(self.source_listWidget, names, [self.selected_listWidget])

    def add_var(self):
        add_to_list_widget(self.source_listWidget, self.selected_listWidget)

    def remove_var(self):
        remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def read_parameters(self):
        return [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())]


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
        self.names = names
        remove_ceased_vars(self.selected_listWidget, names)
        init_source_vars(self.source_listWidget, names, [self.selected_listWidget])
    def add_var(self):
        add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)
    
    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                self.freq_checkbox.isChecked(), str(self.ttest_value.text()))


from .ui import xylims
class xylims_dialog(QtWidgets.QDialog, xylims.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def read_parameters(self):
        return [self.lineEdit.text(), self.lineEdit_2.text()], [self.lineEdit_3.text(), self.lineEdit_4.text()]


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
        self.pushButton.clicked.connect(self.optionsButton_clicked)

        self.xylims_dialog = xylims_dialog(self)
        self.xlims = [None, None]
        self.ylims = [None, None]

        self.init_vars(names)
        self.show()
        
    def init_vars(self, names):
        self.names = names
        remove_ceased_vars(self.selected_listWidget, names)
        init_source_vars(self.source_listWidget, names, [self.selected_listWidget])
    def add_var(self):
        add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def optionsButton_clicked(self):
        if self.xylims_dialog.exec_():
            self.xlims, self.ylims = self.xylims_dialog.read_parameters()
            self.xlims[0] = _float_or_none(self.xlims[0])
            self.xlims[1] = _float_or_none(self.xlims[1])
            self.ylims[0] = _float_or_none(self.ylims[0])
            self.ylims[1] = _float_or_none(self.ylims[1])

    def read_parameters(self):
        return [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())], \
               self.xlims, self.ylims


from .ui import factor
class factor_dialog(QtWidgets.QDialog, factor.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def set_parameters(self, lineEdit='', spinBox=None):
        if lineEdit:
            self.lineEdit.setText(lineEdit)
        if spinBox:
            self.spinBox.setValue(spinBox)

    def read_parameters(self):
        return self.lineEdit.text(), self.spinBox.value()


from .ui import factors
class factors_dialog(QtWidgets.QDialog, factors.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.listWidget.doubleClicked.connect(self.modifyButton_clicked)
        self.pushButton.clicked.connect(self.addButton_clicked)
        self.pushButton_2.clicked.connect(self.modifyButton_clicked)
        self.pushButton_3.clicked.connect(self.removeButton_clicked)

        self.factor_dialog = factor_dialog(self)

    def addButton_clicked(self):
        self.factor_dialog.lineEdit.setFocus()
        if self.factor_dialog.exec_():
            factor_name, level_n = self.factor_dialog.read_parameters()
            self.listWidget.addItem(QString('%s (%d)' % (factor_name, level_n)))

    def modifyButton_clicked(self):
        self.factor_dialog.lineEdit.setFocus()
        for item in self.listWidget.selectedItems():
            t = item.text()
            text_to_modify = t[:t.rfind(' (')]
            value_to_modify = int(t[t.rfind('(')+1:t.rfind(')')])
            self.factor_dialog.set_parameters(text_to_modify, value_to_modify)
            if self.factor_dialog.exec_():
                factor_name, level_n = self.factor_dialog.read_parameters()
                item.setText(QString('%s (%d)' % (factor_name, level_n)))

    def removeButton_clicked(self):
        for item in self.listWidget.selectedItems():
            self.listWidget.takeItem(self.listWidget.row(item))

    def read_parameters(self):
        return [self.listWidget.item(i).text() for i in range(self.listWidget.count())]
        #return [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())]

from .ui import ylims
class ylims_dialog(QtWidgets.QDialog, ylims.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def read_parameters(self):
        return [self.lineEdit.text(), self.lineEdit_2.text()]


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
        self.pushButton.clicked.connect(self.factorsButton_clicked)
        self.pushButton_2.clicked.connect(self.optionsButton_clicked)

        self.factors_dialog = factors_dialog(self)
        self.ylims_dialog = ylims_dialog(self)
        self.factors = []
        self.ylims = [None, None]

        self.init_vars(names)
        self.show()

    def init_vars(self, names):
        self.names = names
        init_source_vars(self.source_listWidget, names, [])
        if len(self.factors) < 2:
            remove_ceased_vars(self.selected_listWidget, names)
            init_source_vars(self.source_listWidget, names, [self.selected_listWidget])

    def add_var(self):
        if len(self.factors) < 2:
            add_to_list_widget(self.source_listWidget, self.selected_listWidget)
        else:
            add_to_list_widget_with_factors(self.source_listWidget, self.selected_listWidget, names=self.names)

    def remove_var(self):
        if len(self.factors) < 2:
            remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)
        else:
            remove_from_list_widget_with_factors(self.source_listWidget, self.selected_listWidget, names=self.names)

    def show_factors(self):
        # remove all items first
        for i in range(self.selected_listWidget.count()):
            item = self.selected_listWidget.takeItem(0)
            if ' :: ' in item.text():
                if not item.text().endswith(' :: '):
                    self.source_listWidget.insertItem(
                        find_previous_item_position(self.source_listWidget, self.names, item.text().split(' :: ')[1]),
                        item.text().split(' :: ')[1])
                    item.setText(item.text().split(' :: ')[0] + ' :: ')
            else:
                self.source_listWidget.insertItem(
                    find_previous_item_position(self.source_listWidget, self.names, item.text()),
                    item.text())

        # add new empty factor levels
        factor_combinations = ['']
        for factor in self.factors:
            factor_combinations = ['%s - %s %s' % (factor_combination, factor[0], level_i + 1) for factor_combination in
                                   factor_combinations for level_i in range(factor[1])]
        factor_combinations = [factor_combination[3:] + ' :: ' for factor_combination in factor_combinations]
        for factor_combination in factor_combinations:
            self.selected_listWidget.addItem(QString(factor_combination))

    def factorsButton_clicked(self):
        if self.factors_dialog.exec_():
            factor_list = self.factors_dialog.read_parameters()
            #print(factor_list)

            self.factors = [[t[:t.rfind(' (')], int(t[t.rfind('(')+1:t.rfind(')')])] for t in factor_list]
            #print(self.factors)
            if len(self.factors) > 1:
                self.show_factors()
            else:  # remove the factor levels if there is one or zero factor level
                for i in range(self.selected_listWidget.count()):
                    item = self.selected_listWidget.takeItem(0)
                    # move formerly selected variables back to the source list
                    if ' :: ' in item.text():
                        if not item.text().endswith(' :: '):  # if there is a factor name and a variable
                            self.source_listWidget.insertItem(
                                find_previous_item_position(self.source_listWidget, self.names,
                                                            item.text().split(' :: ')[1]),
                                item.text().split(' :: ')[1])
                            item.setText(item.text().split(' :: ')[0] + ' :: ')
                    else:
                        self.source_listWidget.insertItem(
                            find_previous_item_position(self.source_listWidget, self.names, item.text()),
                            item.text())

    def optionsButton_clicked(self):
        if self.ylims_dialog.exec_():
            self.ylims = self.ylims_dialog.read_parameters()
            self.ylims[0] = _float_or_none(self.ylims[0])
            self.ylims[1] = _float_or_none(self.ylims[1])

    def read_parameters(self):
        if len(self.factors) > 1:
            return [str(self.selected_listWidget.item(i).text().split(' :: ')[1]) for i in
                    range(self.selected_listWidget.count())], self.factors, self.ylims
        else:
            return [str(self.selected_listWidget.item(i).text()) for i in
                    range(self.selected_listWidget.count())], self.factors, self.ylims


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
        self.names = names
        remove_ceased_vars(self.selected_listWidget, names)
        init_source_vars(self.source_listWidget, names, [self.selected_listWidget])

    def add_var(self):
        if self.selected_listWidget.count() == 0:  # allow only if the list is empty
            add_to_list_widget(self.source_listWidget, self.selected_listWidget)

    def remove_var(self):
        remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())][0],
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
        self.pushButton_2.clicked.connect(self.optionsButton_clicked)

        self.slope_dialog = compare_groups_single_case_slope_dialog(self, names=names)
        self.ylims_dialog = ylims_dialog(self)
        self.single_case_slope_SEs, self.single_case_slope_trial_n = [], 0
        self.ylims = [None, None]

        self.init_vars(names)
        self.show()

    def init_vars(self, names):
        self.names = names
        remove_ceased_vars(self.selected_listWidget, names)
        remove_ceased_vars(self.group_listWidget, names)
        init_source_vars(self.source_listWidget, names, [self.selected_listWidget, self.group_listWidget])
        self.slope_dialog.init_vars(names)

    def add_var(self):
        add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def add_group(self):
        if self.group_listWidget.count() < 2:  # allow maximum two grouping variables
            add_to_list_widget(self.source_listWidget, self.group_listWidget)
    def remove_group(self):
        remove_item_from_list_widget(self.source_listWidget, self.group_listWidget, self.names)

    def on_slopeButton_clicked(self):
        self.slope_dialog.exec_()
        self.single_case_slope_SE, self.single_case_slope_trial_n = self.slope_dialog.read_parameters()

    def optionsButton_clicked(self):
        if self.ylims_dialog.exec_():
            self.ylims = self.ylims_dialog.read_parameters()
            self.ylims[0] = _float_or_none(self.ylims[0])
            self.ylims[1] = _float_or_none(self.ylims[1])

    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                [str(self.group_listWidget.item(i).text()) for i in range(self.group_listWidget.count())],
                self.single_case_slope_SE, int(self.single_case_slope_trial_n), self.ylims)


from .ui import find_text
class find_text_dialog(QtWidgets.QDialog, find_text.Ui_Dialog):
    def __init__(self, parent=None, output_pane=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.output_pane = output_pane
        self.pushButton_next.clicked.connect(self.find_forward_text)
        self.pushButton_previous.clicked.connect(self.find_backward_text)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Close).clicked.connect(self.reject)
        self.lineEdit.setFocus()
        self.show()

    def find_forward_text(self):
        self.output_pane.find(self.lineEdit.text())

    def find_backward_text(self):
        self.output_pane.find(self.lineEdit.text(), QtGui.QTextDocument.FindBackward)


from .ui import preferences
class preferences_dialog(QtWidgets.QDialog, preferences.Ui_Dialog):
    def __init__(self, parent=None):
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

        langs = sorted(['en']+available_langs(domain='cogstat', localedir=os.path.dirname(os.path.abspath(__file__)) +
                                                                          '/locale'))
        # local language names based on https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
        lang_names = {'bg': 'Български (Bulgarian)', 'de': 'Deutsch (German)', 'en': 'English',
                      'el': 'Ελληνικά (Greek)', 'es': 'Español (Spanish)',
                      'et': 'Eesti (Estonian)', 'fa': 'فارسی (Persian)',
                      'fr': 'Français (French)', 'he': 'עברית (Hebrew)',
                      'hr': 'Hrvatski (Croatian)', 'hu': 'Magyar (Hungarian)', 'it': 'Italiano (Italian)',
                      'kk': 'Qazaqsha (Kazakh)', 'ko': '한국어 (Korean)', 'nb': 'Norsk Bokmål (Norvegian Bokmål)',
                      'ro': 'Română (Romanian)', 'ru': 'Русский (Russian)', 'sk': 'Slovenčina (Slovak)',
                      'th': 'ไทย (Thai)'}
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

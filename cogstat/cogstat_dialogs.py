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


def open_data_file(directory):
    #dialog = QtWidgets.QFileDialog()
    #dialog.setFilter(QtCore.QDir.CaseSensitive)
    # TODO how to make the filter case insensitive?
    return str(QtWidgets.QFileDialog.getOpenFileName(None, _('Open data file'), directory,
                                                     '%s *.ods *.xls *.xlsx *.csv *.txt *.tsv *.dat *.log '
                                                     '*.sav *.zsav *.por *.jasp *.omv *.sas7bdat *.xpt *.dta '
                                                     '*.rdata *.Rdata *.rds *.rda;;'
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


def open_demo_data_file(directory):
    return str(QtWidgets.QFileDialog.getOpenFileName(None, _('Open data file'), directory,
                                                     '%s *.ods *.xls *.xlsx *.csv *.txt *.tsv *.dat *.log '
                                                     '*.sav *.zsav *.por *.jasp *.omv *.sas7bdat *.xpt *.dta '
                                                     '*.rdata *.Rdata *.rds *.rda;;'
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
    return str(QtWidgets.QFileDialog.getSaveFileName(None, _('Save result file'), 'CogStat analysis result.html',
                                                     '*.html')[0])

# XXX self.buttonBox.Ok.setEnabled(False) # TODO how can we disable the OK button without the other?
# TODO Some variables are CamelCase - change them

### Various functions ###

def _prepare_list_widgets(source_list_widget, names, selected_list_widgets):
    """Prepare the source and selected list widgets when opening the dialog.

    Dialog list widgets keep the variable list of the previous use. However, the data set may have been changed, and
    the variables should be updated.

    First, if the current dataset has changed, then some former items (variable names) may not be present in the new
    data. Therefore, we remove any items from the selected_list_widgets that are not present in the current dataset
    (names).

    Second, we clear the source_list_widget and add the names (of the actual variables) to the source_list_widget (a
    source list widget), unless they are used in any selected_list_widgets.

    Parameters
    ----------
    source_list_widget : listWidget
        source variables
    names : list of strings
        names of the variables
    selected_list_widgets : list of listWidgets
        list of listWidgets that may contain variables already in use

    Returns
    -------
    Nothing
    """

    # 1. Remove non-existing variables from selected_list_widgets
    for selected_list_widget in selected_list_widgets:
        for item_i in range(selected_list_widget.count()-1, -1, -1):
            if not str(selected_list_widget.item(item_i).text().split(' :: ')[-1]) in list(list(names) + ['']):
                # split(' :: ')[-1] returns the variable name if only a variable name is included, otherwise, the variable
                # after the factor is returned; if there is no variable after ' :: ', then it returns '';
                selected_list_widget.takeItem(item_i)

    # 2. Add variables that are not used already in selected_list_widgets to source_list_widget
    # Collect variable names that are already in use in the already_in_use widgets
    already_in_use_vars = []
    for selected_list_widget in selected_list_widgets:
        already_in_use_vars.extend([selected_list_widget.item(i).text().split(' :: ')[-1]
                                    for i in range(selected_list_widget.count())])
        # split(' :: ')[-1] returns the variable name if only a variable name is included, otherwise, the variable after
        # the factor is returned
    # Clear the list_widget...
    source_list_widget.clear()
    # ...then add the names to them unless they are already in use in other relevant listWidgets
    for var_name in names:
        if not(var_name in already_in_use_vars):
            source_list_widget.addItem(QString(var_name))


def _add_to_list_widget(source_list_widget, target_list_widget, checkable=False):
    """Add the selected item(s) of the source_list_widget to the target_list_widget, and remove the item(s) from the
    source_list_widget.

    Parameters
    ----------
    source_list_widget : qt listWidget
    target_list_widget : qt listWidget
    checkable : bool
        Are the target list widget items checkable?

    Returns
    -------
    int
        number of items added

    """
    # TODO add a maximum parameter, for the maximum number of items that can be added
    number_of_items = len(source_list_widget.selectedItems())
    for item in source_list_widget.selectedItems():
        target_list_widget.addItem(QString(item.text()))
        if checkable:
            #temp_item = QtWidgets.QListWidgetItem(QString(item.text()))
            #temp_item.setCheckState(QtCore.Qt.Unchecked)
            #target_list_widget.addItem(temp_item)
            target_list_widget.findItems(item.text(), QtCore.Qt.MatchExactly)[0].setCheckState(QtCore.Qt.Unchecked)
        source_list_widget.takeItem(source_list_widget.row(item))
    return number_of_items


def _remove_item_from_list_widget(source_list_widget, target_list_widget, names):
    """Remove selected item(s) from target_list_widget, and add it to the source_list_widget.

    Parameters
    ----------
    source_list_widget : qt listWidget
    target_list_widget : qt listWidget
    names : TODO

    Returns
    -------
    int
        number of items removed
    """
    number_of_items = len(target_list_widget.selectedItems())
    for item in target_list_widget.selectedItems():
        target_list_widget.takeItem(target_list_widget.row(item))
        source_list_widget.insertItem(_find_previous_item_position(source_list_widget, names, item.text()), item.text())
    return number_of_items


def _find_previous_item_position(list_widget, names, text_item):
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


def _add_to_list_widget_with_factors(source_list_widget, target_list_widget):
    """Add the selected items of the source_list_widget to the target_list_widget, while using the factor information in
    the target_list_widget.
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


def _remove_from_list_widget_with_factors(source_list_widget, target_list_widget, names=None):
    """Remove selected items from target_list_widget, while using the factor information in the target_list_widget.
    """
    if names is None:
        names = []
    for item in target_list_widget.selectedItems():
        if item.text().split(' :: ')[1]:
            source_list_widget.insertItem(_find_previous_item_position(source_list_widget, names, item.text().
                                                                       split(' :: ')[1]), item.text().split(' :: ')[1])
            item.setText(item.text().split(' :: ')[0]+' :: ')


def _enable_adding_var(button, list_widget, enable):
    """Enable or disable adding variable to a list.

    Parameters
    ----------
    button : qt button
    list_widget : qt listWidget
    enable : bool
        enable or disable

    Returns
    -------

    """
    button.setEnabled(enable)
    if enable:
        list_widget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
    else:
        list_widget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)


def _float_or_none(x):
    """Return the value as float if possible, otherwise return None.

    Parameters
    ----------
    x :
        value to be converted

    Returns
    -------
    float or None

    """
    try:
        return float(x)
    except ValueError:
        return None

### Data dialogs ###

from .ui import pivot
class pivot_dialog(QtWidgets.QDialog, pivot.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.sourceListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.sourceListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addRows.clicked.connect(self.add_rows)
        self.removeRows.clicked.connect(self.remove_rows)
        self.rowsListWidget.doubleClicked.connect(self.remove_rows)
        self.rowsListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.rowsListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addColumns.clicked.connect(self.add_columns)
        self.removeColumns.clicked.connect(self.remove_columns)
        self.columnsListWidget.doubleClicked.connect(self.remove_columns)
        self.columnsListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.columnsListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addPages.clicked.connect(self.add_pages)
        self.removePages.clicked.connect(self.remove_pages)
        self.pagesListWidget.doubleClicked.connect(self.remove_pages)
        self.pagesListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.pagesListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addDependent.clicked.connect(self.add_dependent)
        self.removeDependent.clicked.connect(self.remove_dependent)
        self.dependentListWidget.doubleClicked.connect(self.remove_dependent)
        self.dependentListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.dependentListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.sourceListWidget, names, [self.pagesListWidget, self.columnsListWidget,
                                                             self.rowsListWidget, self.dependentListWidget])

    def add_rows(self):
        _add_to_list_widget(self.sourceListWidget, self.rowsListWidget)
    def remove_rows(self):
        _remove_item_from_list_widget(self.sourceListWidget, self.rowsListWidget, self.names)
    def add_columns(self):
        _add_to_list_widget(self.sourceListWidget, self.columnsListWidget)
    def remove_columns(self):
        _remove_item_from_list_widget(self.sourceListWidget, self.columnsListWidget, self.names)
    def add_pages(self):
        _add_to_list_widget(self.sourceListWidget, self.pagesListWidget)
    def remove_pages(self):
        _remove_item_from_list_widget(self.sourceListWidget, self.pagesListWidget, self.names)
    def add_dependent(self):
        if self.dependentListWidget.count() == 0:  # do this only if the list is empty
            _add_to_list_widget(self.sourceListWidget, self.dependentListWidget)
    def remove_dependent(self):
        _remove_item_from_list_widget(self.sourceListWidget, self.dependentListWidget, self.names)
    
    def read_parameters(self):
        return ([str(self.rowsListWidget.item(i).text()) for i in range(self.rowsListWidget.count())],
                [str(self.columnsListWidget.item(i).text()) for i in range(self.columnsListWidget.count())],
                [str(self.pagesListWidget.item(i).text()) for i in range(self.pagesListWidget.count())], 
                [str(self.dependentListWidget.item(i).text()) for i in range(self.dependentListWidget.count())][0] if
                self.dependentListWidget.count() else [],
                str(self.function.currentText()))


from .ui import diffusion


class diffusion_dialog(QtWidgets.QDialog, diffusion.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.sourceListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.sourceListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addRT.clicked.connect(self.add_RT)
        self.removeRT.clicked.connect(self.remove_RT)
        self.RTListWidget.doubleClicked.connect(self.remove_RT)
        self.RTListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.RTListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addError.clicked.connect(self.add_error)
        self.removeError.clicked.connect(self.remove_error)
        self.errorListWidget.doubleClicked.connect(self.remove_error)
        self.errorListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.errorListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addParticipant.clicked.connect(self.add_participant)
        self.removeParticipant.clicked.connect(self.remove_participant)
        self.participantListWidget.doubleClicked.connect(self.remove_participant)
        self.participantListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.participantListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addCondition.clicked.connect(self.add_condition)
        self.removeCondition.clicked.connect(self.remove_condition)
        self.conditionListWidget.doubleClicked.connect(self.remove_condition)
        self.conditionListWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.conditionListWidget.setDefaultDropAction(QtCore.Qt.MoveAction)

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.sourceListWidget, names,
                              [self.RTListWidget, self.errorListWidget, self.participantListWidget,
                              self.conditionListWidget])
        if self.RTListWidget.count() == 0:
            _enable_adding_var(self.addRT, self.RTListWidget, True)
        if self.errorListWidget.count() == 0:
            _enable_adding_var(self.addError, self.errorListWidget, True)
        if self.participantListWidget.count() == 0:
            _enable_adding_var(self.addParticipant, self.participantListWidget, True)

    # TODO enable and disable relevant elements after drag and drop too

    def add_RT(self):
        if _add_to_list_widget(self.sourceListWidget, self.RTListWidget):
            # only a single variable can be added
            _enable_adding_var(self.addRT, self.RTListWidget, False)


    def remove_RT(self):
        if _remove_item_from_list_widget(self.sourceListWidget, self.RTListWidget, self.names):
            # list is empty, you can add new variable
            _enable_adding_var(self.addRT, self.RTListWidget, True)

    def add_error(self):
        if _add_to_list_widget(self.sourceListWidget, self.errorListWidget):
            # only a single variable can be added
            _enable_adding_var(self.addError, self.errorListWidget, False)

    def remove_error(self):
        if _remove_item_from_list_widget(self.sourceListWidget, self.errorListWidget, self.names):
            # list is empty, you can add new variable
            _enable_adding_var(self.addError, self.errorListWidget, True)

    def add_participant(self):
        if _add_to_list_widget(self.sourceListWidget, self.participantListWidget):
            # only a single variable can be added
            _enable_adding_var(self.addParticipant, self.participantListWidget, False)

    def remove_participant(self):
        if _remove_item_from_list_widget(self.sourceListWidget, self.participantListWidget, self.names):
            # list is empty, you can add new variable
            _enable_adding_var(self.addParticipant, self.participantListWidget, True)

    def add_condition(self):
        _add_to_list_widget(self.sourceListWidget, self.conditionListWidget)

    def remove_condition(self):
        _remove_item_from_list_widget(self.sourceListWidget, self.conditionListWidget, self.names)

    def read_parameters(self):
        return (str(self.errorListWidget.item(0).text()) if range(self.errorListWidget.count()) else '',
                str(self.RTListWidget.item(0).text()) if range(self.RTListWidget.count()) else '',
                str(self.participantListWidget.item(0).text()) if range(self.participantListWidget.count()) else '',
                [str(self.conditionListWidget.item(i).text()) for i in range(self.conditionListWidget.count())],
                str(self.response_coding.currentText()),
                str(self.reaction_time_in.currentText()),
                float(self.scaling_parameter.currentText()))


from .ui import filter_outlier
class filter_outlier(QtWidgets.QDialog, filter_outlier.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.selected_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.selected_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.selected_listWidget])

    def add_var(self):
        _add_to_list_widget(self.source_listWidget, self.selected_listWidget)

    def remove_var(self):
        _remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def read_parameters(self):
        return [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())], \
               self.multivariate_outliers.isChecked()


from .ui import var_properties
class explore_var_dialog(QtWidgets.QDialog, var_properties.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.selected_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.selected_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.selected_listWidget])
    def add_var(self):
        _add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        _remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)
    
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
        return [_float_or_none(self.lineEdit.text()), _float_or_none(self.lineEdit_2.text())], \
               [_float_or_none(self.lineEdit_3.text()), _float_or_none(self.lineEdit_4.text())]


from .ui import explore_var_pairs
class explore_var_pairs_dialog(QtWidgets.QDialog, explore_var_pairs.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.selected_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.selected_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)
        self.pushButton.clicked.connect(self.optionsButton_clicked)

        self.xylims_dialog = xylims_dialog(self)
        self.xlims = [None, None]
        self.ylims = [None, None]

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.selected_listWidget])
    def add_var(self):
        _add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        _remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def optionsButton_clicked(self):
        if self.xylims_dialog.exec_():
            self.xlims, self.ylims = self.xylims_dialog.read_parameters()

    def read_parameters(self):
        return [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())], \
               self.xlims, self.ylims


from .ui import regression
class regression_dialog(QtWidgets.QDialog, regression.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.predicted_listWidget.doubleClicked.connect(self.remove_predicted)
        self.predicted_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.predicted_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addPredicted.clicked.connect(self.add_predicted)
        self.removePredicted.clicked.connect(self.remove_predicted)
        self.predictor_listWidget.doubleClicked.connect(self.remove_predictor)
        self.predictor_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.predictor_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addPredictor.clicked.connect(self.add_predictor)
        self.removePredictor.clicked.connect(self.remove_predictor)
        self.pushButton.clicked.connect(self.optionsButton_clicked)

        self.xylims_dialog = xylims_dialog(self)
        self.xlims = [None, None]
        self.ylims = [None, None]

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.predicted_listWidget,
                                                              self.predictor_listWidget])
        if self.predicted_listWidget.count() == 0:
            _enable_adding_var(self.addPredicted, self.predicted_listWidget, True)

    # TODO enable and disable relevant elements after drag and drop too

    def add_predicted(self):
        if _add_to_list_widget(self.source_listWidget, self.predicted_listWidget):
            # only a single variable can be added
            _enable_adding_var(self.addPredicted, self.predicted_listWidget, False)
    def remove_predicted(self):
        if _remove_item_from_list_widget(self.source_listWidget, self.predicted_listWidget, self.names):
            # list is empty, you can add new variable
            _enable_adding_var(self.addPredicted, self.predicted_listWidget, True)
    def add_predictor(self):
        _add_to_list_widget(self.source_listWidget, self.predictor_listWidget)
    def remove_predictor(self):
        _remove_item_from_list_widget(self.source_listWidget, self.predictor_listWidget, self.names)

    def optionsButton_clicked(self):
        if self.xylims_dialog.exec_():
            self.xlims, self.ylims = self.xylims_dialog.read_parameters()

    def read_parameters(self):
        return [str(self.predicted_listWidget.item(i).text()) for i in range(self.predicted_listWidget.count())], \
               [str(self.predictor_listWidget.item(i).text()) for i in range(self.predictor_listWidget.count())], \
               self.xlims, self.ylims


from .ui import factor
class factor_dialog(QtWidgets.QDialog, factor.Ui_Dialog):
    """Set  a repeated measures factor's name and number of the levels.
    """
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
    """Specify the list of repeated measures factors.
    """
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

    def init_factors(self, factors):
        """Add factors and level values to the list.
        This is needed because the values can be changed by the Display option (adding 'Unnamed factor') too, and not
        only by the Factors dialog itself.

        Parameters
        ----------
        factors : list of list of str and int

        Returns
        -------

        """
        self.listWidget.clear()
        for factor_name, level_n in factors:
            self.listWidget.addItem(QString('%s (%d)' % (factor_name, level_n)))

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


from .ui import display_options_repeated
class display_options_repeated_dialog(QtWidgets.QDialog, display_options_repeated.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.factor_x_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.factor_x_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.factor_x_listWidget.doubleClicked.connect(self.add_color)
        self.factor_color_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.factor_color_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.factor_color_listWidget.doubleClicked.connect(self.remove_color)
        self.add_color_button.clicked.connect(self.add_color)
        self.remove_color_button.clicked.connect(self.remove_color)

    def set_factors(self, factors=None):
        self.factors = factors
        _prepare_list_widgets(self.factor_x_listWidget, self.factors, [self.factor_color_listWidget])
    def add_color(self):
        _add_to_list_widget(self.factor_x_listWidget, self.factor_color_listWidget)
    def remove_color(self):
        _remove_item_from_list_widget(self.factor_x_listWidget, self.factor_color_listWidget, self.factors)
    def read_parameters(self):
        return ([[str(self.factor_x_listWidget.item(i).text()) for i in range(self.factor_x_listWidget.count())] if
                self.factor_x_listWidget.count() else [],
                [str(self.factor_color_listWidget.item(i).text()) for i in range(self.factor_color_listWidget.count())] if
                self.factor_color_listWidget.count() else []],
                [_float_or_none(self.minimum_y.text()), _float_or_none(self.maximum_y.text())])


from .ui import compare_vars
class compare_vars_dialog(QtWidgets.QDialog, compare_vars.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        # TODO drag and drop and moving should handle factor names
        #self.selected_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        #self.selected_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)
        self.pushButton.clicked.connect(self.factorsButton_clicked)
        self.display_options_button.clicked.connect(self.display_options_button_clicked)

        self.factors_dialog = factors_dialog(self)
        self.display_options_repeated_dialog = display_options_repeated_dialog(self)
        self.factors = []
        self.displayfactors = [[], []]
        self.ylims = [None, None]

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.selected_listWidget])

    def add_var(self):
        if self.factors:
            _add_to_list_widget_with_factors(self.source_listWidget, self.selected_listWidget)
        else:
            _add_to_list_widget(self.source_listWidget, self.selected_listWidget)

    def remove_var(self):
        if self.factors:
            _remove_from_list_widget_with_factors(self.source_listWidget, self.selected_listWidget, names=self.names)
        else:
            _remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def show_factors(self):
        # first, remove all items from the selected_listWidget
        previously_used_vars = []
        for i in range(self.selected_listWidget.count()):
            item = self.selected_listWidget.takeItem(0)
            if ' :: ' in item.text():  # factor name and level are present
                if not item.text().endswith(' :: '):  # variable name is also present
                    self.source_listWidget.insertItem(
                        _find_previous_item_position(self.source_listWidget, self.names, item.text().split(' :: ')[1]),
                        item.text().split(' :: ')[1])
                    previously_used_vars.append(item.text().split(' :: ')[1])
                    item.setText(item.text().split(' :: ')[0] + ' :: ')
            else:  # variable name only (without factor name and level)
                self.source_listWidget.insertItem(
                    _find_previous_item_position(self.source_listWidget, self.names, item.text()),
                    item.text())
                previously_used_vars.append(item.text())
        #print(previously_used_vars)

        # add new empty factor levels
        factor_combinations = ['']
        for factor in self.factors:
            factor_combinations = ['%s - %s %s' % (factor_combination, factor[0], level_i + 1) for factor_combination in
                                   factor_combinations for level_i in range(factor[1])]
        factor_combinations = [factor_combination[3:] + ' :: ' for factor_combination in factor_combinations]
        for i, factor_combination in enumerate(factor_combinations):
            if i < len(previously_used_vars):
                self.selected_listWidget.addItem(QString(factor_combination + previously_used_vars[i]))
                self.source_listWidget.takeItem(self.source_listWidget.row(
                    self.source_listWidget.findItems(previously_used_vars[i], QtCore.Qt.MatchExactly)[0]))
            else:
                self.selected_listWidget.addItem(QString(factor_combination))

    def factorsButton_clicked(self):
        self.factors_dialog.init_factors(self.factors)
        if self.factors_dialog.exec_():
            factor_list = self.factors_dialog.read_parameters()
            #print(factor_list)
            # factor list is a list of str, where a str has a 'factorname (x)'format, where x is the number of levels
            self.factors = [[t[:t.rfind(' (')], int(t[t.rfind('(')+1:t.rfind(')')])] for t in factor_list]
            #print(self.factors)
            if self.factors:
                self.show_factors()
                # modify self.displayfactors too because the user possibly changed the factors without changing the
                #  display options (where self.displayfactors are set)
                self.display_options_repeated_dialog.set_factors(factors=[factor[0] for factor in self.factors])
                self.displayfactors, self.ylims = self.display_options_repeated_dialog.read_parameters()
            else:  # remove the factor levels if there is no explicit factor level
                previously_used_vars = []
                for i in range(self.selected_listWidget.count()):
                    item = self.selected_listWidget.takeItem(0)
                    # move formerly selected variables back to the source list
                    if ' :: ' in item.text():  # factor name and level are present
                        if not item.text().endswith(' :: '):  # variable name is also present
                            self.source_listWidget.insertItem(
                                _find_previous_item_position(self.source_listWidget, self.names,
                                                             item.text().split(' :: ')[1]),
                                item.text().split(' :: ')[1])
                            previously_used_vars.append(item.text().split(' :: ')[1])
                            item.setText(item.text().split(' :: ')[0] + ' :: ')
                    else:  # variable name only (without factor name and level)
                        self.source_listWidget.insertItem(
                            _find_previous_item_position(self.source_listWidget, self.names, item.text()),
                            item.text())
                        previously_used_vars.append(item.text())
                for previously_used_var in previously_used_vars:
                    self.selected_listWidget.addItem(QString(previously_used_var))
                    self.source_listWidget.takeItem(self.source_listWidget.row(self.source_listWidget.findItems(previously_used_var, QtCore.Qt.MatchExactly)[0]))

    def display_options_button_clicked(self):
        # If there are several variables but no factors are given, then create a default factor name that can be used in
        #  Display options.
        default_factor_added = False
        if not self.factors and self.selected_listWidget.count() > 1:
            self.factors = [[_('Unnamed factor'), self.selected_listWidget.count()]]
            default_factor_added = True
        self.display_options_repeated_dialog.set_factors(factors=[factor[0] for factor in self.factors])
        if self.display_options_repeated_dialog.exec_():
            self.displayfactors, self.ylims = self.display_options_repeated_dialog.read_parameters()
            self.show_factors()
        else:  # if Display option is cancelled, then remove Unnamed factor
            if default_factor_added:  # do not remove Unnamed factor if dialog is Cancelled but factor was added
                                      #  previously
                self.factors = []

    def read_parameters(self):
        return [str(self.selected_listWidget.item(i).text().split(' :: ')[1])
                for i in range(self.selected_listWidget.count())] \
                if self.factors else \
                [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())], \
                self.factors, self.displayfactors, self.ylims


from .ui import compare_groups_single_case_slope
class compare_groups_single_case_slope_dialog(QtWidgets.QDialog, compare_groups_single_case_slope.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.selected_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.selected_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.selected_listWidget])

    def add_var(self):
        if self.selected_listWidget.count() == 0:  # allow only if the list is empty
            _add_to_list_widget(self.source_listWidget, self.selected_listWidget)

    def remove_var(self):
        _remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())][0] if
                self.selected_listWidget.count() else [],
                str(self.spinBox.text()))


from .ui import display_options_groups
class display_options_groups_dialog(QtWidgets.QDialog, display_options_groups.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.factor_x_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.factor_x_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.factor_color_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.factor_color_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.factor_color_listWidget.doubleClicked.connect(self.remove_color)
        self.factor_panel_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.factor_panel_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.factor_panel_listWidget.doubleClicked.connect(self.remove_panel)
        self.add_color_button.clicked.connect(self.add_color)
        self.remove_color_button.clicked.connect(self.remove_color)
        self.add_panel_button.clicked.connect(self.add_panel)
        self.remove_panel_button.clicked.connect(self.remove_panel)

    def set_factors(self, factors=None):
        self.factors = factors
        _prepare_list_widgets(self.factor_x_listWidget, self.factors, [self.factor_color_listWidget, self.factor_panel_listWidget])
    def add_color(self):
        _add_to_list_widget(self.factor_x_listWidget, self.factor_color_listWidget)
    def remove_color(self):
        _remove_item_from_list_widget(self.factor_x_listWidget, self.factor_color_listWidget, self.factors)
    def add_panel(self):
        _add_to_list_widget(self.factor_x_listWidget, self.factor_panel_listWidget)
    def remove_panel(self):
        _remove_item_from_list_widget(self.factor_x_listWidget, self.factor_panel_listWidget, self.factors)
    def read_parameters(self):
        return ([[str(self.factor_x_listWidget.item(i).text()) for i in range(self.factor_x_listWidget.count())] if
                self.factor_x_listWidget.count() else [],
                [str(self.factor_color_listWidget.item(i).text()) for i in range(self.factor_color_listWidget.count())] if
                self.factor_color_listWidget.count() else [],
                [str(self.factor_panel_listWidget.item(i).text()) for i in range(self.factor_panel_listWidget.count())] if
                self.factor_panel_listWidget.count() else []],
                [_float_or_none(self.minimum_y.text()), _float_or_none(self.maximum_y.text())])


from .ui import compare_groups
class compare_groups_dialog(QtWidgets.QDialog, compare_groups.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.selected_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.selected_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.group_listWidget.doubleClicked.connect(self.remove_group)
        self.group_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.group_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)
        self.add_group_button.clicked.connect(self.add_group)
        self.remove_group_button.clicked.connect(self.remove_group)
        self.pushButton.clicked.connect(self.on_slopeButton_clicked)
        self.display_options_button.clicked.connect(self.display_options_button_clicked)

        self.slope_dialog = compare_groups_single_case_slope_dialog(self)
        self.display_options_groups_dialog = display_options_groups_dialog(self)
        self.displayfactors = [[], [], []]
        self.single_case_slope_SE, self.single_case_slope_trial_n = [], 0
        self.ylims = [None, None]

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.selected_listWidget, self.group_listWidget])
        self.slope_dialog.init_vars(names)

    def add_var(self):
        _add_to_list_widget(self.source_listWidget, self.selected_listWidget)
    def remove_var(self):
        _remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def add_group(self):
        _add_to_list_widget(self.source_listWidget, self.group_listWidget)
    def remove_group(self):
        _remove_item_from_list_widget(self.source_listWidget, self.group_listWidget, self.names)

    def on_slopeButton_clicked(self):
        if self.slope_dialog.exec_():
            self.single_case_slope_SE, self.single_case_slope_trial_n = self.slope_dialog.read_parameters()

    def display_options_button_clicked(self):
        self.display_options_groups_dialog.\
            set_factors(factors=[str(self.group_listWidget.item(i).text()) for i in range(self.group_listWidget.count())])
        if self.display_options_groups_dialog.exec_():
            self.displayfactors, self.ylims = self.display_options_groups_dialog.read_parameters()

    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                [str(self.group_listWidget.item(i).text()) for i in range(self.group_listWidget.count())],
                self.displayfactors,
                self.single_case_slope_SE, int(self.single_case_slope_trial_n), self.ylims)


from .ui import display_options_mixed
class display_options_mixed_dialog(QtWidgets.QDialog, display_options_mixed.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.factor_x_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.factor_x_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.factor_color_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.factor_color_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.factor_color_listWidget.doubleClicked.connect(self.remove_color)
        self.factor_panel_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.factor_panel_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.factor_panel_listWidget.doubleClicked.connect(self.remove_panel)
        self.add_color_button.clicked.connect(self.add_color)
        self.remove_color_button.clicked.connect(self.remove_color)
        self.add_panel_button.clicked.connect(self.add_panel)
        self.remove_panel_button.clicked.connect(self.remove_panel)

    def set_factors(self, factors=None):
        self.factors = factors
        _prepare_list_widgets(self.factor_x_listWidget, self.factors, [self.factor_color_listWidget, self.factor_panel_listWidget])
    def add_color(self):
        _add_to_list_widget(self.factor_x_listWidget, self.factor_color_listWidget)
    def remove_color(self):
        _remove_item_from_list_widget(self.factor_x_listWidget, self.factor_color_listWidget, self.factors)
    def add_panel(self):
        _add_to_list_widget(self.factor_x_listWidget, self.factor_panel_listWidget)
    def remove_panel(self):
        _remove_item_from_list_widget(self.factor_x_listWidget, self.factor_panel_listWidget, self.factors)
    def read_parameters(self):
        return ([[str(self.factor_x_listWidget.item(i).text()) for i in range(self.factor_x_listWidget.count())] if
                self.factor_x_listWidget.count() else [],
                [str(self.factor_color_listWidget.item(i).text()) for i in range(self.factor_color_listWidget.count())] if
                self.factor_color_listWidget.count() else [],
                [str(self.factor_panel_listWidget.item(i).text()) for i in range(self.factor_panel_listWidget.count())] if
                self.factor_panel_listWidget.count() else []],
                [_float_or_none(self.minimum_y.text()), _float_or_none(self.maximum_y.text())])


from .ui import compare_vars_groups
class compare_vars_groups_dialog(QtWidgets.QDialog, compare_vars_groups.Ui_Dialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        # TODO drag and drop and moving should handle factor names
        #self.selected_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        #self.selected_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.group_listWidget.doubleClicked.connect(self.remove_group)
        self.group_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.group_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)
        self.add_group_button.clicked.connect(self.add_group)
        self.remove_group_button.clicked.connect(self.remove_group)
        self.pushButton.clicked.connect(self.factorsButton_clicked)
        self.display_options_button.clicked.connect(self.display_options_button_clicked)
        self.single_case_button.clicked.connect(self.on_slopeButton_clicked)

        self.slope_dialog = compare_groups_single_case_slope_dialog(self)
        self.factors_dialog = factors_dialog(self)
        self.display_options_mixed_dialog = display_options_mixed_dialog(self)
        self.factors = []
        self.displayfactors = [[], []]
        self.single_case_slope_SE, self.single_case_slope_trial_n = [], 0
        self.ylims = [None, None]

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.selected_listWidget])
        self.slope_dialog.init_vars(names)

    def add_var(self):
        if self.factors:
            _add_to_list_widget_with_factors(self.source_listWidget, self.selected_listWidget)
        else:
            _add_to_list_widget(self.source_listWidget, self.selected_listWidget)

    def remove_var(self):
        if self.factors:
            _remove_from_list_widget_with_factors(self.source_listWidget, self.selected_listWidget, names=self.names)
        else:
            _remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def add_group(self):
        if self.group_listWidget.count() < 2:  # allow maximum two grouping variables
            _add_to_list_widget(self.source_listWidget, self.group_listWidget)
    def remove_group(self):
        _remove_item_from_list_widget(self.source_listWidget, self.group_listWidget, self.names)

    def show_factors(self):
        # first, remove all items from the selected_listWidget
        previously_used_vars = []
        for i in range(self.selected_listWidget.count()):
            item = self.selected_listWidget.takeItem(0)
            if ' :: ' in item.text():  # factor name and level are present
                if not item.text().endswith(' :: '):  # variable name is also present
                    self.source_listWidget.insertItem(
                        _find_previous_item_position(self.source_listWidget, self.names, item.text().split(' :: ')[1]),
                        item.text().split(' :: ')[1])
                    previously_used_vars.append(item.text().split(' :: ')[1])
                    item.setText(item.text().split(' :: ')[0] + ' :: ')
            else:  # variable name only (without factor name and level)
                self.source_listWidget.insertItem(
                    _find_previous_item_position(self.source_listWidget, self.names, item.text()),
                    item.text())
                previously_used_vars.append(item.text())
        #print(previously_used_vars)

        # add new empty factor levels
        factor_combinations = ['']
        for factor in self.factors:
            factor_combinations = ['%s - %s %s' % (factor_combination, factor[0], level_i + 1) for factor_combination in
                                   factor_combinations for level_i in range(factor[1])]
        factor_combinations = [factor_combination[3:] + ' :: ' for factor_combination in factor_combinations]
        for i, factor_combination in enumerate(factor_combinations):
            if i < len(previously_used_vars):
                self.selected_listWidget.addItem(QString(factor_combination + previously_used_vars[i]))
                self.source_listWidget.takeItem(self.source_listWidget.row(
                    self.source_listWidget.findItems(previously_used_vars[i], QtCore.Qt.MatchExactly)[0]))
            else:
                self.selected_listWidget.addItem(QString(factor_combination))

    def factorsButton_clicked(self):
        self.factors_dialog.init_factors(self.factors)
        if self.factors_dialog.exec_():
            factor_list = self.factors_dialog.read_parameters()
            #print(factor_list)
            # factor list is a list of str, where a str has a 'factorname (x)'format, where x is the number of levels
            self.factors = [[t[:t.rfind(' (')], int(t[t.rfind('(')+1:t.rfind(')')])] for t in factor_list]
            #print(self.factors)
            if self.factors:
                self.show_factors()
                # modify self.displayfactors too because the user possibly changed the factors without changing the
                #  display options (where self.displayfactors are set)
                self.display_options_mixed_dialog. \
                    set_factors(factors=[str(self.group_listWidget.item(i).text())
                                         for i in range(self.group_listWidget.count())] +
                                        [factor[0] for factor in self.factors])
                self.displayfactors, self.ylims = self.display_options_mixed_dialog.read_parameters()
            else:  # remove the factor levels if there is no explicit factor level
                previously_used_vars = []
                for i in range(self.selected_listWidget.count()):
                    item = self.selected_listWidget.takeItem(0)
                    # move formerly selected variables back to the source list
                    if ' :: ' in item.text():  # factor name and level are present
                        if not item.text().endswith(' :: '):  # variable name is also present
                            self.source_listWidget.insertItem(
                                _find_previous_item_position(self.source_listWidget, self.names,
                                                             item.text().split(' :: ')[1]),
                                item.text().split(' :: ')[1])
                            previously_used_vars.append(item.text().split(' :: ')[1])
                            item.setText(item.text().split(' :: ')[0] + ' :: ')
                    else:  # variable name only (without factor name and level)
                        self.source_listWidget.insertItem(
                            _find_previous_item_position(self.source_listWidget, self.names, item.text()),
                            item.text())
                        previously_used_vars.append(item.text())
                for previously_used_var in previously_used_vars:
                    self.selected_listWidget.addItem(QString(previously_used_var))
                    self.source_listWidget.takeItem(self.source_listWidget.row(self.source_listWidget.findItems(previously_used_var, QtCore.Qt.MatchExactly)[0]))

    def display_options_button_clicked(self):
        # If there are several variables but no factors are given, then create a default factor name that can be used in
        #  Display options.
        default_factor_added = False
        if not self.factors and self.selected_listWidget.count() > 1:
            self.factors = [[_('Unnamed factor'), self.selected_listWidget.count()]]
            restore_factors = True  # if Display option is cancelled, then remove Unnamed factor
            default_factor_added = True
        self.display_options_mixed_dialog.\
            set_factors(factors=[str(self.group_listWidget.item(i).text())
                                 for i in range(self.group_listWidget.count())] +
                                [factor[0] for factor in self.factors])
        if self.display_options_mixed_dialog.exec_():
            self.displayfactors, self.ylims = self.display_options_mixed_dialog.read_parameters()
            self.show_factors()
        else:  # if Display option is cancelled, then remove Unnamed factor
            if default_factor_added:  # do not remove Unnamed factor if dialog is Cancelled but factor was added
                                      #  previously
                self.factors = []

    def on_slopeButton_clicked(self):
        if self.slope_dialog.exec_():
            self.single_case_slope_SE, self.single_case_slope_trial_n = self.slope_dialog.read_parameters()

    def read_parameters(self):
        return [str(self.selected_listWidget.item(i).text().split(' :: ')[1]) for i in range(self.selected_listWidget.count())] \
                if self.factors else [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())], \
                [str(self.group_listWidget.item(i).text()) for i in range(self.group_listWidget.count())], \
                self.factors, self.displayfactors, \
                self.single_case_slope_SE, int(self.single_case_slope_trial_n), self.ylims


from .ui import reliability_internal
class reliability_internal_dialog(QtWidgets.QDialog, reliability_internal.Ui_Dialog):
    """Dialog for internal consistency reliability analysis.

    Unlike other statistical packages (e.g., jamovi or JASP), we use checkboxes next to the chosen variables to set if
     they are reversed (other packages use separate list for the reversed items). This ensures intuitively that (a)
     reversed items are the subset of the chosen variables, and (b) reversed items can be set in the same widget;
     therefore, visually it is more compact.
    """
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.selected_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.selected_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.selected_listWidget])

    def add_var(self):
        _add_to_list_widget(self.source_listWidget, self.selected_listWidget, checkable=True)

    def remove_var(self):
        _remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                [str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())
                 if self.selected_listWidget.item(i).checkState() == QtCore.Qt.Checked])


from .ui import reliability_interrater
class reliability_interrater_dialog(QtWidgets.QDialog, reliability_interrater.Ui_Dialog):
    """Dialog for interrater reliability analysis.

    While the cogstat.py method includes a parameter for ylims (since this parameter is available for the
     chart type that the method uses, the GUI (this dialog) does not have an option for this, since it is a
     highly unusual use case when the y-axis limits are set for interrater reliability analysis.
    """
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setModal(True)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.source_listWidget.doubleClicked.connect(self.add_var)
        self.source_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.source_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.selected_listWidget.doubleClicked.connect(self.remove_var)
        self.selected_listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.selected_listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.addVar.clicked.connect(self.add_var)
        self.removeVar.clicked.connect(self.remove_var)

    def init_vars(self, names):
        self.names = names
        _prepare_list_widgets(self.source_listWidget, names, [self.selected_listWidget])

    def add_var(self):
        _add_to_list_widget(self.source_listWidget, self.selected_listWidget)

    def remove_var(self):
        _remove_item_from_list_widget(self.source_listWidget, self.selected_listWidget, self.names)

    def read_parameters(self):
        return ([str(self.selected_listWidget.item(i).text()) for i in range(self.selected_listWidget.count())],
                self.ratings_averaged_check_box.isChecked())


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
        self.buttonBox.accepted.connect(self.write_and_apply_settings)
        self.buttonBox.rejected.connect(self.reject)
    
        self.init_langs()
        self.init_themes()

        # Init image format
        image_formats = ['png', 'svg']
        self.image_combo_box.addItems(image_formats)
        self.image_combo_box.setCurrentIndex(image_formats.index(csc.image_format))

        # Init detailed error message
        error_messages = [_('Off'), _('On')]
        self.error_combo_box.addItems(error_messages)
        self.error_combo_box.setCurrentIndex(error_messages.index(error_messages[csc.detailed_error_message]))

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
        lang_names = {'ar': 'العربية (Arabic)', 'bg': 'Български (Bulgarian)', 'de': 'Deutsch (German)',
                      'en': 'English', 'el': 'Ελληνικά (Greek)', 'es': 'Español (Spanish)',
                      'et': 'Eesti (Estonian)', 'fa': 'فارسی (Persian)',
                      'fr': 'Français (French)', 'he': 'עברית (Hebrew)',
                      'hr': 'Hrvatski (Croatian)', 'hu': 'Magyar (Hungarian)', 'it': 'Italiano (Italian)',
                      'kk': 'Qazaqsha (Kazakh)', 'ko': '한국어 (Korean)', 'ms': 'Melayu (Malay)',
                      'nb': 'Norsk Bokmål (Norvegian Bokmål)',
                      'ro': 'Română (Romanian)', 'ru': 'Русский (Russian)', 'sk': 'Slovenčina (Slovak)',
                      'th': 'ไทย (Thai)', 'tr': 'Türkçe (Turkish)', 'zh': '汉语 (Chinese)'}
        lang_names_sorted = sorted([lang_names[lang] for lang in langs])
        self.lang_codes = {lang_name: lang_code for lang_code, lang_name in zip(lang_names.keys(), lang_names.values())}

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

    def write_and_apply_settings(self):
        """Save the settings when OK is pressed. Apply the settings so that restart is not needed.
        """
        from . import cogstat_chart as cs_chart

        # Language
        csc.save('language', self.lang_codes[str(self.langComboBox.currentText())])
        # Theme
        csc.theme = str(self.themeComboBox.currentText())
        cs_chart.set_matplotlib_theme()
        csc.save('theme', csc.theme)
        # Image format
        csc.image_format = str(self.image_combo_box.currentText())
        csc.save('image_format', csc.image_format)
        # Detailed error message
        csc.detailed_error_message = bool(self.error_combo_box.currentIndex())
        csc.save('detailed_error_message', str(csc.detailed_error_message))

        self.accept()

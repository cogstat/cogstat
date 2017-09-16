# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pivot.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(360, 350)
        Dialog.setMinimumSize(QtCore.QSize(360, 350))
        Dialog.setMaximumSize(QtCore.QSize(360, 350))
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(160, 310, 191, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label = QtGui.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 151, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.function = QtGui.QComboBox(Dialog)
        self.function.setGeometry(QtCore.QRect(210, 270, 141, 24))
        self.function.setObjectName(_fromUtf8("function"))
        self.function.addItem(_fromUtf8(""))
        self.function.addItem(_fromUtf8(""))
        self.function.addItem(_fromUtf8(""))
        self.function.addItem(_fromUtf8(""))
        self.function.addItem(_fromUtf8(""))
        self.function.addItem(_fromUtf8(""))
        self.function.addItem(_fromUtf8(""))
        self.function.addItem(_fromUtf8(""))
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(210, 130, 55, 13))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(210, 70, 55, 13))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(210, 10, 55, 13))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(210, 200, 141, 16))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_6 = QtGui.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(210, 250, 55, 13))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.removeRows = QtGui.QPushButton(Dialog)
        self.removeRows.setGeometry(QtCore.QRect(180, 170, 21, 21))
        self.removeRows.setObjectName(_fromUtf8("removeRows"))
        self.addRows = QtGui.QPushButton(Dialog)
        self.addRows.setGeometry(QtCore.QRect(180, 150, 21, 21))
        self.addRows.setObjectName(_fromUtf8("addRows"))
        self.addColumns = QtGui.QPushButton(Dialog)
        self.addColumns.setGeometry(QtCore.QRect(180, 90, 21, 21))
        self.addColumns.setObjectName(_fromUtf8("addColumns"))
        self.removeColumns = QtGui.QPushButton(Dialog)
        self.removeColumns.setGeometry(QtCore.QRect(180, 110, 21, 21))
        self.removeColumns.setObjectName(_fromUtf8("removeColumns"))
        self.addPages = QtGui.QPushButton(Dialog)
        self.addPages.setGeometry(QtCore.QRect(180, 30, 21, 21))
        self.addPages.setObjectName(_fromUtf8("addPages"))
        self.removePages = QtGui.QPushButton(Dialog)
        self.removePages.setGeometry(QtCore.QRect(180, 50, 21, 21))
        self.removePages.setObjectName(_fromUtf8("removePages"))
        self.addDependent = QtGui.QPushButton(Dialog)
        self.addDependent.setGeometry(QtCore.QRect(180, 220, 21, 21))
        self.addDependent.setObjectName(_fromUtf8("addDependent"))
        self.removeDependent = QtGui.QPushButton(Dialog)
        self.removeDependent.setGeometry(QtCore.QRect(180, 240, 21, 21))
        self.removeDependent.setObjectName(_fromUtf8("removeDependent"))
        self.sourceListWidget = QtGui.QListWidget(Dialog)
        self.sourceListWidget.setGeometry(QtCore.QRect(10, 30, 150, 230))
        self.sourceListWidget.setMinimumSize(QtCore.QSize(150, 230))
        self.sourceListWidget.setMaximumSize(QtCore.QSize(150, 230))
        self.sourceListWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.sourceListWidget.setObjectName(_fromUtf8("sourceListWidget"))
        self.rowsListWidget = QtGui.QListWidget(Dialog)
        self.rowsListWidget.setGeometry(QtCore.QRect(210, 150, 141, 31))
        self.rowsListWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.rowsListWidget.setObjectName(_fromUtf8("rowsListWidget"))
        self.columnsListWidget = QtGui.QListWidget(Dialog)
        self.columnsListWidget.setGeometry(QtCore.QRect(210, 90, 141, 31))
        self.columnsListWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.columnsListWidget.setObjectName(_fromUtf8("columnsListWidget"))
        self.pagesListWidget = QtGui.QListWidget(Dialog)
        self.pagesListWidget.setGeometry(QtCore.QRect(210, 30, 141, 31))
        self.pagesListWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.pagesListWidget.setObjectName(_fromUtf8("pagesListWidget"))
        self.dependentListWidget = QtGui.QListWidget(Dialog)
        self.dependentListWidget.setGeometry(QtCore.QRect(210, 220, 141, 21))
        self.dependentListWidget.setObjectName(_fromUtf8("dependentListWidget"))

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Pivot table", None))
        self.label.setText(_translate("Dialog", "Available variables", None))
        self.function.setItemText(0, _translate("Dialog", "N", None))
        self.function.setItemText(1, _translate("Dialog", "Sum", None))
        self.function.setItemText(2, _translate("Dialog", "Mean", None))
        self.function.setItemText(3, _translate("Dialog", "Median", None))
        self.function.setItemText(4, _translate("Dialog", "Lower quartile", None))
        self.function.setItemText(5, _translate("Dialog", "Upper quartile", None))
        self.function.setItemText(6, _translate("Dialog", "Standard deviation", None))
        self.function.setItemText(7, _translate("Dialog", "Variance", None))
        self.label_2.setText(_translate("Dialog", "Rows", None))
        self.label_3.setText(_translate("Dialog", "Columns", None))
        self.label_4.setText(_translate("Dialog", "Pages", None))
        self.label_5.setText(_translate("Dialog", "Dependent variable", None))
        self.label_6.setText(_translate("Dialog", "Function", None))
        self.removeRows.setText(_translate("Dialog", "<=", None))
        self.addRows.setText(_translate("Dialog", "=>", None))
        self.addColumns.setText(_translate("Dialog", "=>", None))
        self.removeColumns.setText(_translate("Dialog", "<=", None))
        self.addPages.setText(_translate("Dialog", "=>", None))
        self.removePages.setText(_translate("Dialog", "<=", None))
        self.addDependent.setText(_translate("Dialog", "=>", None))
        self.removeDependent.setText(_translate("Dialog", "<=", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())


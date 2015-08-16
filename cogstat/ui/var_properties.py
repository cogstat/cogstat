# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'var_properties.ui'
#
# Created: Fri Aug  8 11:12:27 2014
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(400, 370)
        Dialog.setMinimumSize(QtCore.QSize(400, 370))
        Dialog.setMaximumSize(QtCore.QSize(400, 370))
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setEnabled(True)
        self.buttonBox.setGeometry(QtCore.QRect(190, 330, 201, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.source_listWidget = QtGui.QListWidget(Dialog)
        self.source_listWidget.setGeometry(QtCore.QRect(20, 30, 161, 192))
        self.source_listWidget.setMouseTracking(False)
        self.source_listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.source_listWidget.setObjectName(_fromUtf8("source_listWidget"))
        self.selected_listWidget = QtGui.QListWidget(Dialog)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setGeometry(QtCore.QRect(220, 30, 171, 192))
        self.selected_listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.selected_listWidget.setObjectName(_fromUtf8("selected_listWidget"))
        self.label = QtGui.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 10, 151, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(220, 10, 121, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.groupBox = QtGui.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(20, 230, 371, 91))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.freq_checkbox = QtGui.QCheckBox(self.groupBox)
        self.freq_checkbox.setGeometry(QtCore.QRect(10, 20, 101, 21))
        self.freq_checkbox.setChecked(True)
        self.freq_checkbox.setTristate(False)
        self.freq_checkbox.setObjectName(_fromUtf8("freq_checkbox"))
        self.dist_checkbox = QtGui.QCheckBox(self.groupBox)
        self.dist_checkbox.setGeometry(QtCore.QRect(10, 40, 121, 21))
        self.dist_checkbox.setChecked(True)
        self.dist_checkbox.setObjectName(_fromUtf8("dist_checkbox"))
        self.ttest_checkbox = QtGui.QCheckBox(self.groupBox)
        self.ttest_checkbox.setGeometry(QtCore.QRect(180, 40, 151, 21))
        self.ttest_checkbox.setChecked(True)
        self.ttest_checkbox.setObjectName(_fromUtf8("ttest_checkbox"))
        self.norm_checkbox = QtGui.QCheckBox(self.groupBox)
        self.norm_checkbox.setGeometry(QtCore.QRect(180, 20, 85, 21))
        self.norm_checkbox.setChecked(True)
        self.norm_checkbox.setObjectName(_fromUtf8("norm_checkbox"))
        self.ttest_value = QtGui.QLineEdit(self.groupBox)
        self.ttest_value.setGeometry(QtCore.QRect(210, 60, 61, 23))
        self.ttest_value.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.ttest_value.setObjectName(_fromUtf8("ttest_value"))
        self.descr_checkbox = QtGui.QCheckBox(self.groupBox)
        self.descr_checkbox.setGeometry(QtCore.QRect(10, 60, 161, 21))
        self.descr_checkbox.setChecked(True)
        self.descr_checkbox.setObjectName(_fromUtf8("descr_checkbox"))
        self.addVar = QtGui.QPushButton(Dialog)
        self.addVar.setGeometry(QtCore.QRect(190, 90, 21, 21))
        self.addVar.setObjectName(_fromUtf8("addVar"))
        self.removeVar = QtGui.QPushButton(Dialog)
        self.removeVar.setGeometry(QtCore.QRect(190, 120, 21, 21))
        self.removeVar.setObjectName(_fromUtf8("removeVar"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Explore variables", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Available variables", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Selected variables", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Dialog", "Statistics", None, QtGui.QApplication.UnicodeUTF8))
        self.freq_checkbox.setText(QtGui.QApplication.translate("Dialog", "Frequencies", None, QtGui.QApplication.UnicodeUTF8))
        self.dist_checkbox.setText(QtGui.QApplication.translate("Dialog", "Distribution", None, QtGui.QApplication.UnicodeUTF8))
        self.ttest_checkbox.setText(QtGui.QApplication.translate("Dialog", "Test central tendency", None, QtGui.QApplication.UnicodeUTF8))
        self.norm_checkbox.setText(QtGui.QApplication.translate("Dialog", "Normality", None, QtGui.QApplication.UnicodeUTF8))
        self.ttest_value.setText(QtGui.QApplication.translate("Dialog", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.descr_checkbox.setText(QtGui.QApplication.translate("Dialog", "Descriptives", None, QtGui.QApplication.UnicodeUTF8))
        self.addVar.setText(QtGui.QApplication.translate("Dialog", "=>", None, QtGui.QApplication.UnicodeUTF8))
        self.removeVar.setText(QtGui.QApplication.translate("Dialog", "<=", None, QtGui.QApplication.UnicodeUTF8))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())


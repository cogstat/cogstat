# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'var_properties.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 370)
        Dialog.setMinimumSize(QtCore.QSize(400, 370))
        Dialog.setMaximumSize(QtCore.QSize(400, 370))
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setEnabled(True)
        self.buttonBox.setGeometry(QtCore.QRect(190, 330, 201, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.source_listWidget = QtWidgets.QListWidget(Dialog)
        self.source_listWidget.setGeometry(QtCore.QRect(20, 30, 161, 221))
        self.source_listWidget.setMouseTracking(False)
        self.source_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.source_listWidget.setObjectName("source_listWidget")
        self.selected_listWidget = QtWidgets.QListWidget(Dialog)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setGeometry(QtCore.QRect(220, 30, 171, 221))
        self.selected_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.selected_listWidget.setObjectName("selected_listWidget")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 10, 151, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(220, 10, 121, 16))
        self.label_2.setObjectName("label_2")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(20, 260, 371, 71))
        self.groupBox.setObjectName("groupBox")
        self.freq_checkbox = QtWidgets.QCheckBox(self.groupBox)
        self.freq_checkbox.setGeometry(QtCore.QRect(10, 20, 101, 21))
        self.freq_checkbox.setChecked(True)
        self.freq_checkbox.setTristate(False)
        self.freq_checkbox.setObjectName("freq_checkbox")
        self.ttest_value = QtWidgets.QLineEdit(self.groupBox)
        self.ttest_value.setGeometry(QtCore.QRect(200, 40, 61, 23))
        self.ttest_value.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.ttest_value.setObjectName("ttest_value")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(170, 20, 171, 16))
        self.label_3.setObjectName("label_3")
        self.addVar = QtWidgets.QPushButton(Dialog)
        self.addVar.setGeometry(QtCore.QRect(190, 90, 21, 21))
        self.addVar.setObjectName("addVar")
        self.removeVar = QtWidgets.QPushButton(Dialog)
        self.removeVar.setGeometry(QtCore.QRect(190, 120, 21, 21))
        self.removeVar.setObjectName("removeVar")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Explore variables"))
        self.label.setText(_translate("Dialog", "Available variables"))
        self.label_2.setText(_translate("Dialog", "Selected variables"))
        self.groupBox.setTitle(_translate("Dialog", "Statistics"))
        self.freq_checkbox.setText(_translate("Dialog", "Frequencies"))
        self.ttest_value.setText(_translate("Dialog", "0"))
        self.label_3.setText(_translate("Dialog", "Central tendency test value"))
        self.addVar.setText(_translate("Dialog", "=>"))
        self.removeVar.setText(_translate("Dialog", "<="))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

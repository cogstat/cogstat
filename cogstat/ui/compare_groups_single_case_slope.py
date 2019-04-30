# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compare_groups_single_case_slope.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(390, 270)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(390, 270))
        Dialog.setMaximumSize(QtCore.QSize(390, 270))
        self.source_listWidget = QtWidgets.QListWidget(Dialog)
        self.source_listWidget.setGeometry(QtCore.QRect(10, 30, 161, 192))
        self.source_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.source_listWidget.setObjectName("source_listWidget")
        self.removeVar = QtWidgets.QPushButton(Dialog)
        self.removeVar.setGeometry(QtCore.QRect(180, 90, 21, 21))
        self.removeVar.setObjectName("removeVar")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 151, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(210, 50, 161, 16))
        self.label_2.setObjectName("label_2")
        self.addVar = QtWidgets.QPushButton(Dialog)
        self.addVar.setGeometry(QtCore.QRect(180, 60, 21, 16))
        self.addVar.setObjectName("addVar")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(40, 230, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.selected_listWidget = QtWidgets.QListWidget(Dialog)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setGeometry(QtCore.QRect(210, 80, 170, 21))
        self.selected_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.selected_listWidget.setObjectName("selected_listWidget")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(210, 150, 120, 16))
        self.label_3.setObjectName("label_3")
        self.spinBox = QtWidgets.QSpinBox(Dialog)
        self.spinBox.setGeometry(QtCore.QRect(210, 170, 71, 27))
        self.spinBox.setObjectName("spinBox")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Compare groups - Single case slope"))
        self.removeVar.setText(_translate("Dialog", "<="))
        self.label.setText(_translate("Dialog", "Available variables"))
        self.label_2.setText(_translate("Dialog", "Slope SE variable"))
        self.addVar.setText(_translate("Dialog", "=>"))
        self.label_3.setText(_translate("Dialog", "Number of trials"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compare_groups_single_case_slope.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.source_listWidget = QtWidgets.QListWidget(Dialog)
        self.source_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.source_listWidget.setObjectName("source_listWidget")
        self.verticalLayout.addWidget(self.source_listWidget)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 2, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.removeVar = QtWidgets.QToolButton(Dialog)
        self.removeVar.setObjectName("removeVar")
        self.gridLayout.addWidget(self.removeVar, 3, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.selected_listWidget = QtWidgets.QListWidget(Dialog)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setMaximumSize(QtCore.QSize(16777215, 25))
        self.selected_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.selected_listWidget.setObjectName("selected_listWidget")
        self.gridLayout.addWidget(self.selected_listWidget, 2, 1, 2, 1)
        self.addVar = QtWidgets.QToolButton(Dialog)
        self.addVar.setObjectName("addVar")
        self.gridLayout.addWidget(self.addVar, 1, 0, 2, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 1, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.spinBox = QtWidgets.QSpinBox(Dialog)
        self.spinBox.setMaximum(1000000000)
        self.spinBox.setObjectName("spinBox")
        self.verticalLayout_4.addWidget(self.spinBox)
        self.gridLayout_3.addLayout(self.verticalLayout_4, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem1, 2, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem2, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout_3, 1, 1, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_2.addWidget(self.buttonBox, 2, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Compare groups - Single case slope"))
        self.label.setText(_translate("Dialog", "Available variables"))
        self.removeVar.setText(_translate("Dialog", "<="))
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

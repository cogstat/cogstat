# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compare_groups.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(410, 308)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(410, 308))
        Dialog.setMaximumSize(QtCore.QSize(410, 308))
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
        self.label_2.setGeometry(QtCore.QRect(210, 30, 161, 16))
        self.label_2.setObjectName("label_2")
        self.addVar = QtWidgets.QPushButton(Dialog)
        self.addVar.setGeometry(QtCore.QRect(180, 60, 21, 16))
        self.addVar.setObjectName("addVar")
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(190, 270, 191, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.selected_listWidget = QtWidgets.QListWidget(Dialog)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setGeometry(QtCore.QRect(210, 50, 170, 71))
        self.selected_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.selected_listWidget.setObjectName("selected_listWidget")
        self.group_listWidget = QtWidgets.QListWidget(Dialog)
        self.group_listWidget.setEnabled(True)
        self.group_listWidget.setGeometry(QtCore.QRect(210, 170, 171, 41))
        self.group_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.group_listWidget.setObjectName("group_listWidget")
        self.remove_group_button = QtWidgets.QPushButton(Dialog)
        self.remove_group_button.setGeometry(QtCore.QRect(180, 190, 21, 21))
        self.remove_group_button.setObjectName("remove_group_button")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(210, 150, 120, 16))
        self.label_3.setObjectName("label_3")
        self.add_group_button = QtWidgets.QPushButton(Dialog)
        self.add_group_button.setGeometry(QtCore.QRect(180, 160, 21, 16))
        self.add_group_button.setObjectName("add_group_button")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setEnabled(True)
        self.pushButton.setGeometry(QtCore.QRect(10, 230, 161, 27))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 270, 88, 28))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Compare groups"))
        self.removeVar.setText(_translate("Dialog", "<="))
        self.label.setText(_translate("Dialog", "Available variables"))
        self.label_2.setText(_translate("Dialog", "Dependent variable(s)"))
        self.addVar.setText(_translate("Dialog", "=>"))
        self.remove_group_button.setText(_translate("Dialog", "<="))
        self.label_3.setText(_translate("Dialog", "Group(s)"))
        self.add_group_button.setText(_translate("Dialog", "=>"))
        self.pushButton.setText(_translate("Dialog", "Single case slope..."))
        self.pushButton_2.setText(_translate("Dialog", "Options..."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

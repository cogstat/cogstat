# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compare_groups.ui'
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
        Dialog.resize(390, 270)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(390, 270))
        Dialog.setMaximumSize(QtCore.QSize(390, 270))
        self.source_listWidget = QtGui.QListWidget(Dialog)
        self.source_listWidget.setGeometry(QtCore.QRect(10, 30, 161, 192))
        self.source_listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.source_listWidget.setObjectName(_fromUtf8("source_listWidget"))
        self.removeVar = QtGui.QPushButton(Dialog)
        self.removeVar.setGeometry(QtCore.QRect(180, 90, 21, 21))
        self.removeVar.setObjectName(_fromUtf8("removeVar"))
        self.label = QtGui.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 151, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(210, 30, 161, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.addVar = QtGui.QPushButton(Dialog)
        self.addVar.setGeometry(QtCore.QRect(180, 60, 21, 16))
        self.addVar.setObjectName(_fromUtf8("addVar"))
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(40, 230, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.selected_listWidget = QtGui.QListWidget(Dialog)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setGeometry(QtCore.QRect(210, 50, 170, 71))
        self.selected_listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.selected_listWidget.setObjectName(_fromUtf8("selected_listWidget"))
        self.group_listWidget = QtGui.QListWidget(Dialog)
        self.group_listWidget.setEnabled(True)
        self.group_listWidget.setGeometry(QtCore.QRect(210, 170, 171, 41))
        self.group_listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.group_listWidget.setObjectName(_fromUtf8("group_listWidget"))
        self.remove_group_button = QtGui.QPushButton(Dialog)
        self.remove_group_button.setGeometry(QtCore.QRect(180, 190, 21, 21))
        self.remove_group_button.setObjectName(_fromUtf8("remove_group_button"))
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(210, 150, 120, 16))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.add_group_button = QtGui.QPushButton(Dialog)
        self.add_group_button.setGeometry(QtCore.QRect(180, 160, 21, 16))
        self.add_group_button.setObjectName(_fromUtf8("add_group_button"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Compare groups", None))
        self.removeVar.setText(_translate("Dialog", "<=", None))
        self.label.setText(_translate("Dialog", "Available variables", None))
        self.label_2.setText(_translate("Dialog", "Dependent variable(s)", None))
        self.addVar.setText(_translate("Dialog", "=>", None))
        self.remove_group_button.setText(_translate("Dialog", "<=", None))
        self.label_3.setText(_translate("Dialog", "Group", None))
        self.add_group_button.setText(_translate("Dialog", "=>", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())


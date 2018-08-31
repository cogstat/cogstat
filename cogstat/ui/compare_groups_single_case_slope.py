# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compare_groups_single_case_slope.ui'
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
        self.label_2.setGeometry(QtCore.QRect(210, 50, 161, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.addVar = QtGui.QPushButton(Dialog)
        self.addVar.setGeometry(QtCore.QRect(180, 60, 21, 16))
        self.addVar.setObjectName(_fromUtf8("addVar"))
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(40, 230, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.selected_listWidget = QtGui.QListWidget(Dialog)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setGeometry(QtCore.QRect(210, 80, 170, 21))
        self.selected_listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.selected_listWidget.setObjectName(_fromUtf8("selected_listWidget"))
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(210, 150, 120, 16))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.spinBox = QtGui.QSpinBox(Dialog)
        self.spinBox.setGeometry(QtCore.QRect(210, 170, 71, 27))
        self.spinBox.setObjectName(_fromUtf8("spinBox"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Compare groups - Single case slope", None))
        self.removeVar.setText(_translate("Dialog", "<=", None))
        self.label.setText(_translate("Dialog", "Available variables", None))
        self.label_2.setText(_translate("Dialog", "Slope SE variable", None))
        self.addVar.setText(_translate("Dialog", "=>", None))
        self.label_3.setText(_translate("Dialog", "Number of trials", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())


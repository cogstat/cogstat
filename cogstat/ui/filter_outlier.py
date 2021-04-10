# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'filter_outlier.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(410, 280)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(410, 280))
        Dialog.setMaximumSize(QtCore.QSize(410, 280))
        Dialog.setSizeGripEnabled(False)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(210, 240, 181, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.addVar = QtWidgets.QPushButton(Dialog)
        self.addVar.setGeometry(QtCore.QRect(190, 90, 21, 21))
        self.addVar.setObjectName("addVar")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 10, 161, 16))
        self.label.setObjectName("label")
        self.selected_listWidget = QtWidgets.QListWidget(Dialog)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setGeometry(QtCore.QRect(220, 30, 171, 192))
        self.selected_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.selected_listWidget.setObjectName("selected_listWidget")
        self.removeVar = QtWidgets.QPushButton(Dialog)
        self.removeVar.setGeometry(QtCore.QRect(190, 120, 21, 21))
        self.removeVar.setObjectName("removeVar")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(220, 10, 171, 16))
        self.label_2.setObjectName("label_2")
        self.source_listWidget = QtWidgets.QListWidget(Dialog)
        self.source_listWidget.setGeometry(QtCore.QRect(20, 30, 161, 192))
        self.source_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.source_listWidget.setObjectName("source_listWidget")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(20, 220, 161, 41))
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_3.setFont(font)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.source_listWidget, self.selected_listWidget)
        Dialog.setTabOrder(self.selected_listWidget, self.addVar)
        Dialog.setTabOrder(self.addVar, self.removeVar)
        Dialog.setTabOrder(self.removeVar, self.buttonBox)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Filter outlier"))
        self.addVar.setText(_translate("Dialog", "=>"))
        self.label.setText(_translate("Dialog", "Available variables"))
        self.removeVar.setText(_translate("Dialog", "<="))
        self.label_2.setText(_translate("Dialog", "Selected variables"))
        self.label_3.setText(_translate("Dialog", "Only interval variables are available"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

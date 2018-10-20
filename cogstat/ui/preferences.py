# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'preferences.ui'
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
        Dialog.resize(390, 160)
        self.buttonBox = QtGui.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(20, 120, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.label = QtGui.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 10, 91, 21))
        self.label.setObjectName(_fromUtf8("label"))
        self.langComboBox = QtGui.QComboBox(Dialog)
        self.langComboBox.setGeometry(QtCore.QRect(120, 10, 151, 22))
        self.langComboBox.setObjectName(_fromUtf8("langComboBox"))
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(40, 40, 321, 16))
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.themeComboBox = QtGui.QComboBox(Dialog)
        self.themeComboBox.setGeometry(QtCore.QRect(120, 70, 151, 22))
        self.themeComboBox.setObjectName(_fromUtf8("themeComboBox"))
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(30, 70, 91, 21))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(40, 100, 321, 16))
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("accepted()")), Dialog.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL(_fromUtf8("rejected()")), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Preferences", None))
        self.label.setText(_translate("Dialog", "Language", None))
        self.label_2.setText(_translate("Dialog", "You should restart CogStat to use the new language", None))
        self.label_3.setText(_translate("Dialog", "Chart theme", None))
        self.label_4.setText(_translate("Dialog", "You should restart CogStat to use the new theme", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())


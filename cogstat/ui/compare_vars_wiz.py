# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compare_vars_wiz.ui'
#
# Created: Fri Aug  8 11:12:26 2014
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Wizard(object):
    def setupUi(self, Wizard):
        Wizard.setObjectName(_fromUtf8("Wizard"))
        Wizard.resize(400, 280)
        Wizard.setMinimumSize(QtCore.QSize(400, 280))
        Wizard.setMaximumSize(QtCore.QSize(400, 280))
        self.wizardPage1 = QtGui.QWizardPage()
        self.wizardPage1.setObjectName(_fromUtf8("wizardPage1"))
        self.radioButton = QtGui.QRadioButton(self.wizardPage1)
        self.radioButton.setGeometry(QtCore.QRect(20, 20, 101, 21))
        self.radioButton.setObjectName(_fromUtf8("radioButton"))
        self.radioButton_2 = QtGui.QRadioButton(self.wizardPage1)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 50, 101, 21))
        self.radioButton_2.setObjectName(_fromUtf8("radioButton_2"))
        Wizard.addPage(self.wizardPage1)
        self.wizardPage = QtGui.QWizardPage()
        self.wizardPage.setObjectName(_fromUtf8("wizardPage"))
        self.label_3 = QtGui.QLabel(self.wizardPage)
        self.label_3.setGeometry(QtCore.QRect(20, 10, 181, 16))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        Wizard.addPage(self.wizardPage)
        self.wizardPage2 = QtGui.QWizardPage()
        self.wizardPage2.setObjectName(_fromUtf8("wizardPage2"))
        self.source_listWidget = QtGui.QListWidget(self.wizardPage2)
        self.source_listWidget.setGeometry(QtCore.QRect(0, 20, 161, 192))
        self.source_listWidget.setObjectName(_fromUtf8("source_listWidget"))
        self.removeVar = QtGui.QPushButton(self.wizardPage2)
        self.removeVar.setGeometry(QtCore.QRect(170, 110, 21, 21))
        self.removeVar.setObjectName(_fromUtf8("removeVar"))
        self.label = QtGui.QLabel(self.wizardPage2)
        self.label.setGeometry(QtCore.QRect(0, 0, 151, 16))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.wizardPage2)
        self.label_2.setGeometry(QtCore.QRect(200, 0, 121, 16))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.addVar = QtGui.QPushButton(self.wizardPage2)
        self.addVar.setGeometry(QtCore.QRect(170, 80, 21, 21))
        self.addVar.setObjectName(_fromUtf8("addVar"))
        self.selected_listWidget = QtGui.QListWidget(self.wizardPage2)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setGeometry(QtCore.QRect(200, 20, 171, 192))
        self.selected_listWidget.setObjectName(_fromUtf8("selected_listWidget"))
        Wizard.addPage(self.wizardPage2)

        self.retranslateUi(Wizard)
        QtCore.QMetaObject.connectSlotsByName(Wizard)

    def retranslateUi(self, Wizard):
        Wizard.setWindowTitle(QtGui.QApplication.translate("Wizard", "Wizard", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton.setText(QtGui.QApplication.translate("Wizard", "One factor", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButton_2.setText(QtGui.QApplication.translate("Wizard", "Several factors", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Wizard", "Not implemented yet", None, QtGui.QApplication.UnicodeUTF8))
        self.removeVar.setText(QtGui.QApplication.translate("Wizard", "<=", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Wizard", "Variables to choose from", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Wizard", "Selected variables", None, QtGui.QApplication.UnicodeUTF8))
        self.addVar.setText(QtGui.QApplication.translate("Wizard", "=>", None, QtGui.QApplication.UnicodeUTF8))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Wizard = QtGui.QWizard()
    ui = Ui_Wizard()
    ui.setupUi(Wizard)
    Wizard.show()
    sys.exit(app.exec_())


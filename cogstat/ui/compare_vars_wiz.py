# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'compare_vars_wiz.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Wizard(object):
    def setupUi(self, Wizard):
        Wizard.setObjectName("Wizard")
        Wizard.resize(400, 280)
        Wizard.setMinimumSize(QtCore.QSize(400, 280))
        Wizard.setMaximumSize(QtCore.QSize(400, 280))
        self.wizardPage1 = QtWidgets.QWizardPage()
        self.wizardPage1.setObjectName("wizardPage1")
        self.radioButton = QtWidgets.QRadioButton(self.wizardPage1)
        self.radioButton.setGeometry(QtCore.QRect(20, 20, 101, 21))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.wizardPage1)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 50, 101, 21))
        self.radioButton_2.setObjectName("radioButton_2")
        Wizard.addPage(self.wizardPage1)
        self.wizardPage = QtWidgets.QWizardPage()
        self.wizardPage.setObjectName("wizardPage")
        self.label_3 = QtWidgets.QLabel(self.wizardPage)
        self.label_3.setGeometry(QtCore.QRect(20, 10, 181, 16))
        self.label_3.setObjectName("label_3")
        Wizard.addPage(self.wizardPage)
        self.wizardPage2 = QtWidgets.QWizardPage()
        self.wizardPage2.setObjectName("wizardPage2")
        self.source_listWidget = QtWidgets.QListWidget(self.wizardPage2)
        self.source_listWidget.setGeometry(QtCore.QRect(0, 20, 161, 192))
        self.source_listWidget.setObjectName("source_listWidget")
        self.removeVar = QtWidgets.QPushButton(self.wizardPage2)
        self.removeVar.setGeometry(QtCore.QRect(170, 110, 21, 21))
        self.removeVar.setObjectName("removeVar")
        self.label = QtWidgets.QLabel(self.wizardPage2)
        self.label.setGeometry(QtCore.QRect(0, 0, 151, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.wizardPage2)
        self.label_2.setGeometry(QtCore.QRect(200, 0, 121, 16))
        self.label_2.setObjectName("label_2")
        self.addVar = QtWidgets.QPushButton(self.wizardPage2)
        self.addVar.setGeometry(QtCore.QRect(170, 80, 21, 21))
        self.addVar.setObjectName("addVar")
        self.selected_listWidget = QtWidgets.QListWidget(self.wizardPage2)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setGeometry(QtCore.QRect(200, 20, 171, 192))
        self.selected_listWidget.setObjectName("selected_listWidget")
        Wizard.addPage(self.wizardPage2)

        self.retranslateUi(Wizard)
        QtCore.QMetaObject.connectSlotsByName(Wizard)

    def retranslateUi(self, Wizard):
        _translate = QtCore.QCoreApplication.translate
        Wizard.setWindowTitle(_translate("Wizard", "Wizard"))
        self.radioButton.setText(_translate("Wizard", "One factor"))
        self.radioButton_2.setText(_translate("Wizard", "Several factors"))
        self.label_3.setText(_translate("Wizard", "Not implemented yet"))
        self.removeVar.setText(_translate("Wizard", "<="))
        self.label.setText(_translate("Wizard", "Variables to choose from"))
        self.label_2.setText(_translate("Wizard", "Selected variables"))
        self.addVar.setText(_translate("Wizard", "=>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Wizard = QtWidgets.QWizard()
    ui = Ui_Wizard()
    ui.setupUi(Wizard)
    Wizard.show()
    sys.exit(app.exec_())


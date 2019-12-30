# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'diffusion.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(360, 350)
        Dialog.setMinimumSize(QtCore.QSize(360, 350))
        Dialog.setMaximumSize(QtCore.QSize(360, 350))
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(160, 310, 191, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 151, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(210, 130, 131, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(210, 70, 55, 13))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(210, 10, 91, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(210, 200, 141, 16))
        self.label_5.setObjectName("label_5")
        self.removeParticipant = QtWidgets.QPushButton(Dialog)
        self.removeParticipant.setGeometry(QtCore.QRect(180, 170, 21, 21))
        self.removeParticipant.setObjectName("removeParticipant")
        self.addParticipant = QtWidgets.QPushButton(Dialog)
        self.addParticipant.setGeometry(QtCore.QRect(180, 150, 21, 21))
        self.addParticipant.setObjectName("addParticipant")
        self.addError = QtWidgets.QPushButton(Dialog)
        self.addError.setGeometry(QtCore.QRect(180, 90, 21, 21))
        self.addError.setObjectName("addError")
        self.removeError = QtWidgets.QPushButton(Dialog)
        self.removeError.setGeometry(QtCore.QRect(180, 110, 21, 21))
        self.removeError.setObjectName("removeError")
        self.addRT = QtWidgets.QPushButton(Dialog)
        self.addRT.setGeometry(QtCore.QRect(180, 30, 21, 21))
        self.addRT.setObjectName("addRT")
        self.removeRT = QtWidgets.QPushButton(Dialog)
        self.removeRT.setGeometry(QtCore.QRect(180, 50, 21, 21))
        self.removeRT.setObjectName("removeRT")
        self.addCondition = QtWidgets.QPushButton(Dialog)
        self.addCondition.setGeometry(QtCore.QRect(180, 220, 21, 21))
        self.addCondition.setObjectName("addCondition")
        self.removeCondition = QtWidgets.QPushButton(Dialog)
        self.removeCondition.setGeometry(QtCore.QRect(180, 240, 21, 21))
        self.removeCondition.setObjectName("removeCondition")
        self.sourceListWidget = QtWidgets.QListWidget(Dialog)
        self.sourceListWidget.setGeometry(QtCore.QRect(10, 30, 150, 230))
        self.sourceListWidget.setMinimumSize(QtCore.QSize(150, 230))
        self.sourceListWidget.setMaximumSize(QtCore.QSize(150, 230))
        self.sourceListWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.sourceListWidget.setObjectName("sourceListWidget")
        self.participantListWidget = QtWidgets.QListWidget(Dialog)
        self.participantListWidget.setGeometry(QtCore.QRect(210, 150, 141, 21))
        self.participantListWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.participantListWidget.setObjectName("participantListWidget")
        self.errorListWidget = QtWidgets.QListWidget(Dialog)
        self.errorListWidget.setGeometry(QtCore.QRect(210, 90, 141, 21))
        self.errorListWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.errorListWidget.setObjectName("errorListWidget")
        self.RTListWidget = QtWidgets.QListWidget(Dialog)
        self.RTListWidget.setGeometry(QtCore.QRect(210, 30, 141, 21))
        self.RTListWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.RTListWidget.setObjectName("RTListWidget")
        self.conditionListWidget = QtWidgets.QListWidget(Dialog)
        self.conditionListWidget.setGeometry(QtCore.QRect(210, 220, 141, 51))
        self.conditionListWidget.setObjectName("conditionListWidget")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Behavioral data diffusion analysis"))
        self.label.setText(_translate("Dialog", "Available variables"))
        self.label_2.setText(_translate("Dialog", "Participant"))
        self.label_3.setText(_translate("Dialog", "Error"))
        self.label_4.setText(_translate("Dialog", "Reaction time"))
        self.label_5.setText(_translate("Dialog", "Condition(s)"))
        self.removeParticipant.setText(_translate("Dialog", "<="))
        self.addParticipant.setText(_translate("Dialog", "=>"))
        self.addError.setText(_translate("Dialog", "=>"))
        self.removeError.setText(_translate("Dialog", "<="))
        self.addRT.setText(_translate("Dialog", "=>"))
        self.removeRT.setText(_translate("Dialog", "<="))
        self.addCondition.setText(_translate("Dialog", "=>"))
        self.removeCondition.setText(_translate("Dialog", "<="))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

# Form implementation generated from reading ui file 'reliability_internal.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(588, 325)
        self.gridLayout_3 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(parent=Dialog)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.label_4 = QtWidgets.QLabel(parent=Dialog)
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.source_listWidget = QtWidgets.QListWidget(parent=Dialog)
        self.source_listWidget.setMouseTracking(False)
        self.source_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.source_listWidget.setObjectName("source_listWidget")
        self.verticalLayout_3.addWidget(self.source_listWidget)
        self.gridLayout_2.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.addVar = QtWidgets.QToolButton(parent=Dialog)
        self.addVar.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.addVar.setObjectName("addVar")
        self.verticalLayout_2.addWidget(self.addVar)
        self.removeVar = QtWidgets.QToolButton(parent=Dialog)
        self.removeVar.setArrowType(QtCore.Qt.ArrowType.LeftArrow)
        self.removeVar.setObjectName("removeVar")
        self.verticalLayout_2.addWidget(self.removeVar)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.gridLayout_2.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(parent=Dialog)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(parent=Dialog)
        font = QtGui.QFont()
        font.setItalic(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.selected_listWidget = QtWidgets.QListWidget(parent=Dialog)
        self.selected_listWidget.setEnabled(True)
        self.selected_listWidget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.selected_listWidget.setObjectName("selected_listWidget")
        self.verticalLayout.addWidget(self.selected_listWidget)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 2, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 2)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=Dialog)
        self.buttonBox.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBox.sizePolicy().hasHeightForWidth())
        self.buttonBox.setSizePolicy(sizePolicy)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Help|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_3.addWidget(self.buttonBox, 1, 1, 1, 1)
        self.label.setBuddy(self.source_listWidget)
        self.label_2.setBuddy(self.selected_listWidget)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Internal consistency reliability"))
        self.label.setText(_translate("Dialog", "Available variables"))
        self.label_2.setText(_translate("Dialog", "Selected variables"))
        self.label_3.setText(_translate("Dialog", "Check the reversed items"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'triplet_ctrl.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget


class TripletControlWidget(QWidget):
    def setupUi(self, Form):
        # Form.setObjectName("Form")
        # Form.resize(640, 480)
        # Form.setStyleSheet("background-color: rgb(46, 52, 54);")
        self.frame = QtWidgets.QFrame(self)
        self.frame.setGeometry(QtCore.QRect(10, 10, 169, 47))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.spinBox_1 = QtWidgets.QSpinBox(self.frame)
        self.spinBox_1.setObjectName("spinBox_1")
        self.horizontalLayout.addWidget(self.spinBox_1)
        self.spinBox_2 = QtWidgets.QSpinBox(self.frame)
        self.spinBox_2.setObjectName("spinBox_2")
        self.horizontalLayout.addWidget(self.spinBox_2)
        self.spinBox_3 = QtWidgets.QSpinBox(self.frame)
        self.spinBox_3.setObjectName("spinBox_3")
        self.horizontalLayout.addWidget(self.spinBox_3)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    @property
    def value(self):
        return self.spinBox_1.value(), self.spinBox_2.value(), self.spinBox_3.value()

    @value.setter
    def value(self, val):
        self.spinBox_1.setValue(val[0])
        self.spinBox_2.setValue(val[1])
        self.spinBox_3.setValue(val[2])

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

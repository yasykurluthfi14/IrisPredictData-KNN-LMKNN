# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Antar_Muka_Skripsi.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import Qt
from os import path
import pandas as pd
import numpy as np
import pathlib
from KNN import *
# from LMKNNGPT import *

class Ui_MainWindow(object):
    dataset = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 602)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 100, 311, 111))
        font = QtGui.QFont()
        font.setFamily('Times New Roman')
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(160, 60, 121, 31))
        font = QtGui.QFont()
        font.setFamily('Times New Roman')
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(lambda: self.tombolklasifikasi())
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(160, 20, 121, 31))
        font = QtGui.QFont()
        font.setFamily('Times New Roman')
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.spinBox = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox.setGeometry(QtCore.QRect(230, 25, 41, 21))
        font = QtGui.QFont()
        font.setFamily('Times New Roman')
        font.setPointSize(12)
        self.spinBox.setFont(font)
        self.spinBox.setObjectName("spinBox")
        self.spinBox.valueChanged.connect(self.get_value_spinbox)
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(20, 30, 82, 17))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 60, 82, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 220, 771, 331))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.textEdit_2 = QtWidgets.QTextEdit(self.groupBox_2)
        self.textEdit_2.setGeometry(QtCore.QRect(10, 40, 111, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(self.groupBox_2)
        self.textEdit_3.setGeometry(QtCore.QRect(320, 10, 141, 31))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_4 = QtWidgets.QTextEdit(self.groupBox_2)
        self.textEdit_4.setGeometry(QtCore.QRect(570, 40, 131, 31))
        self.textEdit_4.setObjectName("textEdit_4")
        self.tableView_3 = QtWidgets.QTableView(self.groupBox_2)
        self.tableView_3.setGeometry(QtCore.QRect(10, 100, 501, 221))
        self.tableView_3.setObjectName("tableView_3")
        self.tableView_3.verticalHeader().setVisible(False)
        self.tableView_3.horizontalHeader().setVisible(False)
        self.tableView = QtWidgets.QTableView(self.groupBox_2)
        self.tableView.setGeometry(QtCore.QRect(530, 100, 231, 221))
        self.tableView.setObjectName("tableView")
        self.tableView_2 = QtWidgets.QTableView(self.centralwidget)
        self.tableView_2.setGeometry(QtCore.QRect(360, 51, 421, 161))
        self.tableView_2.setObjectName("tableView_2")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(40, 60, 271, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setItemText(2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Metode "))
        self.pushButton_4.setText(_translate("MainWindow", "Mulai Pengujian"))
        self.lineEdit.setText(_translate("MainWindow", " Nilai K :"))
        self.radioButton.setText(_translate("MainWindow", "KNN"))
        self.radioButton_2.setText(_translate("MainWindow", "LMKNN"))
        self.textEdit_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Times New Roman\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">Detail Akurasi</span></p></body></html>"))
        self.textEdit_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Times New Roman\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">Output Klasifikasi</span></p></body></html>"))
        self.textEdit_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Times New Roman\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">Confusion Matrix</span></p></body></html>"))
        
        self.comboBox.setItemText(0, _translate("MainWindow", "Iris.xlsx"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Irisd2.xlsx"))
        
        
    def input_data(self, path):
        
        file = pathlib.Path(path)
        if file.exists():
            self.dataset = pd.read_excel(path)
        else:
            print('Path error : tidak ada data pada path ->',path)
             

    def get_dataset(self):
        return self.dataset     
    
    def get_total_feature(self, dataset):
        data = self.dataset
        return data.shape[0]

    def open_file(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "QFileDialog.getOpenFileName()", "", "Excel File (*.xlsx)", options=options)
        if fileName:
            self.textEdit.setText(fileName)
            self.input_data(fileName)
            # self.proses.preprocessing()
            # print(self.proses.preprocessing)
            
            dataset = self.dataset
            print(dataset)

            jumlah_data = self.get_total_feature(dataset)

            tampil = dataset.iloc[:, 0:7].values.tolist()
            #label = dataset.label.tolist()
            data_view = []
            for i in range(1,149):
                temp = []
                temp.append(tampil[i-1])
                #temp.append(label[i])
                data_view.append(temp)
            data_view = np.array(data_view, dtype=object)
            data_model = TabelView(data_view)
            self.tableView_2.setModel(data_model)

    def get_value_spinbox(self):
        value = self.spinBox.value()
        return value

    def tombolklasifikasi(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Apakah anda yakin Ingin melanjutkan Klasifikasi")
        msg.setInformativeText("Pastikan Semua Parameter telah terisi dengan benar")
        msg.setWindowTitle("InformationBox")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes |QtWidgets.QMessageBox.No ) #| QtWidgets.QMessageBox.Cancel)
        if msg.exec_()==QtWidgets.QMessageBox.Yes:
            if self.radioButton.isChecked():

                path = self.comboBox.currentText()
                k = self.get_value_spinbox()                
                knn = KNN()

                proses_knn = knn.proses(k, path)
                score = np.array(proses_knn, dtype=object)
                data_model = TabelView(score)

                self.tableView_3.setModel(data_model)



            elif self.ui.seleksifiturMI.isChecked():
                if self.get_kernel() != -1 and self.get_c() != -1 and self.get_k()!= 0:
                    kernel_clf = self.get_kernel()
                    c_clf = self.get_c()
                    k_fs = self.get_k()
                    dataset = self.model.feature_selectionMI(k_fs)
                    dataset.to_excel('Asset/datasethasilMI.xlsx',index=False)
                    #dataset.to_excel('assets/dataset_seleksiMI.xlsx',index=False)
                    score = self.model.classify(dataset,kernel_clf,c_clf)
                    score = np.array(score)
                    data_model = TabelView(score)
                    self.ui.tampil_hasil_train.setModel(data_model)
                    total_fitur = self.model.get_total_feature(dataset)
                    self.ui.tampil_hasil_jumlah_fitur.setText(str(total_fitur))
                    self.ui.tampil_hasil_selisihjumlah_fitur.setText(str(2004-total_fitur))
                else:
                    self.message_box_warning()
        else:
            self.message_box_warning()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

class TabelView(QtCore.QAbstractTableModel):
    
  #menginisialisasi table model pyqt dan data input
    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data
  #mendefinisikan role dari list
    def data(self, index, role):
        if role == Qt.DisplayRole:
            # Note: self._data[index.row()][index.column()] will also work
            value = self._data[index.row(), index.column()]
            return str(value)
  #melakukan count row pada index di shape 1 dari list
    def rowCount(self, index):
        return self._data.shape[0]
#melakukan count column pada index di shape 1 dari list
    def columnCount(self, index):
        if len(self._data.shape) > 1:
            return self._data.shape[1]
        else:
            return 0
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


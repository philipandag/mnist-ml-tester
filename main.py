import sys

import joblib
import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Wczytanie bazy danych
        self.mnist = joblib.load('mnist.joblib')
        self.X = self.mnist.data
        self.y = self.mnist.target

        # Podział na zbiór treningowy i testowy
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        # Wybór modelu
        self.model_type, ok = QInputDialog.getItem(self, "Wybierz", "Wybierz model",
                                                   ["Drzewo decyzyjne z biblioteki sklearn", "Własne drzewo decyzyjne",
                                                    "Random Forest z biblioteki sklearn"], 0, False)
        # Jeśli użytkownik kliknął "Cancel"
        if not ok:
            sys.exit()

        # Jeśli wybrał własne drzewo decyzyjne itd.
        if self.model_type == "Własne drzewo decyzyjne":
            # Wczytaj tutaj model self.clf = ...
            pass
        elif self.model_type == "Drzewo decyzyjne z biblioteki sklearn":
            self.clf = DecisionTreeClassifier()
        elif self.model_type == "Random Forest z biblioteki sklearn":
            self.clf = RandomForestClassifier()

        # Trenowanie modelu
        msgBox = QMessageBox()
        msgBox.setWindowTitle("Trenowanie w toku             ")
        msgBox.show()

        print("Trenuję...")
        print(self.X_train.shape)

        self.clf.fit(self.X_train, self.y_train)

        msgBox.close()
        msgBox2 = QMessageBox()
        msgBox2.setWindowTitle("Trenowanie zakończone")
        msgBox2.setText(f"Wynik na danych testowych: {self.clf.score(self.X_test, self.y_test)}")
        msgBox2.exec_()

        # Ustawienie interfejsu
        self.canvas = Canvas(self, width=280, height=280)
        self.canvas.move(20, 25)

        # Ustawienie menu
        menu_bar = self.menuBar()
        classify_action = QAction("Klasyfikuj", self)
        classify_action.triggered.connect(self.classify)
        menu_bar.addAction(classify_action)

        clear_action = QAction("Wyczyść", self)
        clear_action.triggered.connect(self.canvas.clear)
        menu_bar.addAction(clear_action)

        self.results_action = QAction("Wyniki", self)
        menu_bar.addAction(self.results_action)

        # Ustawienie okna
        self.setFixedSize(320, 320)
        self.setWindowTitle("Rozpoznawanie cyfr")

    # Funkcja wywoływana po kliknięciu "Klasyfikuj"
    def classify(self):

        image = self.canvas.converted_image
        # Zamiana obrazka na wektor floatów o długości 784
        image = image.reshape(1, -1).astype(np.float32)

        # Tutaj klasyfikacja
        prediction = self.clf.predict(image)

        # Wpisanie wyniku do menu
        self.results_action.setText(f"Wyniki: {prediction}")

        # Wyczyszczenie kanwy
        self.canvas.image.fill(Qt.white)
        self.canvas.update()


class Canvas(QWidget):
    def __init__(self, parent, width, height):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.converted_image = None
        self.setFixedSize(self.width, self.height)

        self.image = QImage(280, 280, QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.drawing = False
        self.last_point = QPoint()

    def paintEvent(self, event):
        # Rysowanie płótna
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        self.drawing = True
        self.last_point = event.pos()

        self.image = self.image.scaled(280, 280)

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 15, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()

            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            # Konwersja obrazka na czarno-biały 28x28 px
            _image = self.image.convertToFormat(QImage.Format_Grayscale8).scaled(28, 28)
            width = _image.width()
            height = _image.height()

            # Wyciągnięcie danych z pikseli i zamiana na macierz numpy
            data = _image.bits().asstring(width * height)
            arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width))

            self.image = self.image.scaled(28, 28)
            self.update()

            self.converted_image = arr

    def clear(self):
        self.image.fill(Qt.white)
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

import sys

import PyQt5.QtGui
import joblib
import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from AbstractModel import Model

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Wczytanie bazy danych
        self.mnist = joblib.load('mnist.joblib')
        self.X = self.mnist.data
        self.y = self.mnist.target

        self.resolution = int(np.sqrt(self.X.shape[1]))

        # Sprawdzenie, czy baza danych została wczytana poprawnie
        print(f"X: {self.X.shape}")
        print(self.mnist.data)
        print(f"y: {self.y.shape}")
        print(self.mnist.target)

        # Podział na zbiór treningowy i testowy
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        # Wybór modelu
        self.model_type, ok = QInputDialog.getItem(self, "Wybierz", "Wybierz model",
                                                   ["Model 1", "Model 2", "Model 3"], 0, False)
        # Jeśli użytkownik kliknął "Cancel"
        if not ok:
            sys.exit()

        # Jeśli wybrał własne drzewo decyzyjne itd.
        if self.model_type == "Model 1":
            self.model = None
        elif self.model_type == "Model 2":
            self.model = None
        elif self.model_type == "Model 3":
            self.model = None

        # Trenowanie modelu
        msgBox = QMessageBox()
        msgBox.setWindowTitle("Trenowanie w toku             ")
        msgBox.show()

        print("Trenuję...")
        print(f"Rozmiar danych treningowych {self.X_train.shape}")

        #self.model.fit(self.X_train, self.y_train)

        #msgBox.close()
        #msgBox2 = QMessageBox()
        #msgBox2.setWindowTitle("Trenowanie zakończone")
        #msgBox2.setText(f"Wynik na danych testowych: {self.model.score(self.X_test, self.y_test)}")
        #msgBox2.exec_()

        # Ustawienie interfejsu
        self.canvas = Canvas(self, width=280, height=280, resolution=self.resolution)
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
        prediction = self.model.predict(image)

        # Jeśli to wektor prawdopodobieństw to wybierz najbardziej prawdopodobną cyfrę
        if len(prediction.shape) == 2:
            prediction = np.argmax(prediction, axis=1)[0]

        # Wpisanie wyniku do menu
        self.results_action.setText(f"Wyniki: {prediction}")

        # Wyczyszczenie kanwy
        self.canvas.image.fill(Qt.white)
        self.canvas.update()


class Canvas(QWidget):
    def __init__(self, parent, width, height, resolution):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.converted_image = None
        self.setFixedSize(self.width, self.height)
        self.resolution = resolution

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
            painter.setPen(QPen(Qt.black, 30, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()

            self.update()

    def image_pixel_average(self, image: PyQt5.QtGui.QImage, posX, posY, width, height):
        pixel_sum = 0
        pixels_in_area = 0
        for x in range(posX-width//4, posX + width//4):
            if x < 0 or x >= image.width():
                continue
            for y in range(posY-width//4, posY + height//4):
                if y < 0 or y >= image.height():
                    continue
                pixel_sum += image.pixel(x, y)
                pixels_in_area += 1
        return int(pixel_sum / pixels_in_area)
    def downsize(self, image: PyQt5.QtGui.QImage, new_width, new_height):
        new_pixel_size = (int(image.width() / new_width), int(image.height() / new_height))
        new_image = QImage(new_width, new_height, QImage.Format_Grayscale8)
        for x in range(new_width):
            for y in range(new_height):
                v = self.image_pixel_average(image, x * new_pixel_size[0], y * new_pixel_size[1], new_pixel_size[0], new_pixel_size[1])
                new_image.setPixel(x, y, v)

        return new_image


    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            # Konwersja obrazka na czarno-biały 28x28 px z antyaliasingiem
            _image = self.downsize(self.image, 28, 28)

            width = _image.width()
            height = _image.height()

            # Wyciągnięcie danych z pikseli i zamiana na macierz numpy
            data = _image.bits().asstring(width * height)
            arr = np.frombuffer(data, dtype=np.uint8).reshape((height, width))

            self.image = _image
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

import os
import pickle
import sys
import time

import joblib
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLabel, QPushButton, QInputDialog, QFileDialog, \
    QMessageBox
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from AbstractModel import DummyModel
from Canvas import Canvas
from downloadDatabase import downloadBase
from ConfusionMatrix import ConfusionMatrix

# Tutaj wstaw swoje modele
models = [
    DummyModel,
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.selected_model = None
        self.selected_base = None
        self.model = None
        self.fitted = False
        self.mnist = None
        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Create the menu bar
        menubar = self.menuBar()

        # Create the Model menu and add actions to it
        model_menu = menubar.addMenu("Model")
        nowy_action = QAction("Nowy", self)
        zapisz_action = QAction("Zapisz", self)
        wczytaj_action = QAction("Wczytaj", self)
        model_menu.addAction(nowy_action)
        model_menu.addAction(zapisz_action)
        model_menu.addAction(wczytaj_action)

        # Create the Baza menu and add actions to it
        baza_menu = menubar.addMenu("Baza")
        wczytaj_baze_action = QAction("Wczytaj", self)
        pobierz_action = QAction("Pobierz", self)
        baza_menu.addAction(wczytaj_baze_action)
        baza_menu.addAction(pobierz_action)

        # Add the checkable QAction to a new menu
        other_menu = menubar.addMenu('Inne')

        # Add button to Inne menu to upload random image
        upload_random_image = QAction("Dodaj losowy obraz", self)
        other_menu.addAction(upload_random_image)
        upload_random_image.triggered.connect(self.set_random_image)

        # Add checkbox for showing plots
        # Create a checkable QAction
        self.plot_checkbox = QAction('Pokazuj wykresy', self, checkable=True)
        self.plot_checkbox.setChecked(True)
        other_menu.addAction(self.plot_checkbox)

        # Add button to Inne menu to create confusion matrix
        self.confusion_matrix_button = QAction("Macierz pomyłek", self)
        other_menu.addAction(self.confusion_matrix_button)
        self.confusion_matrix_button.triggered.connect(self.macierz_konfuzji)

        # Add button to Inne menu to do crossvalidation
        self.crossvalidation_button = QAction("Crossvalidation", self)
        other_menu.addAction(self.crossvalidation_button)
        self.crossvalidation_button.triggered.connect(self.crossvalidation)

        # Connect the actions to their respective methods
        nowy_action.triggered.connect(self.nowy_model)
        zapisz_action.triggered.connect(self.zapisz_model)
        wczytaj_action.triggered.connect(self.wczytaj_model)
        pobierz_action.triggered.connect(self.pobierz_baze)
        wczytaj_baze_action.triggered.connect(self.wczytaj_baze)

        # Add to the menu bar dummy buttons with
        # text about selected base and model

        self.selected_model_label = QAction(f"Wybrany model: {self.selected_model}", self)
        self.selected_model_label.setEnabled(False)

        self.selected_base_label = QAction(f"Wybrana baza: {self.selected_base}", self)
        self.selected_base_label.setEnabled(False)

        menubar.addAction(self.selected_model_label)
        menubar.addAction(self.selected_base_label)

        # Set the window properties
        self.setWindowTitle("Projekt SI 2023")
        self.resize(500, 380)
        self.setFixedSize(self.size())

        # Add label for status bar
        self.label = QLabel(self)
        self.label.setText("Gotowy")

        # Create a status bar
        self.status = self.statusBar()
        self.status.addWidget(self.label)

        # Fit button
        self.button_trenuj = QPushButton('Trenuj', self)
        self.button_trenuj.move(100, 330)
        self.button_trenuj.clicked.connect(self.trenuj_model)

        # Predict button
        self.button_predict = QPushButton('Klasyfikuj', self)
        self.button_predict.move(200, 330)
        self.button_predict.clicked.connect(self.klasyfikuj)

        # Clear button
        self.button_clear = QPushButton('Wyczyść', self)
        self.button_clear.move(300, 330)
        self.button_clear.clicked.connect(self.clear)

        # Add text field for predicted value
        self.predicted_value = QLabel(self)
        self.predicted_value.setText("Predicted value: ")
        self.predicted_value.move(325, 25)
        self.predicted_value.resize(170, 300)
        self.predicted_value.setStyleSheet("border: 1px solid black;"
                                           "background-color: white;"
                                           "font-size: 15px;")
        self.predicted_value.setWordWrap(True)

        # Add canvas for drawing
        self.canvas = Canvas(self, width=300, height=300)
        self.canvas.move(5, 25)

    def komunikat(self, text, color="black"):
        self.label.setText(text)
        self.label.setStyleSheet(f"color: {color};")

    def nowy_model(self):
        self.komunikat("Wybrano opcję Nowy")

        if self.selected_base is None:
            self.komunikat("Nie wybrano bazy", color="red")
            return

        # Combobox with models
        model_type, ok_pressed = QInputDialog.getItem(self, "Wybór modelu", "Wybierz model:",
                                                      (model.__name__ for model in models), 0, False)

        if ok_pressed:
            self.model = globals()[model_type]()

            if self.model is not None:
                self.selected_model = model_type
                self.selected_model_label.setText(f"Wybrany model: {self.selected_model}")
                self.selected_model_label.setEnabled(True)
                self.komunikat(f"Wybrano model {self.selected_model}", color="green")
                self.fitted = False
            else:
                self.komunikat("Model pusty", color="red")
        else:
            self.komunikat("Nie wybrano modelu", color="red")

    def zapisz_model(self):
        self.komunikat("Wybrano opcję Zapisz")

        if self.selected_model is None:
            self.komunikat("Nie wybrano modelu", color="red")
        elif self.model is None:
            self.komunikat("Model pusty", color="red")
        if self.fitted is False:
            self.komunikat("Model nie jest wytrenowany", color="red")
        else:
            pickle.dump(self.model, open(f"{self.selected_model}.pkl", "wb"))
            self.komunikat("Zapisano model", color="green")

    def wczytaj_model(self):
        self.komunikat("Wybrano opcję Wczytaj")

        file_name, ok = QFileDialog.getOpenFileName(self, "Wczytaj model", "", "Plik modelu (*.pkl)")
        if ok:
            self.model = pickle.load(open(file_name, "rb"))
            self.selected_model = file_name.split("/")[-1].split(".")[0]
            self.selected_model_label.setText(f"Wybrany model: {self.selected_model}")
            self.selected_model_label.setEnabled(True)
            self.komunikat(f"Wczytano model {self.selected_model}", color="green")
        else:
            self.komunikat("Nie wybrano modelu", color="red")

    def trenuj_model(self):
        self.komunikat("Wybrano opcję Trenuj")

        if self.selected_model is None:
            self.komunikat("Nie wybrano modelu", color="red")
        elif self.selected_base is None:
            self.komunikat("Nie wybrano bazy", color="red")
        else:
            self.komunikat("Trenowanie modelu...", color="green")
            self.fitted = True

            start = time.time()

            self.model.fit(self.X_train, self.y_train)

            end = time.time()

            try:
                score = self.model.score(self.X_test, self.y_test)
            except:
                score = self.model.evaluate(self.X_test, self.y_test)[1]

            self.komunikat(f"Model wyćwiczony, wynik: {score}, czas {(end - start):.3f}s", color="green")


    def klasyfikuj(self):
        self.komunikat("Wybrano opcję Klasyfikuj")
        if self.fitted is False:
            self.komunikat("Model niewyćwiczony", color="red")
        else:
            self.komunikat("Klasyfikowanie...", color="green")

            predicted = self.model.predict(self.canvas.getConvertedImage())[0]

            print(f"Predicted: {predicted}")

            try:
                text = "<html><body>"
                value = np.argmax(predicted)

                for i in range(len(predicted)):
                    if i == value:
                        text += f"<b>{i}: {predicted[i]:.5f}</b>\n"
                    else:
                        text += f"{i}: {predicted[i]:.5f}\n"

                text += "</body></html>"
                self.predicted_value.setText(text)
            except:
                self.predicted_value.setText(f"Predicted value: {predicted}")
                value = predicted

            if self.plot_checkbox.isChecked():
                self.canvas.showResult(value)
            self.komunikat(f"Klasyfikacja zakończona, wynik: {value}", color="green")


    def pobierz_baze(self):
        self.komunikat("Wybrano opcję Pobierz")

        self.selected_base, ok = QInputDialog.getText(self, "Pobieranie bazy", "Podaj nazwę bazy:")

        if ok:
            self.komunikat(f"Pobieranie bazy {self.selected_base}...")
            downloadBase(self.selected_base)
            self.wczytaj_baze()
        else:
            self.komunikat("Nie podano nazwy bazy", color="red")

    def wczytaj_baze(self):
        self.komunikat("Wybrano opcję Wczytaj Baze")

        # Show file selection dialog
        file_name, ok = QFileDialog.getOpenFileName(self, "Wybierz plik", "", "Pliki joblib (*.joblib)")
        if ok:
            if os.path.exists(f"{file_name}"):
                train_size, ok = QInputDialog.getInt(self, "Wybierz",
                                                     "Podaj rozmiar treningowy (w %):",
                                                     min=1, max=99, step=1, value=80)

                if ok:
                    # From absolute path get only file name without extension
                    self.selected_base = os.path.splitext(os.path.basename(file_name))[0]
                    self.komunikat(f"Wybrano bazę {self.selected_base}", color="green")

                    train_size /= 100
                    self.mnist = joblib.load(f'{self.selected_base}.joblib')
                    self.X = self.mnist.data
                    self.y = self.mnist.target

                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                            train_size=train_size,
                                                                                            random_state=42)
                    self.fitted = False

                    self.canvas.set_resolution(int(np.sqrt(self.X.shape[1])))
                    self.canvas.clear()

                    self.selected_base_label.setText(
                        f"Wybrana baza: {self.selected_base} {self.canvas.resolution}x{self.canvas.resolution}")
                    self.selected_base_label.setEnabled(True)

                    if self.plot_checkbox.isChecked():
                        for i in range(10):
                            plt.subplot(2, 5, i + 1)
                            random = np.random.randint(0, len(self.X))
                            plt.imshow(self.X[random].reshape(self.canvas.resolution, self.canvas.resolution),
                                       cmap='gray',
                                       vmin=0, vmax=255)
                            plt.axis('off')
                        plt.show()

                    print("Wczytano dane:")
                    print(f"X_train: {self.X_train.shape}")
                    print(f"X_test: {self.X_test.shape}")
                    print(f"y_train: {self.y_train.shape}")
                    print(f"y_test: {self.y_test.shape}")

                    print("X:")
                    print(self.X)
                    print("y:")
                    print(self.y)

                    self.komunikat("Wczytano dane", color="green")

                else:
                    self.komunikat("Anulowano", color="red")
            else:
                self.komunikat("Nie znaleziono pliku", color="red")
        else:
            self.komunikat("Anulowano", color="red")

    def clear(self):
        self.komunikat("Wyczyszczono", color="green")
        self.canvas.clear()

    def set_random_image(self):
        if self.X is None:
            self.komunikat("Nie wczytano bazy", color="red")
            return

        # Dialog box to get number of class
        class_number, ok = QInputDialog.getInt(self, "Wybierz",
                                               "Podaj numer klasy:",
                                               min=0, max=np.max(self.y), step=1, value=0)

        if ok:
            # Find all images from selected class
            images = np.where(self.y == class_number)[0]

            if len(images) == 0:
                self.komunikat("Nie znaleziono obrazów", color="red")
                return

            # Select a random image from selected class
            random_idx = np.random.choice(images)
            self.canvas.setImage(self.X[random_idx])

            self.komunikat(f"Wybrano obraz #{random_idx}. Klasa {self.y[random_idx]}", color="green")

        else:
            self.komunikat("Anulowano", color="red")

    def macierz_konfuzji(self):
        self.komunikat("Wybrano opcję Macierz Konfuzji")
        if self.fitted is False:
            self.komunikat("Model niewyćwiczony", color="red")
        else:
            self.komunikat("Generowanie macierzy konfuzji...", color="green")

            confusion_matrix = ConfusionMatrix(self.y_test[0].shape[0])

            for i in range(len(self.X_test)):
                predicted = np.argmax(self.model.predict(self.X_test[i])[0])
                actual = np.argmax(self.y_test[i])
                confusion_matrix.add(predicted, actual)

            print("Confusion matrix:")
            for i in range(len(confusion_matrix.matrix)):
                conf = confusion_matrix.matrix[i]
                print("Class ", i, "TP: ", conf.tp, "FP: ", conf.fp, "FN: ", conf.fn, "TN: ", conf.tn)
                print("\tPrecision: ", conf.precision(), "Jak czesto mial racje wybierajac go")
                print("\tRecall: ", conf.recall(), "Jak często wybieral go gdy powinien")
                print("\tF1: ", conf.fb(1))
                print("\tAccuracy: ", conf.accuracy(), "Jak często miał racje wybierając lub nie wybierając", end="\n\n")

    def crossvalidation(self):

        n_splits, ok = QInputDialog.getInt(self, "Wybierz",
                                            "Podaj ilość podziałów:",
                                            min=2, max=10, step=1, value=5)

        if self.selected_model is None:
            self.komunikat("Nie wybrano modelu", color="red")
            return
        if self.selected_base is None:
            self.komunikat("Nie wczytano bazy", color="red")
            return

        cv_model = self.model.__class__() # nowy model tej samej klasy zeby nie stracic starego

        self.komunikat("Wybrano opcję Cross Validation")
        splitted_data = np.array_split(self.X, n_splits)
        accuracy = []
        for n in range(n_splits):
            self.komunikat(f"Fold {n+1}/{n_splits}", color="green")
            X_test = splitted_data[n]
            X_train = np.concatenate(np.delete(splitted_data, n, axis=0))
            y_test = np.array_split(self.y, n_splits)[n]
            y_train = np.concatenate(np.delete(np.array_split(self.y, n_splits), n, axis=0))
            cv_model.fit(X_train, y_train)
            acc = cv_model.score(X_test, y_test)
            self.komunikat(f"Fold {n+1}/{n_splits} accuracy: {acc}", color="green")
            accuracy.append(acc)
        mean_accuracy = np.mean(accuracy)
        self.komunikat(f"Accuracy: {mean_accuracy}", color="green")

        #popup mean_accuracy
        msg = QMessageBox()
        msg.setWindowTitle("Cross Validation")
        msg.setText(f"Accuracy: {mean_accuracy}")
        msg.exec_()


        return mean_accuracy



    def exit(self):
        QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

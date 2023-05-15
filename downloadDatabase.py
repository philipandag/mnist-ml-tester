from sklearn.datasets import fetch_openml, load_digits
import joblib
import numpy as np


def downloadBase(nazwa_bazy):
    if nazwa_bazy == 'mnist_64':
        mnist = load_digits()
        mnist.data = mnist.data * 16
    else:
        try:
            mnist = fetch_openml(nazwa_bazy, as_frame=False)
        except:
            print(f"Baza danych {nazwa_bazy} nie istnieje")
            return

    mnist.data = 255 - mnist.data  # Odwróć kolory

    mnist.data[mnist.data < 0] = 0  # Zamień wartości ujemne na 0

    # Zamień tablicę klas na inty jeśli jest stringiem
    if type(mnist.target[0]) == str:
        mnist.target = mnist.target.astype(np.int8)

    joblib.dump(mnist, f'{nazwa_bazy}.joblib')

    print(f"Pobrano bazę danych {nazwa_bazy}")
    # X (np.ndarray) – shape (n_samples, n_features)
    # y (np.ndarray) – shape (n_samples,)

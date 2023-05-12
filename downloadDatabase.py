from sklearn.datasets import fetch_openml, load_digits
import joblib
import matplotlib.pyplot as plt
import numpy as np


def downloadBase(nazwa_bazy):
    if nazwa_bazy == 'mnist_64':
        mnist = load_digits()
    else:
        try:
            mnist = fetch_openml(nazwa_bazy, as_frame=False)
        except:
            print(f"Baza danych {nazwa_bazy} nie istnieje")
            return

    rozdzielczosc = int(np.sqrt(mnist.data.shape[1]))

    # Odwróć kolorystykę obrazków
    mnist.data = 255 - mnist.data

    # Wyświetlenie randomowych 10 obrazków
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        random = np.random.randint(0, len(mnist.data))
        plt.imshow(mnist.data[random].reshape(rozdzielczosc, rozdzielczosc), cmap='gray')
        plt.axis('off')
    plt.show()

    joblib.dump(mnist, f'{nazwa_bazy}.joblib')

    print(f"Pobrano bazę danych {nazwa_bazy}")

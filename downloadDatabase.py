from sklearn.datasets import fetch_openml
import joblib
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784', as_frame=False)

# Odwróć kolorystykę obrazków
mnist.data = 255 - mnist.data

# Wyswietlenie randomowych 10 obrazków
for i in range(10):
    plt.subplot(2, 5, i + 1)
    random = np.random.randint(0, len(mnist.data))
    plt.imshow(mnist.data[random].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

joblib.dump(mnist, 'mnist.joblib')

print("Pobrano bazę danych MNIST")

# Żeby wczytać bazę danych do programu

# mnist = joblib.load('mnist.joblib')
# X = mnist.data
# y = mnist.target

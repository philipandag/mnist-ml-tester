from sklearn.datasets import fetch_openml
import joblib

mnist = fetch_openml('mnist_', as_frame=False)

# Odwróć kolorystykę obrazków
mnist.data = 255 - mnist.data

joblib.dump(mnist, 'mnist.joblib')

print("Pobrano bazę danych MNIST")

# Żeby wczytać bazę danych do programu

# mnist = joblib.load('mnist.joblib')
# X = mnist.data
# y = mnist.target
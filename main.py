import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Wczytaj bazę danych MNIST
mnist = joblib.load('mnist.joblib')
X = mnist.data
y = mnist.target

print(f"Kształt danych: {X.shape}")

# Pokaż przykładowy obrazek z bazy danych
plt.imshow(X[0].reshape(28, 28), cmap="gray")
plt.title(f"Klasa: {y[0]}")
plt.show()

# Podziel zbiór danych na zestawy uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utwórz obiekt klasyfikatora drzewa decyzyjnego
clf = DecisionTreeClassifier()

# Wytrenuj klasyfikator na danych uczących
clf.fit(X_train, y_train)

# Przewiduj klasy dla danych testowych
y_pred = clf.predict(X_test)

# Wyświetl dokładność klasyfikatora
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność: {accuracy}")

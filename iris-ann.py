from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import _multilayer_perceptron
from sklearn.metrics import accuracy_score

iris = load_iris()


X, y = iris.data, iris.target

X_test, X_train, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



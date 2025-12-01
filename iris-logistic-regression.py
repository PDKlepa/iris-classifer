from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Data
iris = datasets.load_iris()

## Seperate data into X = features and values and y = actual results
X, y = iris.data, iris.target
# Use only petal length and petal width for visualisation (columns 2 and 3)
X_petal = X[:, 2:4]

## Seperate train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_petal, y, test_size=0.2, random_state=42
)
## Fit data to the median and standard deviation of the columns 
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) 

scaler_petal = StandardScaler()
scaler_petal.fit(Xp_train)

## Create and train logistic regression model 

model = LogisticRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
y_proba = model.predict_proba(X_test_scaled)

print(y_proba[:5])

Xp_train_scaled = scaler_petal.transform(Xp_train)
Xp_test_scaled = scaler_petal.transform(Xp_test)


log_reg_2d = LogisticRegression()
log_reg_2d.fit(Xp_train_scaled, yp_train)
# Create a grid over the feature space
x_min, x_max = Xp_train_scaled[:, 0].min() - 1, Xp_train_scaled[:, 0].max() + 1
y_min, y_max = Xp_train_scaled[:, 1].min() - 1, Xp_train_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = log_reg_2d.predict(grid_points)
Z = Z.reshape(xx.shape)

plt.figure()
# Background: predicted class for each point in the grid
plt.contourf(xx, yy, Z, alpha=0.3)

# Overlay the training points again
for class_value in np.unique(yp_train):
    plt.scatter(
        Xp_train_scaled[yp_train == class_value, 0],
        Xp_train_scaled[yp_train == class_value, 1],
        label=f"Class {class_value}"
    )

plt.xlabel("Standardized petal length")
plt.ylabel("Standardized petal width")
plt.title("Logistic Regression decision regions (Iris, petal features)")
plt.legend()
plt.tight_layout()
plt.show()

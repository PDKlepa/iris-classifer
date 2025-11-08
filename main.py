from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline (scaler -> PCA -> classifier)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("clf", LogisticRegression(max_iter=200))
])

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro")

# Hyperparameter search
param_grid = [
    # With PCA (try LR)
    {
        "pca": [PCA()],
        "pca__n_components": [2, 3],
        "clf": [LogisticRegression(max_iter=500)],
        "clf__C": [0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
    # With PCA (try SVC)
    {
        "pca": [PCA()],
        "pca__n_components": [2, 3],
        "clf": [SVC(probability=True)],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", "auto"],
        "clf__kernel": ["rbf", "linear"],
    },
    # No PCA (passthrough)
    {
        "pca": ["passthrough"],
        "clf": [LogisticRegression(max_iter=500)],
        "clf__C": [0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
    {
        "pca": ["passthrough"],
        "clf": [SVC(probability=True)],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", "auto"],
        "clf__kernel": ["rbf", "linear"],
    },
]

search = GridSearchCV(
    pipe, param_grid=param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True
)
search.fit(X_tr, y_tr)

best_model = search.best_estimator_

# Test evaluation
y_pred = best_model.predict(X_te)
print(classification_report(y_te, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_te, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot()
plt.show()

"""Comparaison de classifieurs sur MNIST et effet de la PCA.

Le test reste séparé de la sélection des hyperparamètres.
Une partie stratifiée de MNIST est utilisée pour limiter les temps de calcul.
"""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


RANDOM_STATE = 42
TRAIN_SIZE = 12000
TEST_SIZE = 3000


def load_mnist():
    mnist = fetch_openml(data_id=554, as_frame=False, parser="auto")
    X = np.asarray(mnist.data, dtype=np.float32) / 255.0
    y = np.asarray(mnist.target, dtype=np.int64)

    # Séparation avant la sélection des sous-ensembles.
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
        X, y, test_size=10000, random_state=RANDOM_STATE, stratify=y
    )
    X_train, _, y_train, _ = train_test_split(
        X_train_all, y_train_all, train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE, stratify=y_train_all
    )
    X_test, _, y_test, _ = train_test_split(
        X_test_all, y_test_all, train_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_test_all
    )
    return X_train, X_test, y_train, y_test


def make_pipeline(estimator, pca_components=None):
    steps = [("scaler", StandardScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def main():
    X_train, X_test, y_train, y_test = load_mnist()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    experiments = {
        "Logistic regression": (
            make_pipeline(LogisticRegression(max_iter=1000)),
            {"model__C": [0.1, 1.0]},
        ),
        "Linear SVM": (
            make_pipeline(LinearSVC(max_iter=5000, dual=False)),
            {"model__C": [0.01, 0.1]},
        ),
        "RBF SVM": (
            make_pipeline(SVC(kernel="rbf", cache_size=500)),
            {"model__C": [1.0, 10.0], "model__gamma": ["scale"]},
        ),
        "k-NN": (
            make_pipeline(KNeighborsClassifier()),
            {"model__n_neighbors": [3, 5], "model__weights": ["uniform"]},
        ),
        "PCA (95%) + linear SVM": (
            make_pipeline(LinearSVC(max_iter=5000, dual=False), pca_components=0.95),
            {"model__C": [0.01, 0.1]},
        ),
        "PCA (95%) + RBF SVM": (
            make_pipeline(SVC(kernel="rbf", cache_size=500), pca_components=0.95),
            {"model__C": [1.0, 10.0], "model__gamma": ["scale"]},
        ),
        "PCA (95%) + k-NN": (
            make_pipeline(KNeighborsClassifier(), pca_components=0.95),
            {"model__n_neighbors": [3, 5], "model__weights": ["uniform"]},
        ),
    }

    results = []
    for name, (pipeline, grid) in experiments.items():
        print(f"Entraînement : {name}", flush=True)
        search = GridSearchCV(
            pipeline, grid, scoring="f1_macro", cv=cv, n_jobs=-1,
            refit=True, error_score="raise"
        )
        start = perf_counter()
        search.fit(X_train, y_train)
        train_time = perf_counter() - start

        start = perf_counter()
        predictions = search.predict(X_test)
        predict_time = perf_counter() - start

        pca = search.best_estimator_.named_steps.get("pca")
        results.append({
            "model": name,
            "cv_f1_macro": round(search.best_score_, 4),
            "test_accuracy": round(accuracy_score(y_test, predictions), 4),
            "test_f1_macro": round(f1_score(y_test, predictions, average="macro"), 4),
            "pca_components": pca.n_components_ if pca is not None else 784,
            "train_cv_seconds": round(train_time, 1),
            "predict_seconds": round(predict_time, 2),
            "best_params": str(search.best_params_),
        })

    table = pd.DataFrame(results).sort_values("cv_f1_macro", ascending=False)
    print("\nRésultats (triés par score de validation croisée) :")
    print(table.to_string(index=False))
    table.to_csv("mnist_benchmark_results.csv", index=False)
    print("\nRésultats enregistrés dans mnist_benchmark_results.csv")


if __name__ == "__main__":
    main()

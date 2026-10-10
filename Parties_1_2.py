import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_data(filename):
    """Charge un jeu de données stocké dans le dossier data."""
    data = np.load(f"data/{filename}")
    x = data[0, :].reshape(-1, 1)
    y = data[1, :]
    return x, y


def linear_regression(x, y, title):
    """Ajuste une régression linéaire OLS et affiche le résultat."""
    model = LinearRegression()
    model.fit(x, y)

    y_pred = model.predict(x)
    mse = np.mean((y - y_pred) ** 2)
    print(f"Erreur d'apprentissage ({title}) : {mse:.4f}")

    x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.scatter(x, y)
    plt.plot(x_line, y_line, color="red", label="Régression OLS")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.show()


def polynomial_regression(x, y, degree=10):
    """Calcule une régression polynomiale OLS."""
    x = x.ravel()
    n = len(x)

    Z = np.ones((n, degree + 1))
    for j in range(1, degree + 1):
        Z[:, j] = x ** j

    # Solution des moindres carrés
    theta = np.linalg.lstsq(Z, y, rcond=None)[0]
    y_pred = Z @ theta
    mse = np.mean((y - y_pred) ** 2)
    print(f"Erreur polynomiale (degré {degree}) : {mse:.4f}")

    xx = np.linspace(x.min(), x.max(), 400)
    Z_xx = np.ones((len(xx), degree + 1))
    for j in range(1, degree + 1):
        Z_xx[:, j] = xx ** j

    plt.scatter(x, y)
    plt.plot(xx, Z_xx @ theta, color="red", label=f"OLS degré {degree}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Régression polynomiale")
    plt.legend()
    plt.show()


# Data 1
x1, y1 = load_data("data1.npy")
linear_regression(x1, y1, "Régression linéaire OLS - data1")

# Data 2
x2, y2 = load_data("data2.npy")
linear_regression(x2, y2, "Régression linéaire OLS - data2")

# Modèle polynomial sur data2
polynomial_regression(x2, y2, degree=10)

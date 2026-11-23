###############################################################################
# MODULES
###############################################################################
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


###############################################################################
# LOAD MNIST
###############################################################################
# Download MNIST
mnist = fetch_openml(data_id=554, parser='liac-arff')
# copy mnist.data (type is pandas DataFrame)
data = mnist.data
# array (70000,784) collecting all the 28x28 vectorized images
img = data.to_numpy()
# array (70000,) containing the label of each image
lb = np.array(mnist.target,dtype=int)
# Splitting the dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    img, lb, 
    test_size=0.25, 
    random_state=0,
    stratify=lb)
# Number of classes
k = len(np.unique(lb))
# Sample sizes and dimension
(n,p) = img.shape
n_train = y_train.size
n_test = y_test.size 


##############################################################################
# Tests sur la base de données MNIST ( Régularisation l2)
##############################################################################
# 1. Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(float))
X_test_scaled = scaler.transform(X_test.astype(float))

# 2. Entraînement avec régularisation L2
log_reg_l2 = LogisticRegression(C=0.01, tol=0.01, max_iter=200)
log_reg_l2.fit(X_train_scaled, y_train)

# 3. Affichage de la matrice de confusion 
y_pred = log_reg_l2.predict(X_test_scaled)
print(f"L2 - accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"L2 - F1 macro : {f1_score(y_test, y_pred, average='macro'):.4f}")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(lb))
disp.plot(cmap='Blues')
plt.title("Matrice de Confusion (Régularisation L2)")
plt.savefig(FIGURES_DIR / "mnist_confusion_l2.png", dpi=200, bbox_inches="tight")
plt.show()

# 4. Visualisation des coefficients Beta sous forme d'image 
poids = log_reg_l2.coef_ # Récupère les coefficients pour chaque classe

plt.figure(figsize=(15, 6))
vmax = np.max(np.abs(poids)) * 1
for i in range(10):
    plt.subplot(2, 5, i + 1)
    # Reshape du vecteur de 784 en image 28x28
    plt.imshow(poids[i].reshape(28, 28), cmap='RdBu', vmin=-vmax, vmax=vmax)
    plt.title(f"Chiffre {i}")
    plt.axis('off')
plt.suptitle("Coefficients Beta (Régularisation L2)")
plt.savefig(FIGURES_DIR / "mnist_coefficients_l2.png", dpi=200, bbox_inches="tight")
plt.show()


##############################################################################
# 5. Tests sur la base de données MNIST ( Régularisation l1)
##############################################################################

# 1. Réentraînement avec pénalité L1
log_reg_l1 = LogisticRegression(l1_ratio=1, solver='saga', C=0.01, tol=0.01, max_iter=200, random_state=0)
log_reg_l1.fit(X_train_scaled, y_train)

# 2. Affichage de la matrice de confusion 
y_pred_l1 = log_reg_l1.predict(X_test_scaled)
print(f"L1 - accuracy : {accuracy_score(y_test, y_pred_l1):.4f}")
print(f"L1 - F1 macro : {f1_score(y_test, y_pred_l1, average='macro'):.4f}")
print(f"L1 - coefficients non nuls : {np.count_nonzero(log_reg_l1.coef_)} / {log_reg_l1.coef_.size}")
cm = confusion_matrix(y_test, y_pred_l1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(lb))
disp.plot(cmap='Blues')
plt.title("Matrice de Confusion (Régularisation L1)")
plt.savefig(FIGURES_DIR / "mnist_confusion_l1.png", dpi=200, bbox_inches="tight")
plt.show()

# 3. Visualisation des coefficients Beta sous forme d'image
poids_l1 = log_reg_l1.coef_

plt.figure(figsize=(15, 6))
vmax = np.max(np.abs(poids)) * 1
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(poids_l1[i].reshape(28, 28), cmap='RdBu', vmin=-vmax, vmax=vmax)
    plt.title(f"Chiffre {i} (L1)")
    plt.axis('off')
plt.suptitle("Coefficients Beta (Régularisation L1)")
plt.savefig(FIGURES_DIR / "mnist_coefficients_l1.png", dpi=200, bbox_inches="tight")
plt.show()


##############################################################################
# Affichage de m images de la base de données
##############################################################################
m=16 # Nombre d'images
plt.figure(figsize=(10,10))
for i in np.arange(m):
  ex_plot = plt.subplot(int(np.sqrt(m)),int(np.sqrt(m)),i+1)
  plt.imshow(img[i,:].reshape((28,28)), cmap='gray')
  ex_plot.set_xticks(()); ex_plot.set_yticks(())
  #lt.title("Label = %i" % lb[i])
plt.savefig(FIGURES_DIR / "mnist_exemples.png", dpi=200, bbox_inches="tight")
plt.show()
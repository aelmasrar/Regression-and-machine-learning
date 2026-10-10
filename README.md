# Régression et classification en Machine Learning

Projet académique de statistiques et de Machine Learning. J'ai étudié plusieurs méthodes de régression et de classification, puis comparé leurs résultats sur des données synthétiques et sur MNIST.

## 1. Régression linéaire et polynomiale

Dans `Parties_1_2.py`, j'ai travaillé sur deux jeux de données (`data/data1.npy` et `data/data2.npy`).

J'ai utilisé la méthode des moindres carrés ordinaires (OLS) et la régression polynomiale. J'ai comparé les ajustements et les erreurs obtenues selon les modèles.

## 2. Classification sur données synthétiques

Dans `Part3_Class_Regre_Logis_contre_Ridge_OLS.py`, j'ai comparé trois méthodes :
- OLS ;
- Ridge ;
- régression logistique.

J'ai aussi ajouté des observations aberrantes pour étudier leur influence sur les frontières de décision.

## 3. Classification des chiffres MNIST

Dans `mnist.py`, j'ai entraîné une régression logistique multiclasse sur MNIST. J'ai comparé les régularisations L1 et L2.

J'ai étudié les prédictions avec une matrice de confusion. J'ai également représenté les coefficients appris pour chaque chiffre sous forme d'images.

Dans `mnist_benchmark.py`, j'ai ensuite comparé :
- la régression logistique ;
- le SVM linéaire et le SVM RBF ;
- le k-NN ;
- les SVM et le k-NN avec réduction de dimension par PCA.

Pour ce benchmark, j'ai utilisé 12 000 images d'entraînement et 3 000 images de test, avec un découpage stratifié. J'ai choisi les hyperparamètres par validation croisée à trois plis, en utilisant le F1-score macro. La standardisation et la PCA sont incluses dans les pipelines pour éviter les fuites de données.

### Résultats du benchmark

| Modèle | F1 macro (CV) | Accuracy test | F1 macro test | Variables | Recherche + CV (s) | Prédiction (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Régression logistique | 0,9050 | 91,17 % | 0,9100 | 784 | 2,4 | 0,00 |
| SVM linéaire | 0,8902 | 89,87 % | 0,8966 | 784 | 363,6 | 0,01 |
| SVM RBF | 0,9460 | 95,50 % | 0,9548 | 784 | 49,8 | 3,91 |
| k-NN | 0,9095 | 91,43 % | 0,9135 | 784 | 2,6 | 0,33 |
| PCA + SVM linéaire | 0,8995 | 90,73 % | 0,9058 | 289 | 148,9 | 0,00 |
| **PCA + SVM RBF** | **0,9486** | **95,83 %** | **0,9581** | **289** | **14,5** | **1,41** |
| PCA + k-NN | 0,9143 | 92,00 % | 0,9190 | 289 | 0,8 | 0,10 |

Avec la PCA, j'ai conservé 95 % de la variance. Le nombre de variables est passé de **784 à 289**, soit **63,1 % de réduction**.

Le meilleur résultat a été obtenu avec **PCA + SVM RBF : 95,83 % d'accuracy** et **95,81 % de F1-score macro** sur le test.

Pour le SVM RBF, le temps de recherche d'hyperparamètres et de validation croisée est passé de **49,8 à 14,5 secondes**, soit une réduction d'environ **70,9 %**. Le temps de prédiction est aussi passé de 3,91 à 1,41 seconde.

Ces mesures correspondent à une exécution sur un sous-ensemble de MNIST. Une répétition sur plusieurs découpages serait nécessaire pour vérifier la stabilité des petits écarts de précision.

## Installation

```bash
pip install -r requirements.txt
```

## Exécution

Régression :

```bash
python Parties_1_2.py
```

Classification sur données synthétiques :

```bash
python Part3_Class_Regre_Logis_contre_Ridge_OLS.py
```

Régression logistique sur MNIST :

```bash
python mnist.py
```

Benchmark MNIST :

```bash
python mnist_benchmark.py
```

MNIST est téléchargé automatiquement lors de la première exécution. Le benchmark enregistre ses résultats dans `mnist_benchmark_results.csv`.

## Outils

Python, NumPy, pandas, Matplotlib, SciPy et scikit-learn.

## Rapport

Le rapport du projet est disponible dans `Rapport_STAT_ELMASRAR_KARA.pdf`.

# TP Deep Learning - Classification Dry Bean avec un MLP (Keras / TensorFlow)

Ce dépôt contient le TP de Deep Learning portant sur la classification multi-classes du dataset **Dry Bean** (UCI id=602) à l'aide d'un réseau de neurones **MLP (Perceptron Multi-Couches)**.

## Contenu du projet

- `TP_Deep_Learning_DryBean.ipynb` : Le notebook Jupyter contenant tout le code exécuté et commenté.
- `modele_drybean.keras` : Le modèle MLP final sauvegardé après entraînement avec *Early Stopping*.
- `scaler_drybean.pkl` : Le `StandardScaler` ajusté sur les données d'entraînement.
- `classes_drybean.json` : La liste des 7 classes de haricots (`BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER`, `SIRA`).

## Étapes réalisées dans le TP

1. **Chargement et exploration des données** : Récupération des 13 611 observations et 16 caractéristiques géométriques depuis `ucimlrepo`.
2. **Prétraitement** : Encodage des labels (`LabelEncoder`), découpage stratifié en Train (60%) / Validation (20%) / Test (20%), et standardisation (`StandardScaler`) sans fuite de données (*data leakage*).
3. **Premier modèle MLP** : Réseau séquentiel `16 -> 32 (ReLU) -> 7 (Softmax)` compilé avec `Adam(lr=0.001)` et `SparseCategoricalCrossentropy` (775 paramètres).
4. **Évaluation** : Courbes d'apprentissage (Loss et Accuracy), rapport de classification (Precision, Recall, F1-score) et matrice de confusion.
5. **Étude des hyperparamètres** :
   - Taux d'apprentissage (`0.0001`, `0.001`, `0.05`)
   - Optimiseurs (`Adam` vs `SGD`)
   - Taille de lot / Batch size (`1`, `32`, `8166`)
   - Architectures (`(2,)`, `(32,)`, `(64, 32)`) et fonctions d'activation (`relu` vs `tanh`)
6. **Régularisation et Arrêt précoce** : Comparaison de `L1`, `L2`, `Dropout`, `BatchNormalization` et mise en place de `EarlyStopping(patience=8, restore_best_weights=True)`.
7. **Sauvegarde et rechargement** : Exportation du modèle `.keras`, du scaler `.pkl` et des classes `.json`.

# Détection du Churn et de la Fraude Multi-SIM par Data Mining

**Projet académique  ISSEA · 2025**
**Auteur : WAMBE Jordan Valère**

> **Note de confidentialité** : Ce dépôt ne contient pas le code source ni les données brutes. Il présente la démarche analytique, les spécifications des modèles, les résultats de validation et les conclusions métier.

---

##  Problématique

Les opérateurs de télécommunications font face à deux défis majeurs qui érodent directement leur rentabilité :

1. **Le churn** : des clients qui résilient ou abandonnent progressivement leur abonnement sans signal d'alerte préalable. Acquérir un nouveau client coûte en moyenne 5 à 7 fois plus cher que retenir un client existant.

2. **La fraude multi-SIM** : des utilisateurs qui détiennent plusieurs cartes SIM simultanément chez le même opérateur, souvent pour contourner les offres tarifaires ou dans le cadre d'activités frauduleuses, ce qui fausse les analyses de clientèle et réduit l'efficacité des campagnes de fidélisation.

**Objectif** : construire un système de détection automatique de ces deux comportements à partir des données transactionnelles des abonnés, en comparant plusieurs familles de modèles et en retenant le plus performant par validation croisée et optimisation des hyperparamètres.

---

## Pipeline d'Analyse

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE DATA MINING                        │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────────────┐
  │   DONNÉES BRUTES     │
  │  (transactions       │
  │   abonnés télécom)   │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  ANALYSE EXPLORATOIRE│
  │  Univariée           │
  │  Bivariée            │
  │  Multivariée         │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  PRÉTRAITEMENT       │
  │  Gestion des NA      │
  │  Encodage variables  │
  │  Détection data      │
  │  leakage             │
  │  Équilibrage classes │
  └──────────┬───────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌──────────────────────────────────────┐
│CLUSTERING│  │      CLASSIFICATION SUPERVISÉE        │
│ K-means  │  │                                       │
│(profils  │  │  Régression logistique                │
│ risque)  │  │  Random Forest                        │
└──────────┘  │  XGBoost                              │
              │  SVM (noyau RBF)                      │
              └──────────────┬───────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  VALIDATION CROISÉE      │
              │  k-Fold (k=5)            │
              │  + GridSearchCV          │
              │  (optimisation           │
              │   hyperparamètres)       │
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  SÉLECTION DU MEILLEUR   │
              │  MODÈLE (AUC-ROC,        │
              │  F1-score, Précision,    │
              │  Rappel)                 │
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  SCORING & ALERTES       │
              │  Classement des abonnés  │
              │  par niveau de risque    │
              └──────────────────────────┘
```

---

## Spécifications des Modèles

### 1. Analyse Exploratoire

**Univariée** : distribution de chaque variable (histogrammes, boxplots, stats descriptives), identification des outliers par IQR :

$$\text{Outlier si } x < Q_1 - 1.5 \cdot \text{IQR} \quad \text{ou} \quad x > Q_3 + 1.5 \cdot \text{IQR}$$

**Bivariée** : corrélations de Pearson et Spearman, test du Chi-² pour les variables catégorielles :

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

**Multivariée** : Analyse en Composantes Principales (ACP) pour visualiser la structure des données et réduire la dimensionnalité avant modélisation :

$$\max_{\|w\|=1} \ \text{Var}(Xw) \quad \Longleftrightarrow \quad \text{décomposition spectrale de } X^TX$$

---

### 2. Précautions Méthodologiques Critiques

#### 2.1 Détection du Data Leakage

Avant toute modélisation, chaque variable candidate a été auditée pour s'assurer qu'elle ne contient pas d'information **postérieure** à l'événement à prédire.

> **Exemple de piège identifié** : une variable "nombre de jours depuis la dernière recharge" peut sembler prédictive du churn mais est renseignée *après* que le client a déjà churné dans certains systèmes. Son inclusion produirait un modèle en apparence très performant mais inexploitable en production.

#### 2.2 Définition rigoureuse de la variable cible

La variable cible (churn = 1 / non-churn = 0 ; fraude = 1 / non-fraude = 0) a été définie avec une **fenêtre temporelle explicite** :

$$Y_i = \mathbf{1}\left[\text{client } i \text{ churne dans les } T \text{ jours suivant l'observation}\right]$$

Le choix de $T$ impacte directement le déséquilibre des classes et la pertinence des alertes.

#### 2.3 Gestion du déséquilibre des classes

Les événements cibles (churn, fraude) étant rares, les classes sont fortement déséquilibrées. La stratégie retenue :

$$\text{SMOTE : } \tilde{x}_i = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim \mathcal{U}[0,1]$$

où $x_{nn}$ est un voisin synthétique du point minoritaire $x_i$.

---

### 3. Modèles de Classification

#### Régression Logistique (baseline)

$$P(Y=1 \mid X) = \sigma(X\beta) = \frac{1}{1 + e^{-X\beta}}$$

Régularisation L2 (Ridge) :

$$\hat{\beta} = \arg\min_\beta \left[ -\ell(\beta) + \lambda \|\beta\|_2^2 \right]$$

---

#### Random Forest

Agrégation de $B$ arbres de décision entraînés sur des sous-échantillons bootstrap :

$$\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Importance des variables par impureté de Gini :

$$\text{Gini}(t) = 1 - \sum_{k} p_{tk}^2$$

---

#### XGBoost (Gradient Boosting)

Minimisation séquentielle d'une fonction de perte avec régularisation :

$$\mathcal{L}^{(t)} = \sum_i \ell\left(y_i,\ \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

avec $\Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$ (nombre de feuilles $T$, poids des feuilles $w$).

---

#### SVM à noyau RBF

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i \quad \text{s.c.} \quad y_i(w \cdot \phi(x_i) + b) \geq 1 - \xi_i$$

Noyau RBF :

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

---

#### K-means (clustering des profils de risque)

$$\min_{\{C_k\}} \sum_{k=1}^{K} \sum_{x \in C_k} \|x - \mu_k\|^2$$

Utilisé pour segmenter les abonnés en profils homogènes *avant* la classification supervisée, permettant d'identifier des groupes à risque structurel.

---

### 4. Validation Croisée et Optimisation

**Stratégie** : k-Fold stratifié ($k=5$) pour préserver le ratio des classes à chaque pli.

Pour chaque combinaison d'hyperparamètres $\theta$ testée par GridSearchCV :

$$\text{Score}_{CV}(\theta) = \frac{1}{k} \sum_{i=1}^{k} \text{AUC-ROC}\left(M_\theta, \mathcal{D}_{\text{val}}^{(i)}\right)$$

Le modèle final est entraîné sur l'ensemble des données d'entraînement avec les hyperparamètres $\theta^*$ maximisant ce score.

---

### 5. Métriques d'Évaluation

$$\text{Précision} = \frac{TP}{TP + FP} \qquad \text{Rappel} = \frac{TP}{TP + FN}$$

$$F_1 = 2 \cdot \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}$$

$$\text{AUC-ROC} = \int_0^1 \text{TPR}(t)\ d\text{FPR}(t)$$

> **Choix de la métrique principale** : l'AUC-ROC a été privilégiée car elle est insensible au déséquilibre des classes et mesure la capacité du modèle à discriminer les deux classes indépendamment du seuil de décision.

---

## Résultats de Comparaison des Modèles

| Modèle | AUC-ROC | F1-Score | Précision | Rappel |
|--------|---------|----------|-----------|--------|
| Régression Logistique | *conf.* | *conf.* | *conf.* | *conf.* |
| Random Forest | *conf.* | *conf.* | *conf.* | *conf.* |
| XGBoost | *conf.* | *conf.* | *conf.* | *conf.* |
| SVM (RBF) | *conf.* | *conf.* | *conf.* | *conf.* |
| logistic Regressor | *conf.* | *conf.* | *conf.* | *conf.* |
| Adaboost | *conf.* | *conf.* | *conf.* | *conf.* |
| Arbre de décision | *conf.* | *conf.* | *conf.* | *conf.* |
| Gradient Boosting | *conf.* | *conf.* | *conf.* | *conf.* |
| **Meilleur modèle retenu** | **conf.** | **conf.** | **conf.** | **conf.** |

> Les métriques exactes sont disponibles sur demande dans le cadre d'un entretien professionnel.

---

## Conclusion et Apports Métier

Ce projet a permis de construire un système de scoring automatique des abonnés télécom selon leur probabilité de churn et leur profil de risque fraude multi-SIM. Les principaux apports sont :

- **Pour la fidélisation** : identifier les clients à fort risque de départ *avant* qu'ils ne partent, pour déclencher des actions de rétention ciblées et économiser les budgets marketing.
- **Pour la lutte contre la fraude** : détecter automatiquement les patterns multi-SIM suspects et les classer par niveau de priorité d'investigation.
- **Pour la méthodologie** : la démarche de détection du data leakage et de définition rigoureuse de la variable cible constitue un cadre réutilisable pour tout projet de prédiction sur données transactionnelles.

**Perspective** : déployer le modèle en production sous forme d'API REST, avec mise à jour mensuelle automatique du modèle sur les nouvelles données.

---

*Projet réalisé dans le cadre du cours de Data Mining  ISSEA  2025*
*Auteur : WAMBE Jordan Valère  wambejordan13@gmail.com*


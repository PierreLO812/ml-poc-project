# Explainable Autonomous Driving - Surrogate Modeling Proof of Concept (POC)

Ce projet est un Proof of Concept (POC) d'intelligence artificielle explicable (XAI) appliquée aux véhicules autonomes, basé sur le jeu de données **NuScenes** (v1.0-mini).

L'objectif de ce projet est de réaliser du **reverse engineering** comportemental : expliquer de façon intelligible (sous forme d'arbres de décision de régression) pourquoi un véhicule autonome (ou son conducteur humain) prend des décisions de conduite spécifiques, telles que tourner (braquage), accélérer ou freiner, à partir d'indicateurs extraits de son environnement.

---

## 🎯 Problématique & Concept

Plutôt que d'entraîner une boîte noire (Deep Learning complexe) pour piloter ou pour décider quand rendre la main à l'humain, ce projet met en œuvre un **modèle de substitution (Surrogate Model)**. 

En entraînant des arbres de décision simples à imiter les sorties d'accélération et de braquage, nous extrayons des **règles logiques transparentes** (des embranchements *SI / ALORS*) basées sur l'environnement immédiat du véhicule :
- *Pourquoi le véhicule a-t-il tourné ?* (ex: courbure de la voie élevée, fin de voie imminente).
- *Pourquoi le véhicule a-t-il freiné ?* (ex: présence d'un piéton ou d'un véhicule à faible distance).

---

## 📊 Caractéristiques du Dataset & Features

Le modèle est entraîné sur le dataset **NuScenes (v1.0-mini)** enrichi avec les cartes vectorielles du **NuScenes Map Expansion Pack**.

### Cibles comportementales (Targets)
- **Taux de changement de cap (`target_heading_rate`)** [rad/s] : Représente l'intention de braquage/virage.
- **Accélération (`target_acceleration`)** [m/s²] : Représente l'intention d'accélérer ou de freiner.

### Variables environnementales (Features)
1. **Contexte dynamique (Obstacles)** :
   - `velocity` : Vitesse actuelle du véhicule (m/s).
   - `nearest_obstacle_distance` : Distance par rapport à l'obstacle le plus proche (véhicule ou piéton).
   - `obstacles_within_10m` : Nombre d'obstacles (véhicules/piétons) présents dans un rayon de 10 mètres.
2. **Contexte spatial (Géométrie de la route)** :
   - `lane_curvature` : Courbure locale de la voie sur laquelle circule le véhicule.
   - `distance_along_lane` : Distance parcourue par le véhicule depuis le début de sa voie actuelle.
   - `distance_to_lane_end` : Distance restante avant la fin de la voie courante.

---

## 🛠️ Pipeline Technique & Optimisations

Le script `src/surrogatemodel.py` implémente les étapes suivantes :

1. **Extraction Spatialisée & Temporelle** : Analyse des annotations NuScenes et interrogation de l'API de cartes vectorielles pour projeter les poses du véhicule sur les voies de circulation.
2. **Filtre Probabiliste "Boucle d'Or" (Data Balancing)** :
   La conduite autonome comporte une écrasante majorité de phases neutres (conduite en ligne droite à vitesse stable). Pour éviter que le modèle ne soit submergé par ces données peu informatives, le pipeline applique un filtre probabiliste : il ne conserve que **15%** des données "normales" (`|acc| <= 0.5` et `|heading_rate| <= 0.05`), tout en gardant **100%** des situations extrêmes (freinage brusque, évitement, virage serré).
3. **Robustesse & Généralisation** :
   - Bridage physique de l'accélération (`.clip(lower=-8.0, upper=4.0)`) pour écarter les anomalies de capteurs.
   - Limitation de l'arbre (`max_depth=4`, `min_samples_leaf=10`) pour forcer des règles simples, explicables et non sur-entraînées (overfitting).
4. **Évaluation Duale** : Évaluation du modèle par le score R² et par la MAE (Mean Absolute Error) avec les unités physiques.

---

## ⚙️ Installation & Lancement

### Prérequis
- Python 3.12 (ou version compatible)
- Le jeu de données **NuScenes v1.0-mini** placé dans `data/sets/`.
- Le **NuScenes Map Expansion Pack** (fichiers JSON comme `singapore-onenorth.json`) placé dans `data/sets/maps/expansion/`.

### Dépendances
Installez les dépendances du projet :
```bash
pip install -r requirements.txt
```

*(Si nécessaire, installez la bibliothèque d'outils NuScenes : `pip install nuscenes-devkit`)*

### Exécution & Visualisation

Vous pouvez exécuter le projet de deux manières :

#### 1. Modèle de substitution en ligne de commande (Terminal)
Générez les arbres et visualisez les règles brutes dans votre terminal :
```bash
python src/surrogatemodel.py
```
Le script affichera la barre de progression de l'extraction, entraînera les modèles, affichera les métriques ($R^2$ et MAE) et sauvegardera automatiquement les données finalisées sous `data/extracted_dataset.csv` pour alimenter le dashboard Streamlit.

#### 2. Dashboard Interactif Streamlit (Web)
Lancez l'interface graphique interactive pour explorer et jouer avec le modèle :
```bash
streamlit run scripts/main.py
```

---

## 🎨 Fonctionnalités du Dashboard Streamlit

L'interface utilisateur Streamlit est structurée en deux grands espaces :

### 1. 🌎 Onglet 1 : Le Dataset NuScenes
* **Visualisation et animations** : Intégration de GIFs et d'illustrations issus du devkit NuScenes pour mieux appréhender les trajectoires du Prediction Challenge.
* **Chiffres clés du projet** : Récapitulatif du nombre de scènes totales (1000) vs mini (10), les villes d'acquisition (Boston & Singapour) et le volume total d'échantillons continus extraits.
* **Visualisation descriptive** : Graphiques de distribution des variables comme la vitesse actuelle du véhicule.

### 2. 🧠 Onglet 2 : Surrogate Model & Explicabilité
* **Métriques clés d'évaluation** : Suivi des scores $R^2$ et MAE en unités réelles pour les modèles de braquage (`rad/s`) et d'accélération (`m/s²`).
* **Simulateur d'environnement interactif** : Des curseurs (sliders) dans le volet latéral permettent d'ajuster en temps réel la vitesse, la courbure de la voie, la distance d'un obstacle ou encore la distance restante avant la fin de la voie.
* **Jauges dynamiques** : Visualisation instantanée de l'accélération et de l'angle de braquage prédits sous forme de cadrans/jauges interactifs (Plotly).
* **Traceur XAI (Explainable AI) pas à pas** : Affiche le cheminement logique exact pris dans l'arbre de décision (les nœuds traversés et conditions validées) pour expliquer précisément *pourquoi* le modèle a pris cette décision.

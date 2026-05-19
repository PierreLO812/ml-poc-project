import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from tqdm import tqdm

# 1. Initialisation
DATAROOT = 'd:/ml-poc/data/sets'
print(f"Chargement de NuScenes à partir de {DATAROOT}...")
nuscenes = NuScenes('v1.0-mini', dataroot=DATAROOT)
helper = PredictHelper(nuscenes)

# Tentative de chargement des cartes pour la géométrie de la route (NuScenesMap)
maps = {}
try:
    from nuscenes.map_expansion.map_api import NuScenesMap
    from nuscenes.map_expansion import arcline_path_utils
    print("Chargement des cartes pour extraire la courbure des voies...")
    for loc in ['singapore-onenorth', 'boston-seaport', 'singapore-hollandvillage', 'singapore-queenstown']:
        try:
            maps[loc] = NuScenesMap(map_name=loc, dataroot=DATAROOT)
        except Exception:
            pass
except ImportError:
    pass

if not maps:
    print("ATTENTION: Les cartes (map_expansion) n'ont pas pu être chargées.")
    print("Assurez-vous d'avoir téléchargé le Map Expansion Pack pour bénéficier de lane_curvature.")

# 2. Extraction des features (Environnement) et Targets (Comportement)
print("\nExtraction des données (avec contexte continu de la conduite)...")
data = []

target_samples = 1500
pbar = tqdm(total=target_samples, desc="Extraction diversifiée")

all_anns = [a for a in nuscenes.sample_annotation if 'vehicle' in a['category_name']]
# Mélanger pour éviter de prendre des échantillons fortement corrélés (de la même scène)
random.seed(42)
random.shuffle(all_anns)

for ann in all_anns:
    if len(data) >= target_samples: 
        break
        
    instance_token = ann['instance_token']
    sample_token = ann['sample_token']
    
    try:
        # --- TARGETS ---
        acc = helper.get_acceleration_for_agent(instance_token, sample_token)
        heading_rate = helper.get_heading_change_rate_for_agent(instance_token, sample_token)
        
        # Ignorer les NaN
        if np.isnan(acc) or np.isnan(heading_rate):
            continue
            
        # --- Filtre "Boucle d'Or" (Probabiliste) ---
        # Si la voiture roule normalement, on ne garde l'échantillon que dans 15% des cas.
        if abs(acc) <= 0.5 and abs(heading_rate) <= 0.05:
            if random.random() > 0.15:
                continue
            
        # --- FEATURES ---
        vel = helper.get_velocity_for_agent(instance_token, sample_token)
        if np.isnan(vel):
            continue
            
        # Récupération de la localisation pour interroger la bonne carte
        sample_record = nuscenes.get('sample', sample_token)
        scene_record = nuscenes.get('scene', sample_record['scene_token'])
        log_record = nuscenes.get('log', scene_record['log_token'])
        location = log_record['location']
        
        other_anns_tokens = sample_record['anns']
        
        x, y = ann['translation'][0], ann['translation'][1]
        
        min_dist = float('inf')
        density_10m = 0
        
        # Distance avec les autres véhicules/piétons
        for other_token in other_anns_tokens:
            if other_token == ann['token']:
                continue
                
            other_ann = nuscenes.get('sample_annotation', other_token)
            if 'vehicle' not in other_ann['category_name'] and 'human' not in other_ann['category_name']:
                continue
                
            ox, oy = other_ann['translation'][0], other_ann['translation'][1]
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            
            if dist < min_dist:
                min_dist = dist
            if dist < 10.0:
                density_10m += 1
                
        if min_dist == float('inf'):
            min_dist = 50.0 
            
        # --- ROAD GEOMETRY (Courbure de la route et distance restante) ---
        curvature = 0.0
        dist_along_lane = 0.0
        dist_to_lane_end = 50.0 # valeur par défaut
        
        if location in maps:
            nusc_map = maps[location]
            closest_lane = nusc_map.get_closest_lane(x, y, radius=5)
            if closest_lane:
                lane_record = nusc_map.get_arcline_path(closest_lane)
                _, dist_along_lane = arcline_path_utils.project_pose_to_lane((x, y, 0), lane_record)
                curvature = arcline_path_utils.get_curvature_at_distance_along_lane(dist_along_lane, lane_record)
                
                # Ajout de la distance restante avant la fin de la voie
                lane_length = arcline_path_utils.length_of_lane(lane_record)
                dist_to_lane_end = lane_length - dist_along_lane
            
        data.append({
            'velocity': vel,
            'nearest_obstacle_distance': min_dist,
            'obstacles_within_10m': density_10m,
            'lane_curvature': curvature,
            'distance_along_lane': dist_along_lane,
            'distance_to_lane_end': dist_to_lane_end,
            'target_acceleration': acc,
            'target_heading_rate': heading_rate
        })
        
        pbar.update(1)
        
    except Exception as e:
        continue

pbar.close()
df = pd.DataFrame(data)

# Sauvegarde du dataset pour le charger instantanément dans Streamlit
import os
os.makedirs('data', exist_ok=True)
df.to_csv('data/extracted_dataset.csv', index=False)
print("Dataset sauvegardé sous data/extracted_dataset.csv")

if len(df) < 10:
    print("Pas assez de données extraites. Essayez d'augmenter le nombre de scènes disponibles.")
else:
    # Bridage physique pour ignorer les erreurs de capteurs
    df['target_acceleration'] = df['target_acceleration'].clip(lower=-8.0, upper=4.0)
    
    print(f"\nDataset finalisé: {len(df)} échantillons continus.")
    
    # --- 3. EXPLICABILITÉ DU BRAQUAGE (Heading Change Rate) ---
    print("\n" + "="*70)
    print("SURROGATE MODEL : POURQUOI A-T-IL TOURNÉ ? (Heading Rate)")
    print("="*70)

    # Ajout du contexte de la route aux features
    X_steering = df[['velocity', 'nearest_obstacle_distance', 'obstacles_within_10m', 'lane_curvature', 'distance_along_lane', 'distance_to_lane_end']]
    y_steering = df['target_heading_rate'] 

    X_train, X_test, y_train, y_test = train_test_split(X_steering, y_steering, test_size=0.2, random_state=42)

    # Paramétrage pour éviter l'overfitting (max_depth=4, min_samples_leaf=10)
    surrogate_steering = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
    surrogate_steering.fit(X_train, y_train)

    score_steering = surrogate_steering.score(X_test, y_test)
    mae_steering = mean_absolute_error(y_test, surrogate_steering.predict(X_test))
    
    print(f"R² du Surrogate Model sur le braquage : {score_steering:.2f}")
    print(f"MAE (Erreur Moyenne Absolue)          : {mae_steering:.3f} rad/s")
    print("\nRègles d'explicabilité (Comment la route et les obstacles dictent le braquage) :")
    print(export_text(surrogate_steering, feature_names=list(X_steering.columns)))

    # --- 4. EXPLICABILITÉ DE L'ACCÉLÉRATION (Freinage/Accélération) ---
    print("\n" + "="*70)
    print("SURROGATE MODEL : POURQUOI A-T-IL ACCÉLÉRÉ OU FREINÉ ? (Acceleration)")
    print("="*70)

    X_acc = df[['velocity', 'nearest_obstacle_distance', 'obstacles_within_10m', 'lane_curvature', 'distance_along_lane', 'distance_to_lane_end']]
    y_acc = df['target_acceleration']
    
    X_train_acc, X_test_acc, y_train_acc, y_test_acc = train_test_split(X_acc, y_acc, test_size=0.2, random_state=42)

    surrogate_acc = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
    surrogate_acc.fit(X_train_acc, y_train_acc)

    score_acc = surrogate_acc.score(X_test_acc, y_test_acc)
    mae_acc = mean_absolute_error(y_test_acc, surrogate_acc.predict(X_test_acc))
    
    print(f"R² du Surrogate Model sur l'accélération : {score_acc:.2f}")
    print(f"MAE (Erreur Moyenne Absolue)             : {mae_acc:.3f} m/s²")
    print("\nRègles d'explicabilité (Comment l'environnement dicte l'accélération) :")
    print(export_text(surrogate_acc, feature_names=list(X_acc.columns)))
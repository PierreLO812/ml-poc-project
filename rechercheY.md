Le Risque de Trajectoire (Planning Uncertainty)On relie la Perception au Motion Planning (très valorisé dans ton cours Coursera). Rendre la main en ligne droite est moins urgent que rendre la main dans un virage ou une intersection.

$Y = 1$ (Danger) : Angle de braquage élevé (virage) + Conditions météo dégradées ou mauvaise visibilité.
$Y = 0$ (Safe) : Ligne droite, même sous la pluie, OU Virage par temps clair.

Comment l'extraire : Croiser les données du CAN bus (steering angle) avec les métadonnées de la scène.

L'avantage pour le prof : Le "Storytelling" est incroyable : "Mon IA ne se contente pas de regarder la pluie, elle rend la main uniquement si la météo met en péril la manœuvre en cours."



ok je viens de parler avec le prof on part sur un modele inverse, un algo qui choisit si c'est une IA ou un humain qui doit conduire ça existe déjà c'est nul, donc pour avoir de la valeur, en fonction de ça, on fait un sorte de reverse engeneering pour trouver la cause du pourquoi on a un humain qui a la main, ou de pourquoi on la laisse à l'IA

Il m'a parlé de surrogate model

# Reste à faire : 
+ Streamlit
+ .gitignore
+ readme



Premier résultat du surrogate model : 

PS D:\ml-poc>  d:; cd 'd:\ml-poc'; & 'c:\Users\pierr\AppData\Local\Programs\Python\Python312\python.exe' 'c:\Users\pierr\.antigravity\extensions\ms-python.debugpy-2026.6.0-win32-x64\bundled\libs\debugpy\launcher' '64705' '--' 'D:\ml-poc\src\surrogatemodel.py' 
Chargement de NuScenes à partir de d:/ml-poc/data/sets...
======
Loading NuScenes tables for version v1.0-mini...
23 category,
8 attribute,
4 visibility,
911 instance,
12 sensor,
120 calibrated_sensor,
31206 ego_pose,
8 log,
10 scene,
404 sample,
31206 sample_data,
18538 sample_annotation,
4 map,
Done loading in 0.246 seconds.
======
Reverse indexing ...
Done reverse indexing in 0.0 seconds.
======
Extraction des données environnementales et comportementales...
Dataset créé avec 1000 échantillons pertinents.

======================================================================
SURROGATE MODEL : POURQUOI A-T-IL TOURNÉ ? (Heading Rate)
======================================================================
R² du Surrogate Model sur le braquage : 0.15

Règles d'explicabilité (Comment l'environnement dicte le braquage) :
|--- velocity <= 4.98
|   |--- velocity <= 1.11
|   |   |--- nearest_obstacle_distance <= 0.79
|   |   |   |--- value: [0.01]
|   |   |--- nearest_obstacle_distance >  0.79
|   |   |   |--- value: [-0.00]
|   |--- velocity >  1.11
|   |   |--- velocity <= 4.32
|   |   |   |--- value: [0.13]
|   |   |--- velocity >  4.32
|   |   |   |--- value: [-0.01]
|--- velocity >  4.98
|   |--- velocity <= 7.30
|   |   |--- nearest_obstacle_distance <= 5.01
|   |   |   |--- value: [0.12]
|   |   |--- nearest_obstacle_distance >  5.01
|   |   |   |--- value: [-0.16]
|   |--- velocity >  7.30
|   |   |--- velocity <= 7.84
|   |   |   |--- value: [-0.04]
|   |   |--- velocity >  7.84
|   |   |   |--- value: [-0.00]


======================================================================
SURROGATE MODEL : POURQUOI A-T-IL ACCÉLÉRÉ OU FREINÉ ? (Acceleration)
======================================================================
R² du Surrogate Model sur l'accélération : -0.26

Règles d'explicabilité (Comment l'environnement dicte l'accélération) :
|--- velocity <= 9.41
|   |--- velocity <= 7.58
|   |   |--- velocity <= 7.44
|   |   |   |--- value: [0.01]
|   |   |--- velocity >  7.44
|   |   |   |--- value: [-6.17]
|   |--- velocity >  7.58
|   |   |--- nearest_obstacle_distance <= 4.21
|   |   |   |--- value: [4.63]
|   |   |--- nearest_obstacle_distance >  4.21
|   |   |   |--- value: [0.45]
|--- velocity >  9.41
|   |--- nearest_obstacle_distance <= 4.44
|   |   |--- nearest_obstacle_distance <= 3.93
|   |   |   |--- value: [1.35]
|   |   |--- nearest_obstacle_distance >  3.93
|   |   |   |--- value: [-0.21]
|   |--- nearest_obstacle_distance >  4.44
|   |   |--- velocity <= 10.06
|   |   |   |--- value: [1.82]
|   |   |--- velocity >  10.06
|   |   |   |--- value: [4.91]
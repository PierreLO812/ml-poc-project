Le Risque de Trajectoire (Planning Uncertainty)On relie la Perception au Motion Planning (très valorisé dans ton cours Coursera). Rendre la main en ligne droite est moins urgent que rendre la main dans un virage ou une intersection.

$Y = 1$ (Danger) : Angle de braquage élevé (virage) + Conditions météo dégradées ou mauvaise visibilité.
$Y = 0$ (Safe) : Ligne droite, même sous la pluie, OU Virage par temps clair.

Comment l'extraire : Croiser les données du CAN bus (steering angle) avec les métadonnées de la scène.

L'avantage pour le prof : Le "Storytelling" est incroyable : "Mon IA ne se contente pas de regarder la pluie, elle rend la main uniquement si la météo met en péril la manœuvre en cours."



ok je viens de parler avec le prof on part sur un modele inverse, un algo qui choisit si c'est une IA ou un humain qui doit conduire ça existe déjà c'est nul, donc pour avoir de la valeur, en fonction de ça, on fait un sorte de reverse engeneering pour trouver la cause du pourquoi on a un humain qui a la main, ou de pourquoi on la laisse à l'IA

Il m'a parlé de surrogate model
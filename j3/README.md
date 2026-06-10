# Jour 3

### Phase A : Prédire les prix immobiliers (régression)

Sur le cas limite (100 observations), les performances de la régression linéaire semblent augmenter (R² de 0.58 à 0.71). Le Random Forest conserve des performances stables.

### Phase B : Segmenter les clients d'AirBnB (non supervisé)

Sans standardisation, les segments sont principalement déterminés par les variables ayant la plus grande échelle, notamment le prix. Le clustering perd alors sa capacité à exploiter l'ensemble des caractéristiques des annonces.

L'introduction d'une annonce à 100 000 € perturbe fortement le clustering. Une seule valeur aberrante peut donc dégrader la qualité de l'ensemble de la segmentation. Ce résultat confirme que le nettoyage du J2 est un pré-requis.
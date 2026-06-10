# Jour 3

### Phase A : Prédire les prix immobiliers (régression)

Sur le cas limite (100 observations), les performances de la régression linéaire semblent augmenter (R² de 0.58 à 0.71). Le Random Forest conserve des performances stables.

### Phase B : Segmenter les clients d'AirBnB (non supervisé)

Sans standardisation, les segments sont principalement déterminés par les variables ayant la plus grande échelle, notamment le prix. Le clustering perd alors sa capacité à exploiter l'ensemble des caractéristiques des annonces.

L'introduction d'une annonce à 100 000 € perturbe fortement le clustering. Une seule valeur aberrante peut donc dégrader la qualité de l'ensemble de la segmentation. Ce résultat confirme que le nettoyage du J2 est un pré-requis.

### Phase C : Courriel vs spam (texte)

Dataset : *SMS Spam Collection (UCI)*

La régression logistique obtient les meilleures performances avec un F1-score `spam` de 0.89 contre 0.83 pour Naive Bayes. Les deux modèles affichent une précision de 100 %, ce qui signifie qu'ils génèrent très peu de faux positifs. En revanche, leur rappel reste inférieur à 85 %, ce qui indique que certains spams passent encore au travers du filtre. Le cas du message vide montre que les modèles reviennent naturellement à la classe majoritaire lorsqu'ils ne disposent d'aucune information. Enfin, un spam déguisé utilisant un vocabulaire proche des messages normaux peut tromper le modèle.

### Phase D : Décrypter les signaux d'un sonar (classification binaire)

Dataset : *Sonar / Connectionist Bench (Mines vs Rocks), UCI*

Le SVM RBF obtient les meilleures performances. La régression logistique est correcte. Le Random Forest reste compétitif mais légèrement en retrait.

Sans standardisation, les performances du SVM et de la régression logistique diminuent. Le Random Forest est beaucoup moins affecté et robuste aux modifications d'échelles.

Lorsque toutes les entrées du signal sont nulles, les trois modèles produisent malgré tout une prédiction avec une certaine confiance, alors qu'un tel signal correspond en réalité à une panne de capteur. En pratique, un système sonar devrait d’abord inclure un module de détection d'anomalies ou de validation du signal avant toute classification.


### Phase E : Le Fight des IA (ouverte)

Le Fight des IA montre qu'il n'existe pas de modèle universellement meilleur. Le classement dépend fortement du type de données et de la métrique choisie. Sur Sonar, le SVM RBF domine tandis que sur le dataset Spam, le classement change avec des modèles optimisés pour le rappel. Enfin, sur Airbnb, le problème n'est plus supervisé et la qualité est évaluée par le score de silhouette, qui suggère ici une segmentation naturelle en six groupes. Les temps d'entraînement rappellent également qu'un modèle légèrement meilleur n'est pas toujours le meilleur choix opérationnel.
# Jour 4

Travail sur le jeu de données du cancer (*breast_cancer*)

### Phase 1 : Séparer les données proprement, train / validation / test

La stratification permet de conserver la distribution des classes dans les ensembles d’entraînement, de validation et de test, même en présence d’un fort déséquilibre. Le ratio 95/5 est conservé dans les 3 jeux. La gestion des cas limites (val_size=0) empêche des usages incohérents du pipeline.

### Phase 2 : Bootstrap et bagging, comprendre le rééchantillonnage

Les résultats montrent que les performances moyennes sont relativement stables entre les modèles, mais que l'écart-type devient un indicateur de robustesse. L'oubli de remplacement ne change pas fortement les scores dans ce dataset. Enfin, un nombre d’itérations trop faible (n=1) empêche toute estimation fiable de l'écart-type.
# Jour 4

Travail sur le jeu de données du cancer (*breast_cancer*)

### Phase 1 : Séparer les données proprement, train / validation / test

La stratification permet de conserver la distribution des classes dans les ensembles d’entraînement, de validation et de test, même en présence d’un fort déséquilibre. Le ratio 95/5 est conservé dans les 3 jeux. La gestion des cas limites (val_size=0) empêche des usages incohérents du pipeline.

### Phase 2 : Bootstrap et bagging, comprendre le rééchantillonnage

Les résultats montrent que les performances moyennes sont relativement stables entre les modèles, mais que l'écart-type devient un indicateur de robustesse. L'oubli de remplacement ne change pas fortement les scores dans ce dataset. Enfin, un nombre d’itérations trop faible (n=1) empêche toute estimation fiable de l'écart-type.

### Phase 3 : La validation croisée k-fold

La validation croisée permet d'estimer la performance moyenne d'un modèle. La stratification a un impact variable selon les modèles et la distribution des données. Sur ce dataset relativement équilibré à grande échelle, l'effet reste limité, mais elle reste essentielle pour garantir la représentativité des folds.


### Phase 4 : Choisir la bonne métrique selon le coût métier

Les modèles avec meilleure accuracy ne sont pas forcément les meilleurs en coût métier. Ici, Logistic Regression et Random Forest dominent car ils minimisent les faux négatifs, qui sont critiques en diagnostic médical.

Un modèle peut atteindre un recall parfait en prédisant systématiquement la classe positive, mais cela entraîne un nombre élevé de faux positifs et un coût opérationnel important.

L'évaluation doit donc intégrer les coûts des erreurs et non uniquement les métriques globales.

### Phase 5 : Sérialiser le modèle et le servir derrière une API

Les cas limites et adversariaux sont correctement gérés. Une requête sans la clé *features* renvoie une erreur HTTP 400 explicite. Une requête avec un nombre incorrect de variables est également rejetée avant l'appel au modèle. Les validations empêchent le modèle de recevoir des données incohérentes.

### Phase 6 : Déployer une WebApp de prédiction

La WebApp permet à un utilisateur de saisir les caractéristiques d'une tumeur et d'obtenir instantanément une prédiction ainsi que le niveau de confiance du modèle.
# Arène des Algos

Objectif : monter un pipeline de Machine Learning (ML) sur un ou plusieurs dataset(s) existant(s) pour comparer plusieurs algorithmes et effectuer un classement

### Jour 1
### Phase 1: Chargement et exploration du dataset
Dataset utilisé : *load_breast_cancer* de `sklearn`

Étude de 3 situations: 
- Cas normal: 
```
Lignes, colonnes : (569, 30)
Class 0 (malignant): 212 cases
Class 1 (benign): 357 cases
```
- Cas limite (avec filtre): 
```
Lignes, colonnes : (212, 30)
Class 0 (malignant) : 212 cases 
Lignes, colonnes : (357, 30)
Class 1 (benign) : 357 cases 
```
- Cas adversarial (répartition en pourcentage):
```
Lignes, colonnes : (569, 30)
Class 0 (malignant) : 212 cases (37.26%)
Class 1 (benign) : 357 cases (62.74%)
```

### Phase 3: 1er classement des modèles
Ayant utilisé le modèle de régression logistique, le programme a affiché un avertissement de non-convergence, car le nombre d'itérations par défaut (`max_iter=100`) est petit et le modèle s'arrête avant convergence. Il est donc nécessaire d'augmenter cette variable (jusqu'à >1000) pour pouvoir se rapprocher de la convergence souhaitée et obtenir une précision élevée.
```
🏆 Classement des modèles :
1. Logistic Regression → 96.5%
2. KNN → 93.0%
3. Decision Tree → 91.2%
```



### Phase 4: Bascule non-supervisé

```
col_0    0    1
row_0          
0      130   82
1        1  356
```
On remarque que la classe 1 (bénigne) est quasi entièrement regroupée dans un seul cluster alors que la classe 0 (maligne) est plus dispersée à travers les 2 clusters. Nous obtenons un score de correspondance à 49%, ce qui indique une correspondance partielle. Ce score montre qu'il existe une structure dans les données alignée avec les classes réelles mais aussi qu'elle ne semble pas suffisante pour reconstruire parfaitement les labels sans supervision.

### Phase 5: Changement de terrain
Dataset utilisé : *load_wine* de `sklearn`

```
🏆 Classement des modèles :
1. Logistic Regression → 94.4%
2. Decision Tree → 94.4%
3. KNN → 75.0%
```

Le classement a changé. On constate que le KNN est moins performant sur ce nouveau dataset, tandis que la régression logistique et l'arbre de décision restent très performants.

La fonction `arene` encaisse également un dataset à 3 classes sans modification nécessaire aux fonctions d'exploration, d'entraînement, d'évaluation, de prédiction, de comparaison.

Les résultats d'exploration du dataset montre qu'une classe peut être moins représentée que les autres à travers l'affichage des pourcentages.

### Phase 6: Voir pour comprendre (graphiques)

Le graphique est bien lisible grâce au titre et axes X et Y nommées pour montrer les indicateurs d'analyse.

Le champion s'est trompé plus dans le sens "Prédit Bénigne --> Réel Maligne" avec 3 erreurs par rapport à 1 erreur "Prédit Maligne --> Réel Bénigne".
Pour le wine, il n'y a eu qu'un seul cas d'erreur.

Sur le cancer du sein, dans le contexte médical, la situation la plus grave est d'obtenir un faux négatif (tumeur maligne --> prédite bénigne) car cela indique que le patient est rassuré alors qu'il est malade. Un faux positif (tumeur bénigne --> prédite maligne) peut être source de stress dû à des examents non urgents, même si c'est souvent moins grave qu'un faux négatif.

Ainsi, l'accuracy seule (même élevée) ne suffit pas à évaluer un modèle car le résultat du modèle peut avoir des conséquences différentes (faux positifs/négatifs).

La matrice de confusion était plus informative qu'une simple accuracy.

### Phase 7: Le buff scaling (et la triche qui se retourne contre vous)

**Manche 1 : le buff**

Qui profite du buff ? **KNN : fort gain**, notamment sur le *wine_dataset*

Gain modéré pour la **régression logistique**

Aucun effet avec **l'arbre de décision**.

Ces résultats confirment bien qu'un algo qui raisonne par distances n'a pas le même rapport aux échelles qu'un algo qui découpe par seuils.
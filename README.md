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

### Phase 4: Bascule non-supervisé

```
col_0    0    1
row_0          
0      130   82
1        1  356
```
On remarque que la classe 1 (bénigne) est quasi entièrement regroupée dans un seul cluster alors que la classe 0 (maligne) est plus dispersée à travers les 2 clusters. Nous obtenons un score de correspondance à 49%, ce qui indique une correspondance partielle. Ce score montre qu'il existe une structure dans les données.
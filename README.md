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
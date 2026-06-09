## Collecte et traitement de données

Travail sur le dataset **Telco Customer Churn** *(source: Kaggle)*

### Phase 1 : L'audit qualité

Aucun manquant détecté. En filtrant le dataset à une seule classe de churn, on remarque que le rapport reste honnête. La cible est en effet déséquilibrée :
- La classe **Non** est majoritaire (73,5 %)
- La classe **Oui** est minoritaire (26,5 %).
L'audit l'a rendu visible en un coup d'oeil.

### Phase 2 : La colonne piégée (types incohérents et trous cachés)

Ayant détecté 11 trous cachés, cela reste une proportion assez faible sur plus de 7000 lignes. Donc on garde un avantage à conserver toutes les lignes, l'impact étant faible. Supprimer les lignes correspondantes risque de provoquer une perte d'observations et de ne pas obtenir une qualité forte des données. C'est pourquoi il est préférable d'imputer ces trous (via la médiane).

Si la colonne est entièrement composée de texte non numérique, la fonction gérera ce cas et lèvera une erreur au lieu de transformer silencieusement toute la colonne en NaN.

Une corruption comme `"29,90"` peut être détectée en inspectant les valeurs non convertibles avant la conversion.


### Phase 3 : Encoder les catégorielles

Bien que les modalités de `Contract` indique un ordre apparent (Month-to-Month = 0, One Year = 1, Two year = 2), il est plus sûr d'encoder ces variables nominales en One-Hot afin de ne pas imposer arbitrairement une relation numérique linéaire entre les catégories. Cette approche laisse également au modèle le soin d'apprendre l'effet spécifique de chaque type de contrat.

Si la colonne contient une catégorie ultra-rare présente sur 1 seul
client, le One-Hot créera quand même une colonne entière pour un seul client, ce qui provoquera une augmentation de la dimension du dataset.

`customerID` possède presque autant de modalités que de lignes (7043 identifiants uniques pour 7043 clients). Si l'on encode par erreur `customerID`, le One-Hot créera donc plusieurs milliers de colonnes supplémentaires. Le dataset est passé de 21 colonnes à 6572 colonnes, ce qui illustre le phénomène d'explosion de dimensions. `customerID` doit être supprimé avant l'encodage.
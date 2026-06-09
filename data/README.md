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
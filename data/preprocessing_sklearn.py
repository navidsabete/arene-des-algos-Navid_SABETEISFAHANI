from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import numpy as np

def construire_preprocesseur(X):
    colonnes_num = (
        X.select_dtypes(include=[np.number, np.float64])
        .columns
        .tolist()
    )

    colonnes_cat = (
        X.select_dtypes(include=["object", "string"])
        .columns
        .tolist()
    )

    if "customerID" in colonnes_cat:
        colonnes_cat.remove("customerID")

    pipeline_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    pipeline_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    preprocesseur = ColumnTransformer([
        ("num", pipeline_num, colonnes_num),
        ("cat", pipeline_cat, colonnes_cat)
    ])

    return preprocesseur
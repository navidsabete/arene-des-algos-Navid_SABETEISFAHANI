import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


def audit_qualite(df):
    print(f"Forme : {df.shape}")
    print("\nTypes des colonnes :")
    print(df.dtypes)

    print("\nPourcentage de valeurs manquantes :")

    manquants = (
        df.isna().mean().mul(100).sort_values(ascending=False)
    )
    nb_colonnes_manquantes = (manquants > 0).sum()
    print(
        f"Manquants détectés : "
        f"{nb_colonnes_manquantes} colonne(s)"
    )

    for colonne, pct in manquants.items():
        print(f"{colonne:<25} {pct:.2f}%")
    
    if "Churn" not in df.columns:
        print("\n⚠️ Colonne 'Churn' absente.")
        return

    print("\nRépartition de Churn :")

    counts = df["Churn"].value_counts().sort_index()
    total = len(df)

    for valeur, nb in counts.items():
        pct = 100 * nb / total if total > 0 else 0
        print(f"Churn {valeur} : {nb} ({pct:.1f}%)")


def reparer_total_charges(df):
    df_repare = df.copy()
    
    if "TotalCharges" not in df.columns:
        print("\n⚠️ Colonne 'TotalCharges' absente.")
        return
    
    audit_virgule = (
        df["MonthlyCharges"]
        .astype(str)
        .str.contains(",", regex=False)
    )

    print(
        "Valeurs contenant une virgule :",
        audit_virgule.sum()
    )

    total_charges_num = pd.to_numeric(
        df_repare["TotalCharges"],
        errors="coerce" 
    )

    nb_nan = total_charges_num.isna().sum()
    print(f"Trous démasqués : {nb_nan}")

    if nb_nan == len(df_repare):
        print("\n⚠️ Échec de conversion : la colonne est 100% non numérique.")
        return
    
    df_repare["TotalCharges"] = total_charges_num

    mediane = df_repare["TotalCharges"].median()
    df_repare["TotalCharges"] = (
        df_repare["TotalCharges"]
        .fillna(mediane)
    )
    print(
        f"Type final : {df_repare['TotalCharges'].dtype}"
    )

    return df_repare

def encoder_features(df):
    df_encode = df.copy()
    if "customerID" in df_encode.columns:
        df_encode = df_encode.drop(columns=["customerID"])

    colonnes_obj = df_encode.select_dtypes(include=["object", "string"]).columns.tolist()

    # On ne touche pas à la cible si elle existe
    if "Churn" in colonnes_obj:
        colonnes_obj.remove("Churn")

    colonnes_binaires_y_n = []
    for col in colonnes_obj:
        valeurs = set(df_encode[col].dropna().unique())
        if valeurs <= {"Yes", "No"}:
            colonnes_binaires_y_n.append(col)
    
    for col in colonnes_binaires_y_n:
        df_encode[col] = df_encode[col].map({
            "No": 0,
            "Yes": 1
        })
    
    colonnes_nominales = [
        c for c in colonnes_obj
        if c not in colonnes_binaires_y_n
    ]

    # One-Hot Encoding
    df_encode = pd.get_dummies(
        df_encode,
        columns=colonnes_nominales,
        drop_first=False,
        dtype=int
    )

    if "Churn" in df_encode.columns:
        df_encode["Churn"] = df_encode["Churn"].map({
            "No": 0,
            "Yes": 1
        })
    
    print(f"Forme après encodage : {df_encode.shape}")

    return df_encode

def detecter_outliers_iqr(df, colonne):
    if colonne not in df.columns:
        print(f"Colonne {colonne} absente.")
        return
    
    serie = df[colonne].dropna()

    if serie.empty:
        return None, None, 0
    
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)

    IQR = Q3 - Q1

    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR

    outliers = serie[(serie < borne_basse) | (serie > borne_haute)]

    nb_outliers = len(outliers)

    print(f"\nColonne : {colonne}")
    print(f"Borne basse : {borne_basse:.2f}")
    print(f"Borne haute : {borne_haute:.2f}")
    print(f"Outliers détectés : {nb_outliers}")

    return borne_basse, borne_haute, nb_outliers

def boxplot_colonne(df, colonne):
    plt.figure(figsize=(6, 6))
    plt.boxplot(df[colonne].dropna(), vert=False)
    plt.title(f"Boxplot - {colonne}")
    plt.xlabel(colonne)
    plt.show()


def rapport_multicolinearite(df, colonnes_num):
    data = df[colonnes_num].copy()

    plt.figure(figsize=(8, 6))
    corr = data.corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Heatmap des corrélations")
    plt.show()

    vif_data = []

    for i in range(data.shape[1]):
        try:
            vif = variance_inflation_factor(data.values, i)
        except Exception:
            vif = np.inf

        vif_data.append((data.columns[i], vif))

    vif_df = pd.DataFrame(vif_data, columns=["Variable", "VIF"])
    vif_df = vif_df.sort_values(by="VIF", ascending=False)

    print("\n📊 VIF des variables :\n")
    print(vif_df)

    print("\n⚠️ Variables avec VIF > 5 :")
    print(vif_df[vif_df["VIF"] > 5])

    return vif_df


def features_discriminantes(df, cible="Churn"):    
    data = df.copy()
    
    y = data[cible]
    X = data.drop(columns=[cible])

    X = X.select_dtypes(include=[np.number])

    corr = X.corrwith(y).abs().sort_values(ascending=False)

    corr_df = pd.DataFrame({
        "feature": corr.index,
        "corr": corr.values
    })

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X, y)

    importance = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    rf_df = pd.DataFrame({
        "feature": importance.index,
        "rf_importance": importance.values
    })

    final = pd.merge(corr_df, rf_df, on="feature")

    final["rank_corr"] = final["corr"].rank(ascending=False)
    final["rank_rf"] = final["rf_importance"].rank(ascending=False)
    final["rank_moyen"] = (final["rank_corr"] + final["rank_rf"]) / 2

    final = final.sort_values("rank_moyen")

    print("\n🏆 TOP 10 FEATURES (comparaison)\n")
    print(final.head(10))

    return final


def split_et_scale_proprement(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def split_et_scale_proprement_sans_stratify(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def comparer_fuite(X, y):
    X_train_scaled, X_test_scaled, y_train, y_test = split_et_scale_proprement(X, y)

    modele = LogisticRegression(max_iter=5000)

    modele.fit(X_train_scaled, y_train)

    score_honnete = accuracy_score(
        y_test,
        modele.predict(X_test_scaled)
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    modele.fit(X_train, y_train)

    score_triche = accuracy_score(
        y_test,
        modele.predict(X_test)
    )

    delta = score_triche - score_honnete

    print(f"Accuracy honnête : {score_honnete:.2%}")
    print(f"Accuracy triche  : {score_triche:.2%}")
    print(f"Mensonge  : {delta:+.2%}")

    return score_honnete, score_triche, delta


def comparer_fuite_imputation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    modele = LogisticRegression(max_iter=5000)

    modele.fit(X_train_imp, y_train)
    score_honnete = accuracy_score(
        y_test,
        modele.predict(X_test_imp)
    )

    X_imp = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    modele.fit(X_train, y_train)
    score_triche = accuracy_score(
        y_test,
        modele.predict(X_test)
    )

    delta = score_triche - score_honnete

    print(f"Accuracy honnête : {score_honnete:.2%}")
    print(f"Accuracy triche  : {score_triche:.2%}")
    print(f"Mensonge  : {delta:+.2%}")

    return score_honnete, score_triche, delta
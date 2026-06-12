from DataCleaner import DataCleaner
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing_sklearn import construire_preprocesseur

df_telco = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

cleaner = DataCleaner()

y = df_telco["Churn"]
X = df_telco.drop(columns=["Churn"])

X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

X_train_clean = cleaner.fit_transform(X_train)
def test_no_missing_values():
    assert X_train_clean.isna().sum().sum() == 0

def test_same_columns():
    X_test_clean = cleaner.transform(X_test)

    assert list(X_train_clean.columns) == list(X_test_clean.columns)

def test_unknown_category():
    X_train_clean = cleaner.fit_transform(X_train)
    X_test_copy = X_test.copy()
    X_test_copy.loc[
        X_test_copy.index[0],
        "PaymentMethod"
    ] = "CryptoCoin"
    X_test_clean = cleaner.transform(X_test_copy)
    assert (
        list(X_test_clean.columns)
        == list(X_train_clean.columns)
    )



def all_numeric():
    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X_train_clean.dtypes
    )


def test_single_row():
    X_test_one_row = X_test.head(1)

    X_test_one_row_clean = cleaner.transform(X_test_one_row)

    assert X_test_one_row_clean.shape[0] == 1

    assert list(X_test_one_row_clean.columns) == list(X_train_clean.columns)

    assert X_test_one_row_clean.isna().sum().sum() == 0


def test_adversarial_case():
    X_test_adversarial = X_test.copy()

    X_test_adversarial.loc[
        X_test_adversarial.index[0],
        "PaymentMethod"
    ] = "CryptoCoin"

    X_test_adv_clean = cleaner.transform(
        X_test_adversarial
    )

    assert list(X_test_adv_clean.columns) == list(
        X_train_clean.columns
    )

    assert X_test_adv_clean.shape[1] == X_train_clean.shape[1]

    assert X_test_adv_clean.isna().sum().sum() == 0

print("\n =======DataCleaner=======")
print("\n =======Happy Path=======")
test_no_missing_values()
test_same_columns()
test_unknown_category()
all_numeric()
print("\n =======Edge Case=======")
test_single_row()
print("\n =======Adversarial Case=======")
test_adversarial_case()
print("\n✅ Tous les tests DataCleaner sont passés")


print("\n =======Sklearn=======")
preprocesseur = construire_preprocesseur(X_train)
X_train_v2 = preprocesseur.fit_transform(X_train)
X_test_v2 = preprocesseur.transform(X_test)
assert X_train_v2.shape[1] == X_test_v2.shape[1]
print("\n✅ Tous les tests du Sklearn sont passés")

print("\nComparaison DataCleaner vs Sklearn")
print("Colonnes DataCleaner :", X_train_clean.shape[1])
print("Colonnes sklearn :", X_train_v2.shape[1])
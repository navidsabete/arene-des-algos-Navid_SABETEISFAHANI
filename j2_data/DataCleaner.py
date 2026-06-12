import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    def __init__(self):
        self.numeric_cols = None
        self.categorical_cols = None

        self.medians = {}
        self.categories = {}

        self.columns_to_drop = []

        self.scaler = StandardScaler()

        self.final_columns = None

    def fit(self, df):
        data = df.copy()
        if "customerID" in data.columns:
            self.columns_to_drop.append("customerID")

        data = data.drop(
            columns=self.columns_to_drop,
            errors="ignore"
        )

        self.numeric_cols = (
            data.select_dtypes(include=[np.number, np.float64])
            .columns
            .tolist()
        )

        self.categorical_cols = (
            data.select_dtypes(include=["object", "string"])
            .columns
            .tolist()
        )

        for col in self.numeric_cols:
            self.medians[col] = data[col].median()

        for col in self.categorical_cols:
            self.categories[col] = sorted(
                data[col].dropna().unique()
            )
        
        transformed = self._transform_internal(data)

        self.scaler.fit(
            transformed[self.numeric_cols]
        )

        self.final_columns = transformed.columns.tolist()

        return self
    
    
    def _transform_internal(self, data):
        for col, median in self.medians.items():
            if col in data.columns:
                data[col] = data[col].fillna(median)

        encoded_parts = []

        for col in self.categorical_cols:

            for category in self.categories[col]:

                new_col = f"{col}_{category}"

                encoded_parts.append(
                    pd.Series(
                        (data[col] == category).astype(int),
                        name=new_col,
                        index=data.index
                    )
                )

        data = data.drop(
            columns=self.categorical_cols,
            errors="ignore"
        )

        if encoded_parts:
            data = pd.concat(
                [data] + encoded_parts,
                axis=1
            )

        return data
    

    def transform(self, df):
        data = df.copy()

        data = data.drop(
            columns=self.columns_to_drop,
            errors="ignore"
        )

        data = self._transform_internal(data)
        for col in self.final_columns:

            if col not in data.columns:
                data[col] = 0

        data = data[self.final_columns]

        data[self.numeric_cols] = self.scaler.transform(
            data[self.numeric_cols]
        )

        return data
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)
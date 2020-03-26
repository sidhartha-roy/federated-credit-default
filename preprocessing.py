import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fancyimpute import MICE
from config import cfg
from dataloader import DataLoader


class Imputer:
    def __init__(self, data):
        self.data = data
        self.column_titles = list(self.data.df)
        self.categorical_cols = []
        self.data.df_imputed = None

    def transform(self):
        self.extract_cat_columns()
        self.encode()
        self.impute()
        self.save_imputed()
        return self.data.df_imputed

    def extract_cat_columns(self):
        for dtype, col in zip(self.data.df.dtypes, self.column_titles):
            if dtype == object:
                print(col)
                self.data.df[col] = self.data.df[col].astype(str)
                self.categorical_cols.append(col)

    def encode(self):
        le = LabelEncoder()
        self.data.df[self.categorical_cols] \
            = self.data.df[self.categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))

    def impute(self):
        mice = MICE(n_imputations=cfg.CONST.IMPUTATION_ITERS)
        mice_results = mice.complete(np.array(self.data.df))
        self.data.df_imputed = pd.DataFrame(mice_results, columns=self.column_titles)

    def save_imputed(self):
        file_path = os.path.join(cfg.DATASETS.PATH, cfg.DATASETS.IMPUTED_FILENAME)
        self.data.df_imputed.to_csv(file_path, index=False, header=True)

def main():

    data = DataLoader()
    data.load()

    imputer = Imputer(data)
    df = imputer.transform()

if __name__ == "__main__":
    main()
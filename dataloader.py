from config import cfg
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


class DatasetLoader:
    def __init__(self):
        self.dataset_path = cfg.DATASETS.PATH
        self.filename = cfg.DATASETS.FILENAME
        self.df = None
        self.df_X = None
        self.df_Y = None
        self.df_X_train = None
        self.df_Y_train = None
        self.df_X_test = None
        self.df_Y_test = None
        self.torch_X_train = None
        self.torch_Y_train = None
        self.torch_X_test = None
        self.torch_Y_test = None
        self.train = None
        self.test = None
        self.train_loader = None
        self.test_loader = None

    def load(self):
        dataset_path = cfg.DATASETS.PATH
        csv_path = os.path.join(dataset_path, cfg.DATASETS.FILENAME)
        self.df = pd.read_csv(csv_path)

    def load_imputed(self):
        dataset_path = cfg.DATASETS.PATH
        csv_path = os.path.join(dataset_path, cfg.DATASETS.IMPUTED_FILENAME)
        self.df = pd.read_csv(csv_path)

    def train_test_split(self):
        self.df_X = self.df.drop(columns=[cfg.CONST.TARGET_NAME])
        self.df_Y = self.df[[cfg.CONST.TARGET_NAME]]
        stratification_type = cfg.CONST.STRAT_TYPE
        test_size = cfg.CONST.TEST_RATIO
        random_state = cfg.CONST.RANDOM_SEED

        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in split.split(self.df, self.df[stratification_type]):
            self.df_X_train = self.df_X.loc[train_index]
            self.df_Y_train = self.df_Y.loc[train_index]
            self.df_X_test = self.df_X.loc[test_index]
            self.df_Y_test = self.df_Y.loc[test_index]

    def oversample_training_data(self):
        ros = RandomOverSampler()
        self.df_X_train, self.df_Y_train = ros.fit_sample(self.df_X_train, self.df_Y_train)

    def standard_scale(self):
        std_scaler = StandardScaler()
        x_col_titles = list(self.df_X_train)
        self.df_X_train = pd.DataFrame(std_scaler.fit_transform(self.df_X_train), columns=x_col_titles)
        self.df_X_test = pd.DataFrame(std_scaler.fit_transform(self.df_X_test), columns=x_col_titles)

    def to_PyTorch(self):
        torch.manual_seed(cfg.ARGS.SEED)
        self.torch_X_train = torch.from_numpy(self.df_X_train.to_numpy()).float()
        self.torch_Y_train = torch.from_numpy(self.df_Y_train.to_numpy().flatten()).float()
        self.torch_X_test = torch.from_numpy(self.df_X_test.to_numpy()).float()
        self.torch_Y_test = torch.from_numpy(self.df_Y_test.to_numpy().flatten()).float()
        self.train = TensorDataset(self.torch_X_train, self.torch_Y_train)
        self.test = TensorDataset(self.torch_X_test, self.torch_Y_test)
        self.train_loader = DataLoader(self.train, batch_size=cfg.ARGS.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(self.test, batch_size=cfg.ARGS.BATCH_SIZE, shuffle=True)

def main():
    data = DatasetLoader()
    #data.load()
    data.load_imputed()
    data.train_test_split()
    data.oversample_training_data()
    data.standard_scale()
    data.to_PyTorch()
    print("Length of oversampled data", data.df_Y_train.shape)


if __name__ == "__main__":
    main()
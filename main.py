import os
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from contextlib import contextmanager
from time import time
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error

INPUT_DIR = "/Users/potsbo/.go/src/github.com/potsbo/atmacup-15/dataset"
OUTPUT_DIR = "/Users/potsbo/.go/src/github.com/potsbo/atmacup-15/output"

class Timer:
    def __init__(self, logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None, sep=" "):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)


def read_csv(name: str, **kwrgs) -> pd.DataFrame:
    p = os.path.join(INPUT_DIR, name + ".csv")
    return pd.read_csv(p, **kwrgs)


def root_mean_squared_error(y_true, y_pred):
    """mean_squared_error の root (0.5乗)"""
    return mean_squared_error(y_true, y_pred) ** .5


anime_df = read_csv("anime")
train_df = read_csv("train")
test_df = read_csv("test")

def merge_by_anime_id(left_df, right_df):
    return pd.merge(left_df["anime_id"], right_df, on="anime_id", how="left").drop(columns=["anime_id"])


def create_anime_numeric_feature(input_df: pd.DataFrame):
    use_columns = [
        "members", 
        "plan_to_watch",
        "dropped",
        "on_hold",
        "completed",
        "watching",
    ]
    
    return merge_by_anime_id(input_df, anime_df)[use_columns]

def create_anime_type_one_hot_encoding(input_df):
    
    # 対象の列のユニーク集合を取る
    target_colname = "type"
    target_series = anime_df[target_colname]
    unique_values = target_series.unique()

    # ユニークな値ごとに列を作る
    out_df = pd.DataFrame()
    for value in unique_values:
        is_value = target_series == value
        out_df[value] = is_value.astype(int)
        
    out_df["anime_id"] = anime_df["anime_id"]
    
    return merge_by_anime_id(input_df, out_df)

def create_feature(input_df):
    functions = [
        create_anime_numeric_feature,
        create_anime_type_one_hot_encoding,
    ]
    
    out_df = pd.DataFrame()
    for func in functions:
        func_name = str(func.__name__)
        with Timer(prefix=f"create {func_name}"):
            _df = func(input_df)
        out_df = pd.concat([out_df, _df], axis=1)
        
    return out_df

with Timer(prefix="train..."):
    train_feat_df = create_feature(train_df)

with Timer(prefix="test..."):
    test_feat_df = create_feature(test_df)

X = train_feat_df.values
y = train_df["score"].values

fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=510)
cv = fold.split(X, y)
cv = list(cv) # split の返り値は generator なので list 化して何度も iterate できるようにしておく


def fit_lgbm(X, y, cv, params: dict=None, verbose: int=50):
    """lightGBM を CrossValidation の枠組みで学習を行なう function"""

    # パラメータがないときは、空の dict で置き換える
    if params is None:
        params = {}

    models = []
    n_records = len(X)
    # training data の target と同じだけのゼロ配列を用意
    oof_pred = np.zeros((n_records, ), dtype=np.float32)


    for i, (idx_train, idx_valid) in enumerate(cv): 
        # この部分が交差検証のところです。データセットを cv instance によって分割します
        # training data を trian/valid に分割
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMRegressor(**params)

        with Timer(prefix="fit fold={} ".format(i)):
            
            # cv 内で train に定義された x_train で学習する
            clf.fit(x_train, y_train, 
                    eval_set=[(x_valid, y_valid)],  
                    callbacks=[
                        lgbm.early_stopping(stopping_rounds=50, verbose=True),
                        lgbm.log_evaluation(100),
                        ],
                    )

        # cv 内で validation data とされた x_valid で予測をして oof_pred に保存していく
        # oof_pred は全部学習に使わなかったデータの予測結果になる → モデルの予測性能を見る指標として利用できる
        pred_i = clf.predict(x_valid)
        oof_pred[idx_valid] = pred_i
        models.append(clf)
        score = root_mean_squared_error(y_valid, pred_i)
        print(f" - fold{i + 1} - {score:.4f}")

    score = root_mean_squared_error(y, oof_pred)

    print("=" * 50)
    print(f"FINISHI: Whole Score: {score:.4f}")
    return oof_pred, models

params = {
    # 目的関数. これの意味で最小となるようなパラメータを探します. 
    "objective": "rmse", 

    # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    "n_estimators": 10000, 

     # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、
    # がそれだけ木を作るため学習に時間がかかります
    "learning_rate": .1,

    # 特徴重要度計算のロジック(後述)
    "importance_type": "gain", 
    "random_state": 510,
}

oof, models = fit_lgbm(X, y=y, params=params, cv=cv)

print(root_mean_squared_error(y_true=y, y_pred=oof))

# k 個のモデルの予測を作成. shape = (5, N_test,).
pred = np.array([model.predict(test_feat_df.values) for model in models])
# k 個のモデルの予測値の平均 shape = (N_test,).
pred = np.mean(pred, axis=0) # axis=0 なので shape の `k` が潰れる

pd.DataFrame({
    "score": pred
}).to_csv(os.path.join(OUTPUT_DIR, "#1__submission.csv"), index=False)

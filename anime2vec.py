# https://www.guruguru.science/competitions/21/discussions/57f5ea4e-69ad-439d-bbe3-887240cf5cf2/
import random
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold
from gensim.models import word2vec

import time
from contextlib import contextmanager
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import os

SEED = 0

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

def add_w2v_features(train_df, val_df, test_df=None, consider_score=True):
    anime_ids = train_df['anime_id'].unique().tolist()
    user_anime_list_dict = {user_id: anime_ids.tolist() for user_id, anime_ids in train_df.groupby('user_id')['anime_id']}


    title_sentence_list = []
    for _user_id, user_df in train_df.groupby('user_id'):
        user_title_sentence_list = []
        for anime_id, anime_score in user_df[['anime_id', 'score']].values:
            for i in range(anime_score):
                user_title_sentence_list.append(anime_id)
        title_sentence_list.append(user_title_sentence_list)


    # ユーザごとにshuffleしたリストを作成
    shuffled_sentence_list = [random.sample(sentence, len(sentence)) for sentence in title_sentence_list]  ## <= 変更点

    # 元のリストとshuffleしたリストを合わせる
    train_sentence_list = title_sentence_list + shuffled_sentence_list

    # word2vecのパラメータ
    vector_size = 64
    w2v_params = {
        "vector_size": vector_size,  ## <= 変更点
        "seed": SEED,
        "min_count": 1,
        "workers": 1
    }

    # word2vecのモデル学習
    model = word2vec.Word2Vec(train_sentence_list, **w2v_params)

    # ユーザーごとの特徴ベクトルと対応するユーザーID
    user_factors = {user_id: np.mean([model.wv[anime_id] for anime_id in user_anime_list], axis=0) for user_id, user_anime_list in user_anime_list_dict.items()}

    # アイテムごとの特徴ベクトルと対応するアイテムID
    item_factors = {aid: model.wv[aid] for aid in anime_ids}

    # データフレームを作成
    user_factors_df = pd.DataFrame(user_factors).T.reset_index().rename(columns={"index": "user_id"})
    item_factors_df = pd.DataFrame(item_factors).T.reset_index().rename(columns={"index": "anime_id"})

    # データフレームのカラム名をリネーム
    user_factors_df.columns = ["user_id"] + [f"user_factor_{i}" for i in range(vector_size)]
    item_factors_df.columns = ["anime_id"] + [f"item_factor_{i}" for i in range(vector_size)]

    train_df = train_df.merge(user_factors_df, on="user_id", how="left")
    train_df = train_df.merge(item_factors_df, on="anime_id", how="left")

    val_df = val_df.merge(user_factors_df, on="user_id", how="left")
    val_df = val_df.merge(item_factors_df, on="anime_id", how="left")

    if test_df is not None:
        test_df = test_df.merge(user_factors_df, on="user_id", how="left")
        test_df = test_df.merge(item_factors_df, on="anime_id", how="left")
        return train_df, val_df, test_df

    return train_df, val_df

def load_data():
    train_df = pd.read_csv('./dataset/train.csv')
    test_df = pd.read_csv('./dataset/test.csv')
    test_df['score'] = 0 # dummy

    # Initialize submission file
    submission_df = pd.read_csv('./dataset/sample_submission.csv')
    submission_df['score'] = 0
    return train_df, test_df, submission_df

def stratified_and_group_kfold_split(train_df):
    # https://www.guruguru.science/competitions/21/discussions/45ffc8a1-e37c-4b95-aac4-c4e338aa6a9b/

    # 20%のユーザを抽出
    n_user = train_df["user_id"].nunique()
    unseen_users = random.sample(sorted(train_df["user_id"].unique()), k=n_user // 5)
    train_df["unseen_user"] = train_df["user_id"].isin(unseen_users)
    unseen_df = train_df[train_df["unseen_user"]].reset_index(drop=True)
    train_df = train_df[~train_df["unseen_user"]].reset_index(drop=True)

    # train_dfの80%をStratifiedKFoldで分割
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_id, (_, valid_idx) in enumerate(skf.split(train_df, train_df["user_id"])):
        train_df.loc[valid_idx, "fold"] = fold_id

    # 20%をGroupKFoldで分割
    gkf = GroupKFold(n_splits=5)
    unseen_df["fold"] = -1
    for fold_id, (_, valid_idx) in enumerate(gkf.split(unseen_df, unseen_df["user_id"], unseen_df["user_id"])):
        unseen_df.loc[valid_idx, "fold"] = fold_id

    # concat
    train_df = pd.concat([train_df, unseen_df], axis=0).reset_index(drop=True)
    train_df.drop(columns=["unseen_user"], inplace=True)
    return train_df

def train(train_df, original_test_df, submission_df, consider_score=True):
    train_df['oof'] = 0

    for fold in range(5):
        # Prepare the train and validation data
        trn_df = train_df[train_df['fold'] != fold].copy()
        val_df = train_df[train_df['fold'] == fold].copy()

        trn_df, val_df, test_df = add_w2v_features(trn_df, val_df, original_test_df.copy(), consider_score=consider_score)

        # Define the features and the target
        unused_cols = ['user_id', 'anime_id', 'score', 'fold', 'oof']
        feature_cols = [col for col in trn_df.columns if col not in unused_cols]
        target_col = 'score'

        # Prepare the LightGBM datasets
        lgb_train = lgb.Dataset(trn_df[feature_cols], trn_df[target_col])
        lgb_val = lgb.Dataset(val_df[feature_cols], val_df[target_col])

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.1,
            # 'reg_lambda': 1.0
        }

        # Train the model
        callbacks = [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
        model_lgb = lgb.train(params, lgb_train, valid_sets=[
                              lgb_train, lgb_val], callbacks=callbacks, num_boost_round=10000)

        # Predict
        train_preds = model_lgb.predict(trn_df[feature_cols], num_iteration=model_lgb.best_iteration)
        val_preds = model_lgb.predict(val_df[feature_cols], num_iteration=model_lgb.best_iteration)
        test_preds = model_lgb.predict(test_df[feature_cols], num_iteration=model_lgb.best_iteration)

        # Evaluate the model
        train_score = np.sqrt(mean_squared_error(trn_df['score'], train_preds))
        val_score = np.sqrt(mean_squared_error(val_df['score'], val_preds))
        print(f"fold{fold} RMSE: {train_score:.3f}, val RMSE: {val_score:.3f}")

        submission_df['score'] += test_preds / 5

        train_df.loc[train_df['fold'] == fold, 'oof'] = val_preds

    total_score = np.sqrt(mean_squared_error(train_df['score'], train_df['oof']))
    print(f"Total RMSE: {total_score}")

    submission_df.to_csv('./output/submission.csv', index=False)

def main():
    with timer("Load the data"):
        train_df, test_df, submission_df = load_data()

    with timer("Stratified & Group split"):
        train_df = stratified_and_group_kfold_split(train_df)

    with timer("Training and evaluation with LightGBM"):
        train(train_df, test_df, submission_df, consider_score=True)

if __name__ == "__main__":
    main()

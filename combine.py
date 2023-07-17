import pandas as pd

df1 = pd.read_csv("/Users/potsbo/.go/src/github.com/potsbo/atmacup-15/output/#1__submission.csv")
df2 = pd.read_csv("/Users/potsbo/.go/src/github.com/potsbo/atmacup-15/dataset/submission.csv")

# 1列目の値を取得する
column1 = df1.iloc[:, 0]  # df1の1列目
column2 = df2.iloc[:, 0]  # df2の1列目

# 加重平均を計算する
weighted_avg = column1 * 0.2 + column2 * 0.8  # 8:2の加重

# 結果を新しいDataFrameに保存する
df_result = pd.DataFrame(weighted_avg, columns=['score'])

# 結果をcsvファイルに出力する
df_result.to_csv('result.csv', index=False)

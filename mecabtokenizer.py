import pandas as pd
import subprocess

cache = {}

stop_words = set([
    '(', ')', '"', "'", "!", "!!"
    ])

def tokenize(s):
    if s in cache:
        return cache[s]

    result = subprocess.run(f"echo \"{s}\" | mecab", shell=True, stdout=subprocess.PIPE)

    # 実行結果はbytes形式なので、decode関数を使って文字列に変換します。
    lines = result.stdout.decode().split("\n")
    lines.pop()
    lines.pop()
    words = [w.split("\t")[0] for w in lines]
    words = [w for w in words if w not in stop_words]
     
    assert type(words) == list
    assert all(isinstance(e, str) for e in words)

    cache[s] = words
    return words

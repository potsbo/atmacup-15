import pandas as pd
import subprocess

cache = {}

stop_words = set([
    '(', ')', '"', "'", "!", "!!"
    ])

class Word():
    def __init__(self, line):
        l = line.split("\t")
        self.w = l[0]
        ts = l[1].split(",")
        self.hinshi = ts[0]

    def valid(self):
        return self.w not in stop_words and self.hinshi != "助詞"


def _tokenize(row):
    s = row['japanese_name']
    result = subprocess.run(f"echo \"{s}\" | mecab", shell=True, stdout=subprocess.PIPE)

    # 実行結果はbytes形式なので、decode関数を使って文字列に変換します。
    lines = result.stdout.decode().split("\n")
    lines.pop()
    lines.pop()
    words = [Word(l) for l in lines]
    words = [w.w for w in words if w.valid()]
     
    assert type(words) == list
    assert all(isinstance(e, str) for e in words)
    words.append(s)
    return words

def tokenize(s):
    key = s['anime_id']
    if key in cache:
        return cache[key]
    words = _tokenize(s)
    cache[key] = words
    return words


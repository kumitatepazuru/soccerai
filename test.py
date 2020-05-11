import itertools
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def _update(frame, x, y):
    """グラフを更新するための関数"""
    # 現在のグラフを消去する
    plt.cla()
    # データを更新 (追加) する
    x.append(frame)
    y.append(math.sin(frame))
    # 折れ線グラフを再描画する
    plt.plot(x, y)


def main():
    # 描画領域
    fig = plt.figure(figsize=(10, 6))
    # 描画するデータ
    x = []
    y = []

    params = {
        'fig': fig,
        'func': _update,  # グラフを更新する関数
        'fargs': (x, y),  # 関数の引数 (フレーム番号を除く)
        'interval': 0.1,  # 更新間隔 (ミリ秒)
        'frames': itertools.count(0, 0.1),  # フレーム番号を無限に生成するイテレータ
    }
    anime = animation.FuncAnimation(**params)

    # グラフを表示する
    plt.show()


if __name__ == '__main__':
    main()
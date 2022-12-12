from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

class KMedoids(object):

    def __init__(self, n_cluster,
                 max_iter=300):

        self.n_cluster = n_cluster
        self.max_iter = max_iter

    def fit_predict(self, D):

        m, n = D.shape

        initial_medoids = np.random.choice(range(m), self.n_cluster, replace=False)
        tmp_D = D[:, initial_medoids]

        # 初期セントロイドの中で距離が最も近いセントロイドにクラスタリング
        labels = np.argmin(tmp_D, axis=1)

        # 各点に一意のIDが振ってあった方が分類したときに取り扱いやすいため
        # IDをもつようにするデータフレームを作っておく
        results = pd.DataFrame([range(m), labels]).T
        results.columns = ['id', 'label']

        col_names = ['x_' + str(i + 1) for i in range(m)]
        # 各点のIDと距離行列を結びつかせる
        # 距離行列の列に名前をつけて後々処理しやすいようにしている
        results = pd.concat([results, pd.DataFrame(D, columns=col_names)], axis=1)

        before_medoids = initial_medoids
        new_medoids = []

        loop = 0
        # medoidの群に変化があり、ループ回数がmax_iter未満なら続く
        while len(set(before_medoids).intersection(set(new_medoids))) != self.n_cluster and loop < self.max_iter:

            if loop > 0:
                before_medoids = new_medoids.copy()
                new_medoids = []

            # 各クラスタにおいて、クラスタ内の他の点との距離の合計が最小の点を新しいクラスタとしている
            for i in range(self.n_cluster):
                tmp = results.ix[results['label'] == i, :].copy()

                # 各点において他の点との距離の合計を求めている
                tmp['distance'] = np.sum(tmp.ix[:, ['x_' + str(id + 1) for id in tmp['id']]].values, axis=1)
                tmp = tmp.reset_index(drop=True)
                # 上記で求めた距離が最初の点を新たなmedoidとしている
                new_medoids.append(tmp.loc[tmp['distance'].idxmin(), 'id'])

            new_medoids = sorted(new_medoids)
            tmp_D = D[:, new_medoids]

            # 新しいmedoidのなかで距離が最も最小なクラスタを新たに選択
            clustaling_labels = np.argmin(tmp_D, axis=1)
            results['label'] = clustaling_labels

            loop += 1

        # resultsに必要情報を追加
        results = results.ix[:, ['id', 'label']]
        results['flag_medoid'] = 0
        for medoid in new_medoids:
            results.ix[results['id'] == medoid, 'flag_medoid'] = 1
        # 各medoidまでの距離
        tmp_D = pd.DataFrame(tmp_D, columns=['medoid_distance'+str(i) for i in range(self.n_cluster)])
        results = pd.concat([results, tmp_D], axis=1)

        self.results = results
        self.cluster_centers_ = new_medoids

        return results['label'].values

"""
n_cluster = 5

x_data, labels = make_blobs(n_samples=150,
                            n_features=2,
                            centers=n_cluster,
                            cluster_std=1.5,
                            shuffle=True,
                            random_state=0)

# k-meansと同じように初期化ではクラスタ数を指定する
km = KMedoids(n_cluster=n_cluster)

D = squareform(pdist(x_data, metric='euclidean'))
predicted_labels = km.fit_predict(D) # クラスタリング結果
centroids = km.cluster_centers_ # medoidsの番号


print(centroids)

print(predicted_labels)
"""

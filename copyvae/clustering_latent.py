#!/usr/bin/envs python3

import os
import click
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pylab as plt
from collections import Counter


def _cluster_with_kmeans(data_in, k=None, metric='euclidean'):
    """
        data (np.ndarray): data to cluster
        metric (str): braycurtis|canberra|chebyshev|cityblock|correlation|cosine
            |dice|euclidean|hamming|jaccard|jensenshannon|kulsinski|mahalanobis
            |matching|minkowski|rogerstanimoto|russellrao|seuclidean|sokalmichener
            |sokalsneath|sqeuclidean|yule
    """

    from sklearn.cluster import KMeans
    from sklearn import metrics

    cluster_val = []

    data = data_in.copy()
    data[np.isnan(data)] = 3

    if not k:
        for k in range(2, 5, 1):
            clustering = KMeans(n_clusters=k, tol=1e-6, n_init=500).fit(data)
            score = metrics.silhouette_score(
                data, clustering.labels_, metric=metric
            )
            cluster_val.append((score, clustering))

    else:
        clustering = KMeans(n_clusters=k, tol=1e-6, n_init=500).fit(data)
        score = metrics.silhouette_score(
            data, clustering.labels_, metric=metric
        )
        cluster_val.append((score, clustering))

    best_clustering = sorted(cluster_val, key=lambda x: x[0])[-1][-1]

    cell_clusters = sorted(
        [(i, j) for i, j in enumerate(best_clustering.labels_)],
        key=lambda x: x[1]
    )
    return [i[0] for i in cell_clusters], best_clustering.labels_


@click.command(short_help='script for clustering the latent space')
@click.option(
    '-da', '--data', default='', help='copy number data'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(data, output):
    # from julia.api import Julia
    # jl = Julia(compiled_modules=False)
    # from dpmmpython.priors import niw
    # from dpmmpython.dpmmwrapper import DPMMPython
    
    X = np.load(data)
    # prior = niw(1, np.zeros(10), 20, np.eye(10))
    
    # labels, clusters ,sub_labels= DPMMPython.fit(
    #     X.T, 10000, prior = prior, verbose=False)
    aa, labels = _cluster_with_kmeans(X)
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
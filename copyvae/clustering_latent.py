#!/usr/bin/envs python3

import os
import click
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats
import matplotlib.pylab as plt
from collections import Counter
from matplotlib.patches import Ellipse

from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import v_measure_score


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
        for k in range(2, 10, 1):
            clustering = KMeans(n_clusters=k, n_init=100).fit(data)
            score = metrics.silhouette_score(
                data, clustering.labels_, metric=metric
            )
            cluster_val.append((score, clustering))

    else:
        clustering = KMeans(n_clusters=k, n_init=100).fit(data)
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


def draw_umap(adata, feature, file_name):

    sc.pp.neighbors(adata, use_rep=feature)
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(
                adata,
                color=["cell_type"],
                frameon=False,
                save='{}.png'.format(file_name)
                )

    return None
    

@click.command(short_help='script for clustering the latent space')
@click.option(
    '-da', '--data', default='', help='copy number data'
)
@click.option(
    '-dk', '--data_copykat', default='', help='Results form copykat'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(data, data_copykat, output):
    # from julia.api import Julia
    # jl = Julia(compiled_modules=False)
    # from dpmmpython.priors import niw
    # from dpmmpython.dpmmwrapper import DPMMPython
    # labels, clusters ,sub_labels= DPMMPython.fit(
    #     X.T, 10000, prior = prior, verbose=False)
    
    X = np.load(data)
    normal = np.ones(379)
    tumor = np.zeros(1480 - 379)
    gt = np.concatenate([normal, tumor])

    tumor_data = X[379:]

    if data_copykat:
        copykat = pd.read_csv(data_copykat, sep='\t')
        copykat_pred = copykat.loc['copykat.pred'].values
        cluster_pred = copykat.loc['cluster.pred'].values

        acc_ck = accuracy_score (copykat_pred, cluster_pred)
        v_mes_ck = v_measure_score(copykat_pred, cluster_pred)
        # import pdb;pdb.set_trace()

    # prior = niw(1, np.zeros(10), 20, np.eye(10))
    
    
    # aa, labels = _cluster_with_kmeans(X)
    # aa_2, labels_2 = _cluster_with_kmeans(X, k=2)


    # gaussian mixture clustering
    # define the model
    model = GaussianMixture(n_components=2, n_init=5, max_iter=100000)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    acc_vae = accuracy_score (yhat, gt)
    v_mes_vae = v_measure_score(yhat, gt)
    print(acc_ck, v_mes_ck)
    print(acc_vae, v_mes_vae)

    import pdb;pdb.set_trace()  

    tumor_cells = X[np.argwhere(yhat == 0).flatten()]
    normal_cells = X[np.argwhere(yhat == 1).flatten()]
    print(tumor_cells.shape, normal_cells.shape)
    
    n_components = np.arange(1,10)
    models = [GaussianMixture(n, covariance_type='full', random_state=0, 
        n_init=5, max_iter=1000).fit(tumor_cells) for n in n_components]

    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    plt.show()
    import pdb;pdb.set_trace()

    models_normal = [GaussianMixture(n, covariance_type='full', random_state=0, 
        n_init=5, max_iter=1000).fit(normal_cells) for n in n_components]

    plt.plot(n_components, [m.bic(X) for m in models_normal], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    plt.show()

if __name__ == '__main__':
    main()
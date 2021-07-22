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
from matplotlib.ticker import FormatStrFormatter

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


def get_GMM(X, gt):
    # define the model
    model = GaussianMixture(n_components=2, n_init=5, max_iter=100000)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)

    acc_vae = accuracy_score (yhat, gt)
    v_mes_vae = v_measure_score(yhat, gt)

    return model, yhat, acc_vae, v_mes_vae



def plot_BIC(cluster, output, label):
    n_components = np.arange(1,10)
    models = [GaussianMixture(n, covariance_type='full', random_state=0, 
        n_init=5, max_iter=1000).fit(cluster) for n in n_components]

    plt.plot(n_components, [m.bic(cluster) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'cluster_{}.pdf'.format(label)))
    plt.close()


def plot_accuracies(df, output):
    import pdb;pdb.set_trace()
    fig, (ax, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(5, 5), facecolor='white', 
        gridspec_kw={'height_ratios':[7,1]})

    ax.set_ylim(.65, 1.) 
    ax2.set_ylim(0, .12)

    sns.barplot(
        x=1, y=0, hue=2, data=df, ax=ax, palette=['#08519c', '#f03b20']
    )
    sns.barplot(
        x=1, y=0, hue=2, data=df, ax=ax2, palette=['#08519c', '#f03b20']
    )

    custom_lines = []
    for el in [('copyVAE', '#08519c'), ('copykat', '#f03b20')]:
        custom_lines.append(
                plt.plot([],[], marker="o", ms=7, ls="", mec='black', 
                mew=0, color=el[1], label=el[0])[0] 
            )

    ax.set_ylabel("Performance", fontsize=12)
    ax2.set_ylabel("", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    ax2.set_xlabel("", fontsize=12)
    # plt.xticks(rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax.tick_params(bottom = False)
    ax.tick_params(labelbottom = False)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax.get_xaxis().set_visible(False)

    ax.legend(
        bbox_to_anchor=(0., 1.2, 1., .102),
        handles=custom_lines, loc='upper center', 
        facecolor='white', ncol=2, fontsize=8, frameon=False
    )

    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  

    ax2.get_legend().remove()

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'cluster_accuracy_comparisons.pdf'))
    plt.close()


@click.command(short_help='script for clustering the latent space')
@click.option(
    '-da', '--data', default='', help='copy number data'
)
@click.option(
    '-dk', '--data_copykat', default='', help='Results form copykat'
)
@click.option(
    '-ed', '--expression_data', default='', help='input data to the VAE'
)
@click.option(
    '-np', '--normals_path', default='', help='input normals from copykat idea'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(data, data_copykat, expression_data, normals_path, output):

    X = np.load(data)
    normal = np.ones(379)
    tumor = np.zeros(1480 - 379)
    gt = np.concatenate([normal, tumor])


    if data_copykat:
        copykat = pd.read_csv(data_copykat, sep='\t')
        copykat_pred = copykat.loc['copykat.pred'].values
        cluster_pred = copykat.loc['cluster.pred'].values

        acc_ck = accuracy_score (copykat_pred, cluster_pred)
        v_mes_ck = v_measure_score(copykat_pred, cluster_pred)
    

    model, yhat, acc_vae, v_mes_vae = get_GMM(X, gt)
    print(acc_ck, v_mes_ck)
    print(acc_vae, v_mes_vae)

    cluster_0 = X[np.argwhere(yhat == 0).flatten()]
    cluster_1 = X[np.argwhere(yhat == 1).flatten()]
    print(cluster_0.shape, cluster_1.shape)

    plot_BIC(cluster_0, output, '0')
    plot_BIC(cluster_1, output, '1')
    
    exp_data = pd.read_csv(expression_data, sep='\t')

    exp_clean = exp_data.drop(['Unnamed: 0', 'cluster.pred'], axis=1).values
    exp_0 = exp_clean[np.argwhere(yhat == 0).flatten()]
    exp_1 = exp_clean[np.argwhere(yhat == 1).flatten()]


    if exp_1.mean(axis=1).std() > exp_0.mean(axis=1).std():
        tum_clust = cluster_1; tum_exp = exp_1
        nor_clust = cluster_0; nor_exp = exp_0
    else:
        tum_clust = cluster_0; tum_exp = exp_0
        nor_clust = cluster_1; nor_exp = exp_1
    
    df_to_plot = pd.DataFrame(
        [[acc_vae, v_mes_vae, acc_ck, v_mes_ck], 
        ['Accuracy', 'V-measure', 'Accuracy', 'V-measure'], 
        ['copyVAE', 'copyVAE', 'copykat', 'copykat']]).T

    plot_accuracies(df_to_plot, output)

    normal_ck = pd.read_csv(normals_path, sep='\t')
    exp_data.reset_index(inplace=True)
    exp_data.set_index('Unnamed: 0', inplace=True)
    normal_cells = exp_data.loc[normal_ck['x'].tolist()]
    clust_and_norms = dict(Counter(yhat[normal_cells['index'].tolist()]))

    import operator
    print('Cluster {} contains the normal cells'.format(
        max(clust_and_norms.items(), key=operator.itemgetter(1))[0]
    ))
    # import pdb;pdb.set_trace()

    



if __name__ == '__main__':
    main()
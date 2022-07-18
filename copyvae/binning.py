#! /usr/bin/env python3

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from os import path
from copyvae.preprocess import annotate_data, build_gene_map


def bin_genes_from_text(umi_counts, bin_size, gene_metadata):
    """ Gene binning for UMI counts from text file

    Args:
        umi_counts: text file
        bin_size: number of genes per bin (int)
        gene_metadata: gene name map

    Returns:
        data: (anndata object) high expressed gene counts,
                        number of genes fully divided by bin_size
        chrom_list: list of chromosome boundry bins
    """

    umis = pd.read_csv(umi_counts, sep='\t', low_memory=False).reset_index()
    labels = umis.iloc[1, :]

    gene_map = build_gene_map(gene_metadata)

    # remove cell cycle genes
    umis = umis[~umis['index'].str.contains("HLA")]
    # gene name mapping
    sorted_gene = pd.merge(
        umis, gene_map, right_on=['Gene name'],
        left_on=['index'], how='inner'
    )

    # remove non-expressed genes
    ch = sorted_gene.iloc[:, 1:-7].astype('int32')
    nonz = np.count_nonzero(ch, axis=1)
    sorted_gene['expressed'] = nonz
    expressed_gene = sorted_gene[sorted_gene['expressed'] > 0]

    # bin and remove exceeded genes
    n_exceeded = expressed_gene.chr.value_counts() % bin_size
    for chrom in n_exceeded.index:
        n = n_exceeded[chrom]
        ind = expressed_gene[
            expressed_gene['chr'] == chrom
        ].sort_values(by=['expressed', 'Gene start (bp)']
                      )[:n].index
        expressed_gene = expressed_gene.drop(index=ind)

    abs_pos = expressed_gene['abspos'].values
    with open('abs.npy', 'wb') as f:
        np.save(f, abs_pos)

    # extract chromosome boundary
    bin_number = expressed_gene.chr.value_counts() // bin_size
    chrom_bound = bin_number.sort_index().cumsum()
    chrom_list = [(0, chrom_bound[1])]
    for i in range(2, 24):
        start_p = chrom_bound[i - 1]
        end_p = chrom_bound[i]
        chrom_list.append((start_p, end_p))

    # clean and add labels
    expressed_gene.drop(columns=expressed_gene.columns[-8:],
                        axis=1, inplace=True)
    expressed_gene = pd.concat([expressed_gene.T, labels], axis=1)
    expressed_gene.rename(columns=expressed_gene.iloc[0], inplace=True)
    expressed_gene.drop(expressed_gene.index[0], inplace=True)
    expressed_gene = expressed_gene.sort_values(by='cluster.pred')
    expressed_gene.to_csv('bined_expressed_cell.csv', sep='\t')
    data = annotate_data(expressed_gene, abs_pos)

    return data, chrom_list


def bin_genes_from_anndata(file, bin_size, gene_metadata):
    """ Gene binning for UMI counts for 10X data

    Args:
        file: h5 file
        bin_size: number of genes per bin (int)
        gene_metadata: gene name map

    Returns:
        data: (anndata object) high expressed gene counts,
                        number of genes fully divided by bin_size
        chrom_list: list of chromosome boundry bins
    """

    #adata = sc.read_10x_h5(file)
    adata = anndata.read_h5ad(file)
    gene_map = build_gene_map(gene_metadata)

    # normalize UMI counts
    #sc.pp.filter_cells(adata, min_genes=1000)
    sc.pp.normalize_total(adata, inplace=True)
    adata.X = np.round(adata.X)

    # extract genes
    gene_df = pd.merge(
        adata.var,
        gene_map,
        right_on=['Gene stable ID'],
        left_on=['gene_ids'],
        how='right').dropna()
    adata_clean = adata[:, adata.var.gene_ids.isin(gene_df.gene_ids.values)]

    # add position in genome
    adata_clean.var['chr'] = gene_df['chr'].values
    adata_clean.var['abspos'] = gene_df['abspos'].values

    # filter out low expressed genes
    sc.pp.filter_genes(adata_clean, min_cells=1)
    # remove exceeded genes
    n_exceeded = adata_clean.var['chr'].value_counts() % bin_size
    ind_list = []
    for chrom in n_exceeded.index:
        n = n_exceeded[chrom]
        ind = adata_clean.var[adata_clean.var['chr'] == chrom].sort_values(
                                by=['n_cells', 'abspos'])[:n].index.values
        ind_list.append(ind)
    data = adata_clean[:, ~adata_clean.var.index.isin(
                            np.concatenate(ind_list))]
    with open('abs.npy', 'wb') as f:
        np.save(f, data.var['abspos'].values)

    # find chromosome boundry bins
    bin_number = adata_clean.var['chr'].value_counts() // bin_size
    chrom_bound = bin_number.sort_index().cumsum()
    chrom_list = [(0, chrom_bound[1])]
    for i in range(2, 24):
        start_p = chrom_bound[i - 1]
        end_p = chrom_bound[i]
        chrom_list.append((start_p, end_p))

    return data, chrom_list

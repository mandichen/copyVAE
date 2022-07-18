#! /usr/bin/env python3

import csv
import pandas as pd
import numpy as np
from os import path
import logging
import anndata
import pybiomart

logger = logging.getLogger(__name__)

LOCAL_DIRECTORY = path.dirname(path.abspath(__file__))
GENE_META = path.join(LOCAL_DIRECTORY, '../data/mart_export.txt')

CHR_BASE_PAIRS = np.array([
    248956422,
    242193529,
    198295559,
    190214555,
    181538259,
    170805979,
    159345973,
    145138636,
    138394717,
    133797422,
    135086622,
    133275309,
    114364328,
    107043718,
    101991189,
    90338345,
    83257441,
    80373285,
    58617616,
    64444167,
    46709983,
    50818468,
    156040895])


def build_gene_map2(gene_metadata=GENE_META, chr_pos=CHR_BASE_PAIRS):
    """ Build gene map from meta data

    """
    gene_info = pd.read_csv(gene_metadata, sep='\t')
    gene_info.rename(columns={'Chromosome/scaffold name': 'chr'}, inplace=True)

    gene_info.loc[gene_info.chr == 'X', 'chr'] = '23'
    gene_info = gene_info[
        gene_info['chr'].isin(
            np.arange(1, 24).astype(str)
        )
    ]
    gene_info = gene_info.copy()
    gene_info.loc[:, 'chr'] = gene_info.chr.astype(int)
    gene_info = gene_info.sort_values(by=['chr', 'Gene start (bp)'])

    gene_map = gene_info[gene_info['Gene type']=='protein_coding'].copy()
    #gene_map = gene_info
    gene_map['abspos'] = gene_map['Gene start (bp)']
    for i in range(len(chr_pos)):
        gene_map.loc[gene_map['chr'] == i + 1, 'abspos'] += chr_pos[:i].sum()

    return gene_map


def build_gene_map(chr_pos=CHR_BASE_PAIRS):
    """ Build gene map from meta data

    """
    server = pybiomart.Server(host='http://www.ensembl.org')
    mart = server['ENSEMBL_MART_ENSEMBL']
    dataset = mart['hsapiens_gene_ensembl']
    attributes = ['external_gene_name', 
                  'ensembl_gene_id',
                  'chromosome_name',
                  'start_position',
                  'end_position']
    
    gene_info = dataset.query(attributes= attributes)
    gene_info.rename(columns={'Chromosome/scaffold name': 'chr'}, inplace=True)

    gene_info.loc[gene_info.chr == 'X', 'chr'] = '23'
    gene_info = gene_info[
        gene_info['chr'].isin(
            np.arange(1, 24).astype(str)
        )
    ]
    gene_info = gene_info.copy()
    gene_info.loc[:, 'chr'] = gene_info.chr.astype(int)
    gene_map = gene_info.sort_values(by=['chr', 'Gene start (bp)'])

    gene_map['abspos'] = gene_map['Gene start (bp)']
    for i in range(len(chr_pos)):
        gene_map.loc[gene_map['chr'] == i + 1, 'abspos'] += chr_pos[:i].sum()

    return gene_map


def load_cortex_txt(path_to_file: str) -> anndata.AnnData:
    logger.info("Loading Cortex data from {}".format(path_to_file))
    rows = []
    gene_names = []
    with open(path_to_file, "r") as csvfile:
        data_reader = csv.reader(csvfile, delimiter="\t")
        for i, row in enumerate(data_reader):
            if i == 1:
                precise_clusters = np.asarray(row, dtype=str)[2:]
            if i == 8:
                clusters = np.asarray(row, dtype=str)[2:]
            if i >= 11:
                rows.append(row[1:])
                gene_names.append(row[0])
    cell_types, labels = np.unique(clusters, return_inverse=True)
    _, precise_labels = np.unique(precise_clusters, return_inverse=True)
    data = np.asarray(rows, dtype=int).T[1:]
    gene_names = np.asarray(gene_names, dtype=str)
    gene_indices = []
    extra_gene_indices = []
    gene_indices = np.concatenate(
                                [gene_indices, 
                                extra_gene_indices]).astype(np.int32)
    if gene_indices.size == 0:
        gene_indices = slice(None)

    data = data[:, gene_indices]
    gene_names = gene_names[gene_indices]
    data_df = pd.DataFrame(data, columns=gene_names)
    adata = anndata.AnnData(X=data_df)
    adata.obs["labels"] = labels
    adata.obs["precise_labels"] = precise_clusters
    adata.obs["cell_type"] = clusters
    logger.info("Finished loading Cortex data")
    return adata


def load_copykat_data(file):

    umis = pd.read_csv(file, sep='\t', low_memory=False).reset_index()
    labels = umis.iloc[1, :]
    gene_map = build_gene_map()

    sorted_gene = pd.merge(
        umis, gene_map, right_on=['Gene name'],
        left_on=['index'], how='inner')

    # remove non-expressed genes
    ch = sorted_gene.iloc[:, 1:-7].astype('int32')
    nonz = np.count_nonzero(ch, axis=1)
    sorted_gene['expressed'] = nonz
    expressed_gene = sorted_gene[sorted_gene['expressed'] > 0]
    abs_pos = expressed_gene['abspos'].values

    expressed_gene.drop(columns=expressed_gene.columns[-8:],
                        axis=1, inplace=True)
    expressed_gene = pd.concat([expressed_gene.T, labels], axis=1)
    expressed_gene.rename(columns=expressed_gene.iloc[0], inplace=True)
    expressed_gene.drop(expressed_gene.index[0], inplace=True)
    data = annotate_data(expressed_gene, abs_pos)
    
    return data


def load_data(file):

    X = pd.read_csv(file, sep='\t', low_memory=False, index_col=0)
    clusters = np.asarray(X['cluster.pred'], dtype=str)
    cell_types, labels = np.unique(clusters, return_inverse=True)
    data_df = X.iloc[:,:-1]
    adata = anndata.AnnData(X=data_df)
    adata.obs["labels"] = labels
    adata.obs["cell_type"] = clusters

    return adata


def annotate_data(data, abs_pos):
    """ Pack data into anndata class

    Args:
        data: pandas DataFrame
    Returns:
        adata: anndata class
    """

    clusters = np.asarray(data['cluster.pred'], dtype=str)
    cell_types, labels = np.unique(clusters, return_inverse=True)
    data_df = data.iloc[:,:-1].astype(int)
    # normalization
    total_counts = data_df.sum(axis=1)
    m_total = total_counts.median(axis=0)
    data_arr = data_df.values
    new_arr = (data_arr.T/total_counts.values * m_total).T
    adata = anndata.AnnData(X=np.round(new_arr))
    adata.var['name'] = data.columns.values[:-1]
    adata.var['abspos'] = abs_pos
    adata.obs["labels"] = labels
    adata.obs["cell_type"] = clusters
    adata.obs['barcode'] = data.index.values

    return adata

### example
"""
from scvi.data._anndata import setup_anndata

data_path_scvi = 'scvi_data/'
data_path_kat = 'copykat_data/txt_files/'
adata = load_cortex_txt(data_path_scvi + 'expression_mRNA_17-Aug-2014.txt')
# copyKAT DCIS1
data = load_copykat_data(data_path_kat + 'GSM4476485_combined_UMIcount_CellTypes_DCIS1.txt')
setup_anndata(data, labels_key="labels")
"""

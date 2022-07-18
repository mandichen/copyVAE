#! /usr/bin/env python3

import argparse
import numpy as np
import tensorflow as tf
import scanpy as sc
import anndata
import pandas as pd

from copyvae.preprocess import load_copykat_data, build_gene_map
from copyvae.binning import bin_genes_from_text, bin_genes_from_anndata
from copyvae.vae2 import VAE, CopyVAE, train_vae, train_cpvae
from copyvae.clustering import find_clones_gmm, find_clones_dbscan, find_clones_kmeans
from copyvae.segmentation import bin_to_segment
from copyvae.cell_tools import Clone
from copyvae.graphics import draw_umap, draw_heatmap, plot_breakpoints


def annotate_data(data, abs_pos):
    """ Pack data into anndata class

    Args:
        data: pandas DataFrame
    Returns:
        adata: anndata class
    """


    data_df = data.iloc[:,:].astype(int)
    # normalization
    total_counts = data_df.sum(axis=1)
    m_total = total_counts.median(axis=0)
    data_arr = data_df.values
    new_arr = (data_arr.T/total_counts.values * m_total).T
    adata = anndata.AnnData(X=np.round(new_arr))
    adata.var['name'] = data.columns.values
    adata.var['abspos'] = abs_pos
    adata.obs['barcode'] = data.index.values

    return adata


def load_gliomas_data(umi_counts):

    umis = pd.read_csv(umi_counts, sep='\t', index_col=0, low_memory=False).reset_index()
    gene_map = build_gene_map()

    sorted_gene = pd.merge(
            umis, gene_map, right_on=['Gene name'],
            left_on=['index'], how='inner')
    sorted_gene = sorted_gene.sort_values(by=['abspos'])

    ch = sorted_gene.iloc[:, 1:-8].astype('int32')
    nonz = np.count_nonzero(ch, axis=1)
    sorted_gene['expressed'] = nonz
    expressed_gene = sorted_gene[sorted_gene['expressed'] > 0]
    abs_pos = expressed_gene['abspos'].values

    expressed_gene.drop(columns=expressed_gene.columns[-8:],
                            axis=1, inplace=True)
    expressed_gene = pd.concat([expressed_gene.T, None], axis=1)
    expressed_gene.rename(columns=expressed_gene.iloc[0], inplace=True)
    expressed_gene.drop(expressed_gene.index[0], inplace=True)

    return expressed_gene, abs_pos


def bin_genes_from_text(umi_counts, bin_size):
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

    umis = pd.read_csv(umi_counts,
                        sep='\t',
                        index_col=0,
                        low_memory=False).reset_index()

    gene_map = build_gene_map()

    # remove cell cycle genes
    umis = umis[~umis['index'].str.contains("HLA")]
    # gene name mapping
    sorted_gene = pd.merge(
        umis, gene_map, right_on=['Gene name'],
        left_on=['index'], how='inner'
    )
    sorted_gene = sorted_gene.sort_values(by=['abspos'])
    # remove non-expressed genes
    ch = sorted_gene.iloc[:, 1:-8].astype('int32')
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
    expressed_gene = pd.concat([expressed_gene.T, None], axis=1)
    expressed_gene.rename(columns=expressed_gene.iloc[0], inplace=True)
    expressed_gene.drop(expressed_gene.index[0], inplace=True)
 
    data = annotate_data(expressed_gene, abs_pos)

    return data, chrom_list


def run_pipeline(umi_counts, is_anndata):
    """ Main pipeline

    Args:
        umi_counts: umi file
        is_anndata: set to True when using 10X data
    Params:
        max_cp: maximum copy number
        bin_size: number of genes per bin
        intermediate_dim: number of intermediate dimensions for vae
        latent_dim: number of latent dimensions for vae
        batch_size: batch size for training
        epochs = number of epochs training
    """

    bin_size = 25
    max_cp = 15

    expressed_gene, abs_pos = load_gliomas_data(umi_counts)
    data = annotate_data(expressed_gene, abs_pos)
    sc.pp.filter_genes(data, min_cells=3000)
    #sc.pp.highly_variable_genes(data,n_top_genes=5000, flavor='seurat_v3', subset=True)

    intermediate_dim = 128
    latent_dim = 10
    batch_size = 128
    epochs = 300
    x_train = data.X
    clus_model = VAE(x_train.shape[-1],
                            intermediate_dim,
                            latent_dim)
    clus_vae = train_vae(clus_model, x_train, batch_size, epochs)

    z_mean, _, z = clus_vae.encoder.predict(data.X)
    #pred_label = find_clones_gmm(z, n_clones=5)
    pred_label = find_clones_kmeans(z, n_clones=10)
    adata, chroms = bin_genes_from_text(umi_counts, bin_size)

    min_std = 10000
    for lab in np.unique(pred_label):
        i_std = adata.X[pred_label==lab].mean(axis=1).std()
        if i_std < min_std:
            min_std = i_std
            nor_label = lab
    
    confident_norm_x = adata.X[pred_label==nor_label]
    baseline = np.median(confident_norm_x, axis=0)
    with open('base.npy', 'wb') as f:
        np.save(f, baseline)
    
    baseline[baseline == 0] = 1
    norm_x = adata.X / baseline * 2
    norm_x[pred_label==nor_label] = 2.


    intermediate_dim = 128
    latent_dim = 10
    batch_size = 128
    epochs = 300
    x_train = norm_x
    mymodel = CopyVAE(x_train[0].shape[-1],
                                intermediate_dim,
                                latent_dim)

    copy_vae = train_cpvae(mymodel, (x_train, z), batch_size, epochs)

    rho_z, cp = copy_vae.encoder(norm_x)
    np.save('z.npy', z)
    np.save('cp.npy',cp)
    np.save('label.npy', pred_label)
    print(cp)

    return

    # assign genes to bins
    if is_anndata:
        adata, chroms = bin_genes_from_anndata(umi_counts, bin_size)
        # TODO add cell_type for 10X data
        adata.obs["cell_type"] = 1
        x_train = adata.X.todense()
        
    else:
        adata, chroms = bin_genes_from_text(umi_counts, bin_size)
        x_train = adata.X

    #bin_numb = x_train.shape[-1]//25
    #model = CopyVAE(bin_numb, bin_size)
    #copy_vae = train_vae(model, x_train, batch_size, epochs)
    #copy_vae.save("models/")
    #z_mean, z_var, z = copy_vae.encoder.predict(x_train)
    #bin_cn = z
    #gene_cn = tf.repeat(z, repeats=bin_size, axis=1)

   
    # train model
    # TODO remove try except after debugging vae training
    #for attempt in range(3):
    #    try:
    model = CopyVAE(x_train.shape[-1],
                            intermediate_dim,
                            latent_dim,
                            bin_size=bin_size,
                            max_cp=max_cp)
    copy_vae = train_vae(model, x_train, batch_size, epochs)
    #    except BaseException:
    #        tf.keras.backend.clear_session()
    #        continue
    #    else:
    #        break
 

    # get copy number and latent output
    """
    base = tf.ones_like(x_train)[0] * 2
    baseline = model.decoder.k_layer(base)
    with open('baseline.npy', 'wb') as f:
        np.save(f, baseline)

    z_mean, _, z = copy_vae.encoder.predict(x_train)
    reconstruction, gene_cn, _ = copy_vae.decoder(z)

    recon = - zinb_pos(x_train, reconstruction)
    #recon = - nb_pos(x_train[379:], reconstruction)
    print("infer:")
    print(np.sum(recon))

    gtcp = np.load('gtcp.npy')
    mu = model.decoder.k_layer(gtcp)
    reconstruction[0] = mu

    recon = - zinb_pos(x_train, reconstruction)
    #recon = - nb_pos(x_train[379:], reconstruction)
    print("gound truth:")
    print(np.sum(recon))

    """
    """
    # split into batch to avoid OOM
    input_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    input_dataset = input_dataset.batch(batch_size)
    for step, x in enumerate(input_dataset):
        l = tf.expand_dims(
                            tf.math.log(
                                        tf.math.reduce_sum(x, axis=1)
                                    ), 
                        axis=1)
        if step == 0:
            z_mean, _, z = copy_vae.encoder.predict(x)
            reconstruction, gene_cn, _ = copy_vae.decoder([z,l])
        else:
            z_mean_old = z_mean
            gene_cn_old = gene_cn
            z_mean, _, z = copy_vae.encoder.predict(x)
            reconstruction, gene_cn, _ = copy_vae.decoder([z,l])
            z_mean = tf.concat([z_mean_old, z_mean], 0)
            gene_cn = tf.concat([gene_cn_old, gene_cn], 0)

    adata.obsm['latent'] = z_mean
    draw_umap(adata, 'latent', '_latent')
    adata.obsm['copy_number'] = gene_cn
    draw_umap(adata, 'copy_number', '_copy_number')
    #draw_heatmap(gene_cn,'gene_copies')
    with open('copy.npy', 'wb') as f:
        np.save(f, gene_cn)

    # compute bin copy number
    gn = x_train.shape[1]
    bin_number = gn // bin_size
    tmp_arr = np.split(gene_cn, bin_number, axis=1)
    tmp_arr = np.stack(tmp_arr, axis=1)
    bin_cn = np.median(tmp_arr, axis=2)
    draw_heatmap(bin_cn,'bin_copies')
    with open('median_cp.npy', 'wb') as f:
        np.save(f, bin_cn)
    """
    if is_anndata:
        pred_labels =  find_clones_gmm(z_mean)#find_clones_dbscan(z_mean, min_members=10)
        clone_ids = np.unique(pred_labels)
        clone_seg_arr = []
        for id in clone_ids:
            mask = pred_labels==id
            cells = bin_cn[mask]
            clone_size = np.shape(cells)[0]
            clone = Clone(id,
                            clone_size,
                            bin_size,
                            cell_gene_cn=gene_cn[mask],
                            cell_bin_cn=bin_cn[mask],
                            chrom_bound=chroms
                            )
            clone.call_breakpoints()
            clone.generate_profile()
            clone_seg = clone.segment_cn
            clone_gene_cn = np.repeat(clone_seg, bin_size)
            with open('clone'+str(id)+'_cn.npy', 'wb') as f:
                np.save(f, clone_gene_cn)
            seg_profile = bin_to_segment(clone.cell_bin_cn, clone.breakpoints)
            #draw_heatmap(seg_profile, "seg_clone"+str(id))
            clone_seg_arr.append(seg_profile)
        all_cell_profile = np.vstack(clone_seg_arr)
        draw_heatmap(all_cell_profile, "all_cell_profile")
    return

    
    # seperate tumour cells from normal
    tumour_mask = find_clones_gmm(z_mean, x_train, 2)
    cells = bin_cn[tumour_mask]

    clone_size = np.shape(cells)[0]
    t_clone = Clone(1,
                    clone_size,
                    bin_size,
                    cell_gene_cn=gene_cn[tumour_mask],
                    cell_bin_cn=bin_cn[tumour_mask],
                    chrom_bound=chroms
                    )
    n_cells = bin_cn[~tumour_mask]
    n_clone_size = np.shape(n_cells)[0]
    n_clone = Clone(0,
                    n_clone_size,
                    bin_size,
                    cell_gene_cn=gene_cn[~tumour_mask],
                    cell_bin_cn=bin_cn[~tumour_mask],
                    chrom_bound=chroms
                    )
    n_clone.call_breakpoints()
    n_clone.generate_profile()
    clone_seg = n_clone.segment_cn
    clone_gene_cn = np.repeat(clone_seg, bin_size)
    with open('nclone_gene_cn.npy', 'wb') as f:
        np.save(f, clone_gene_cn)

    # call clone breakpoints
    t_clone.call_breakpoints()
    #print(t_clone.breakpoints)
    #cp_arr = np.mean(cells, axis=0)
    #plot_breakpoints(cp_arr, t_clone.breakpoints, 'bp_plot')

    # generate clone profile
    t_clone.generate_profile()
    clone_seg = t_clone.segment_cn
    #print(clone_seg)
    clone_gene_cn = np.repeat(clone_seg, bin_size)
    with open('clone_gene_cn.npy', 'wb') as f:
        np.save(f, clone_gene_cn)

    # generate consensus segment profile
    bp_arr = t_clone.breakpoints
    seg_profile = bin_to_segment(bin_cn, bp_arr)
    with open('segments.npy', 'wb') as f:
        np.save(f, seg_profile)
    draw_heatmap(seg_profile, "tumour_seg")
    return None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="input UMI")
    parser.add_argument('-a',
                        nargs='?',
                        const=True,
                        default=False,
                        help="flag for 10X data")
    parser.add_argument('-g', '--gpu', type=int, help="GPU id")

    args = parser.parse_args()
    file = args.input
    is_10x = args.a

    if args.gpu:
        dvc = '/device:GPU:{}'.format(args.gpu)
    else:
        dvc = '/device:GPU:0'

    with tf.device(dvc):
        run_pipeline(file, is_anndata=is_10x)


if __name__ == "__main__":
    main()

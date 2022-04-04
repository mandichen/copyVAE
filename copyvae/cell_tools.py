#! /usr/bin/env python3

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import List
from copyvae.segmentation import pelt_multi


@dataclass
class Clone:
    """ Class for cell clones """
    id: int
    clone_size: int
    bin_size: int
    cell_gene_cn: np.ndarray = np.array([])
    cell_bin_cn: np.ndarray = np.array([])
    chrom_bound: List[int] = field(default_factory=list)
    breakpoints: np.ndarray = np.array([])
    segment_cn: np.ndarray = np.array([])


    def call_breakpoints(self):
        """ Detect clone breakpoints """

        k = self.clone_size
        n = np.shape(self.cell_bin_cn)[1]
        beta = 0.5 * k * np.log(n)
        bp_arr = np.array([])

        for tup in self.chrom_bound:
            start_bin = tup[0]
            end_bin = tup[1]
            chrom_bps = pelt_multi(
                                    self.cell_bin_cn[:, start_bin:end_bin], 
                                    beta
                                    )
            bps = chrom_bps + start_bin
            bp_arr = np.concatenate((bp_arr, bps))

        self.breakpoints = bp_arr.astype(int)



    def generate_profile(self):
        """ Generate clone profile """

        seg_array = np.zeros_like(self.cell_bin_cn)

        for cell in range(self.clone_size):
            for i in range(len(self.breakpoints) - 1):
                start_p = self.breakpoints[i]
                end_p = self.breakpoints[i + 1]
                seg_array[cell, start_p:end_p] = stats.mode(
                    self.cell_gene_cn[cell,
                                    start_p * self.bin_size:end_p * self.bin_size]
                                    ).mode
            seg_array[cell, end_p:] = stats.mode(
                            self.cell_gene_cn[cell, end_p * self.bin_size:]
                            ).mode
        self.segment_cn = np.mean(seg_array, axis=0)

"""
from copyvae.binning import bin_genes_from_text 
from copyvae.segmentation import bin_to_segment
from copyvae.graphics import draw_heatmap

gene_cn = np.load('cn_per_gene.npy')
bin_cn = np.load('cn_per_bin.npy')
binned_genes, chroms = bin_genes_from_text('../data/copykat_data/txt_files/GSM4476485_combined_UMIcount_CellTypes_DCIS1.txt', 25)

t_clone = Clone(1,
                    1021,
                    25,
                    cell_gene_cn=gene_cn[379:,:],
                    cell_bin_cn=bin_cn[379:,:],
                    chrom_bound=chroms
                    )

t_clone.call_breakpoints()
t_clone.generate_profile()
clone_seg = t_clone.segment_cn
#print(clone_seg)
clone_gene_cn = np.repeat(clone_seg, 25)
with open('clone_gene_cn.npy', 'wb') as f:
    np.save(f, clone_gene_cn)

bp_arr = t_clone.breakpoints
seg_profile = bin_to_segment(bin_cn, bp_arr)
with open('segments.npy', 'wb') as f:
        np.save(f, seg_profile)
draw_heatmap(seg_profile, "tumour_seg")
"""
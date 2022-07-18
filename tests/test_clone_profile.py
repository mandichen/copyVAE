#! /usr/bin/env python3

import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import distance
from scipy.stats import pearsonr
#from copyvae.binning import CHR_BASE_PAIRS

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
    156040895,
    57227415])

def construct_position_df(base_per_region=180000):
    """ Generate position dataframe for given resolution

    Args:
        base_per_region: genome resolution
    Returns:
        dataframe containing chromosome name, position in chromosome,
        and absolute position
    """

    chr_arr = np.array([])
    abs_arr = np.array([])
    name_arr = np.array([])
    chr_pos = np.array(CHR_BASE_PAIRS)

    for i in range(len(chr_pos)):
        chrpos = np.arange(0, chr_pos[i], base_per_region)
        abspos = chrpos + chr_pos[:i].sum()
        chr_name = np.ones_like(chrpos) + i
        chr_arr = np.concatenate((chr_arr, chrpos), axis=0)
        abs_arr = np.concatenate((abs_arr, abspos), axis=0)
        name_arr = np.concatenate((name_arr, chr_name), axis=0)

    narr = np.stack((name_arr, chr_arr, abs_arr), axis=-1).astype(int)
    df = pd.DataFrame(data=narr, columns=['chrom', 'chrompos', 'abspos'])
    #df = df.iloc[5:].reset_index(drop=True)
    return df


def test_dcis_inference(clone_profile, abs_position, gt_profile):
    """ Test DCIS1 data inference

    Args:
        clone_profile: inferred clone profile
        abs_position: absolute gene position in the genome
        gt_profile: ground truth clone profile
    Returns:
        distance1: Euclidean distance
        distance2: Cosine similarity
        distance3: Pearson correlation coefficient
        distance4: Manhattan distance
    """

    # process groud truth
    gt = pd.read_csv(gt_profile, sep='\t')
    gt['abspos'] = gt['abspos'].astype('int')

    #gt['copy'] = round(2**gt['med.DNA']*2)
    gt['copy'] = 2**gt['med.DNA'] * 2

    # estimate copy of positions in ground truth
    clone = {'abspos': abs_position, 'cp_inf': clone_profile}
    df = pd.DataFrame(data=clone)
    compdf = pd.merge_asof(
        gt.sort_values('abspos'),
        df.sort_values('abspos'),
        on='abspos')

    #distance1 = np.linalg.norm(compdf['copy'].values - compdf['cp_inf'].values)
    distance1 = distance.euclidean(compdf['copy'].values, compdf['cp_inf'].values)
    distance2 = distance.cosine(compdf['copy'].values, compdf['cp_inf'].values)
    distance3 = pearsonr(compdf['copy'].values, compdf['cp_inf'].values)
    distance4 = distance.cityblock(compdf['copy'].values, compdf['cp_inf'].values)

    #compdf.loc[compdf['copy'] > 6, 'copy'] = 6.0
    gtcp = compdf['copy'].values
    infcp = compdf['cp_inf'].values
    plt.figure(figsize=(17, 5), dpi=120)
    plt.plot(infcp,label='inference')
    plt.plot(gtcp, label='GT')
    plt.legend()
    plt.savefig('dcis1.png')

    return distance1, distance2, distance3, distance4


def test_dlp_inference(profile_path, abs_position, gt_profile):
    """ Test DLP 10x data inference

    Args:
        profile_path: path to inferred clone profiles
        abs_position: absolute gene position in the genome
        gt_profile: ground truth clone profile
    """

    gt = pd.read_csv(gt_profile)
    gt['abspos']=gt['start']
    gb = gt.groupby('clone_id')    
    clone_list = [gb.get_group(x).reset_index(0, drop=True) for x in gb.groups]
    sample_gt = clone_list[3]
    sample_gt.loc[sample_gt.chr == 'X', 'chr'] = '23'
    sample_gt.loc[sample_gt.chr == 'Y', 'chr'] = '24'
    sample_gt.loc[:, 'chr'] = sample_gt.chr.astype(int)
    sample_gt = sample_gt.sort_values(by=['chr', 'start'])
    chr_pos=CHR_BASE_PAIRS
    for i in range(len(chr_pos)):
        sample_gt.loc[sample_gt['chr'] == i + 1, 'abspos'] += chr_pos[:i].sum()

    c_list = []    
    for file in glob.glob(profile_path+'*.npy'):
        clone_profile = np.load(file)
        clone = {'abspos': abs_position, 'cp_inf': clone_profile}
        df = pd.DataFrame(data=clone)
        compdf = pd.merge_asof(
                sample_gt.sort_values('abspos'),
                df.sort_values('abspos'),
                on='abspos')
        c_list.append(compdf['cp_inf'].values)
    
    plt.figure(figsize=(17, 5), dpi=120)
    plt.ylim(0,6)
    for i in range(len(c_list)):
        plt.plot(c_list[i], label='clone '+str(i+1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('dlp.png')

    return None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="clone profile")
    parser.add_argument('-a', '--abs', help="absolute position")
    parser.add_argument('-gt', '--gt', help="ground truth")

    args = parser.parse_args()
    #clone_profile = np.load(args.input)
    abs_pos = np.load(args.abs)
    gt_profile = args.gt

    #dis = test_dcis_inference(clone_profile, abs_pos, gt_profile)
    #print('Euclidean:')
    #print(dis)
    profile_path = args.input
    test_dlp_inference(profile_path, abs_pos, gt_profile)


if __name__ == "__main__":
    main()

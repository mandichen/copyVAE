#!/usr/bin/envs python3

import os 
import click 
import numpy as np
import pandas as pd

from tqdm import tqdm


def get_mean(group, j, group_name, df_all):
    group = group[group.columns[1:-6]]
            
    mean_df = pd.DataFrame([group.astype(int).mean()])
    mean_df['group'] = group_name
    df_all = pd.concat([df_all, mean_df])

    group_name += 1

    return df_all, group_name


def compute_corrected_mean(column):
    try: 
        return sum(column) / len(column[column != 0])
    except: 
        return 0


def get_mean_corrected(group, j, group_name, df_all):
    group = group[group.columns[1:-6]].astype(int)
    
    mean_df = pd.DataFrame([group.apply(compute_corrected_mean, axis=0)])
    mean_df['group'] = group_name
    df_all = pd.concat([df_all, mean_df])

    group_name += 1

    return df_all, group_name


@click.command(short_help='script to group umi counts ')
@click.option(
    '-umi', '--umi_counts', help='path to umi counts'
)
@click.option(
    '-gm', '--gene_metadata', help='Table containing gene names and metadata'
)
@click.option(
    '-o', '--output', help='output folder'
)
@click.option(
    '-mt', '--mean_type', required=True,
    type=click.Choice(['mean', 'corrected']),
    help='select the type of mean to perform'
)
def main(umi_counts, gene_metadata, output, mean_type):
    umis = pd.read_csv(umi_counts, sep='\t', low_memory=False).reset_index()
    gene_info = pd.read_csv(gene_metadata, sep='\t')

    gene_merge = pd.merge(
        umis, gene_info, right_on=['Gene name'], left_on=['index'], how='inner'
    )

    group_name = 0; df_all = pd.DataFrame()
    for i, df in tqdm(gene_merge.groupby('Chromosome/scaffold name')):
        for j in tqdm(range(25, len(df), 25)):
            group = df[j-25:j]

            if mean_type == 'mean':
                df_all, group_name = get_mean(group, j, group_name, df_all)
            else:
                df_all, group_name = get_mean_corrected(group, j, group_name, df_all)

        group = df[j: len(df)]
        if mean_type == 'mean':
            df_all, group_name = get_mean(group, j, group_name, df_all)
        else:
            df_all, group_name = get_mean_corrected(group, j, group_name, df_all)
    

    if mean_type == 'mean':
        df_all.to_csv(os.path.join(output, 'mean_data.tsv'), sep='\t', index=None)
    
    else:
        df_all.to_csv(
            os.path.join(output, 'mean_data_corrected.tsv'), sep='\t', index=None)
    

if __name__ == '__main__':
    main()
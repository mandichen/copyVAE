#!/usr/bin/envs python3

import os
import click
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt 

def plot_scatter(data, output, outfile):
    fig, ax = plt.subplots(figsize=(20, 4), facecolor='white')
    # import pdb;pdb.set_trace()
    try:
        sns.scatterplot(list(range(data.shape[1])) * len(data), data.flatten())
    except: 
        sns.scatterplot(list(range(len(data))),data)
        plt.ylim([1.25, 3.5])

   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig_out = os.path.join(output, outfile)
    plt.tight_layout()
    plt.savefig(fig_out)
    plt.close()



@click.command(short_help='explore input data pdpaola')
@click.option(
    '-d', '--data', default='', help='tsv file containing the data'
)
@click.option(
    '-o', '--output', default='', help='Path to save file'
)
def main(data, output):

    data = np.load(data)
    import pdb;pdb.set_trace()
    normal = data[:374]
    tumor = data[374:]
    
    print(Counter(normal.flatten()))
    print(Counter(tumor.flatten()))

    plot_scatter(np.mean(normal, axis=0), output, 'normal.png')
    plot_scatter(np.mean(tumor, axis=0), output, 'tumor.png')

    # plot_scatter(normal, output, 'normal.png')
    # plot_scatter(tumor, output, 'tumor.png')



if __name__ == '__main__':
    main()
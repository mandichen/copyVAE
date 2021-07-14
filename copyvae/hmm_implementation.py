#!/usr/bin/envs/ python3

import os
import click
import numpy as np

import numpy as np
import pandas as pd

class Initialization:
    def __init__(self):
        self.A_transmat = [1, 2, 3]
        self.B_obsmat = [3, 4]
        self.pi_init_statedist = [5]


class ForwardPass:
    def __init__(self, states):
        self.N = states
        import pdb;pdb.set_trace()


class BackwardPass:
    def __init__(self, ):
        import pdb;pdb.set_trace()


class HMMTraining:
    def __init__(self, ):
        import pdb;pdb.set_trace()



@click.command(short_help='script for the segmentation throuhg an HMM ')
@click.option(
    '-da', '--data', default='', help='copy number data'
)
@click.option(
    '-o', '--output', default='', help='output folder'
)
def main(data, output):
    data = np.load(data)
    
    normal = data[:374]
    tumor = data[374:]

    import pdb;pdb.set_trace()

    


if __name__ == '__main__':
    main()
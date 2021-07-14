#!/usr/bin/envs/ python3

import os
import click
from sklearn import mixture

import numpy as np

from hmmlearn import hmm


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

    #this should be done with all cells but to start we select only one: 
    to_analyze = tumor[139]
    import pdb;pdb.set_trace()

    # model = hmm.GaussianHMM(n_components=6, covariance_type="diag", init_params="cm", params="cmt", n_iter=100)
    # # model.startprob_prior = np.array([0.00, 0.00, 1.0, 0.00, 0.00, 0.00])
    # model.transmat_ = np.array([[0.6, 0.2, 0.1, 0.05, 0.03, 0.02], [0.1, 0.6, 0.2, 0.05, 0.03, 0.02], [0.05, 0.1, 0.6, 0.2, 0.03, 0.02], \
    #     [0.05, 0.05, 0.1, 0.6, 0.18, 0.02], [0.02, 0.03, 0.05, 0.2, 0.6, 0.1], [0.02, 0.03, 0.05, 0.1, 0.2, 0.6]])
    # # model.transmat_ = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], \
    # #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    # model.startprob_ = np.array([0.03, 0.05, 0.8, 0.05, 0.04, 0.03])

    model = hmm.GMMHMM(n_components=6, n_mix=2, n_iter=100, init_params="smtcw")
    # model.gmms_ = [mixture.GaussianMixture(),mixture.GaussianMixture(),mixture.GaussianMixture()]
    model.fit(tumor.T)
    print(model.predict(tumor.T))

    model.monitor_
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
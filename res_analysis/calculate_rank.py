# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:38:07 2024

@author: xiato
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

data = """0.550 ± 0.015	0.580 ± 0.001	0.549 ± 0.001	0.565 ± 0.001	0.586 ± 0.008	0.551 ± 0.010	0.605 ± 0.001
0.649 ± 0.006	0.557 ± 0.005	0.616 ± 0.001	0.648 ± 0.003	0.701 ± 0.002	0.629 ± 0.006	0.677 ± 0.001
0.571 ± 0.006	0.571 ± 0.003	0.583 ± 0.003	0.611 ± 0.006	0.603 ± 0.005	0.610 ± 0.004	0.613 ± 0.002
0.633 ± 0.012	0.605 ± 0.004	0.659 ± 0.001	0.669 ± 0.002	0.680 ± 0.006	0.665 ± 0.001	0.673 ± 0.001
0.546 ± 0.008	0.602 ± 0.001	0.549 ± 0.005	0.603 ± 0.013	0.609 ± 0.004	0.584 ± 0.003	0.575 ± 0.006
0.639 ± 0.010	0.608 ± 0.000	0.666 ± 0.002	0.684 ± 0.002	0.801 ± 0.000	0.722 ± 0.004	0.762 ± 0.001
0.579 ± 0.043	0.605 ± 0.077	0.886 ± 0.017	0.933 ± 0.005	0.855 ± 0.012	0.872 ± 0.011	0.741 ± 0.011
0.534 ± 0.06	0.507 ± 0.027	0.549 ± 0.022	0.680 ± 0.009	0.685 ± 0.012	0.674 ± 0.013	0.650 ± 0.005
0.753 ± 0.008	0.606 ± 0.003	0.724 ± 0.001	0.742 ± 0.001	0.874 ± 0.000	0.801 ± 0.002	0.825 ± 0.001
0.502 ± 0.08	0.505 ± 0.11	0.614 ± 0.040	0.703 ± 0.036	0.719 ± 0.018	0.742 ± 0.014	0.700 ± 0.013
0.494 ± 0.054	0.590 ± 0.034	0.510 ± 0.021	0.635 ± 0.040	0.625 ± 0.038	0.683 ± 0.007	0.615 ± 0.019
0.772 ± 0.005	0.657 ± 0.002	0.649 ± 0.001	0.702 ± 0.001	0.781 ± 0.000	0.769 ± 0.000	0.742 ± 0.001
0.965 ± 0.589	1.545 ± 2.084	1.345 ± 0.792	1.138 ± 0.962	1.606 ± 1.312	1.023 ± 0.854	1.191 ± 0.721
0.859 ± 0.815	1.738 ± 2.967	1.081 ± 0.720	1.130 ± 0.845	1.459 ± 1.074	0.771 ± 0.752	0.996 ± 0.732
0.194 ± 0.397	0.279 ± 0.629	0.143 ± 0.153	0.178 ± 0.151	0.155 ± 0.155	0.148 ± 0.165	0.155 ± 0.172
0.179 ± 0.204	0.227 ± 0.301	0.150 ± 0.184	0.276 ± 0.300	0.179 ± 0.127	0.220 ± 0.217	0.245 ± 0.185
0.724 ± 0.532	0.900 ± 1.377	0.983 ± 0.721	0.710 ± 0.585	1.737 ± 1.041	0.672 ± 0.535	0.593 ± 0.414
0.605 ± 0.541	1.103 ± 1.466	0.960 ± 0.741	0.838 ± 0.694	1.488 ± 1.005	0.736 ± 0.566	0.561 ± 0.348
3.852 ± 1.060	2.611 ± 0.786	2.630 ± 0.832	2.615 ± 0.804	2.567 ± 0.785	2.623 ± 0.831	2.537 ± 0.782"""

data = data.split("\n")
num_tasks = len(data)
num_methods = 7
metrics = []
for i in range(num_tasks):
    metric = []
    line = data[i]
    dps = line.split("\t")
    metrics.append([float(dp.split(" ± ")[0]) for dp in dps])
for line in metrics:
    for v in line:
        print(v, end="\t")
    print()
# print(metrics)
data = np.array(metrics)

plt.figure(figsize=(5,5), dpi=300)
# Define the number of variables (metrics) and methods
num_vars = num_tasks
num_methods = 7

methods = ['Opensmile', 'VGGish', 'AudioMAE', 'CLAP', 'OPERA-CT', 'OPERA-CE', 'OPERA-GT']

def compute_all():
     
    num_vars = len(metrics)

    print(data)
    print(len(data))

    #  compute the Mean Reciprocal Rank
    ranks = np.zeros_like(data)

    # Rank the methods for each task
    for i in range(num_vars):
        if i <12:
            # ranks[i] = np.argsort(np.argsort(-data[i])) + 1
            ranks[i] = rankdata(-data[i], method="min")
        else:
            # ranks[i] = np.argsort(np.argsort(data[i])) + 1 
            ranks[i] = rankdata(data[i], method="min")

    print(ranks)
    # Calculate the reciprocal ranks
    reciprocal_ranks = 1.0 / ranks

    # Calculate the Mean Reciprocal Rank (MRR) for each method
    mrrs = reciprocal_ranks.mean(axis=0)

    # Print the MRR for each method
    for method in range(num_methods):
        print(f'{mrrs[method]:.4f}')


def compute_group1():
     
    selected = range(12)
    mask = np.ma.make_mask([1 if i in selected else 0 for i in range(num_tasks)])
    data_selected = data[mask]
    num_vars = len(selected)

    print(data_selected)
    print(len(data_selected))

    #  compute the Mean Reciprocal Rank
    ranks = np.zeros_like(data_selected)

    # Rank the methods for each task
    for i in range(num_vars):
        ranks[i] = np.argsort(np.argsort(-data_selected[i])) + 1

    print(ranks)
    # Calculate the reciprocal ranks
    reciprocal_ranks = 1.0 / ranks

    # Calculate the Mean Reciprocal Rank (MRR) for each method
    mrrs = reciprocal_ranks.mean(axis=0)

    # Print the MRR for each method
    for method in range(num_methods):
        print(f'{mrrs[method]:.4f}')


def compute_group2():
     
    selected = range(12, 19)
    mask = np.ma.make_mask([1 if i in selected else 0 for i in range(num_tasks)])
    data_selected = data[mask]
    num_vars = len(selected)

    print(data_selected)
    print(len(data_selected))

    #  compute the Mean Reciprocal Rank
    ranks = np.zeros_like(data_selected)

    # Rank the methods for each task
    for i in range(num_vars):
        ranks[i] = np.argsort(np.argsort(data_selected[i])) + 1 

    print(ranks)
    # Calculate the reciprocal ranks
    reciprocal_ranks = 1.0 / ranks

    # Calculate the Mean Reciprocal Rank (MRR) for each method
    mrrs = reciprocal_ranks.mean(axis=0)

    # Print the MRR for each method
    for method in range(num_methods):
        print(f'{mrrs[method]:.4f}')

compute_all()
compute_group1()
compute_group2()
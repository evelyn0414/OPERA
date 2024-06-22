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
0.537 ± 0.011	0.538 ± 0.028	0.554 ± 0.004	0.599 ± 0.007	0.578 ± 0.001	0.566 ± 0.008	0.552 ± 0.003
0.677 ± 0.005	0.600 ± 0.001	0.628 ± 0.001	0.665 ± 0.001	0.795 ± 0.001	0.721 ± 0.001	0.735 ± 0.000
0.579 ± 0.043	0.605 ± 0.077	0.886 ± 0.017	0.933 ± 0.005	0.855 ± 0.012	0.872 ± 0.011	0.741 ± 0.011
0.534 ± 0.060	0.507 ± 0.027	0.549 ± 0.022	0.680 ± 0.009	0.685 ± 0.012	0.674 ± 0.013	0.650 ± 0.005
0.753 ± 0.008	0.606 ± 0.003	0.724 ± 0.001	0.742 ± 0.001	0.874 ± 0.000	0.801 ± 0.002	0.825 ± 0.001
0.502 ± 0.080	0.505 ± 0.110	0.614 ± 0.040	0.703 ± 0.036	0.719 ± 0.018	0.742 ± 0.014	0.700 ± 0.013
0.494 ± 0.054	0.590 ± 0.034	0.510 ± 0.021	0.635 ± 0.040	0.625 ± 0.038	0.683 ± 0.007	0.615 ± 0.019
0.772 ± 0.005	0.657 ± 0.002	0.649 ± 0.001	0.702 ± 0.001	0.781 ± 0.000	0.769 ± 0.000	0.742 ± 0.001
0.985 ± 0.743	0.904 ± 0.568	0.900 ± 0.551	0.896 ± 0.542	0.924 ± 0.583	0.848 ± 0.607	0.892 ± 0.618
0.756 ± 0.721	0.839 ± 0.563	0.821 ± 0.590	0.840 ± 0.547	0.837 ± 0.563	0.834 ± 0.581	0.825 ± 0.560
0.141 ± 0.185	0.131 ± 0.146	0.129 ± 0.146	0.134 ± 0.146	0.128 ± 0.140	0.132 ± 0.141	0.128 ± 0.141
0.850 ± 0.592	0.895 ± 0.559	0.833 ± 0.588	0.883 ± 0.560	0.885 ± 0.553	0.761 ± 0.544	0.878 ± 0.550
0.730 ± 0.497	0.842 ± 0.559	0.876 ± 0.561	0.859 ± 0.541	0.780 ± 0.542	0.830 ± 0.561	0.774 ± 0.554
0.138 ± 0.166	0.130 ± 0.145	0.131 ± 0.141	0.137 ± 0.147	0.132 ± 0.140	0.136 ± 0.150	0.130 ± 0.138
2.714 ± 0.902	2.605 ± 0.759	2.641 ± 0.813	2.650 ± 0.947	2.636 ± 0.858	2.525 ± 0.782	2.416 ± 0.885"""

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
        print(f'{mrrs[method]:.4f}', end="\t")
    print()

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
        ranks[i] = rankdata(-data_selected[i], method="min")

    print(ranks)
    # Calculate the reciprocal ranks
    reciprocal_ranks = 1.0 / ranks

    # Calculate the Mean Reciprocal Rank (MRR) for each method
    mrrs = reciprocal_ranks.mean(axis=0)

    # Print the MRR for each method
    for method in range(num_methods):
        print(f'{mrrs[method]:.4f}', end="\t")
    print()

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
        ranks[i] = rankdata(data_selected[i], method="min")

    print(ranks)
    # Calculate the reciprocal ranks
    reciprocal_ranks = 1.0 / ranks

    # Calculate the Mean Reciprocal Rank (MRR) for each method
    mrrs = reciprocal_ranks.mean(axis=0)

    # Print the MRR for each method
    for method in range(num_methods):
        print(f'{mrrs[method]:.4f}', end="\t")
    print()

compute_all()
compute_group1()
# compute_group2()
# Transitive graph projection - greedy heuristic

A greedy algorithm for solving the transitive graph projection problem.

Paper:

Exact and Heuristic Algorithms for Weighted Cluster Editing.
Sven Rahmann, Tobias Wittkop, Jan Baumbach, Marcel Martin, Anke Truß, Sebastian Böcker
https://www.lifesciencessociety.org/CSB2007/toc/391.2007.html


The original code from 2007 was written in Python 2.
It can be run using Conda:

    conda create -n tgp python==2.7
    conda activate tgp
    easy_install networkx==0.37
    python rewedge.py data/*.cm


# Transitive graph projection - greedy heuristic

A greedy algorithm for solving the transitive graph projection problem.

This is old code written for my diploma thesis in 2007.

The algorithm has also been published in a paper:

Exact and Heuristic Algorithms for Weighted Cluster Editing.
Sven Rahmann, Tobias Wittkop, Jan Baumbach, Marcel Martin, Anke Truß, Sebastian Böcker
https://www.lifesciencessociety.org/CSB2007/toc/391.2007.html

The original code was written in Python 2.
It can be run using Conda:

    conda create -n tgp python==2.7
    conda activate tgp
    easy_install networkx==0.99
    python rewedge.py data/*.cm

Run the tests:

    easy_install pytest==3.0.0
    pytest

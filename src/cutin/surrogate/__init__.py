"""A small finite element (FEM) plus graph neural network (GNN) surrogate study.

The rest of cutin-atlas asks how few expensive simulation runs are enough to answer a question.
This subpackage asks the same question for a finite element simulation: a graph neural network
learns to predict how a plate with a hole deforms under load, and the study measures how its
error falls as it sees more FEM runs.
"""

from __future__ import division
import networkx as nx
import sys
import numpy as np

def normalize(X):
    Sum = sum(X)
    X = [x / Sum for x in X]
    return X

def load_groups(path):
    groups = []
    f = open(path, 'r')
    for line in f:
        line = line.strip().split()
        groups.append(map(int, line))
    f.close()
    return groups

def load_network(format, path):
    if format == 'edge_list':
        G = nx.read_edgelist(path, nodetype=int)
    elif format == 'adj_list':
        G = nx.read_adjlist(path, nodetype=int)
    elif format == 'adj_matrix':
        am = np.matrix(np.loadtxt(path))
        G = nx.from_numpy_matrix(am)
    else:
        print 'Network Format Wrong!'
        sys.exit()
    return G

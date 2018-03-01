from __future__ import division
import os
from collections import defaultdict, Counter
import pynauty
import numpy as np
import itertools
import networkx as nx
import random
from tqdm import tqdm
from multiprocessing import Pool
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.utils import shuffle as skshuffle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score
import re
import cPickle as pickle
import math
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from group2vec.utils import load_data, normalize
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run CPNE.")
    parser.add_argument('--edge_list', default='edges_list', help='Input edge list path.')
    parser.add_argument('--group_members', default='group_members', help='Input group members path.')
    parser.add_argument('--group_embs', default='group_embs', help='Output group embeddings path.')
    parser.add_argument('--num_threads', type=int, default=10, help='Number of threads.')
    parser.add_argument('--num_walks', type=int, default=1000, help='Number of walks for generating the group corpus.')
    parser.add_argument('--num_walks_trans', type=int, default=100,
                        help='Number of walks for generating the transition matrices with DeepWalk.')
    parser.add_argument('--walk_length', type=int, default=100, help='Length of walks for generating the group corpus.')
    parser.add_argument('--walk_length_trans', type=int, default=100,
                        help='Length of walks for generating the transition matrices with DeepWalk.')
    parser.add_argument('--neg_cnt', type=int, default=5,
                        help='Number of negtive instances for generating the group corpus.')
    parser.add_argument('--neg_cnt_trans', type=int, default=5,
                        help='Number of negtive instances for generating the transition matrices with DeepWalk.')
    parser.add_argument('--dimension', type=int, default=128, help='Number of dimensions for group embedding.')
    parser.add_argument('--dimension_trans', type=int, default=128,
                        help='Number of dimensions for node embedding used to generating the transition matrices.')
    parser.add_argument('--window_size', type=int, default=3, help='Window size for group embedding.')
    parser.add_argument('--window_size_trans', type=int, default=3,
                        help='Window size for node embedding used to generating the transition matrices with DeepWalk.')
    parser.add_argument('--K', type=int, default=100, help='Top K nodes to save in A for each node.')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold lambda to generating group corpus.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate to learn group embedding.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted for generating the transition matrices with DeepWalk. Default is weighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)
    parser.add_argument('--deepwalk', dest='deepwalk', action='store_true', help='Generate transition matrix A with DeepWalk. Default is true.')
    parser.set_defaults(deepwalk=True)
    return parser.parse_args()


def generate_motif(motif_size):
    cert2idx = {}
    idx2size = {}
    file_counter = open("canonical_maps/canonical_map_n%s.p" % motif_size, "rb")
    canonical_map = pickle.load(file_counter)
    for canonical, values in canonical_map.iteritems():
        if values['idx'] > 1:  # rm the case idx == 0 and idx == 1
            cert2idx[canonical] = values['idx'] - 2
            idx2size[values['idx'] - 2] = values['n']
    return cert2idx, idx2size



if __name__ == '__main__':
    args = parse_args()

    # generatint motifs
    cert2idx, idx2size = generate_motif(args.motif_size)
    num_motifs = len(cert2idx)

    # loading the data
    G = nx.read_edgelist(args.edge_list, nodetype=int)
    groups = load_data(args.group_members)
    #nodes = G.nodes()
    #num_nodes = G.number_of_nodes()
    num_groups = len(groups)

    group_ams = []
    for group in groups:
        group_ams.append(nx.adjacency_matrix(G.subgraph(group)).toarray())
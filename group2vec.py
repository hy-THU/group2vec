from __future__ import division
import os
from collections import Counter
from collections import defaultdict
import re
import networkx as nx
import itertools
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.utils import shuffle as skshuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import argparse
from group2vec.utils import load_data, normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Run CPNE.")
    parser.add_argument('--edge_list', default='edges_week', help='Input edge list path.')
    parser.add_argument('--group_members', default='pos_week', help='Input group members path.')
    parser.add_argument('--num_threads', type=int, default=10, help='Number of threads.')
    parser.add_argument('--num_walks', type=int, default=100, help='Number of walks for generating the group corpus.')
    parser.add_argument('--num_walks_trans', type=int, default=100, help='Number of walks for generating the transition matrices with DeepWalk.')
    parser.add_argument('--walk_length', type=int, default=100, help='Length of walks for generating the group corpus.')
    parser.add_argument('--walk_length_trans', type=int, default=100, help='Length of walks for generating the transition matrices with DeepWalk.')
    parser.add_argument('--neg_cnt', type=int, default=5, help='Number of negtive instances for generating the group corpus.')
    parser.add_argument('--neg_cnt_trans', type=int, default=5, help='Number of negtive instances for generating the transition matrices with DeepWalk.')
    parser.add_argument('--dimension', type=int, default=128, help='Number of dimensions for group embedding.')
    parser.add_argument('--dimension_trans', type=int, default=128, help='Number of dimensions for node embedding used to generating the transition matrices.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate to learn group embedding.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted for generating the transition matrices with DeepWalk. Default is weighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    return parser.parse_args()

def generate_A():
    pass

def generate_B():
    pass

def generate_group_corpus():
    pass

def compute_embs(corpus):
    pass

def output_embs():
    pass

if __name__ == '__main__':
    args = parse_args()

    # loading the data
    G = nx.read_edgelist(args.edge_list, nodetype = int)
    groups = load_data(args.group_members)
    num_nodes = G.number_of_nodes()
    num_groups = len(groups)

    # generating the transition matrices
    neighbors_node_node, weights_node_node = generate_A()
    neighbors_node_group, weights_node_group, neighbors_group_node, weights_group_node = generate_B()

    # generating group corpus
    corpus = generate_group_corpus()

    # embedding
    embs = compute_embs(corpus)

    output_embs()

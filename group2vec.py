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
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    return parser.parse_args()


if __name__ == '__main__':
    # loading the data
    f = open('G.dat', 'r')
    G, groups, labels, _ = pickle.load(f)
    f.close()
    num_nodes = G.number_of_nodes()
    num_groups = len(groups)

    # generating the transition matrices
    neighbors
    weights

    # generating group corpus

    # embedding

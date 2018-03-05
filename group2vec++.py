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
    parser.add_argument('--adj_list', default='adj_list', help='Input adjacency list path.')
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
    parser.add_argument('--deepwalk', dest='deepwalk', action='store_true',
                        help='Generate transition matrix A with DeepWalk. Default is true.')
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


def generate_transition_matrices():
    corpus, groups = sample_motifs()
    neighbors_motif_motif, weights_motif_motif = generate_A(corpus)
    neighbors_motif_group, weights_motif_group, neighbors_group_motif, weights_group_motif = generate_B()
    return neighbors_motif_motif, weights_motif_motif, neighbors_motif_group, weights_motif_group, neighbors_group_motif, weights_group_motif


def sample_motifs():
    corpus = []
    corpus_augmented = []

    gidx = num_motifs
    groups = []
    for am in group_ams:
        groups.append({'am': am, 'gidx': gidx})
        gidx += 1
    pool = Pool(args.num_threads)
    for _ in tqdm(range(args.num_walks_motif)):
        corpus_augmented += pool.map(motif_sampler, groups)
    pool.close()
    pool.join()
    groupsdict = defaultdict(list)

    for gidx, seq in corpus_augmented:
        corpus.append(seq)
        for motif in seq:
            groupsdict[gidx].append(motif)
    groups = []  # different meaning from above
    for i in range(num_groups):
        groups.append(' '.join(a for a in groupsdict[i + num_motifs]))

    random.shuffle(corpus)
    return corpus, groups


def motif_sampler():
    am = group['am']
    gidx = group['gidx']
    seq = []
    while len(seq) < args.walk_length_trans:
        size = args.motif_size
        nodes = np.random.permutation(range(len(am)))[:size]
        motif_am = am[np.ix_(nodes, nodes)]
        motif_idx = cert2idx[get_motif(motif_am, size)]
        seq.append(str(motif_idx))
    return (gidx, seq)

def get_motif(motif_am, size):
    adj_mat = {idx: [i for i in list(np.where(edge)[0]) if i != idx] for idx, edge in enumerate(motif_am)}
    g = pynauty.Graph(number_of_vertices=size, directed=False, adjacency_dict=adj_mat)
    cert = pynauty.certificate(g)
    return cert

def generate_A(corpus):
    motif_counter = Counter()
    for seq in corpus:
        motif_counter.update(seq)
    #print motif_counter

    model = Word2Vec(corpus, size=args.dimension_trans, window=args.window_size_trans, min_count=0, workers=args.num_threads)

    neighbors_motif_motif = {}
    weights_motif_motif = {}
    for motif in motif_counter:
        motif = int(motif)
        neighbors, weights = zip(*model.wv.most_similar(positive=str(motif), topn=args.K))
        neighbors = np.asarray(neighbors)
        weights = np.asarray(weights)
        idxs = np.ix_(np.where(weights > 0)[0])
        neighbors_motif_motif[motif] = map(int, neighbors[idxs])
        weights = weights[idxs]
        weights_motif_motif[motif] = normalize(weights)
    return neighbors_motif_motif, weights_motif_motif


def generate_B(groups):
    neighbors_motif_group = {}
    weights_motif_group = {}
    neighbors_group_motif = {}
    weights_group_motif = {}
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(groups))
    tfidf = tfidf.toarray()
    word = np.array(vectorizer.get_feature_names())
    for i in range(num_groups):
        idxs = np.ix_(np.where(tfidf[i] > 0)[0])
        weights_group_motif[i + num_motifs] = normalize(tfidf[i][idxs])
        neighbors_group_motif[i + num_motifs] = map(int, word[idxs])
    for i in range(len(word)):
        motif = int(word[i])
        idxs = np.ix_(np.where(tfidf[:, i] > 0)[0])
        neighbors_motif_group[motif] = list(idxs[0] + num_motifs)
        weights_motif_group[motif] = normalize(tfidf[idxs, i][0])
    return neighbors_motif_group, weights_motif_group, neighbors_group_motif, weights_group_motif


def generate_group_corpus():
    walks = []
    nodes = neighbors_motif_group.keys() + neighbors_group_motif.keys()
    pool = Pool(args.num_threads)
    for _ in tqdm(range(args.num_walks)):
        walks += pool.map(walk, nodes)
    pool.close()
    pool.join()
    random.shuffle(walks)
    walks = [map(str, walk) for walk in walks]
    return walks

def walk(start_node):
    walk = [start_node]
    while len(walk) < args.walk_length:
        cur = walk[-1]
        if cur >= num_motifs and len(neighbors_group_motif[cur]) > 0:
            walk.append(np.random.choice(neighbors_group_motif[cur], p = weights_group_motif[cur]))
        elif cur < num_motifs and random.random() > args.threshold:
            walk.append(np.random.choice(neighbors_motif_motif[cur], p = weights_motif_motif[cur]))
        elif cur < num_motifs and len(neighbors_motif_group[cur]) > 0:
            walk.append(np.random.choice(neighbors_motif_group[cur], p = weights_motif_group[cur]))
        else:
            break
    walk = np.asarray(walk)
    return walk[np.ix_(np.where(walk >= num_motifs)[0])]

def compute_embs(corpus):
    model = Word2Vec(corpus, size=args.dimension, window=args.window_size, min_count=0, workers=args.num_threads)
    return model.wv

def output_embs():
    embs.save(args.group_embs)


if __name__ == '__main__':
    args = parse_args()

    # generatint motifs
    cert2idx, idx2size = generate_motif(args.motif_size)
    num_motifs = len(cert2idx)

    # loading the data
    G = nx.read_edgelist(args.edge_list, nodetype=int)
    groups = load_data(args.group_members)
    # nodes = G.nodes()
    # num_nodes = G.number_of_nodes()
    num_groups = len(groups)

    group_ams = []
    for group in groups:
        group_ams.append(nx.adjacency_matrix(G.subgraph(group)).toarray())

    # generating the transition matrices
    neighbors_motif_motif, weights_motif_motif, neighbors_motif_group, weights_motif_group, neighbors_group_motif, weights_group_motif = generate_transition_matrices()

    # generating group corpus
    corpus = generate_group_corpus()

    # embedding
    embs = compute_embs(corpus)

    output_embs()
from __future__ import division
from gensim.models import Word2Vec
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import argparse
from collections import defaultdict
from sklearn import preprocessing
from utils import load_network, load_groups, normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Run CPNE.")
    parser.add_argument('--network_format', default='adj_list', help='Input adjacency list path.')
    parser.add_argument('--network', default='adj.txt', help='Input adjacency list path.')
    parser.add_argument('--groups', default='group.txt', help='Input group members path.')
    parser.add_argument('--group_embs', default='group_embs', help='Output group embeddings path.')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers.')
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


def generate_A_adjacency():
    neighbors_node_node = defaultdict(list)
    weights_node_node = defaultdict(list)
    for node in nodes:
        for a, b in G[node].iteritems():
            if node != a:
                neighbors_node_node[node].append(a)
                if b.has_key('weight'):
                    weights_node_node[node].append(b['weight'])
                else:
                    weights_node_node[node].append(1)
        weights_node_node[node] = normalize(weights_node_node[node])
    return neighbors_node_node, weights_node_node


def generate_A_deepwalk():
    corpus = []
    pool = Pool(args.num_workers)
    for _ in tqdm(range(args.num_walks_trans)):
        corpus += pool.map(nodes_walk, nodes)
    pool.close()
    pool.join()
    corpus = [map(str, seq) for seq in corpus]
    random.shuffle(corpus)

    model = Word2Vec(corpus, size=args.dimension_trans, window=args.window_size_trans, min_count=0, workers=args.num_workers)

    for node in nodes:
        neighbors, weights = calculate_nodes_proximities(model, node)
        neighbors_node_node[node] = map(int, neighbors)
        weights_node_node[node] = weights
    return neighbors_node_node, weights_node_node


def calculate_nodes_proximities(model, node):
    neighbors, weights = zip(*model.wv.most_similar(positive=str(node), topn=args.K))
    neighbors = np.asarray(neighbors)
    weights = np.asarray(weights)
    idxs = np.ix_(np.where(weights > 0)[0])
    neighbors = neighbors[idxs]
    weights = weights[idxs]
    weights = normalize(weights)
    return neighbors, weights


def nodes_walk(start_node):
    seq = [start_node]
    while len(seq) < args.walk_length_trans:
        cur = seq[-1]
        if len(neighbors_node_node[cur]) > 0:
            if args.weighted:
                seq.append(np.random.choice(neighbors_node_node[cur], p=weights_node_node[cur]))
            else:
                seq.append(random.choice(neighbors_node_node[cur]))
        else:
            break
    return seq


def generate_B():
    neighbors_node_group = {}
    weights_node_group = {}
    neighbors_group_node = {}
    weights_group_node = {}
    # compute tf-idf values for group members
    groups_str = []
    for group in groups:
        groups_str.append(' '.join(str(i) for i in group))
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(groups_str))
    tfidf = tfidf.toarray()
    word = np.array(vectorizer.get_feature_names())
    for i in range(num_groups):
        idxs = np.ix_(np.where(tfidf[i] > 0)[0])
        weights_group_node[i + num_nodes] = normalize(tfidf[i][idxs])
        neighbors_group_node[i + num_nodes] = map(int, word[idxs])
    for i in range(num_nodes):
        node = int(word[i])
        idxs = np.ix_(np.where(tfidf[:, i] > 0)[0])
        neighbors_node_group[node] = list(idxs[0] + num_nodes)
        weights_node_group[node] = normalize(tfidf[idxs, i][0])
    return neighbors_node_group, weights_node_group, neighbors_group_node, weights_group_node


def generate_group_corpus():
    walks = []
    # nodes = range(num_nodes + num_groups)
    group_ids = neighbors_group_node.keys()
    pool = Pool(args.num_workers)
    for _ in tqdm(range(args.num_walks)):
        walks += pool.map(walker, group_ids)
    pool.close()
    pool.join()
    random.shuffle(walks)
    walks = [map(str, walk) for walk in walks]
    return walks


def walker(start_node):
    walk = [start_node]
    while len(walk) < args.walk_length:
        cur = walk[-1]
        if cur >= num_nodes:
            walk.append(np.random.choice(neighbors_group_node[cur], p = weights_group_node[cur]))
        elif random.random() > args.threshold and len(neighbors_node_node[cur]) > 0:
            walk.append(np.random.choice(neighbors_node_node[cur], p = weights_node_node[cur]))
        elif len(neighbors_node_group[cur]) > 0:
            walk.append(np.random.choice(neighbors_node_group[cur], p = weights_node_group[cur]))
        else:
            break
    walk = np.asarray(walk)
    return walk[np.ix_(np.where(walk >= num_nodes)[0])]


def compute_embs(corpus):
    model = Word2Vec(corpus, size=args.dimension, window=args.window_size, min_count=0, workers=args.num_workers)
    group_embs = []
    for gid in range(num_groups):
        group_embs.append(model[str(num_nodes + gid)])
    group_embs = preprocessing.normalize(group_embs, norm = 'l2')
    return np.array(group_embs)


def output_embs():
    np.save(args.group_embs, embs)


if __name__ == '__main__':
    args = parse_args()

    # loading the data
    G = load_network(args.network_format, args.network)
    groups = load_groups(args.groups)
    nodes = G.nodes()
    num_nodes = G.number_of_nodes()
    num_groups = len(groups)

    # generating the transition matrices
    neighbors_node_node, weights_node_node = generate_A_adjacency()
    if args.deepwalk:
        neighbors_node_node, weights_node_node = generate_A_deepwalk()
    neighbors_node_group, weights_node_group, neighbors_group_node, weights_group_node = generate_B()

    # generating group corpus
    corpus = generate_group_corpus()

    # embedding
    embs = compute_embs(corpus)

    output_embs()

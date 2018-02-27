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
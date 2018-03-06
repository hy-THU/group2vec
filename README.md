# group2vec
Group2vec and group2vec++ by Yu Han (yuhanthu@126.com)

## Input files
- network:\
This is the network input file. You should specify the format.

```--network_format adj_list``` for an adjacency list, e.g.:
```
0 1 2
1 2
2 3
...
```
```--network_format edge_list``` for an edge list, e.g.:
```
0 1
0 2
1 2
2 3
...
``````
```--network_format adj_matrix``` for a numpy adjacency matrix.

- groups:\
This file denotes the membership of the nodes for each group, such as
```
0 1 2 3 4 5
1 3 6 7
...
```
where each line contains all the member nodes of a group. The line indexes correspond to the group indexes starting with 0 by default.

## Output files
- group_embs:\
This file stores the group embeddings learned by group2vec or group2vec++. The results are in the form of a numpy array.

## Usage
```angular2html
python group2vec.py/group2vec++.py [options]
```

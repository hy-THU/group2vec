# group2vec
Group2vec and group2vec++ by Yu Han (yuhanthu@126.com)

##Input files
- edge_list:\
It is the network topology represented by an edge list, such as
```
0 1
1 2
1 3
...
```
where each line contains two node indexes separated by a space.
- group_members:\
This file denotes the membership of the nodes for each group, such as
```
0 1 2 3 4 5
1 3 6 7
...
```
where each line contains all the member nodes of a group. The line indexes correspond to the group indexes starting with 0 by default.

##Output files
- group_embs:\
This file stores the group embeddings learned by group2vec or group2vec++

##Usage
```angular2html
python group2vec.py/group2vec++.py [options]
```

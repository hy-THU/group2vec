def normalize(X):
    Sum = sum(X)
    X = [x / Sum for x in X]
    return X

def load_data(path):
    groups = []
    f = open(path, 'r')
    for line in f:
        line = line.strip().split()
        groups.append(map(int, line))
    f.close()
    return groups
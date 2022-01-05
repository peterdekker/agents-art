import numpy as np
import io
import time

np.random.seed(11)

def map_int(string_list):
    return [int(s) for s in string_list]

def flatten(t):
    return [item for sublist in t for item in sublist]

def load_vectors(fname, read_first_words=10000, sample_words=1000):
    # t0 = time.time()
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, dim = map(int, fin.readline().split())
    file_data = []
    words = []
    sample_indices = np.random.choice(np.arange(read_first_words), sample_words, replace=False)
    for i,line in enumerate(fin):
        if i >= read_first_words:
            break
        if i in sample_indices: # Maybe inefficient if sample_words is high
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            words.append(word)
            bin_values = [f"{int(decimal_value):064b}" for decimal_value in tokens[1:]]
            file_data.append(bin_values)
    data_flatten = "".join(flatten(file_data))
    data_int = map_int(list(data_flatten))
    assert len(data_int)== dim*sample_words
    data_array = np.array(data_int).reshape(sample_words, dim)
    # Split array into n_words. This also joins the groups of 4 longs that represent one vector together
    # t1 = time.time()
    # print(t1-t0)
    return data_array, words, dim

# def load_vectors(fname):
#     t0 = time.time()
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n_words, d = map(int, fin.readline().split())
#     data = []
#     words = []
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         word = tokens[0]
#         words.append(word)
#         bin_values = [map_int(list(f"{int(decimal_value):064b}")) for decimal_value in tokens[1:]]
#         data.append(bin_values)
#     data_array = np.array(data)
#     print(data_array.shape)
#     t1 = time.time()
#     print(t1-t0)
#     return data

# TODO: also load real-valued vectors, to compare
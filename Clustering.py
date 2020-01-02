import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
#from sklearn.cluster import KMeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.utils import read_sample

def data_preparation(n_cluster, data):
    #initial_medians = [[0.0, 0.1], [2.5, 0.7]]
    # For now we will intialize the intial_medians to the first n_cluster points in the data and just shuffle the data array
    np.random.shuffle(data)
    intial_medians = data[:n_cluster]
    data = read_sample(data)
    kmedians_instance = kmedians(data, initial_medians,itermax=300)
    kmedians_instance.process()
    return kmedians_instance




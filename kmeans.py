import numpy as np
import sys


threshold = 0.1
converged = False

# sys.argv[2] is second argument -- path to file
dataset = np.loadtxt(str(sys.argv[2]), dtype=np.float)
k = int(sys.argv[1])
n = dataset.shape[0]

# get k random centroids from dataset (one centroid is one row from the dataset)
ying = dataset[np.random.choice(n, size=k, replace=False), :]
yang = np.zeros(shape=(k, dataset.shape[1]))
# x, y are two arrays of same length
def distance(x, y):
    return np.linalg.norm(x-y)

def calculate_means(cluster_sums, cluster_size):
    for i in range(len(cluster_size)):
        cluster_sums[i] /= cluster_size[i]
    return cluster_sums

# partitioning -- which cluster does the datapoint belong to?
clustering = np.full(n, fill_value=-1, dtype=np.int)

# distance between data point and centroid
distance_d_centroid = np.zeros(k, dtype=np.float)
changed_datapoints = n
while changed_datapoints > n*threshold:
    changed_datapoints = 0
    for i, d in enumerate(dataset):
        for j, c in enumerate(ying):
            distance_d_centroid[j] = distance(d, c)
        argmin = np.argmin(distance_d_centroid)
        if argmin != clustering[i]:
            changed_datapoints += 1
            clustering[i] = argmin
        yang[argmin] += d
    cluster_size = np.histogram(clustering, bins=range(k+1))[0]  # how big is each cluster?
    print('cluster_size:', cluster_size)
    print('yang:', yang)
    yang = calculate_means(yang, cluster_size)  # calculate new centroid
    ying, yang = yang, ying  # swap centroids
    yang = np.zeros(shape=(k, dataset.shape[1]))



    # print(np.histogram(clustering, bins=range(k))[0]) # how big is each cluster?
   # print(n)
   # print(np.sum(np.histogram(clustering, bins=range(k))[0]))
   # print("test")





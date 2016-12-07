import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import sys

def kmeans(dataset, k, threshold=0.001):
    n = dataset.shape[0]
    #ying/yang --> centroid array double buffer
    ying = dataset[np.random.choice(n, size=k, replace=False), :] #choose k initial centroids at random
    yang = np.zeros(shape=(k, dataset.shape[1]))

    #euclidian distance
    def distance(x, y):
        return np.linalg.norm(x-y)

    def calculate_means(cluster_sums, cluster_size):
        for i in range(len(cluster_size)):
            cluster_sums[i] /= cluster_size[i]
        return cluster_sums

    # data partition induced by kmeans clustering
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
            if argmin != clustering[i]: #assign point to cluster of the closest centroid
                changed_datapoints += 1
                clustering[i] = argmin
            yang[argmin] += d
        cluster_size = np.histogram(clustering, bins=range(k+1))[0]  # how big is each cluster?
        yang = calculate_means(yang, cluster_size)  # calculate new centroids
        ying, yang = yang, ying  # swap centroid arrays
        yang = np.zeros(shape=(k, dataset.shape[1]))

    return clustering+1 #off by one regularization

def purity(clustering, k1, labels, k2):
    frequencies = np.zeros((k1,k2), dtype=np.int)
    n = len(clustering)
    purity = 0.0
    for i in range(n):
        frequencies[clustering[i]-1][labels[i]-1] += 1
    for i in range(k2):
        purity += np.max(frequencies[i])
    return purity / n

k = int(sys.argv[1])
# sys.argv[2] is second argument -- path to file
dataset = np.loadtxt(str(sys.argv[2]), dtype=np.float)#
labels  = dataset[:,-1].astype(np.int)
dataset = np.delete(dataset,-1,1)

clustering = kmeans(dataset, k)
#print labels
#print clustering

print "purity:\t", purity(clustering, k, labels, k)
print "RI:\t", metrics.adjusted_rand_score(clustering, labels) #TODO implement this
print "NMI:\t", metrics.normalized_mutual_info_score(clustering, labels)

plt.close('all')
f, axarr = plt.subplots(2, sharex=True)
axarr[0].scatter(dataset[:,0], dataset[:,1], c=labels)
axarr[0].set_title('labels true')
axarr[1].scatter(dataset[:,0], dataset[:,1], c=clustering)
axarr[1].set_title('labels pred')
plt.show()

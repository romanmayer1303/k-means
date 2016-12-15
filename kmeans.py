import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import sys
import time

#euclidian distance
def distance(data, centroid):
    return np.sum(np.square(np.subtract(data, centroid)), axis=1)

#assign data objects to clusters
def assign(data, centroids):
    distances = []
    for c in centroids:
        distances.append(distance(data, c))
    return np.argmin(np.vstack(distances), axis=0)

#how many elements transitioned to another cluster
def num_changed(old, new):
    return np.sum(np.not_equal(old, new))

#evaluate mean centroids
def update_centroids(data, clustering):
    centroids = []
    for c in range(np.max(clustering)+1):
        mask = np.equal(clustering, c)
        if np.sum(mask) == 0:
            print "ERROR: empty cluster"
            sys.exit(1)
        centroids.append(np.sum(data[mask], axis=0)/np.sum(mask))
    return np.stack(centroids)

def plus_plus(data, k):
    centroids = []
    centroids.append(data[np.random.choice(data.shape[0]), :])
    for i in range(1, k):
        distances = []
	for c in centroids:
	    distances.append(np.square(distance(data, c)))
	min_dist  = np.min(np.stack(distances), axis=0)
	min_dist /= np.sum(min_dist)
	centroids.append(data[np.random.choice(data.shape[0], p=min_dist), :])
    return np.stack(centroids)
	    

def kmeans(data, k, threshold=0.001, plusplus=False):
    n = data.shape[0] #num data objects
    d = data.shape[1] #data dimension
    #clustering arrays (double buffer)
    ying = np.full(n, fill_value=-1, dtype=np.int)
    yang = np.array(ying)

    if plusplus:
	centroids = plus_plus(data, k)
    else:
        centroids = data[np.random.choice(n, size=k, replace=False), :]

    while True: #do while workaround
        ying = assign(data, centroids)
        #if converged
        if num_changed(ying, yang) <= n*threshold:
            break
        centroids = update_centroids(data, ying)
        #swap double buffer
        ying, yang = yang, ying
    return ying, centroids

def mini_batch_kmeans(data, k, batch=0.1, iterations=100, plusplus=False):
    n = data.shape[0] #num data objects
    b = int(batch*n)  #batch size
    d = data.shape[1] #data dimension
    #clustering arrays (double buffer)
    clustering = np.full(b, fill_value=-1, dtype=np.int)

    if plusplus:
	centroids = plus_plus(data, k)
    else:
        centroids = data[np.random.choice(n, size=k, replace=False), :]

    i = 0
    while True:
        batch = data[np.random.choice(n, size=b, replace=False), :]
        clustering = assign(batch, centroids)
        if i == iterations:
            break
        i += 1
        centroids = update_centroids(batch, clustering)
    return assign(data, centroids), centroids


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

t_start = time.clock()

clustering, centroids = kmeans(dataset, k, plusplus=True)
#clustering, centroids = mini_batch_kmeans(dataset, k, float(sys.argv[3]), int(sys.argv[4]))

t_elapsed = time.clock()-t_start

clustering += 1 #regularization with labels

print "purity:\t", purity(clustering, k, labels, k)
print "RI:\t", metrics.adjusted_rand_score(clustering, labels)
print "NMI:\t", metrics.normalized_mutual_info_score(clustering, labels)
print "TIME:\t", t_elapsed

sys.exit(0) #print and plot

#b = sys.argv[3]+"_"+sys.argv[4]
b = "kmeansplusplus"
print str(sys.argv[2])+"_"+str(b)
with open(str(sys.argv[2])+"_"+str(b), "a") as f:
    f.write(str(purity(clustering, k, labels, k))+"\t"+str(metrics.adjusted_rand_score(clustering, labels))+"\t"+str(metrics.normalized_mutual_info_score(clustering, labels))+"\t"+str(t_elapsed)+"\n")

plt.close('all')
f, axarr = plt.subplots(2, sharex=True)
axarr[0].scatter(dataset[:,0], dataset[:,1], c=labels)
axarr[0].set_title('labels true')
axarr[1].scatter(dataset[:,0], dataset[:,1], c=clustering)
axarr[1].set_title('labels pred')
plt.show()

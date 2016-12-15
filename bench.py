import os
import sys
from tqdm import tqdm
import numpy as np

ba = ["0.1", "0.2", "0.3", "0.4", "0.5"]
it = ["5", "10", "20", "30", "40"]

for i in tqdm(ba):
    for j in tqdm(it):
        for _ in tqdm(range(100)):
            os.system("python kmeans.py 31 datasets/D31.txt "+i+" "+j)
        f = "datasets/D31.txt_"+i+"_"+j
        data = np.loadtxt(f)
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        np.savetxt(f+"_", np.stack((mu, sigma), axis=-1), delimiter='\t', fmt='%1.4f')

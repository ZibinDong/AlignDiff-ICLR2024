import matplotlib.pyplot as plt
import pickle
import numpy as np

task = "humanoid"
label_type = "syn"

mae = []
for seed in range(3):
    with open(f'results/evaluation/{task}_{label_type}_{seed}.pkl', 'rb') as f:
        data = pickle.load(f)
    mae.append(data["mae"])

mae = np.array(mae)[None,...]
n_algos, n_seeds, n_samples, n_seg = mae.shape
area = np.empty((n_algos,n_seeds,n_seg))
x = np.exp(np.linspace(0, -4.6, 90))
x = np.concatenate([x, np.linspace(x[-1], 0, 10)])
dx = x - np.concatenate([x[1:], x[-1:]*0])
for i in range(n_algos):
    for seed in range(n_seeds):
        for t in range(n_seg):
            area[i,seed,t] = np.array([(np.where(mae[i,seed][:,:(t+1)].min(-1)<j)[0]).size/n_samples for j in x]).sum()*1/100
area_mean = area.mean(1)
area_std = area.std(1)
print(area_mean, area_std)
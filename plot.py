import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="walker")
    parser.add_argument("--label_type", type=str, default="syn")
    
    task = parser.parse_args().task
    label_type = parser.parse_args().label_type

    mae = []
    file_names = os.listdir(f'results/evaluation/')
    file_names = [f for f in file_names if f.startswith(f'{task}_{label_type}')]
    for f in file_names:
        with open(f'results/evaluation/{f}', 'rb') as f:
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
    print(f'Mean: {area_mean[0,-1]:.3f}, Std: {area_std[0,-1]:.3f}')
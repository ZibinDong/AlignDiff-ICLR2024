import argparse
import os
import pickle

import environments
import dmc2gym
import gym
import h5py
import numpy as np
import torch

from diffusion import ODE, Planner
from utils import (AttrFunc, GaussianNormalizer, count_parameters,
                   set_mujoco_state, set_seed)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="walker")
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--label_type", type=str, default="syn")

    device = parser.parse_args().device
    task = parser.parse_args().task
    label_type = parser.parse_args().label_type
    seed = parser.parse_args().seed
    
    traj_len = 32 if task == "walker" else 100
    episode_len = 500 if task == "hopper" else 1000
    seg_len = 50 if task == "walker" else 100
    n_segs = 3 if task == "walker" else 1
    if task == "humanoid": model_size_cfg = {"d_model": 512, "n_heads": 8, "depth": 14}
    else: model_size_cfg = {"d_model": 384, "n_heads": 6, "depth": 12}
    
    n_tests = 100

    set_seed(seed)

    def generate_random_mask(n, p):
        while True:
            cond_mask = np.random.choice([0, 1], size=n, p=[1-p, p])
            if np.any(cond_mask): return cond_mask

    # Load dataset to create normalizer
    print(f'Load dataset for {task}-{label_type}...')
    with h5py.File(f"datasets/behaviors/{task}.hdf5", "r") as f:
        _obs, _act = f["observations"][:], f["actions"][:]
    with h5py.File(f"datasets/attr_label/{task}_{label_type}.hdf5", "r") as f:
        attr_ds = torch.from_numpy(f["attr_strength"][:]).float().to(device)
        max_attr, min_attr = attr_ds.max(0)[0], attr_ds.min(0)[0]
    obs_ds, act_ds = torch.FloatTensor(_obs).to(device), torch.FloatTensor(_act).to(device)
    normalizer = GaussianNormalizer(obs_ds)
    nor_obs_ds = normalizer.normalize(obs_ds)
    attr_dim = attr_ds.shape[-1]
    x_ds = torch.cat([nor_obs_ds, act_ds], dim=-1)
    sigma_data = x_ds.std().item()
    print(f'max_attr:{max_attr} min_attr:{min_attr}')

    # Load initial states
    with h5py.File(f"datasets/behaviors/{task}.hdf5", "r") as f:
        if task == "walker": 
            dataset_states = f["states"][:]
            dataset_obs = f["observations"][:]
        elif task == "humanoid":
            dataset_qpos = f["qpos"][:]
            dataset_qvel = f["qvel"][:]
            dataset_obs = f["observations"][:]

    if task == "walker":
        env = dmc2gym.make(domain_name="walker", task_name="walk", seed=seed, visualize_reward=False)
    elif task == "hopper":
        env = gym.make("MO-Hopper-v2")
    elif task == "humanoid":
        env = gym.make("MO-Humanoid-v4")
    o_dim, a_dim = env.observation_space.shape[0], env.action_space.shape[0]

    ode = ODE(o_dim, a_dim, attr_dim, sigma_data, device=device, **model_size_cfg)
    ode.load(f'{task}_{label_type}')
    ode.F.eval()
    attr_func = AttrFunc(o_dim, attr_dim).to(device)
    attr_func.load(f'{task}_{label_type}', device)
    attr_func.eval()

    planner = Planner(ode, attr_func, normalizer, max_attr, min_attr)

    o = env.reset()

    target_cond = np.empty((n_tests, attr_dim))
    cond_mask = np.empty((n_tests, attr_dim))
    real_cond = np.empty((n_tests, n_segs, attr_dim))
    mae = np.empty((n_tests, n_segs))
    length = seg_len*n_segs

    for e in range(n_tests):
        
        attr = np.random.rand(attr_dim)
        if task == "hopper": attr[1] = np.clip((1-attr[0])+(np.random.rand()-0.5)*(1.5/5.), 0., 1.)
        _attr_mask = generate_random_mask(attr_dim, 0.5)

        target_cond[e] = attr
        cond_mask[e] = _attr_mask
        
        print(f'[Eval {e+1}/{n_tests}] target attr: {attr} mask: {_attr_mask}')
        
        o = env.reset()
        if task == "walker":
            idx = np.random.randint(dataset_states.shape[0])
            env.physics.set_state(dataset_states[idx])
            env.physics.forward()
            o = dataset_obs[idx]
        elif task == "humanoid":
            idx = np.random.randint(dataset_qpos.shape[0])
            set_mujoco_state(env, dataset_qpos[idx], dataset_qvel[idx])
            o = dataset_obs[idx]
        
        traj = np.empty((1, length, o_dim))
        
        # walker 3.0 / hopper 2.0 / 
        # walker/ hopper 5  / humanoid 10
        for t in range(length):
            a, _ = planner.plan(o, traj_len, attr, _attr_mask, n_samples=256, w=3., sample_steps=10)
            o, r, d, info = env.step(a)
            traj[0, t] = o

        with torch.no_grad():
            traj = torch.from_numpy(traj).float().to(device)
            traj = normalizer.normalize(traj[0])[None,]
            for k in range(n_segs):
                pred_attr = attr_func.predict_attr(traj[:, seg_len*k:seg_len*(k+1)])
                normalized_attr = (pred_attr-min_attr[None,])/(max_attr-min_attr)[None,]
                real_cond[e, k] = normalized_attr.cpu().numpy()[0]
                mae[e, k] = ((_attr_mask/_attr_mask.sum())*np.abs(real_cond[e, k]-target_cond[e])).sum()
            print(f'[Eval {e+1}/{n_tests}] Finished. real attr: {real_cond[e]}. MAE: {mae[e]}')
            
    save_file_name = f"{task}_{label_type}_{seed}.pkl"
    if not os.path.exists(f"results/evaluation"): os.makedirs(f"results/evaluation")
    with open(f"results/evaluation/"+save_file_name, "wb") as f:
        data = {"target_cond": target_cond, "real_cond": real_cond, "mae": mae, "cond_mask": cond_mask}
        pickle.dump(data, f)
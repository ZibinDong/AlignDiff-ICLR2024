import argparse

import h5py
import numpy as np
import torch
from tqdm import tqdm

from diffusion import ODE
from utils import GaussianNormalizer, count_parameters, set_seed

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="walker")
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--label_type", type=str, default="syn")
    parser.add_argument("--n_gradient_steps", type=int, default=1000_000)
    parser.add_argument("--batch_size", type=int, default=32)
    
    device = parser.parse_args().device
    task = parser.parse_args().task
    label_type = parser.parse_args().label_type
    traj_len = 32 if task == "walker" else 100
    episode_len = 500 if task == "hopper" else 1000
    seed = parser.parse_args().seed
    n_gradient_steps = parser.parse_args().n_gradient_steps
    batch_size = parser.parse_args().batch_size
    
    if "humanoid" in task: model_size_cfg = {"d_model": 512, "n_heads": 8, "depth": 14}
    else: model_size_cfg = {"d_model": 384, "n_heads": 6, "depth": 12}
    
    set_seed(seed)

    print(f'Load dataset for {task}-{label_type}...')
    with h5py.File(f"datasets/behaviors/{task}.hdf5", "r") as f:
        _obs, _act = f["observations"][:], f["actions"][:]
        _tml = f["terminals"][:] if "terminals" in f else None
    with h5py.File(f"datasets/attr_label/{task}_{label_type}.hdf5", "r") as f:
        attr_ds = torch.from_numpy(f["attr_strength"][:]).float().to(device)
        attr_idx_ds = torch.from_numpy(f["attr_indices"][:]).long().to(device)
        max_attr, min_attr = attr_ds.max(0)[0], attr_ds.min(0)[0]
    valid_idx = np.empty(((_obs.shape[0]//episode_len)*(episode_len-traj_len),),dtype=np.int32)
    sample_weights = np.ones_like(valid_idx,dtype=np.float32)
    if _tml is None:
        for i in range(_obs.shape[0] // episode_len):
            valid_idx[i*(episode_len-traj_len):(i+1)*(episode_len-traj_len)] = np.arange(i*episode_len, (i+1)*episode_len-traj_len)
            if "humanoid" in task:
                sample_weights[i*(episode_len-traj_len):i*(episode_len-traj_len)+100] = 20.
    else:
        ptr = 0
        end_point = np.where(_tml == 1.)[0]
        for p in end_point:
            valid_idx += list(range(ptr, p-traj_len+2))
            ptr = p+1
    obs_ds, act_ds = torch.FloatTensor(_obs).to(device), torch.FloatTensor(_act).to(device)
    normalizer = GaussianNormalizer(obs_ds)
    nor_obs_ds = normalizer.normalize(obs_ds)
    o_dim, a_dim, attr_dim = obs_ds.shape[-1], act_ds.shape[-1], attr_ds.shape[-1]
    x_ds = torch.cat([nor_obs_ds, act_ds], dim=-1)
    sigma_data = x_ds.std().item()
    print(f'size:{valid_idx.shape[0]} max_attr:{max_attr} min_attr:{min_attr}')

    ode = ODE(o_dim, a_dim, attr_dim, sigma_data, device=device, **model_size_cfg)
    print(f'Initialized ODE model with {count_parameters(ode.F)} parameters.')
    print(f'Begins training for {task}-{label_type}...')
    loss_avg = 0.
    pbar = tqdm(range(n_gradient_steps))
    for step in range(n_gradient_steps):
        if "humanoid" in task: idx = np.random.choice(valid_idx.shape[0], batch_size, p=sample_weights/sample_weights.sum())
        else: idx = np.random.randint(0, valid_idx.shape[0], batch_size)
        raw_idx = valid_idx[idx]
        x, attr = x_ds[raw_idx[:,None]+np.arange(traj_len)[None,:]], attr_idx_ds[idx]
        loss, grad_norm = ode.update(x, attr)
        loss_avg += loss
        if (step+1) % 1000 == 0:
            pbar.set_description(f'step: {step+1} loss: {loss_avg / 1000.} grad_norm: {grad_norm}')
            pbar.update(1000)
            loss_avg = 0.
            ode.save(f'{task}_{label_type}')
    print(f'Finished!')
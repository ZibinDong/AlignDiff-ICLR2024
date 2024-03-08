import argparse
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils import AttrFunc, GaussianNormalizer, get_grid_indices, set_seed

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="walker")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--label_type", type=str, default="syn")
    parser.add_argument("--n_gradient_steps", type=int, default=3000)
    
    seed = parser.parse_args().seed
    task = parser.parse_args().task
    device = parser.parse_args().device
    label_type = parser.parse_args().label_type
    n_gradient_steps = parser.parse_args().n_gradient_steps
    traj_len = 32 if task == "walker" else 100
    episode_len = 500 if task == "hopper" else 1000
    
    set_seed(seed)
    # Create normalizer
    with h5py.File("datasets/behaviors/"+task+".hdf5", "r") as f:
        raw_obs = torch.FloatTensor((f["observations"][:])).to(device)
        tml = torch.FloatTensor(f["terminals"][:]).to(device) if "terminals" in f else None
    normalizer = GaussianNormalizer(raw_obs)

    # Prepare dataset
    with h5py.File(f"datasets/feedbacks/{task}_{label_type}_train.hdf5", "r") as f:
        pref_ds = torch.FloatTensor((f["pref"][:]+1.)/2.).to(device)
        obs1_ds, obs2_ds = torch.FloatTensor(f["obs1"][:]).to(device), torch.FloatTensor(f["obs2"][:]).to(device)
    nor_obs1_ds, nor_obs2_ds = normalizer.normalize(obs1_ds), normalizer.normalize(obs2_ds)
    
    if label_type == "syn":
        with h5py.File(f"datasets/feedbacks/{task}_{label_type}_eval.hdf5", "r") as f:
            eval_pref_ds = torch.FloatTensor((f["pref"][:]+1.)/2.).to(device)
            eval_obs1_ds, eval_obs2_ds = torch.FloatTensor(f["obs1"][:]).to(device), torch.FloatTensor(f["obs2"][:]).to(device)
        nor_eval_obs1_ds, nor_eval_obs2_ds = normalizer.normalize(eval_obs1_ds), normalizer.normalize(eval_obs2_ds)

    o_dim, attr_dim = nor_obs1_ds.shape[-1], pref_ds.shape[-1]

    # Create attribute function
    attr_func = AttrFunc(o_dim, attr_dim).to(device)
    attr_func.train()
    
    # Train attribute function
    print(f"Start training {task} attribute function with {label_type} labels.")
    mean_sr, ckpt_mean_sr, erly_stop_cnt = 0., 0., 0
    pbar = tqdm(range(n_gradient_steps))
    for n in range(n_gradient_steps):
        log = {"loss": np.zeros(attr_func.ensemble_size)}
        for k in range(attr_func.ensemble_size):
            idx = torch.randint(obs1_ds.shape[0], (256,), device=device)
            pref, obs1, obs2 = pref_ds[idx], nor_obs1_ds[idx], nor_obs2_ds[idx]
            loss = attr_func.update(obs1, obs2, pref, k)
            log["loss"][k] += loss
            
        if (n+1) % 100 == 0 and label_type == "syn":
            with torch.no_grad():
                attr_func.eval()
                prob = attr_func.predict_pref_prob(nor_eval_obs1_ds, nor_eval_obs2_ds)
                mean_sr = ((prob >= 0.5) == (eval_pref_ds == 0)).float().mean(0).cpu().numpy()
                attr_func.train()
            log["mean_sr"] = mean_sr
            pbar.update(100)
            pbar.set_description(f"Loss {log['loss']/100.} Mean success rate: {mean_sr}")
            log = {"loss": np.zeros(attr_func.ensemble_size)}
            ckpt_mean_sr = mean_sr.copy()
            attr_func.save(f'{task}_{label_type}')
    print(f"Finish training.")

    # Relabel behavior dataset
    print(f"Relabeling {task} behavior dataset.")
    attr_func.eval()
    nor_raw_obs = normalizer.normalize(raw_obs)
    with torch.no_grad():
        if tml is None:
            n_episode = raw_obs.shape[0] // episode_len
            attr_stength = []
            for e in tqdm(range(n_episode)):
                batch_traj = torch.empty((episode_len-traj_len, traj_len, o_dim)).to(device)
                for t in range(episode_len - traj_len):
                    batch_traj[t] = nor_raw_obs[episode_len*e+t:episode_len*e+t+traj_len]
                attr_stength.append(attr_func.predict_attr(batch_traj, None))
            attr_stength = torch.cat(attr_stength, dim=0).cpu().numpy()
        else:
            valid_idx = []
            attr_stength = []
            for i in range(raw_obs.shape[0]):
                if i <= raw_obs.shape[0] - traj_len:
                    if tml[i:i+traj_len].sum() == 0. or tml[i+traj_len-1] == 1.:
                        valid_idx.append(i)
            valid_idx = np.array(valid_idx)
            ptr = 0
            while ptr < len(valid_idx):
                batch_traj = torch.empty(
                    min(1000, len(valid_idx)-ptr), traj_len, o_dim).to(device)
                for i in range(min(1000, len(valid_idx)-ptr)):
                    batch_traj[i] = nor_raw_obs[valid_idx[ptr+i]:valid_idx[ptr+i]+traj_len]
                attr_stength.append(attr_func.predict_attr(batch_traj, None))
                ptr += 1000
            attr_stength = torch.cat(attr_stength, dim=0).cpu().numpy()
        
    attr_max = attr_stength.max(axis=0)[None,]
    attr_min = attr_stength.min(axis=0)[None,]
    indices = get_grid_indices(attr_stength, attr_min, attr_max, 100)
    file_path = "datasets/attr_label"
    if not os.path.exists(file_path): os.makedirs(file_path)
    with h5py.File(f"{file_path}/{task}_{label_type}.hdf5", "w") as f:
        f["attr_strength"] = attr_stength
        f["attr_indices"] = indices
    print(f"Finish relabeling.")
    
    
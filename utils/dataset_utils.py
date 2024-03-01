from pathlib import Path
from typing import Optional, Union
import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import scipy.interpolate as interpolate
    

class GaussianNormalizer():
    def __init__(self, x: torch.Tensor):
        self.mean, self.std = x.mean(0), x.std(0)
        self.std[torch.where(self.std==0.)] = 1.
    def normalize(self, x: torch.Tensor):
        return (x - self.mean[None,]) / self.std[None,]
    def unnormalize(self, x: torch.Tensor):
        return x * self.std[None,] + self.mean[None,]

class TruncatedNormalizer():
    def __init__(self, x: torch.Tensor):
        self.low, self.high = x.min(0).values, x.max(0).values
    def normalize(self, x: torch.Tensor):
        return ((x - self.low[None,]) / (self.high[None,] - self.low[None,])) * 2. - 1.
    def unnormalize(self, x: torch.Tensor):
        return (x+1)/2 * (self.high[None,] - self.low[None,]) + self.low[None,]

# class AttrFuncDataset(Dataset):
#     ''' Dataset for attribute function training.
    
#     The dataset is loaded from a hdf5 file, which is supposed to contain the following keys:
#         - pref: (N, attr_dim) array, preference label, in {-1, 0, 1}.
#         - obs1: (N, seq_len, obs_dim) array, observations in trajectories 1.
#         - act1: (N, seq_len, act_dim) array, actions in trajectories 1.
#         - obs2: (N, seq_len, obs_dim) array, observations in trajectories 2.
#         - act2: (N, seq_len, act_dim) array, actions in trajectories 2.
#     '''
#     def __init__(self, filename="walker0246_vec_labelled_2k", normalizer: Optional[GaussianNormalizer] = None):
#         super().__init__()
#         self._load_file(filename)
#         if normalizer is not None:
#             self.obs_normalizer = normalizer
#             self._normalized_obs1 = normalizer.normalize(self._obs1)
#             self._normalized_obs2 = normalizer.normalize(self._obs2)
#         else:
#             self._normalized_obs1 = self._obs1
#             self._normalized_obs2 = self._obs2
        
#     def _load_file(self, filename):
#         '''
#         In order to align with Enhanced-RLHF Labeller, we suppose 
#         values in `preference` are in {-1, 0, 1} where 
#         -1 means obs1 is preferred, 
#         1 means obs2 is preferred, 
#         0 means no preference.
#         '''
#         file_path = Path(__file__).parents[1]/"datasets"/"attr_func"/filename
#         if not os.path.exists(file_path): raise FileNotFoundError(f'File {file_path} not found')
#         with h5py.File(file_path, "r") as f:
#             self._pref = (f["pref"][:]+1.)/2.
#             self._obs1 = f["obs1"][:]
#             self._act1 = f["act1"][:]
#             self._obs2 = f["obs2"][:]
#             self._act2 = f["act2"][:]
            
#     def __len__(self):
#         return self._pref.shape[0]
    
#     def __getitem__(self, index):
#         return (
#             self._pref[index],
#             self._normalized_obs1[index],
#             self._act1[index],
#             self._normalized_obs2[index],
#             self._act2[index],
#         )
    
# class DiffusionDataset(Dataset):
#     ''' Dataset for diffusion model training. Each batch contains a trajectory of length `traj_len` and its corresponding attribute.
    
#     Args:
#         - task: (str), name of the task.
#         - label_type: (str), type of label, in {"fake", "real"}.
#         - episode_len: (int), length of each episode.
#         - traj_len: (int), length of each trajectory.
#         - obs_normalizer: (str), type of observation normalizer, in {"gaussian", "cdf"}.
#         - act_normalizer: (str), type of action normalizer, in {"gaussian", "cdf", None}. If None, no normalization is applied.
#         - attr_type: (str), type of attribute, in {"discrete", "normalized", "raw"}. 
#         "discrete" means the attribute is a discrete label, "normalized" means the attribute is normalized to [0, 1], "raw" means the attribute is the raw value.
#     '''
#     def __init__(
#         self,
#         task: str = "walker",
#         label_type: str = "fake",
#         episode_len: int = 1000,
#         traj_len: int = 32,
#         obs_normalizer: str = "gaussian",
#         act_normalizer: Optional[str] = None,
#         attr_type: str = "discrete",
#     ):
#         super().__init__()
#         self.attr_type = attr_type
#         self.episode_len = episode_len
#         self.traj_len = traj_len
#         self.label_type = label_type
#         self.normalizer = obs_normalizer
#         self._load_file(task)
        
#         print(f"[DiffusionDataset] Creating normalizer...")
#         if obs_normalizer == "gaussian": self.obs_normalizer = GaussianNormalizer(self._obs)
#         elif obs_normalizer == "cdf": self.obs_normalizer = CDFNormalizer(self._obs)
        
#         if act_normalizer == "gaussian": self.act_normalizer = GaussianNormalizer(self._act)
#         elif act_normalizer == "cdf": self.act_normalizer = CDFNormalizer(self._act)
#         elif act_normalizer is None: self.act_normalizer = None
#         print(f"[DiffusionDataset] Normalizer created.")

#         print(f"[DiffusionDataset] Normalizing observations and actions...")
#         self._normalized_obs = self.obs_normalizer.normalize(self._obs)
#         self._normalized_act = self.act_normalizer.normalize(self._act) if act_normalizer is not None else self._act
#         self._normalized_obs = torch.from_numpy(self._normalized_obs).float()
#         self._normalized_act = torch.from_numpy(self._normalized_act).float()
#         print(f"[DiffusionDataset] Normalized.")

#     def _load_file(self, task: str):
#         with h5py.File(Path(__file__).parents[1]/"datasets/raw_datasets"/f"{task}.hdf5", "r") as f:
#             self._obs = f["observations"][:]
#             self._act = f["actions"][:]
#             self._tml = f["terminals"][:] if "terminals" in f else None
#         with h5py.File(Path(__file__).parents[1]/"datasets/attr_label"/f"{task}_{self.label_type}_{self.normalizer}_{self.traj_len}.hdf5", "r") as f:
#             self._attr = torch.from_numpy(f["attr_strength"][:]).float()
#             self._attr_idx = torch.from_numpy(f["attr_indices"][:]).long()
#             self.max_attr = self._attr.max(0)[0]
#             self.min_attr = self._attr.min(0)[0]
#         self.valid_idx = []
#         if self._tml is None:
#             weight = np.ones((self.episode_len-self.traj_len))
#             weight[:100] = 20.
#             self.weights = []
#             for i in range(self._obs.shape[0] // self.episode_len):
#                 self.valid_idx += list(range(i*self.episode_len, (i+1)*self.episode_len-self.traj_len))
#                 self.weights.append(weight)
#         else:
#             ptr = 0
#             end_point = np.where(self._tml == 1.)[0]
#             for p in end_point:
#                 self.valid_idx += list(range(ptr, p-self.traj_len+2))
#                 ptr = p+1
#         self.valid_idx = np.array(self.valid_idx)
#         self.weights = np.concatenate(self.weights)
#         print(f'[DiffusionDataset] size:{self.valid_idx.shape[0]} max_attr:{self.max_attr} min_attr:{self.min_attr}')
        
#     def __len__(self):
#         return self.valid_idx.shape[0]
    
#     def __getitem__(self, index):
#         raw_index = self.valid_idx[index]
#         if self.attr_type == "discrete":
#             attr_batch = self._attr_idx[index]
#         elif self.attr_type == "normalized":
#             attr_batch = (self._attr[index] - self.min_attr) / (self.max_attr - self.min_attr)
#         else:
#             attr_batch = self._attr[index]
#         return (
#             attr_batch,
#             self._normalized_obs[raw_index: raw_index+self.traj_len],
#             self._normalized_act[raw_index: raw_index+self.traj_len])
        
        
# class AddRewardDatasetWrapper(Dataset):
#     def __init__(
#         self,
#         diffusion_dataset: DiffusionDataset,
#         reward_func,
#     ):
#         self.dataset = diffusion_dataset
#         self.reward_func = reward_func
#         self._add_rew()
        
#     @torch.no_grad()
#     def _add_rew(self):
#         print("[AddRewardDatasetWrapper] Adding reward...")
#         self._rew = torch.empty((self.__len__(), 1))
#         self._cond_mask = torch.bernoulli(torch.empty_like(self.dataset._attr), 0.8)
#         n_batch = np.ceil(self.__len__() / 1000).astype(int)
#         for i in range(n_batch):
#             raw_idx = self.dataset.valid_idx[i*1000:(i+1)*1000]
#             self._rew[i*1000:(i+1)*1000] = \
#                 self.reward_func(
#                     self.dataset._normalized_obs[raw_idx].to(self.reward_func.device), 
#                     self.dataset._attr[i*1000:(i+1)*1000].to(self.reward_func.device),
#                     self._cond_mask[i*1000:(i+1)*1000].to(self.reward_func.device)).cpu()
#         print(self._rew.max())
#         print(self._rew.min())
#         print(self._rew.mean())
#         print(self._rew.std())
#         print("[AddRewardDatasetWrapper] Done.")
    
#     def __len__(self):
#         return self.dataset.__len__()
        
#     def __getitem__(self, index):
#         raw_index = self.dataset.valid_idx[index]
#         if self.dataset.attr_type == "discrete":
#             attr_batch = self.dataset._attr_idx[index]
#         elif self.dataset.attr_type == "normalized":
#             attr_batch = (self.dataset._attr[index] - self.dataset.min_attr) / (self.dataset.max_attr - self.dataset.min_attr)
#         else:
#             attr_batch = self.dataset._attr[index]
#         return (
#             attr_batch,
#             self._cond_mask[index],
#             self.dataset._normalized_obs[raw_index],
#             self.dataset._act[raw_index],
#             self.dataset._normalized_obs[raw_index+1],
#             self._rew[index],
#         )
        
        
# class GCPolicyDataset():
#     def __init__(
#         self,
#         traj_filename="walker_attr",
#         episode_len=1000):
#         super().__init__()
#         self.episode_len = episode_len
#         self._load_file(traj_filename)
        
#         self.obs_normalizer = GaussianNormalizer(self._obs)
#         self._normalized_obs = self.obs_normalizer.normalize(self._obs)
        
#         self.sample_p = self.generate_probabilities(self.episode_len-1, 5)

#     def _load_file(self, traj_filename):
#         with h5py.File(Path(__file__).parents[1]/"datasets/raw_datasets"/(traj_filename+".hdf5"), "r") as f:
#             self._obs = torch.from_numpy(f["observations"][:]).float()
#             self._act = torch.from_numpy(f["actions"][:]).float()
#             self._tml = torch.from_numpy(f["terminals"][:]).float() if "terminals" in f else None
#         self.n_episodes = self._obs.shape[0] // self.episode_len
            
#     def generate_probabilities(self, n, k):
#         p = np.linspace(1., 1./k, n)
#         return p/p.sum()

#     def sample(self, batch_size):
#         ep = np.random.randint(0, self.n_episodes, (batch_size,))
#         obs = np.empty((batch_size, self._obs.shape[1]))
#         act = np.empty((batch_size, self._act.shape[1]))
#         goal_obs = np.empty((batch_size, self._obs.shape[1]))
        
#         for i in range(batch_size):
#             idx = np.random.choice(self.episode_len-1, p=self.sample_p)
#             goal_idx = np.random.randint(idx+1, self.episode_len)
#             obs[i] = self._normalized_obs[ep[i]*self.episode_len+idx]
#             act[i] = self._act[ep[i]*self.episode_len+idx]
#             goal_obs[i] = self._normalized_obs[ep[i]*self.episode_len+goal_idx]
        
#         return obs, goal_obs, act 
    
    

# def empirical_cdf(sample):
#     quantiles, counts = np.unique(sample, return_counts=True)
#     cumprob = np.cumsum(counts).astype(np.double) / sample.size
#     return quantiles, cumprob

# def atleast_2d(x):
#     if x.ndim < 2:
#         x = x[:,None]
#     return x

# class Normalizer:
#     def __init__(self, X):
#         self.X = X.astype(np.float32)
#         self.mins = X.min(axis=0)
#         self.maxs = X.max(axis=0)
#     def __repr__(self):
#         return (
#             f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
#             f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n''')
#     def __call__(self, x):
#         return self.normalize(x)
#     def normalize(self, *args, **kwargs):
#         raise NotImplementedError()
#     def unnormalize(self, *args, **kwargs):
#         raise NotImplementedError()

# class CDFNormalizer(Normalizer):
#     def __init__(self, X):
#         super().__init__(atleast_2d(X))
#         self.dim = self.X.shape[1]
#         self.cdfs = [CDFNormalizer1d(self.X[:, i]) for i in range(self.dim)]
#     def __repr__(self):
#         return f'[ CDFNormalizer ] dim: {self.mins.size}\n' + '    |    '.join(
#             f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
#         )
#     def wrap(self, fn_name, x):
#         shape = x.shape
#         x = x.reshape(-1, self.dim)
#         out = np.zeros_like(x)
#         for i, cdf in enumerate(self.cdfs):
#             fn = getattr(cdf, fn_name)
#             out[:, i] = fn(x[:, i])
#         return out.reshape(shape)
#     def normalize(self, x):
#         return self.wrap('normalize', x)
#     def unnormalize(self, x):
#         return self.wrap('unnormalize', x)

# class CDFNormalizer1d:
#     def __init__(self, X):
#         assert X.ndim == 1
#         self.X = X.astype(np.float32)
#         if self.X.max() == self.X.min():
#             self.constant = True
#         else:
#             self.constant = False
#             quantiles, cumprob = empirical_cdf(self.X)
#             self.fn = interpolate.interp1d(quantiles, cumprob)
#             self.inv = interpolate.interp1d(cumprob, quantiles)
#             self.xmin, self.xmax = quantiles.min(), quantiles.max()
#             self.ymin, self.ymax = cumprob.min(), cumprob.max()
#     def __repr__(self):
#         return (f'[{np.round(self.xmin, 2):.4f}, {np.round(self.xmax, 2):.4f}')
#     def normalize(self, x):
#         if self.constant:
#             return x
#         x = np.clip(x, self.xmin, self.xmax)
#         y = self.fn(x)
#         y = 2 * y - 1
#         return y

#     def unnormalize(self, x, eps=1e-4):
#         if self.constant:
#             return x
#         x = (x + 1) / 2.
#         # if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
#             # print(
#             #     f'''[ dataset/normalization ] Warning: out of range in unnormalize: '''
#             #     f'''[{x.min()}, {x.max()}] | '''
#             #     f'''x : [{self.xmin}, {self.xmax}] | '''
#             #     f'''y: [{self.ymin}, {self.ymax}]'''
#             # )
#         x = np.clip(x, self.ymin, self.ymax)
#         y = self.inv(x)
#         return y
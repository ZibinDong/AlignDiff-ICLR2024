import environments
import gym
import matplotlib.pyplot as plt
import h5py
import numpy as np
from utils import set_mujoco_state
import numpy as np
import torch
import random


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)

for _ in range(4):
    print(np.random.rand(2))


# env = gym.make("MO-Hopper-v2")
# env = gym.make("MO-Humanoid-v4")

# o = env.reset()
# img = env.render("rgb_array")
# plt.imshow(img)
    
# with h5py.File("datasets/feedbacks/humanoid_syn_train.hdf5", "r") as f:
#     obs = f["obs1"][:]
#     act = f["act1"][:]
#     pref = f["pref"][:]

# with h5py.File("datasets/feedbacks/humanoid_noisy_syn_train.hdf5", "r") as f:
#    obs1, obs2, act1, act2 = f["obs1"][:], f["obs2"][:], f["act1"][:], f["act2"][:]
#    pref = f["pref"][:]
#with h5py.File("datasets/feedbacks/humanoid_noisy_syn_train.hdf5", "w") as f:
#    f['obs1'], f['obs2'], f['act1'], f['act2'] = obs2, obs1, act2, act1
#    f['pref'] = pref[:,:2]
    

# with h5py.File("datasets/behaviors/humanoid.hdf5", "w") as f:
#     f["actions"] = act
#     f["observations"] = obs
#     f["qpos"] = qpos
#     f["qvel"] = qvel
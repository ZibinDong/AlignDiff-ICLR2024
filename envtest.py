import environments
import gym
import matplotlib.pyplot as plt
import h5py
import numpy as np
from utils import set_mujoco_state

env = gym.make("MO-Hopper-v2")
env = gym.make("MO-Humanoid-v4")

# o = env.reset()
# img = env.render("rgb_array")
# plt.imshow(img)

with h5py.File("datasets/behaviors/humanoid.hdf5", "r") as f:
    act = f["actions"][:]
    obs = f["observations"][:]

with h5py.File("datasets/behaviors/humanoid_ori.hdf5", "r") as f:
    qpos = f["qpos"][:]
    qvel = f["qvel"][:]

with h5py.File("datasets/behaviors/humanoid.hdf5", "w") as f:
    f["actions"] = act
    f["observations"] = obs
    f["qpos"] = qpos
    f["qvel"] = qvel
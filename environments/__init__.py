from gym.envs.registration import register


register(
    id = 'MO-Hopper-v2',
    entry_point = 'environments.hopper:HopperEnv',
    max_episode_steps=500,
)

register(
    id = 'MO-Humanoid-v4',
    entry_point = 'environments.humanoid:MaskedHumanoidEnv',
    max_episode_steps=1000,
)
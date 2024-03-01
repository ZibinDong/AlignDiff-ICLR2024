import mujoco_py

    
def set_mujoco_state(env, qpos, qvel):
    state = env.sim.get_state()
    state = mujoco_py.MjSimState(state.time, qpos, qvel, state.act, state.udd_state)
    env.sim.set_state(state)
    env.sim.forward()
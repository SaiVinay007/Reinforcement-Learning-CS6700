from gym.envs.registration import register

register(
    id='four-v0',
    entry_point='gym_four.envs:FourEnv',
)

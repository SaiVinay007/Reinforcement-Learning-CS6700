from gym.envs.registration import register

register(
    id='pdw-v0',
    entry_point='gym_pdw.envs:PdwEnv',
)
register(
    id='pdw-extrahard-v0',
    entry_point='gym_pdw.envs:PdwExtraHardEnv',
)
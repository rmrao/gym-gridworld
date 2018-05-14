from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gym_gridworld.envs:GridworldEnv',
)

register(
    id='gridworld-v1',
    entry_point='gym_gridworld.envs:GridworldEnvV2'
)

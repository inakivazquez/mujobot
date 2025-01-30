from gymnasium.envs.registration import register

register(
    id='mujobot/ur5-reach-v0',
    entry_point='mujobot.envs.ur5_reach_v0:UR5ReachEnv',
    max_episode_steps=100,
)


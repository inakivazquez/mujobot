from gymnasium.envs.registration import register

register(
    id='mujobot/ur5-reach-v0',
    entry_point='mujobot.envs.ur5_reach_v0:UR5ReachEnv',
    max_episode_steps=100,
)

register(
    id='mujobot/ur5-gripper-v0',
    entry_point='mujobot.envs.ur5_gripper_v0:UR5GripperEnv',
    max_episode_steps=100,
)

register(
    id='mujobot/ur5-paddle-v1',
    entry_point='mujobot.envs.ur5_paddle_v1:UR5PaddleEnv',
    max_episode_steps=2000,
)
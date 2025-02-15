from __future__ import annotations
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
#from gymnasium.envs.mujoco import MujocoEnv
from mujobot.envs.mujoco_env import MujocoEnv
import mujoco
import numpy as np
import math
import os

class UR5PaddleEnv(MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple",],
    }

    def __init__(self, **kwargs):

        default_camera_config = {
            "distance": 2.5,
            "elevation": -30.0,
            "azimuth": 90.0,
            "lookat": [0.0, 0.0, 0.6],
        }

        screen_width = screen_height = 800

        MujocoEnv.__init__(
            self,
            model_path=os.path.dirname(os.path.abspath(__file__)) + "/ur5_paddle/scene.xml",
            frame_skip=5,
            observation_space=None,
            default_camera_config=default_camera_config,
            width=screen_width,
            height=screen_height,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # Create a data structure to hold the simulation state
        #self.data = mujoco.MjData(self.model)

        # Target pos, current pos of ee, joint positions
        self.observation_space = gym.spaces.Box(low=np.array([-2]*3 + [-2*math.pi]*6, dtype=np.float32), high=np.array([+2]*3 + [+2*math.pi]*6, dtype=np.float32), shape=(9,))

        # Action space is 6 joint angles and the gripper open/close level (0 to 1)
        action_max = 2*math.pi / 36
        self.action_space = gym.spaces.Box(low=np.array([-action_max]*6, dtype=np.float32), high=np.array([+action_max]*6, dtype=np.float32), shape=(6,))
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ball")
        self.paddle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "paddle")


    def reset_model(self):
        self.data.qpos[0:6] = [math.pi/2, -math.pi/2, 0, 0, 0, 0]
        # Starting position
        self.data.ctrl[0:6] = [math.pi/2, -math.pi/2, 0, 0, 0, 0]
        self.data.ctrl[6] = 255  # Gripper closed
        self.initial_position = self.data.qpos[0:6].copy()

        # Set random position of the ball 
        random_diff = np.random.uniform(low=[-0.05, -0.10, 0], high=[+0.05, +0.05, 0])
        ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        qpos_idx = self.model.jnt_qposadr[ball_joint_id]  # Get the index in qpos
        self.data.qpos[qpos_idx:qpos_idx+3] += random_diff

        # Set random speed of the ball
        # Ensure we get the correct `qvel` indices
        ball_dof = self.model.body_dofadr[self.ball_id]  # Index of velocity in `qvel`

        # Assign a random velocity (linear + angular)
        random_linear_velocity = np.random.uniform(-0.5, 0.5, size=3)  # Random velocity in x, y, z
        random_angular_velocity = np.random.uniform(-1, +1, size=3)  # Random spin
        random_linear_velocity[2] = 0 # No vertical velocity

        # Set the velocity in `qvel`
        self.data.qvel[ball_dof : ball_dof + 3] = random_linear_velocity  # Set linear velocity
        self.data.qvel[ball_dof + 3 : ball_dof + 6] = random_angular_velocity  # Set angular velocity

        return self.get_observation()

    def wait_until_stable(self, control, sim_steps=5, tolerance=1e-1):
        joint_pos = np.array(self.data.qpos[0:6], dtype=np.float32)

        for _ in range(sim_steps):
            self.do_simulation(control, self.frame_skip)
            if self.render_mode == "human":
                self.render()
            new_joint_pos = np.array(self.data.qpos[0:6], dtype=np.float32)

            if np.linalg.norm(new_joint_pos - joint_pos, ord=np.inf) < tolerance:
                return True
            
            joint_pos = new_joint_pos
            print(".", end="")
        print("Warning: The robot configuration did not stabilize")
        return False

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        new_control = np.array(self.data.qpos[0:6], np.float32) + np.array(action[0:6], np.float32)
        new_control = np.clip(new_control, -2*math.pi, +2*math.pi) # Clip final configuration to be within limits
        new_control = np.append(new_control, 255) # Gripper closed at all times

        self.do_simulation(new_control, self.frame_skip)

        ball_pos = self.get_ball_position()
        paddle_pos = self.get_paddle_position()
        distance = np.linalg.norm(paddle_pos - ball_pos)
        reward = +1 # Keep alive reward
        reward += paddle_pos[2] - 0.7 # Reward for keeping the paddle high
        terminated = False
        if ball_pos[2] < paddle_pos[2] - 0.1 or distance > 2: # Ball on the ground or too low, or jumped away fallen
            print("Ball fell!")
            reward = -500
            terminated= True
        truncated = False
        obs = self.get_observation()
        info = {}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def get_observation(self):
        ball_pos = self.get_ball_position()
        paddle_pos = self.get_paddle_position() # Geometrical center of the paddle
        pos_diff = paddle_pos - ball_pos
        joint_pos = self.data.qpos[0:6]
        return np.concatenate((pos_diff, joint_pos), dtype=np.float32)

    def get_ee_pose(self):
        pinch_name = "pinch"
        pinch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, pinch_name)
        ee_pos_absolute = self.data.site_xpos[pinch_id]
        rotation_matrix = self.data.xmat[pinch_id].reshape(3, 3)  # Extract 3×3 rotation matrix
        quat = [0]*4  # Convert to quaternion (x, y, z, w)

        # site_* methods to access global coordinates, other * methods (xpos, xquat) access local coordinates
        return ee_pos_absolute, quat
    
    def get_ball_position(self):
        return self.data.geom_xpos[self.ball_id]
    
    def get_paddle_position(self):
        return self.data.site_xpos[self.paddle_id]
        

import time
import mujobot
if __name__ == "__main__":
    env = gym.make("mujobot/ur5-paddle-v1", render_mode="human")
    for _ in range(100):
        print("Resetting")
        obs = env.reset()
        for _ in range(1000):
            action = [0]*6
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
    env.close()